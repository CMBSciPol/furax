import functools
from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._observation import ReaderField
from .acquisition import build_acquisition_operator
from .config import (
    GapTreatment,
    MapMakingConfig,
    Methods,
    NoiseSource,
    PolynomialOrders,
    TemplatesConfig,
    WeightingMode,
    _legendre_leg_groups,
)
from .gram import gram_inverse, shared_gram
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel, padding_aware_welch
from .pomme import PommeProjectionOperator
from .templates import (
    AbstractTemplateOperator,
    Basis,
    SharedBasis,
    StokesTemplateOperator,
    TemplateOperator,
    azimuth_hwp_synchronous_basis,
    binned_azimuth_hwp_synchronous_basis,
    binned_azimuth_synchronous_basis,
    common_mode_basis,
    hwp_synchronous_basis,
    is_basis,
    polynomial_basis,
    scan_synchronous_basis,
    spline_hwp_synchronous_basis,
    t2p_basis,
)
from .weight import NestedWeightOperator, WeightOperator


@register_dataclass
@dataclass
class ObservationModel:
    """Holds the operators and data for one or more observations.

    When stacked via ``jax.lax.scan``, the array fields carry a leading batch
    dimension over observations while static fields (structures, chunk sizes)
    remain shared.
    """

    H: AbstractLinearOperator
    """Acquisition operator"""

    W: WeightOperator | NestedWeightOperator
    """Weighting operator (noise weights + mask)"""

    F: AbstractLinearOperator
    """Pomme filter/deprojector (identity when not enabled)"""

    noise_model: PyTree[NoiseModel]
    """Noise model"""

    sample_rate: Array
    """Data sampling rate"""

    @classmethod
    def create(
        cls, data: Any, padding: Any, config: MapMakingConfig, landscape: StokesLandscape
    ) -> Self:
        H = build_acquisition_operator(
            landscape,
            Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
            Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
            data.get(ReaderField.HWP_ANGLES),
            demodulated=config.demodulated,
            pointing_on_the_fly=config.pointing.on_the_fly,
            pointing_batch_samples=config.pointing.batch_samples,
            pointing_interpolate=config.pointing.interpolation == 'bilinear',
            dtype=config.dtype,
        )
        tod_struct = H.out_structure
        M = _mask_projector(_sample_mask(data, config), structure=tod_struct)
        noise_model, sample_rate = _noise_model(
            data, config, tod_structure=tod_struct, padding=padding
        )
        Ninv = _noise_operator(
            noise_model,
            tod_struct,
            sample_rate,
            config.weighting.correlation_length,
            inverse=True,
        )
        W: WeightOperator | NestedWeightOperator
        if config.gaps.treatment == GapTreatment.NESTED and not config.binned:
            # Minimum-variance correlated-noise weight using iterative solve.
            cov = None
            if config.gaps.nested.precondition:
                # Precondition inner flagged-subspace CG using covariance N
                cov = _noise_operator(
                    noise_model,
                    tod_struct,
                    sample_rate,
                    config.weighting.correlation_length,
                    inverse=False,
                )
            W = NestedWeightOperator.create(Ninv, M, config.gaps.nested, cov=cov)
        else:
            # Plain masked weights, only exact for diagonal W
            W = WeightOperator.create(Ninv, M)
        F = (
            PommeProjectionOperator(config.pomme_tau, in_structure=tod_struct)
            if config.method == Methods.POMME
            else IdentityOperator(in_structure=tod_struct)
        )
        return cls(H, W, F, noise_model, sample_rate)

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build an [`ObservationModel`][] via [`create`][]."""
        fields: set[str] = {
            ReaderField.BORESIGHT_QUATERNIONS,
            ReaderField.DETECTOR_QUATERNIONS,
            ReaderField.VALID_SAMPLE_MASKS,
            ReaderField.TIMESTAMPS,
        }
        if not config.demodulated:
            # FIXME: this does not handle the case of a telescope without HWP
            fields.add(ReaderField.HWP_ANGLES)
        if config.scanning_mask:
            fields.add(ReaderField.VALID_SCANNING_MASKS)
        if config.weighting.mode != WeightingMode.IDENTITY:
            if config.weighting.source == NoiseSource.FIT:
                fields.update({ReaderField.SAMPLE_DATA, ReaderField.HWP_ANGLES})
            else:
                fields.add(ReaderField.NOISE_MODEL_FITS)
        if config.gaps.treatment == GapTreatment.FILL and not config.binned:
            fields.add(ReaderField.METADATA)
        return fields

    @property
    def tod_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.out_structure

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    @property
    def M(self) -> MaskOperator:
        return self.W.mask

    @M.setter
    def M(self, mask: MaskOperator) -> None:
        # rebuild the weight around the new mask (W is the only holder of M)
        self.W = self.W.with_mask(mask)

    @property
    def rhs_operator(self) -> AbstractLinearOperator:
        return (self.H.T @ self.W @ self.F).reduce()

    @property
    def rhs_operator_prefilled(self) -> AbstractLinearOperator:
        """RHS operator for gap-filled data: the data-side mask is dropped.

        Gap-filling already replaced the flagged samples with a constrained realization so that
        ``N⁻¹`` applies cleanly across the gaps; re-zeroing them with the inner mask of ``W`` would
        defeat the fill. Keep the outer mask (applied after ``N⁻¹``) and skip the inner one.

        Only reached under ``GapTreatment.FILL``, where ``W`` is the plain inner-mask weight.
        """
        assert isinstance(self.W, WeightOperator)
        return (self.H.T @ self.M @ self.W.weight @ self.F).reduce()

    def noise_operator(
        self, correlation_length: int, *, inverse: bool = True
    ) -> AbstractLinearOperator:
        """Build the (inverse) noise covariance operator."""
        return _noise_operator(
            self.noise_model,
            self.tod_structure,
            self.sample_rate,
            correlation_length,
            inverse=inverse,
        )

    def diag_W(self) -> WeightOperator:
        """Build the inverse white-noise weight for a *single* observation.

        Assumes ``self`` is a single-observation model (single-observation ``noise_model`` and
        ``tod_structure``). The observation-stacked preconditioner is obtained by mapping this over
        the observation axis in [`MultiObservationMapMaker.make_maps`][], mirroring how ``W`` is
        stacked inside the accumulation scan.
        """
        white = self.noise_model.to_white_noise_model()
        inv = _noise_operator(white, self.tod_structure, self.sample_rate, inverse=True)
        return WeightOperator.create(inv, self.M)


def _noise_model(
    data: Any,
    config: MapMakingConfig,
    tod_structure: jax.ShapeDtypeStruct | None = None,
    padding: Any | None = None,
) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data[ReaderField.TIMESTAMPS])

    # The demodulated TOD is a single-array Stokes; the noise model runs on its backing array, with
    # per-detector parameters carrying any leading axes (the Stokes axis) so a single model covers
    # every leg. The sample axis is always last.
    def _as_array(x: Any) -> Array:
        return x.data if isinstance(x, Stokes) else x

    if config.weighting.mode == WeightingMode.IDENTITY:
        if tod_structure is None:
            raise ValueError('tod_structure is required when config.weighting.mode is IDENTITY')
        struct = _as_array(tod_structure)
        return WhiteNoiseModel(sigma=jnp.ones(struct.shape[:-1], dtype=struct.dtype)), fs
    if config.weighting.source == NoiseSource.FIT:
        fit_config = config.weighting.fitting
        noise_model_class = WhiteNoiseModel if config.binned else AtmosphericNoiseModel
        fhwp = _hwp_frequency(data[ReaderField.TIMESTAMPS], data[ReaderField.HWP_ANGLES])

        tod = _as_array(data[ReaderField.SAMPLE_DATA])  # (*lead, nsamp)
        lead = tod.shape[:-1]
        sample_padding = 0 if padding is None else _as_array(padding[ReaderField.SAMPLE_DATA])[-1]
        f, Pxx = padding_aware_welch(
            tod.reshape(-1, tod.shape[-1]),
            sample_padding,
            fs=fs,
            nperseg=fit_config.nperseg,
        )
        flat_model = noise_model_class.fit_psd_model(
            f, Pxx, sample_rate=fs, hwp_frequency=fhwp, config=fit_config
        )
        # restore the leading axes on each (flattened-detector) parameter array
        noise_model = jax.tree.map(lambda p: p.reshape(lead + p.shape[1:]), flat_model)
    else:
        fits = data[ReaderField.NOISE_MODEL_FITS]  # (*lead, 4), already a plain array
        noise_model = AtmosphericNoiseModel(*jnp.moveaxis(fits, -1, 0))
        if config.binned:
            noise_model = noise_model.to_white_noise_model()
    return noise_model, fs


def _sample_mask(data: Any, config: MapMakingConfig) -> Array:
    """The valid-sample mask, `(ndet, nsamp)`, combining the sample and scanning masks.

    For Pomme, every tau-interval with a masked sample is masked whole, and so is the partial
    interval at the end. The weight is then constant on each interval, which makes it commute with
    the Pomme projector (see [`PommeProjectionOperator`][]). The widening applies to the combined
    mask: an interval straddling a scan boundary is dropped too.
    """
    mask = data[ReaderField.VALID_SAMPLE_MASKS]
    if (scanning := data.get(ReaderField.VALID_SCANNING_MASKS)) is not None:
        mask = jnp.logical_and(mask, scanning)

    if config.method == Methods.POMME:
        tau = config.pomme_tau
        F = PommeProjectionOperator(config.pomme_tau, in_structure=tree.as_structure(mask))
        # Mask all tau-intervals that are partially masked
        interval_mask = jnp.abs(F(mask)) < 0.5 / tau
        mask = jnp.logical_and(mask, interval_mask)
        # The partial interval at the end is unchanged by Pomme operator
        # -> True samples get interval_mask = False (since 1 > 0.5/tau)
        # -> False samples have mask = False
        # in both cases the logical and eliminates the tail

    return mask


def _noise_operator(
    noise_model: NoiseModel,
    tod_structure: jax.ShapeDtypeStruct,
    sample_rate: Array,
    correlation_length: int | None = None,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """Build the (inverse) noise covariance operator for this block.

    ``correlation_length`` sets the Toeplitz band for correlated (atmospheric) models; it is unused
    by white-noise models and may be omitted for them.
    """
    build = noise_model.inverse_operator if inverse else noise_model.operator
    return build(tod_structure, sample_rate=sample_rate, correlation_length=correlation_length)


def _sample_rate(timestamps: Float[Array, '...']) -> Float[Array, '']:
    # Note that the reader extrapolates timestamps in the padded region, keeping sample rate constant
    return (timestamps.size - 1) / jnp.ptp(timestamps)


def _hwp_frequency(
    timestamps: Float[Array, '...'], hwp_angles: Float[Array, '...']
) -> Float[Array, '']:
    # Note that the reader extrapolates hwp_angles in the padded region, keeping hwp freq constant
    return (jnp.unwrap(hwp_angles)[-1] - hwp_angles[0]) / jnp.ptp(timestamps) / (2 * jnp.pi)


def _mask_projector(*valid_masks: Array | None, structure: jax.ShapeDtypeStruct) -> MaskOperator:
    """Mask operator combining a series of boolean masks (logical AND)."""
    masks = [mask for mask in valid_masks if mask is not None]
    combined = functools.reduce(jnp.logical_and, masks) if masks else jnp.array(True)
    # A per-sample mask (ndet, nsamp) broadcasts right-aligned over a demodulated Stokes TOD's
    # leading Stokes axis (n, ndet, nsamp), and the sample axis stays last (packed by MaskOperator).
    return MaskOperator.from_boolean_mask(combined, in_structure=structure)


@register_dataclass
@dataclass
class TemplateBundle:
    r"""One template operator $T$ and the accompanying Gram inverse $(T^\top W T)^{-1}$."""

    operator: AbstractTemplateOperator
    gram_inverse: AbstractLinearOperator

    @classmethod
    def create(
        cls,
        operator: AbstractTemplateOperator,
        weight: AbstractLinearOperator,
        config: TemplatesConfig,
        *,
        allow_probe: bool = False,
    ) -> Self:
        r"""Pair a template operator $T$ with its Gram inverse given a weight matrix $W$.

        Args:
            operator: The template operator.
            weight: The weight matrix.
            config: Supplies the Gram regularization and batch size.
            allow_probe: See [`gram_inverse`][].
        """
        ginv = gram_inverse(
            operator,
            weight,
            config.regularization,
            allow_probe=allow_probe,
            batch_size=config.gram_batch_size,
        )
        return cls(operator, ginv)


@register_dataclass
@dataclass
class SharedTemplates:
    """Templates whose amplitudes are shared by the detectors, with their Gram on some of them.

    The Gram covers the detectors the templates were built for, one batch of an observation's
    detectors for instance: the observation's Gram is the sum over its batches.
    """

    operator: AbstractTemplateOperator
    gram: PyTree[Array]
    """Per template (and Stokes leg), the Gram over the flattened amplitudes."""

    @classmethod
    def create(
        cls,
        operator: AbstractTemplateOperator,
        weight: AbstractLinearOperator,
        config: TemplatesConfig,
    ) -> Self:
        """Pair shared templates with their Gram, given the diagonal weight matrix `weight`."""
        diag = weight(tree.ones_like(weight.in_structure))

        def gram_of(basis, weights):
            assert isinstance(basis, SharedBasis)  # the operator holds shared templates only
            return shared_gram(basis, weights, config.gram_batch_size)

        if isinstance(operator, StokesTemplateOperator):
            gram = {
                name: {leg: gram_of(basis, getattr(diag, leg)) for leg, basis in legged.items()}
                for name, legged in operator.bases_by_leg.items()
            }
        else:
            gram = {name: gram_of(basis, diag) for name, basis in operator.bases.items()}
        return cls(operator, gram)


@register_dataclass
@dataclass
class ObservationTemplates:
    """One observation's templates, stackable across observations via ``jax.lax.scan``.

    Active templates are partitioned by their ``explicit`` config flag when built, because the two
    kinds enter the map-making system in different places.
    """

    explicit: TemplateBundle | None
    """Templates whose amplitudes are solved jointly with the map."""

    implicit: TemplateBundle | None
    """Templates whose amplitudes are marginalised over."""

    shared: SharedTemplates | None
    """Templates whose amplitudes are shared by the detectors, solved jointly with the map."""

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build the active templates via [`create`][]."""
        tcfg = config.templates
        if tcfg is None:
            return set()
        fields: set[str] = set()
        if tcfg.polynomial is not None:
            fields |= {
                ReaderField.SCANNING_INTERVALS,
                ReaderField.TIMESTAMPS,
                ReaderField.VALID_SCANNING_MASKS,
            }
        if tcfg.scan_synchronous is not None:
            fields |= {ReaderField.AZIMUTH}
        if tcfg.binned_azimuth_synchronous is not None:
            fields |= {ReaderField.AZIMUTH}
        if tcfg.hwp_synchronous is not None:
            fields |= {ReaderField.HWP_ANGLES}
        if tcfg.azimuth_hwp_synchronous is not None:
            fields |= {ReaderField.AZIMUTH, ReaderField.HWP_ANGLES}
            if tcfg.azimuth_hwp_synchronous.split_scans:
                fields |= {ReaderField.LEFT_SCAN_MASK, ReaderField.RIGHT_SCAN_MASK}
        if tcfg.binned_azimuth_hwp_synchronous is not None:
            fields |= {ReaderField.AZIMUTH, ReaderField.HWP_ANGLES}
        if tcfg.spline_hwp_synchronous is not None:
            fields |= {ReaderField.TIMESTAMPS, ReaderField.HWP_ANGLES}
        if tcfg.t2p is not None:
            fields |= {ReaderField.SAMPLE_DATA, ReaderField.TIMESTAMPS}
        if tcfg.common_mode is not None:
            fields |= {ReaderField.TIMESTAMPS}
        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-observation path.'
            )
        return fields

    @classmethod
    def create(
        cls,
        data: Any,
        config: MapMakingConfig,
        model: ObservationModel,
        tod: PyTree[Array],
        temperature: Float[Array, 'det samp'] | None = None,
    ) -> tuple[Self, PyTree[Array]]:
        """Build one observation's templates and weight its TOD, implicit templates folded in.

        Args:
            data: The fields read for the observation.
            config: The mapmaking configuration, whose templates are built.
            model: The observation's model.
            tod: The TOD to weight.
            temperature: The I timestream, from which the T2P template is built; by default, the
                I leg of the sample data.
        """
        if (tcfg := config.templates) is None:
            raise ValueError('templates config required to build template operators')
        n_dets = model.tod_structure.shape[0]
        dtype = config.dtype
        # Demodulation splits the raw stream into differently filtered I/Q/U streams, so a template
        # covering several legs fits an independent amplitude on each.
        legs = config.landscape.stokes if config.demodulated else None

        explicit_bases: dict[str, Any] = {}
        implicit_bases: dict[str, Any] = {}

        def grouped(bases):
            if legs is not None and is_basis(bases):
                return {legs.lower(): bases}  # one group: stored once for every leg
            return bases

        def add(name: str, bases: Basis | dict[str, Basis], explicit: bool) -> None:
            (explicit_bases if explicit else implicit_bases)[name] = grouped(bases)

        if (poly := tcfg.polynomial) is not None:

            def poly_basis(orders: PolynomialOrders) -> Basis:
                return polynomial_basis(
                    max_poly_order=orders.max_order,
                    intervals=data[ReaderField.SCANNING_INTERVALS],
                    times=data[ReaderField.TIMESTAMPS],
                    dtype=dtype,
                    valid_mask=data[ReaderField.VALID_SCANNING_MASKS],
                    min_poly_order=orders.min_order,
                )

            if legs is not None:
                # legs fitted with the same orders share one basis
                groups = _legendre_leg_groups(poly.legendre, legs)
                add('polynomial', {g: poly_basis(o) for g, o in groups.items()}, poly.explicit)
            else:
                assert isinstance(poly.legendre, PolynomialOrders)  # per-leg needs demodulation
                add('polynomial', poly_basis(poly.legendre), poly.explicit)

        if (scan := tcfg.scan_synchronous) is not None:
            azimuth = data[ReaderField.AZIMUTH]
            if legs is not None:
                groups = _legendre_leg_groups(scan.legendre, legs)
                bases: dict[str, Basis] = {
                    g: scan_synchronous_basis(o, azimuth, dtype) for g, o in groups.items()
                }
                add('scan_synchronous', bases, scan.explicit)
            else:
                assert isinstance(scan.legendre, PolynomialOrders)  # per-leg needs demodulation
                add(
                    'scan_synchronous',
                    scan_synchronous_basis(scan.legendre, azimuth, dtype),
                    scan.explicit,
                )

        if (binned_az := tcfg.binned_azimuth_synchronous) is not None:
            basis = binned_azimuth_synchronous_basis(
                binned_az.bins, data[ReaderField.AZIMUTH], dtype
            )
            add('binned_azimuth_synchronous', basis, binned_az.explicit)

        if (hwp := tcfg.hwp_synchronous) is not None:
            basis = hwp_synchronous_basis(hwp.n_harmonics, data[ReaderField.HWP_ANGLES], dtype)
            add('hwp_synchronous', basis, hwp.explicit)

        if (az_hwp := tcfg.azimuth_hwp_synchronous) is not None:
            if az_hwp.split_scans:
                for side, scan_mask_field in (
                    ('left', ReaderField.LEFT_SCAN_MASK),
                    ('right', ReaderField.RIGHT_SCAN_MASK),
                ):
                    basis = azimuth_hwp_synchronous_basis(
                        az_hwp.legendre,
                        az_hwp.n_harmonics,
                        data[ReaderField.AZIMUTH],
                        data[ReaderField.HWP_ANGLES],
                        dtype,
                        scan_mask=data[scan_mask_field],
                    )
                    add(f'azimuth_hwp_synchronous_{side}', basis, az_hwp.explicit)
            else:
                basis = azimuth_hwp_synchronous_basis(
                    az_hwp.legendre,
                    az_hwp.n_harmonics,
                    data[ReaderField.AZIMUTH],
                    data[ReaderField.HWP_ANGLES],
                    dtype,
                )
                add('azimuth_hwp_synchronous', basis, az_hwp.explicit)

        if (binned_az_hwp := tcfg.binned_azimuth_hwp_synchronous) is not None:
            basis = binned_azimuth_hwp_synchronous_basis(
                binned_az_hwp.bins,
                binned_az_hwp.n_harmonics,
                data[ReaderField.AZIMUTH],
                data[ReaderField.HWP_ANGLES],
                dtype,
            )
            add('binned_azimuth_hwp_synchronous', basis, binned_az_hwp.explicit)

        if (spline_hwp := tcfg.spline_hwp_synchronous) is not None:
            times = data[ReaderField.TIMESTAMPS]
            basis = spline_hwp_synchronous_basis(
                times,
                data[ReaderField.HWP_ANGLES],
                spline_hwp.resolve_n_knots(times.size),
                spline_hwp.harmonics,
                dtype,
            )
            add('spline_hwp_synchronous', basis, spline_hwp.explicit)

        if (t2p := tcfg.t2p) is not None:
            if temperature is None:
                temperature = data[ReaderField.SAMPLE_DATA].i
            sample_rate = _sample_rate(data[ReaderField.TIMESTAMPS])
            # Q and U each fit their own leakage amplitude from the same temperature stream, so
            # they share one basis.
            qu = ''.join(s.lower() for s in config.landscape.stokes if s in 'QU')
            bases = {}
            if qu:
                bases[qu] = t2p_basis(
                    temperature,
                    dtype,
                    fit_band=t2p.fit_band,
                    sample_rate=sample_rate,
                    decimation_factor=t2p.decimation_factor,
                )
            add('t2p', bases, t2p.explicit)

        # Templates whose amplitudes are shared by the detectors enter the system on their own.
        shared_bases: dict[str, Any] = {}
        if (common := tcfg.common_mode) is not None:
            times = data[ReaderField.TIMESTAMPS]
            n_knots = common.resolve_n_knots(times.size)
            shared_bases['common_mode'] = grouped(common_mode_basis(times, n_knots, n_dets, dtype))

        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-observation path.'
            )

        if not explicit_bases and not implicit_bases and not shared_bases:
            raise ValueError('config.templates is set but no template is active.')

        def build(bases: dict[str, Any]) -> AbstractTemplateOperator | None:
            if not bases:
                return None
            if config.demodulated:
                return StokesTemplateOperator(bases, n_dets, config.landscape.stokes)
            return TemplateOperator(bases, n_dets)

        # `Weff` bundles the sample mask (via `model.W`) and the deprojector `model.F`
        Weff = (model.W @ model.F).reduce()
        wd = Weff(tod)

        implicit = None
        if (op := build(implicit_bases)) is not None:
            # Pomme + templates is rejected in _check_config, so F = I and Weff = W (diagonal)
            implicit = TemplateBundle.create(op, model.W, tcfg)
            ginv = implicit.gram_inverse
            wd = wd - Weff(op(ginv(op.T(wd))))  # W'd = W d − W Tᵢ G⁻¹ Tᵢᵀ W d

        explicit = None
        if (op := build(explicit_bases)) is not None:
            # T2P templates are always explicit and per-detector, so we need to allow probing.
            explicit = TemplateBundle.create(op, model.W, tcfg, allow_probe=True)

        shared = None
        if (op := build(shared_bases)) is not None:
            shared = SharedTemplates.create(op, model.W, tcfg)

        return cls(explicit=explicit, implicit=implicit, shared=shared), wd


def restrict_legs(
    data: dict[str, Any], tod: Stokes, legs: ValidStokesLiteral
) -> tuple[dict[str, Any], Stokes]:
    """An observation's sample data, noise fits and TOD, restricted to the Stokes `legs`.

    The reader may load more demodulated legs than the map has (see
    [`MapMakingConfig.read_stokes`][furax.mapmaking.config.MapMakingConfig.read_stokes]).
    """
    index = jnp.array([tod.stokes.index(leg) for leg in legs])

    def select(x: Stokes) -> Stokes:
        return Stokes.class_for(legs).from_array(x.data[index])

    data = {**data, ReaderField.SAMPLE_DATA: select(data[ReaderField.SAMPLE_DATA])}
    if (fits := data.get(ReaderField.NOISE_MODEL_FITS)) is not None:
        data[ReaderField.NOISE_MODEL_FITS] = fits[index]  # one row per leg
    return data, select(tod)
