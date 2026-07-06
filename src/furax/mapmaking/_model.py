import functools
from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes

from ._observation import ReaderField
from .acquisition import build_acquisition_operator
from .config import GapTreatment, MapMakingConfig, Methods, NoiseSource, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .templates import (
    ATOPProjectionOperator,
    Basis,
    TemplateFamily,
    TemplateOperator,
    azhwp_synchronous_basis,
    binaz_synchronous_basis,
    binazhwp_synchronous_basis,
    bspline_hwpss_basis,
    hwp_synchronous_basis,
    polynomial_basis,
    scan_synchronous_basis,
    temperature_basis,
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
    """ATOP filter/deprojector (identity when not enabled)"""

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
            data[ReaderField.BORESIGHT_QUATERNIONS],
            data[ReaderField.DETECTOR_QUATERNIONS],
            data.get(ReaderField.HWP_ANGLES),
            demodulated=config.demodulated,
            pointing_on_the_fly=config.pointing.on_the_fly,
            pointing_batch_size=config.pointing.batch_size,
            pointing_interpolate=config.pointing.interpolation == 'bilinear',
            dtype=config.dtype,
        )
        tod_struct = H.out_structure
        M = _mask_projector(
            _sample_mask(data, config),
            data.get(ReaderField.VALID_SCANNING_MASKS),
            structure=tod_struct,
        )
        noise_model, sample_rate = _noise_model(data, config, tod_structure=tod_struct)
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
            ATOPProjectionOperator(config.atop_tau, in_structure=tod_struct)
            if config.method == Methods.ATOP
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
        f, Pxx = jax.scipy.signal.welch(
            tod.reshape(-1, tod.shape[-1]), fs=fs, nperseg=fit_config.nperseg
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
    """Get the sample mask from data.

    For ATOP mapmaker, extra pixels may be masked depending on atop_tau.
    """
    mask = data[ReaderField.VALID_SAMPLE_MASKS]

    if config.method == Methods.ATOP:
        tau = config.atop_tau
        F = ATOPProjectionOperator(config.atop_tau, in_structure=tree.as_structure(mask))
        # Mask all tau-intervals that are partially masked
        interval_mask = jnp.abs(F(mask)) < 0.5 / tau
        mask = jnp.logical_and(mask, interval_mask)
        # The partial interval at the end is unchanged by ATOP operator
        # -> True samples get interval_mask = False (since 1 > 0.5/tau)
        # -> False samples have mask = False
        # in both cases the logical and eliminates the tail

    return mask  # type: ignore[no-any-return]


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


def _demod_bases(basis: Basis, config: MapMakingConfig) -> Basis | dict[str, Basis]:
    """Bare basis if not demodulated; the same basis repeated for every Stokes leg otherwise.

    A kind-agnostic family (e.g. scan-synchronous pickup) shares one functional form across
    legs, but demodulation splits the raw stream into differently-filtered I/Q/U streams, so
    each leg still needs its own independently fitted amplitude — hence a per-leg dict (with
    the same [`Basis`][] object repeated), not a single shared amplitude. The TOD structure
    a [`TemplateOperator`][] produces must match the actual TOD
    (bare array vs. ``Stokes`` container): mixing the two within one operator's families breaks
    ``out_structure``.
    """
    if not config.demodulated:
        return basis
    return {s.lower(): basis for s in config.landscape.stokes}


@register_dataclass
@dataclass
class ObservationTemplates:
    """Per-observation template operators, stackable across observations via ``jax.lax.scan``.

    Active families are partitioned by their ``explicit`` config flag:

    - ``explicit`` — families whose amplitudes are solved jointly with the map and returned
      (``explicit=True``). ``None`` if every active family is implicit.
    - ``implicit`` — families folded into the noise weight ``W → W_m`` and never solved
      (``explicit=False``). ``None`` if every active family is explicit (then ``W_m = W``).

    Each is a [`TemplateOperator`][] mapping a dict of per-family
    amplitudes to a single observation's TOD; under ``jax.lax.scan`` the array leaves gain a
    leading observation axis.
    """

    explicit: AbstractLinearOperator | None
    implicit: AbstractLinearOperator | None

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build the active template families via [`create`][]."""
        tcfg = config.templates
        if tcfg is None:
            return set()
        fields: set[str] = set()
        if tcfg.polynomial is not None:
            fields.update(
                {
                    ReaderField.SCANNING_INTERVALS,
                    ReaderField.TIMESTAMPS,
                    ReaderField.VALID_SCANNING_MASKS,
                }
            )
        if tcfg.scan_synchronous is not None:
            fields.add(ReaderField.AZIMUTH)
        if tcfg.binaz_synchronous is not None:
            fields.add(ReaderField.AZIMUTH)
        if tcfg.hwp_synchronous is not None:
            fields.add(ReaderField.HWP_ANGLES)
        if tcfg.azhwp_synchronous is not None:
            fields.update({ReaderField.AZIMUTH, ReaderField.HWP_ANGLES})
            if tcfg.azhwp_synchronous.split_scans:
                fields.update({ReaderField.LEFT_SCAN_MASK, ReaderField.RIGHT_SCAN_MASK})
        if tcfg.binazhwp_synchronous is not None:
            fields.update({ReaderField.AZIMUTH, ReaderField.HWP_ANGLES})
        if tcfg.spline_hwpss is not None:
            fields.update({ReaderField.TIMESTAMPS, ReaderField.HWP_ANGLES})
        if tcfg.t2p is not None:
            fields.update({ReaderField.SAMPLE_DATA, ReaderField.TIMESTAMPS})
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
        tod_structure: jax.ShapeDtypeStruct,
    ) -> Self:
        if (tcfg := config.templates) is None:
            raise ValueError('templates config required to build template operators')
        n_dets = tod_structure.shape[0]
        dtype = config.dtype
        families: list[TemplateFamily] = []

        if (poly := tcfg.polynomial) is not None:
            if config.demodulated:
                legendre_qu = poly.legendre_qu if poly.legendre_qu is not None else poly.legendre
                bases: Basis | dict[str, Basis] = {
                    s.lower(): polynomial_basis(
                        max_poly_order=(poly.legendre if s == 'I' else legendre_qu).max_order,
                        intervals=data[ReaderField.SCANNING_INTERVALS],
                        times=data[ReaderField.TIMESTAMPS],
                        dtype=dtype,
                        valid_mask=data[ReaderField.VALID_SCANNING_MASKS],
                    )
                    for s in config.landscape.stokes
                }
            else:
                bases = polynomial_basis(
                    max_poly_order=poly.legendre.max_order,
                    intervals=data[ReaderField.SCANNING_INTERVALS],
                    times=data[ReaderField.TIMESTAMPS],
                    dtype=dtype,
                    valid_mask=data[ReaderField.VALID_SCANNING_MASKS],
                )
            families.append(TemplateFamily(name='polynomial', bases=bases, explicit=poly.explicit))

        if (sss := tcfg.scan_synchronous) is not None:
            basis = scan_synchronous_basis(sss.legendre, data[ReaderField.AZIMUTH], dtype)
            families.append(
                TemplateFamily(
                    name='scan_synchronous',
                    bases=_demod_bases(basis, config),
                    explicit=sss.explicit,
                )
            )

        if (baz := tcfg.binaz_synchronous) is not None:
            basis = binaz_synchronous_basis(baz.bins, data[ReaderField.AZIMUTH], dtype)
            families.append(
                TemplateFamily(
                    name='binaz_synchronous',
                    bases=_demod_bases(basis, config),
                    explicit=baz.explicit,
                )
            )

        if (hwpss := tcfg.hwp_synchronous) is not None:
            basis = hwp_synchronous_basis(hwpss.n_harmonics, data[ReaderField.HWP_ANGLES], dtype)
            families.append(
                TemplateFamily(
                    name='hwp_synchronous',
                    bases=_demod_bases(basis, config),
                    explicit=hwpss.explicit,
                )
            )

        if (azhwpss := tcfg.azhwp_synchronous) is not None:
            if azhwpss.split_scans:
                for side, scan_mask_field in (
                    ('left', ReaderField.LEFT_SCAN_MASK),
                    ('right', ReaderField.RIGHT_SCAN_MASK),
                ):
                    basis = azhwp_synchronous_basis(
                        azhwpss.legendre,
                        azhwpss.n_harmonics,
                        data[ReaderField.AZIMUTH],
                        data[ReaderField.HWP_ANGLES],
                        dtype,
                        scan_mask=data[scan_mask_field],
                    )
                    families.append(
                        TemplateFamily(
                            name=f'azhwp_synchronous_{side}',
                            bases=_demod_bases(basis, config),
                            explicit=azhwpss.explicit,
                        )
                    )
            else:
                basis = azhwp_synchronous_basis(
                    azhwpss.legendre,
                    azhwpss.n_harmonics,
                    data[ReaderField.AZIMUTH],
                    data[ReaderField.HWP_ANGLES],
                    dtype,
                )
                families.append(
                    TemplateFamily(
                        name='azhwp_synchronous',
                        bases=_demod_bases(basis, config),
                        explicit=azhwpss.explicit,
                    )
                )

        if (binazhwpss := tcfg.binazhwp_synchronous) is not None:
            basis = binazhwp_synchronous_basis(
                binazhwpss.bins,
                binazhwpss.n_harmonics,
                data[ReaderField.AZIMUTH],
                data[ReaderField.HWP_ANGLES],
                dtype,
            )
            families.append(
                TemplateFamily(
                    name='binazhwp_synchronous',
                    bases=_demod_bases(basis, config),
                    explicit=binazhwpss.explicit,
                )
            )

        if (shwpss := tcfg.spline_hwpss) is not None:
            times = data[ReaderField.TIMESTAMPS]
            basis = bspline_hwpss_basis(
                times,
                data[ReaderField.HWP_ANGLES],
                shwpss.resolve_n_knots(times.size),
                shwpss.harmonics,
                dtype,
            )
            families.append(
                TemplateFamily(
                    name='spline_hwpss', bases=_demod_bases(basis, config), explicit=shwpss.explicit
                )
            )

        if (t2p := tcfg.t2p) is not None:
            temperature = data[ReaderField.SAMPLE_DATA].i
            sample_rate = _sample_rate(data[ReaderField.TIMESTAMPS])
            # Q and U each fit their own leakage amplitude from the same temperature stream.
            bases = {
                s.lower(): temperature_basis(
                    temperature,
                    dtype,
                    fit_band=t2p.fit_band,
                    sample_rate=sample_rate,
                    decimation_factor=t2p.decimate,
                )
                for s in config.landscape.stokes
                if s in 'QU'
            }
            families.append(
                TemplateFamily(name='t2p', bases=bases, explicit=t2p.explicit, shared=False)
            )

        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-observation path.'
            )

        if not families:
            raise ValueError('config.templates is set but no template family is active.')

        stokes = config.landscape.stokes if config.demodulated else None
        op = TemplateOperator.create(families, n_dets, stokes)
        return cls(explicit=op.explicit, implicit=op.implicit)
