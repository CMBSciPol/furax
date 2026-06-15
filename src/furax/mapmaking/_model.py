import functools
from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.core import BlockDiagonalOperator, BlockRowOperator
from furax.obs.landscapes import StokesLandscape

from ._observation import ReaderField
from .acquisition import build_acquisition_operator
from .config import GapTreatment, MapMakingConfig, Methods, NoiseSource, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .templates import ATOPProjectionOperator, PerDetectorTemplate
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
    """Deprojection operator"""

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
            pointing_chunk_size=config.pointing.chunk_size,
            pointing_on_the_fly=config.pointing.on_the_fly,
            pointing_interpolate=config.pointing.interpolation == 'bilinear',
            dtype=config.dtype,
        )
        M = _mask_projector(
            _sample_mask(data, config),
            data.get(ReaderField.VALID_SCANNING_MASKS),
            structure=H.out_structure,
        )
        noise_model, sample_rate = _noise_model(data, config, tod_structure=H.out_structure)
        Ninv = _noise_operator(
            noise_model,
            H.out_structure,
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
                    H.out_structure,
                    sample_rate,
                    config.weighting.correlation_length,
                    inverse=False,
                )
            W = NestedWeightOperator.create(Ninv, M, config.gaps.nested, cov=cov)
        else:
            # Plain masked weights, only exact for diagonal W
            W = WeightOperator.create(Ninv, M)
        F = _template_deprojector(config, H.out_structure)
        return cls(H, W, F, noise_model, sample_rate)

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build an :class:`ObservationModel` via :meth:`create`."""
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
        the observation axis in :meth:`MultiObservationMapMaker.make_maps`, mirroring how ``W`` is
        stacked inside the accumulation scan.
        """
        white = jax.tree.map(
            lambda nm: nm.to_white_noise_model(),
            self.noise_model,
            is_leaf=lambda nm: isinstance(nm, NoiseModel),
        )
        inv = _noise_operator(white, self.tod_structure, self.sample_rate, inverse=True)
        return WeightOperator.create(inv, self.M)


def _noise_model(
    data: Any,
    config: MapMakingConfig,
    tod_structure: jax.ShapeDtypeStruct | None = None,
) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data[ReaderField.TIMESTAMPS])
    if config.weighting.mode == WeightingMode.IDENTITY:
        if tod_structure is None:
            raise ValueError('tod_structure is required when config.weighting.mode is IDENTITY')
        noise_model = jax.tree.map(
            lambda s: WhiteNoiseModel(sigma=jnp.ones(s.shape[0], dtype=s.dtype)),
            tod_structure,
        )
        return noise_model, fs
    if config.weighting.source == NoiseSource.FIT:
        fit_config = config.weighting.fitting
        noise_model_class = WhiteNoiseModel if config.binned else AtmosphericNoiseModel
        fhwp = _hwp_frequency(data[ReaderField.TIMESTAMPS], data[ReaderField.HWP_ANGLES])

        def _compute_Pxx_and_fit(tod):  # type: ignore[no-untyped-def]
            f, Pxx = jax.scipy.signal.welch(tod, fs=fs, nperseg=fit_config.nperseg)
            return noise_model_class.fit_psd_model(
                f,
                Pxx,
                sample_rate=fs,
                hwp_frequency=fhwp,
                config=fit_config,
            )

        noise_model = jax.tree.map(_compute_Pxx_and_fit, data[ReaderField.SAMPLE_DATA])
    else:
        noise_model = jax.tree.map(
            lambda x: AtmosphericNoiseModel(*x.T), data[ReaderField.NOISE_MODEL_FITS]
        )
        if config.binned:
            noise_model = jax.tree.map(
                lambda m: m.to_white_noise_model(),
                noise_model,
                is_leaf=lambda x: isinstance(x, AtmosphericNoiseModel),
            )
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
    operator_tree = jax.tree.map(
        lambda model, struct: (model.inverse_operator if inverse else model.operator)(
            struct, sample_rate=sample_rate, correlation_length=correlation_length
        ),
        noise_model,
        tod_structure,
        is_leaf=lambda x: isinstance(x, NoiseModel),
    )
    return BlockDiagonalOperator(operator_tree)


def _template_deprojector(
    config: MapMakingConfig,
    tod_structure: jax.ShapeDtypeStruct,
) -> AbstractLinearOperator:
    """Build the template deprojection operator."""
    if config.method == Methods.ATOP:
        return ATOPProjectionOperator(config.atop_tau, in_structure=tod_structure)
    return IdentityOperator(in_structure=tod_structure)


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
    return MaskOperator.from_boolean_mask(combined, in_structure=structure)


@register_dataclass
@dataclass
class ObservationTemplates:
    """Per-observation template operators, stackable across observations via ``jax.lax.scan``.

    Active families are partitioned by their ``explicit`` config flag:

    - ``explicit`` — families whose amplitudes are solved jointly with the map and returned
      (``explicit=True``). ``None`` if every active family is implicit.
    - ``implicit`` — families folded into the noise weight ``W → W_m`` and never solved
      (``explicit=False``; see :mod:`._deprojection`). ``None`` if every active family is
      explicit (then ``W_m = W``).

    Each is a :class:`~furax.core.BlockRowOperator` mapping a dict of per-family amplitudes
    to a single observation's TOD; under ``jax.lax.scan`` the array leaves gain a leading
    observation axis.
    """

    explicit: AbstractLinearOperator | None
    implicit: AbstractLinearOperator | None

    @staticmethod
    def required_reader_fields(config: MapMakingConfig) -> set[str]:
        """Reader fields needed to build the active template families via :meth:`create`."""
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
        blocks: dict[str, AbstractLinearOperator] = {}
        explicit_flags: dict[str, bool] = {}

        if (poly := tcfg.polynomial) is not None:
            blocks['polynomial'] = PerDetectorTemplate.polynomial(
                max_poly_order=poly.legendre.max_order,
                intervals=data[ReaderField.SCANNING_INTERVALS],
                times=data[ReaderField.TIMESTAMPS],
                n_dets=n_dets,
                dtype=dtype,
                valid_mask=data[ReaderField.VALID_SCANNING_MASKS],
            )
            explicit_flags['polynomial'] = poly.explicit

        if (sss := tcfg.scan_synchronous) is not None:
            blocks['scan_synchronous'] = PerDetectorTemplate.scan_synchronous(
                legendre=sss.legendre,
                azimuth=data[ReaderField.AZIMUTH],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['scan_synchronous'] = sss.explicit

        if (baz := tcfg.binaz_synchronous) is not None:
            blocks['binaz_synchronous'] = PerDetectorTemplate.binaz_synchronous(
                bins=baz.bins,
                azimuth=data[ReaderField.AZIMUTH],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['binaz_synchronous'] = baz.explicit

        if (hwpss := tcfg.hwp_synchronous) is not None:
            blocks['hwp_synchronous'] = PerDetectorTemplate.hwp_synchronous(
                n_harmonics=hwpss.n_harmonics,
                hwp_angles=data[ReaderField.HWP_ANGLES],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['hwp_synchronous'] = hwpss.explicit

        if (azhwpss := tcfg.azhwp_synchronous) is not None:
            if azhwpss.split_scans:
                for side, scan_mask_field in (
                    ('left', ReaderField.LEFT_SCAN_MASK),
                    ('right', ReaderField.RIGHT_SCAN_MASK),
                ):
                    blocks[f'azhwp_synchronous_{side}'] = PerDetectorTemplate.azhwp_synchronous(
                        legendre=azhwpss.legendre,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=data[ReaderField.AZIMUTH],
                        hwp_angles=data[ReaderField.HWP_ANGLES],
                        n_dets=n_dets,
                        dtype=dtype,
                        scan_mask=data[scan_mask_field],
                    )
                    explicit_flags[f'azhwp_synchronous_{side}'] = azhwpss.explicit
            else:
                blocks['azhwp_synchronous'] = PerDetectorTemplate.azhwp_synchronous(
                    legendre=azhwpss.legendre,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data[ReaderField.AZIMUTH],
                    hwp_angles=data[ReaderField.HWP_ANGLES],
                    n_dets=n_dets,
                    dtype=dtype,
                )
                explicit_flags['azhwp_synchronous'] = azhwpss.explicit

        if (binazhwpss := tcfg.binazhwp_synchronous) is not None:
            blocks['binazhwp_synchronous'] = PerDetectorTemplate.binazhwp_synchronous(
                bins=binazhwpss.bins,
                n_harmonics=binazhwpss.n_harmonics,
                azimuth=data[ReaderField.AZIMUTH],
                hwp_angles=data[ReaderField.HWP_ANGLES],
                n_dets=n_dets,
                dtype=dtype,
            )
            explicit_flags['binazhwp_synchronous'] = binazhwpss.explicit

        if (shwpss := tcfg.spline_hwpss) is not None:
            times = data[ReaderField.TIMESTAMPS]
            blocks['spline_hwpss'] = PerDetectorTemplate.bspline_hwpss(
                times=times,
                hwp_angles=data[ReaderField.HWP_ANGLES],
                n_dets=n_dets,
                n_knots=shwpss.resolve_n_knots(times.size),
                harmonics=shwpss.harmonics,
                dtype=dtype,
            )
            explicit_flags['spline_hwpss'] = shwpss.explicit

        if tcfg.ground is not None:
            raise NotImplementedError(
                'Ground templates are not supported in the multi-observation path.'
            )

        if not blocks:
            raise ValueError('config.templates is set but no template family is active.')

        explicit = {k: v for k, v in blocks.items() if explicit_flags[k]}
        implicit = {k: v for k, v in blocks.items() if not explicit_flags[k]}
        return cls(
            explicit=BlockRowOperator(explicit) if explicit else None,
            implicit=BlockRowOperator(implicit) if implicit else None,
        )
