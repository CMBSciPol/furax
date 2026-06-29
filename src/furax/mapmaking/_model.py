import functools
from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Inexact, PyTree

from furax import AbstractLinearOperator, MaskOperator, symmetric, tree
from furax.core import BlockDiagonalOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape

from ._observation import ReaderField
from .acquisition import build_acquisition_operator
from .config import MapMakingConfig, Methods, NoiseSource, WeightingMode
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .templates import ATOPProjectionOperator


@symmetric
class WeightOperator(AbstractLinearOperator):
    """Masked noise weight `Z W Z` as a single operator."""

    weight: AbstractLinearOperator  # symmetric
    mask: MaskOperator

    @classmethod
    def create(cls, weight: AbstractLinearOperator, mask: MaskOperator) -> Self:
        return cls(weight, mask, in_structure=mask.in_structure)

    def mv(self, x: PyTree[Inexact[jax.Array, '...']]) -> PyTree[Inexact[jax.Array, '...']]:
        return self.mask(self.weight(self.mask(x)))


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

    W: WeightOperator
    """Weighting operator (noise weights + mask)"""

    F: AbstractLinearOperator | None
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
        Z = _mask_projector(
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
        W = WeightOperator.create(Ninv, Z)  # equivalent to Z @ Ninv @ Z
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
        if config.gaps.fill and not config.binned:
            fields.add(ReaderField.METADATA)
        return fields

    @property
    def tod_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.out_structure

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    @property
    def Z(self) -> MaskOperator:
        return self.W.mask

    @Z.setter
    def Z(self, mask: MaskOperator) -> None:
        # rebuild the weight around the new mask (W is the only holder of Z)
        self.W = WeightOperator.create(self.W.weight, mask)

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
        """Build the inverse white noise covariance operator."""
        operator_tree = jax.tree.map(
            lambda noise, s: noise.to_white_noise_model().inverse_operator(s),
            self.noise_model,
            self.tod_structure,
            is_leaf=lambda nm: isinstance(nm, NoiseModel),
        )
        return WeightOperator.create(BlockDiagonalOperator(operator_tree), self.Z)

    def _get_indexer(self) -> IndexOperator:
        """Get the IndexOperator for gap-filling"""
        return IndexOperator(self.Z.to_boolean_mask(), in_structure=self.tod_structure)


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
    correlation_length: int,
    *,
    inverse: bool = True,
) -> AbstractLinearOperator:
    """Build the (inverse) noise covariance operator for this block."""
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
) -> AbstractLinearOperator | None:
    """Build the template deprojection operator."""
    if config.method == Methods.ATOP:
        return ATOPProjectionOperator(config.atop_tau, in_structure=tod_structure)
    else:
        return None


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
