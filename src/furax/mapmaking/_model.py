from dataclasses import dataclass
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.core import BlockDiagonalOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape

from .acquisition import build_acquisition_operator
from .config import MapMakingConfig, Methods
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .templates import ATOPProjectionOperator


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

    W: AbstractLinearOperator
    """Weighting operator"""

    masker: AbstractLinearOperator
    """Sample masking operator"""

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
            data['boresight_quaternions'],
            data['detector_quaternions'],
            data.get('hwp_angles'),
            demodulated=config.demodulated,
            pointing_chunk_size=config.pointing.chunk_size,
            pointing_on_the_fly=config.pointing.on_the_fly,
            pointing_interpolate=config.pointing.interpolation == 'bilinear',
            dtype=config.dtype,
        )
        masker = _mask_projector(
            _sample_mask(data, config),
            data.get('valid_scanning_masks'),
            structure=H.out_structure,
        )
        noise_model, sample_rate = _noise_model(data, config, tod_structure=H.out_structure)
        W = _noise_operator(
            noise_model, H.out_structure, sample_rate, config.noise.correlation_length, inverse=True
        )
        if F_T := _template_deprojector(config, H.out_structure):
            W = W @ F_T
        return cls(H, W, masker, noise_model, sample_rate)

    @property
    def tod_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.out_structure

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

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

    def diag_W(self) -> AbstractLinearOperator:
        """Build the inverse white noise covariance operator."""
        operator_tree = jax.tree.map(
            lambda noise, s: noise.to_white_noise_model().inverse_operator(s),
            self.noise_model,
            self.tod_structure,
            is_leaf=lambda nm: isinstance(nm, NoiseModel),
        )
        return BlockDiagonalOperator(operator_tree)

    def _get_indexer(self) -> IndexOperator:
        """Get the IndexOperator for gap-filling"""
        if isinstance(self.masker, MaskOperator):
            mask = self.masker.to_boolean_mask()
        elif isinstance(self.masker, IdentityOperator):
            mask = tree.ones_like(self.tod_structure).astype(bool)
        else:
            raise NotImplementedError
        return IndexOperator(mask, in_structure=self.tod_structure)


def pad_model(models: ObservationModel, n_pad: int) -> ObservationModel:
    """Pad models with n_pad dummy observations that contribute nothing to sums.

    The dummy observations are copies of the last real observation with their
    masker array leaves zeroed out, so masker(x) = 0 for any x.
    """
    if n_pad == 0:
        return models
    last = jax.tree.map(lambda a: jnp.repeat(a[-1:], n_pad, axis=0), models)
    if len(jax.tree.leaves(last.masker)) == 0:
        # Leaf-free masker (e.g. IdentityOperator): jax.tree.map produces no zeros.
        # Build an explicit all-False MaskOperator so padding obs contribute nothing.
        structure = last.masker.in_structure
        first_leaf = jax.tree.leaves(structure)[0]
        zero_mask = jnp.zeros((n_pad, *first_leaf.shape), dtype=bool)
        zero_masker = MaskOperator.from_boolean_mask(zero_mask, in_structure=structure)
    else:
        zero_masker = jax.tree.map(jnp.zeros_like, last.masker)
    padded = ObservationModel(
        H=last.H,
        W=last.W,
        masker=zero_masker,
        noise_model=last.noise_model,
        sample_rate=last.sample_rate,
    )
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), models, padded)  # type: ignore[no-any-return]


def _noise_model(
    data: Any, config: MapMakingConfig, tod_structure: Any = None
) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data['timestamps'])
    if config.noise.identity:
        noise_model = jax.tree.map(
            lambda s: WhiteNoiseModel(sigma=jnp.ones(s.shape[0], dtype=s.dtype)),
            tod_structure,
        )
        return noise_model, fs
    if config.noise.fit_from_data:
        noise_model_class = WhiteNoiseModel if config.binned else AtmosphericNoiseModel
        fhwp = _hwp_frequency(data['timestamps'], data['hwp_angles'])

        def _compute_Pxx_and_fit(tod):  # type: ignore[no-untyped-def]
            f, Pxx = jax.scipy.signal.welch(tod, fs=fs, nperseg=config.noise.fitting.nperseg)
            return noise_model_class.fit_psd_model(
                f,
                Pxx,
                sample_rate=fs,
                hwp_frequency=fhwp,
                config=config.noise.fitting,
            )

        noise_model = jax.tree.map(_compute_Pxx_and_fit, data['sample_data'])
    else:
        noise_model = jax.tree.map(lambda x: AtmosphericNoiseModel(*x.T), data['noise_model_fits'])
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

    mask = data['valid_sample_masks']

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


def _mask_projector(
    *valid_masks: Array | None, structure: jax.ShapeDtypeStruct
) -> AbstractLinearOperator:
    """Mask operator built from a series of boolean masks."""

    def _masker(valid_mask: Array | None) -> AbstractLinearOperator:
        if valid_mask is None:
            return IdentityOperator(in_structure=structure)
        return MaskOperator.from_boolean_mask(valid_mask, in_structure=structure)

    combined_masker = CompositionOperator([_masker(valid_mask) for valid_mask in valid_masks])
    return combined_masker.reduce()
