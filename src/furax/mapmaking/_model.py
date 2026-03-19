from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int64, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.core import BlockDiagonalOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, StokesI

from .acquisition import build_acquisition_operator
from .config import MapMakingConfig
from .gap_filling import GapFillingOperator
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .pointing import PointingOperator


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
    ) -> 'ObservationModel':
        H = build_acquisition_operator(
            landscape,
            data['boresight_quaternions'],
            data['detector_quaternions'],
            data.get('hwp_angles'),
            demodulated=config.demodulated,
            pointing_chunk_size=config.pointing.chunk_size,
            pointing_on_the_fly=config.pointing.on_the_fly,
            dtype=config.dtype,
        )
        masker = _mask_projector(
            data['valid_sample_masks'],
            data.get('valid_scanning_masks'),
            structure=H.out_structure,
        )
        noise_model, sample_rate = _noise_model(data, config)
        W = _noise_operator(
            noise_model, H.out_structure, sample_rate, config.noise.correlation_length, inverse=True
        )
        return cls(H, W, masker, noise_model, sample_rate)

    @property
    def tod_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.out_structure

    @property
    def map_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.H.in_structure

    def hits(self) -> Int64[Array, ' pixels']:
        assert isinstance(self.H, CompositionOperator)  # mypy assert
        # the pointing operator should be the first in the acquisition chain, so the last operand...
        # there is a unit test for that
        pointing = self.H.operands[-1]
        assert isinstance(pointing, PointingOperator)  # mypy assert
        pointing_i = pointing.as_stokes_i()
        ones = tree.ones_like(self.tod_structure)
        # the masker could be acting on a Stokes pytree or an Array
        masked_ones = jax.tree.leaves(self.masker(ones))[0]
        hits_stokes = pointing_i.T(StokesI(masked_ones))
        return jnp.int64(hits_stokes.i)  # type: ignore[no-any-return]

    def white_noise_W(self) -> AbstractLinearOperator:
        """Build the inverse white noise covariance operator."""
        operator_tree = jax.tree.map(
            lambda noise, s: noise.to_white_noise_model().inverse_operator(s),
            self.noise_model,
            self.tod_structure,
            is_leaf=lambda nm: isinstance(nm, NoiseModel),
        )
        return BlockDiagonalOperator(operator_tree)

    def apply_system(self, x: Stokes, *, white_noise: bool = False) -> Stokes:
        """Applies H.T @ W @ H (+ TOD masking) to x for this observation."""
        weight = self.white_noise_W() if white_noise else self.W
        y = self.masker(self.H(x))
        y = weight(y)
        return self.H.T(self.masker.T(y))  # type: ignore[no-any-return]

    def rhs(self, data: Any, config: MapMakingConfig) -> Stokes:
        """Accumulates data into the r.h.s. of the mapmaking equation.

        Also performs gap-filling on the data before projecting into map domain.
        """
        tod = data['sample_data']
        if config.gaps.fill and not config.binned:
            # FIXME: check with demodulated data
            N = _noise_operator(
                self.noise_model,
                self.tod_structure,
                self.sample_rate,
                config.noise.correlation_length,
                inverse=False,
            )
            gapfill = GapFillingOperator(
                N,  # type: ignore[arg-type]
                self._get_indexer(),
                data['metadata'],
                self.W,  # type: ignore[arg-type]
                rate=self.sample_rate,
                max_cg_steps=config.gaps.fill_options.max_steps,
                rtol=config.gaps.fill_options.rtol,
            )
            key = jax.random.key(config.gaps.fill_options.seed)
            tod = gapfill(key, tod)
        rhs: Stokes = (self.H.T @ self.masker @ self.W)(tod)
        return rhs

    def _get_indexer(self) -> IndexOperator:
        """Get the IndexOperator for gap-filling"""
        if isinstance(self.masker, MaskOperator):
            mask = self.masker.to_boolean_mask()
        elif isinstance(self.masker, IdentityOperator):
            mask = tree.ones_like(self.tod_structure).astype(bool)
        else:
            raise NotImplementedError
        return IndexOperator(mask, in_structure=self.tod_structure)


def _noise_model(data: Any, config: MapMakingConfig) -> tuple[PyTree[NoiseModel], Array]:
    """Compute the noise model and sample rate for a single observation block."""
    fs = _sample_rate(data['timestamps'])
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
