from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int64, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, tree
from furax.core import BlockDiagonalOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesPyTreeType

from ._reader import ObservationReader
from ._scan_blocks import ScanBlockColumnOperator, ScanBlockDiagonalOperator
from .acquisition import build_acquisition_operator
from .config import MapMakingConfig, Methods
from .gap_filling import GapFillingOperator
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
    ) -> 'ObservationModel':
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
        noise_model, sample_rate = _noise_model(data, config)
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

    @property
    def n_observations(self) -> int:
        # assuming stacked pytree
        return jax.tree.leaves(self.H)[0].shape[0]  # type: ignore[no-any-return]

    def get_system_operator(self, *, diag: bool = False) -> AbstractLinearOperator:
        n = self.n_observations
        H = ScanBlockColumnOperator(self.H, n)
        M = ScanBlockDiagonalOperator(self.masker, n)
        weight = self.diag_W() if diag else self.W
        W = ScanBlockDiagonalOperator(weight, n)
        return (H.T @ M @ W @ M @ H).reduce()

    @jax.jit
    def accumulate_hits(self) -> Int64[Array, ' pixels']:
        """Accumulate hit counts across all observations."""
        n = self.n_observations
        assert isinstance(self.H, CompositionOperator)  # mypy assert
        # pointing operator is the last operand of the acquisition chain (see unit test)
        pointing = self.H.operands[-1]
        assert isinstance(pointing, PointingOperator)  # mypy assert
        P = ScanBlockColumnOperator(pointing.as_stokes_i(interpolate=False), n)
        M = ScanBlockDiagonalOperator(self.masker, n)
        ones = tree.ones_like(M.in_structure)
        masked_i = jax.tree.leaves(M(ones))[0]
        return jnp.int64(P.T(StokesI(masked_i)).i)  # type: ignore[no-any-return]

    @jax.jit(static_argnames=('config',))
    def accumulate_rhs(self, reader: ObservationReader[Any], config: MapMakingConfig) -> Stokes:
        """Accumulate RHS vector across all observations."""

        def prepare_tod(model, data):  # type: ignore[no-untyped-def]
            tod = data['sample_data']
            if not config.gaps.fill or config.binned:
                return tod

            # FIXME: check with demodulated data
            N = _noise_operator(
                model.noise_model,
                model.tod_structure,
                model.sample_rate,
                config.noise.correlation_length,
                inverse=False,
            )
            return GapFillingOperator(
                N,  # type: ignore[arg-type]
                model._get_indexer(),
                data['metadata'],
                model.W,
                rate=model.sample_rate,
                max_cg_steps=config.gaps.fill_options.max_steps,
                rtol=config.gaps.fill_options.rtol,
            )(jax.random.key(config.gaps.fill_options.seed), tod)

        def step(carry, args):  # type: ignore[no-untyped-def]
            i, model = args
            data, _ = reader.read(i)
            tod = prepare_tod(model, data)
            rhs = (model.H.T @ model.masker @ model.W)(tod)
            return carry + rhs, None

        init = tree.zeros_like(self.map_structure)
        total, _ = jax.lax.scan(step, init, (jnp.arange(reader.count), self))
        return total  # type: ignore[no-any-return]

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


_StokesPyTree = TypeVar('_StokesPyTree', bound=StokesPyTreeType)


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
