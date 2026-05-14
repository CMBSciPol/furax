from dataclasses import dataclass, field
from functools import partial
from typing import Any, Self, TypeVar

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int64, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, symmetric, tree
from furax.core import BlockDiagonalOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesPyTreeType

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

    def hits(self) -> Int64[Array, ' pixels']:
        assert isinstance(self.H, CompositionOperator)  # mypy assert
        # the pointing operator should be the first in the acquisition chain, so the last operand...
        # there is a unit test for that
        pointing = self.H.operands[-1]
        assert isinstance(pointing, PointingOperator)  # mypy assert
        pointing_i = pointing.as_stokes_i(interpolate=False)
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
        rhs: Stokes = self.H.T(self.masker(self.W(tod)))
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


def pad_model(models: ObservationModel, n_pad: int) -> ObservationModel:
    """Pad models with n_pad dummy observations that contribute nothing to sums.

    The dummy observations are copies of the last real observation with their
    masker array leaves zeroed out, so masker(x) = 0 for any x.
    """
    last = jax.tree.map(lambda a: jnp.repeat(a[-1:], n_pad, axis=0), models)
    zero_masker = jax.tree.map(jnp.zeros_like, last.masker)
    padded = ObservationModel(
        H=last.H,
        W=last.W,
        masker=zero_masker,
        noise_model=last.noise_model,
        sample_rate=last.sample_rate,
    )
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), models, padded)  # type: ignore[no-any-return]


_StokesPyTree = TypeVar('_StokesPyTree', bound=StokesPyTreeType)


@partial(jax.jit, static_argnames=['diag', 'mesh'])
def _system_scan(
    model: ObservationModel,
    x: _StokesPyTree,
    *,
    diag: bool = False,
    mesh: jax.sharding.Mesh | None = None,
) -> _StokesPyTree:
    """Apply H^T W H to x, summed over all observations in model.

    `model` is an explicit JIT argument so it is never captured as an XLA constant.

    Args:
        model: Stacked ObservationModel with a flat leading obs axis sharded across
            devices. Each device scans its slice sequentially; psum reduces globally.
        x: Input sky map pytree.
        diag: If True, use the diagonal white-noise approximation for W.
        mesh: The concrete Mesh on which `model` is sharded. When set, shard_map is
            used to parallelise the scan across devices. Must be the same Mesh object
            that was used when sharding the model arrays.
    """

    if mesh is None:

        def step(lhs: _StokesPyTree, obs: ObservationModel) -> tuple[_StokesPyTree, None]:
            return tree.add(lhs, obs.apply_system(x, white_noise=diag)), None

        lhs, _ = jax.lax.scan(step, tree.zeros_like(x), model)
        return lhs

    # lineax's abstract eval drops the NamedSharding on closure-captured arrays,
    # so shard_map sees replicated leaves and rejects the P('obs') in_specs.
    # Reshard restores it (the fix JAX's error message points to).
    obs_sharding = NamedSharding(mesh, P('obs'))
    model = jax.tree.map(lambda a: jax.reshard(a, obs_sharding), model)

    # check_vma=False: furax operator chain inside the body lacks manual-axis annotations.
    @jax.shard_map(mesh=mesh, in_specs=(P('obs'), P()), out_specs=P(), check_vma=False)
    def local_scan(local_model: ObservationModel, local_x: _StokesPyTree) -> _StokesPyTree:
        def step(lhs: _StokesPyTree, obs: ObservationModel) -> tuple[_StokesPyTree, None]:
            return tree.add(lhs, obs.apply_system(local_x, white_noise=diag)), None

        lhs, _ = jax.lax.scan(step, tree.zeros_like(local_x), local_model)
        return jax.lax.psum(lhs, axis_name='obs')  # type: ignore[no-any-return]

    return local_scan(model, x)


@symmetric
class SystemOperator(AbstractLinearOperator):
    """System operator for multiple observations: `A = Σ_i H_i^T W_i H_i`."""

    models: ObservationModel
    diag: bool = field(metadata={'static': True})
    mesh: jax.sharding.Mesh | None = field(metadata={'static': True})

    def __init__(
        self,
        models: ObservationModel,
        *,
        diag: bool = False,
        mesh: jax.sharding.Mesh | None = None,
    ):
        object.__setattr__(self, 'models', models)
        object.__setattr__(self, 'diag', diag)
        object.__setattr__(self, 'mesh', mesh)
        object.__setattr__(self, 'in_structure', models.map_structure)

    def mv(self, x: _StokesPyTree) -> _StokesPyTree:
        return _system_scan(self.models, x, diag=self.diag, mesh=self.mesh)  # type: ignore[no-any-return]


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
