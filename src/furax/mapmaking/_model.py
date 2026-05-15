import functools
from dataclasses import dataclass, field
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int64, PyTree

from furax import AbstractLinearOperator, IdentityOperator, MaskOperator, symmetric, tree
from furax.core import BlockDiagonalOperator, BlockRowOperator, CompositionOperator, IndexOperator
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesPyTreeType

from . import templates as _templates
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


_StokesPyTree = TypeVar('_StokesPyTree', bound=StokesPyTreeType)


@functools.partial(jax.jit, static_argnames=['diag'])
def _system_scan(model: ObservationModel, x: _StokesPyTree, *, diag: bool = False) -> _StokesPyTree:
    """Apply H^T W H to x, summed over the batch of observations in model.

    `model` is an explicit JIT argument so it is never captured as an XLA constant.

    Args:
        model: Stacked ObservationModel with a leading batch dimension over observations.
        x: Input sky map pytree.
        diag: If True, use the diagonal white-noise approximation for W.
    """

    def accumulate(lhs, obs):  # type: ignore[no-untyped-def]
        return tree.add(lhs, obs.apply_system(x, white_noise=diag)), None

    lhs, _ = jax.lax.scan(accumulate, tree.zeros_like(x), model)
    return lhs


@symmetric
class SystemOperator(AbstractLinearOperator):
    """System operator for multiple observations: `A = Σ_i H_i^T W_i H_i`."""

    models: ObservationModel
    diag: bool = field(metadata={'static': True})

    def __init__(self, models: ObservationModel, *, diag: bool = False):
        object.__setattr__(self, 'models', models)
        object.__setattr__(self, 'diag', diag)
        object.__setattr__(self, 'in_structure', models.map_structure)

    def mv(self, x: _StokesPyTree) -> _StokesPyTree:
        return _system_scan(self.models, x, diag=self.diag)  # type: ignore[no-any-return]


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


@register_dataclass
@dataclass
class ObservationTemplates:
    """Per-observation template operator, stackable across observations via ``jax.lax.scan``.

    The wrapped ``obs_template_op`` is a ``BlockRowOperator`` whose blocks are the active
    :class:`~furax.mapmaking.templates.TemplateOperator` subclasses. Static fields
    (``n_params``, ``n_dets``, ``n_samps``) must match across observations, which is the
    case as long as template construction depends only on reader-padded fields.
    """

    obs_template_op: AbstractLinearOperator

    @classmethod
    def create(
        cls,
        data: Any,
        config: MapMakingConfig,
        tod_structure: jax.ShapeDtypeStruct,
    ) -> 'ObservationTemplates':
        return cls(obs_template_op=_build_obs_templates_op(config, data, tod_structure))


def _build_obs_templates_op(
    config: MapMakingConfig,
    data: Any,
    tod_structure: jax.ShapeDtypeStruct,
) -> BlockRowOperator:
    """Build the per-observation template ``BlockRowOperator`` from a single-obs data dict.

    Mirrors :meth:`MapMaker.get_template_operator` but sources fields from the multi-obs
    ``data`` dict (reader-padded arrays) instead of an :class:`AbstractGroundObservation`.
    """
    tcfg = config.templates
    assert tcfg is not None, 'templates config required to build template operator'
    n_dets, n_samps = tod_structure.shape
    dtype = config.dtype
    blocks: dict[str, AbstractLinearOperator] = {}

    if tcfg.polynomial is not None:
        raise NotImplementedError(
            'Polynomial templates are not supported in the multi-obs two-step path: '
            'PolynomialTemplateOperator stores per-interval blocks as a Python list of '
            'arrays with varying lengths and cannot be stacked across observations.'
        )

    if (sss := tcfg.scan_synchronous) is not None:
        blocks['scan_synchronous'] = _templates.ScanSynchronousTemplateOperator.create(
            min_poly_order=sss.min_poly_order,
            max_poly_order=sss.max_poly_order,
            azimuth=data['azimuth'],
            n_dets=n_dets,
            dtype=dtype,
        )

    if (hwpss := tcfg.hwp_synchronous) is not None:
        blocks['hwp_synchronous'] = _templates.HWPSynchronousTemplateOperator.create(
            n_harmonics=hwpss.n_harmonics,
            hwp_angles=data['hwp_angles'],
            n_dets=n_dets,
            dtype=dtype,
        )

    if (azhwpss := tcfg.azhwp_synchronous) is not None:
        if azhwpss.split_scans:
            blocks['azhwp_synchronous_left'] = (
                _templates.AzimuthHWPSynchronousTemplateOperator.create(
                    n_polynomials=azhwpss.n_polynomials,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data['azimuth'],
                    hwp_angles=data['hwp_angles'],
                    n_dets=n_dets,
                    dtype=dtype,
                    scan_mask=data['left_scan_mask'],
                )
            )
            blocks['azhwp_synchronous_right'] = (
                _templates.AzimuthHWPSynchronousTemplateOperator.create(
                    n_polynomials=azhwpss.n_polynomials,
                    n_harmonics=azhwpss.n_harmonics,
                    azimuth=data['azimuth'],
                    hwp_angles=data['hwp_angles'],
                    n_dets=n_dets,
                    dtype=dtype,
                    scan_mask=data['right_scan_mask'],
                )
            )
        else:
            blocks['azhwp_synchronous'] = _templates.AzimuthHWPSynchronousTemplateOperator.create(
                n_polynomials=azhwpss.n_polynomials,
                n_harmonics=azhwpss.n_harmonics,
                azimuth=data['azimuth'],
                hwp_angles=data['hwp_angles'],
                n_dets=n_dets,
                dtype=dtype,
            )

    if (binazhwpss := tcfg.binazhwp_synchronous) is not None:
        blocks['binazhwp_synchronous'] = _templates.BinAzimuthHWPSynchronousTemplateOperator.create(
            n_azimuth_bins=binazhwpss.n_azimuth_bins,
            n_harmonics=binazhwpss.n_harmonics,
            interpolate_azimuth=binazhwpss.interpolate_azimuth,
            smooth_interpolation=binazhwpss.smooth_interpolation,
            azimuth=data['azimuth'],
            hwp_angles=data['hwp_angles'],
            n_dets=n_dets,
            dtype=dtype,
        )

    if tcfg.ground is not None:
        raise NotImplementedError(
            'Ground templates are not supported in the multi-obs two-step path: a '
            'shared HorizonLandscape across observations is required.'
        )

    if not blocks:
        raise ValueError('config.templates is set but no template is active.')

    return BlockRowOperator(blocks=blocks)


@functools.partial(jax.jit, static_argnames=['transpose'])
def _template_scan(
    templates: ObservationTemplates,
    x: PyTree[Array],
    *,
    transpose: bool = False,
) -> PyTree[Array]:
    """Apply per-observation template operator over a stacked batch.

    Args:
        templates: Stacked :class:`ObservationTemplates` with a leading batch dimension.
        x: Stacked input with the same leading batch dimension (amplitudes if
            ``transpose=False``, TODs otherwise).
        transpose: If True, apply ``T^T`` per obs (TOD → amplitudes); else ``T`` (amplitudes
            → TOD).
    """

    def step_forward(_, args):  # type: ignore[no-untyped-def]
        obs_t, x_i = args
        return None, obs_t.obs_template_op(x_i)

    def step_transpose(_, args):  # type: ignore[no-untyped-def]
        obs_t, x_i = args
        return None, obs_t.obs_template_op.T(x_i)

    step = step_transpose if transpose else step_forward
    _, out = jax.lax.scan(step, None, (templates, x))
    return out


class MultiObsTemplateOperator(AbstractLinearOperator):
    """Template operator for multiple observations: block-diag over obs.

    Input: stacked amplitudes (each leaf has a leading ``n_obs`` dim).
    Output: stacked TODs (leading ``n_obs`` dim).
    """

    templates: ObservationTemplates
    n_obs: int = field(metadata={'static': True})

    def __init__(self, templates: ObservationTemplates, n_obs: int):
        per_obs_in = templates.obs_template_op.in_structure
        stacked_in = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct((n_obs, *s.shape), s.dtype), per_obs_in
        )
        object.__setattr__(self, 'templates', templates)
        object.__setattr__(self, 'n_obs', n_obs)
        object.__setattr__(self, 'in_structure', stacked_in)

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return _template_scan(self.templates, x, transpose=False)

    def transpose(self) -> AbstractLinearOperator:
        return _MultiObsTemplateTransposeOperator(self)


class _MultiObsTemplateTransposeOperator(AbstractLinearOperator):
    """Transpose of :class:`MultiObsTemplateOperator`."""

    operator: MultiObsTemplateOperator

    def __init__(self, operator: MultiObsTemplateOperator):
        per_obs_out = operator.templates.obs_template_op.out_structure
        stacked_out = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct((operator.n_obs, *s.shape), s.dtype), per_obs_out
        )
        object.__setattr__(self, 'operator', operator)
        object.__setattr__(self, 'in_structure', stacked_out)

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        return _template_scan(self.operator.templates, x, transpose=True)

    def transpose(self) -> AbstractLinearOperator:
        return self.operator
