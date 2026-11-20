"""The pass over the data: what each detector tile contributes to the system, and its buckets.

The mapmaker reads its observations once, a bucket at a time. Each detector tile of a bucket goes
through [`pass_kernel`][], which builds the tile and returns its hit map, its right-hand side, its
share of the Gram of the templates its observation's detectors share, and the
[`Tile`][furax.mapmaking._tile.Tile] the normal system keeps. The engine combines them over the
tiles as [`PASS_REDUCTIONS`][] says, and [`bucket_tiles`][] makes the tiles of the bucket a
stacked ensemble. The pass returns the tiles of each bucket, and what it sums over all of them as
[`PassTotals`][].
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Int, Int64, PyTree

from furax import AbstractLinearOperator
from furax.core import Policy, StackedEnsemble, lift, map_reduce
from furax.obs import TODFormat
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType

from ._diagnostics import tile_statistics
from ._gaps import GapSplines
from ._injection import inject, unpadded
from ._observation import ReaderField
from ._tile import (
    AMPLITUDE_STRIDES,
    SHARED_TEMPLATES,
    TemplateBundle,
    Tile,
    acquisition_operator,
    assemble_templates,
    low_rank_tile_spectra,
    restrict_streams,
    shared_deprojection,
)
from .config import DiagnosticsConfig, GapTreatment, InjectionMode, MapMakingConfig, WeightingMode
from .gap_filling import gap_fill
from .layout import TileBucket, TilePlacement
from .lowrank import LowRankNoiseModel, ModesCore, fit_low_rank_spectra
from .noise import NoiseModel
from .templates.gram import BlockGram, factor, factor_diagonal
from .weight import WeightOperator

_REDUCTIONS_OF_STRIDES = {1: 1, 'segment': 'segment_sum', None: 'sum'}
"""How the pass combines the right-hand side of amplitudes held with each stride."""

PASS_REDUCTIONS = {
    'hit_map': 'sum',
    # keyed as the unknowns: the map RHS summed, that of each kind of amplitudes held as the
    # tiles hold them
    'rhs': {'map': 'sum'}
    | {kind: _REDUCTIONS_OF_STRIDES[stride] for kind, stride in AMPLITUDE_STRIDES.items()},
    # the Gram of the templates an observation's detectors share, summed over its tiles, paired
    # with its coarse amplitudes, which follow from the bases alone, the same in every tile
    'shared_gram': {'gram': 'segment_sum', 'coarse': 'segment'},
    # the diagonal of the Gram of the templates every observation shares, summed over every tile
    'global_diagonal': 'sum',
    # what the tiles of an observation share (boresight pointing, bases of the shared templates,
    # ...) is kept once per segment, the rest per tile
    'tile': 'auto',
    # under a low-rank noise model, the amplitudes of the correlated modes of the data, summed
    # over the tiles of each observation
    'modes': 'segment_sum',
}
"""How the engine combines the outputs of [`pass_kernel`][] over the tiles of a bucket (see
`map_reduce`): the hit map and the map RHS are summed, and the rest is kept where it lives, per
tile or per observation."""


@register_dataclass
@dataclass(frozen=True)
class PassTotals:
    """What the pass over the data sums over the tiles of every bucket."""

    hit_map: Int64[Array, '...']
    """Hit map over every observation, replicated on every device."""
    rhs: dict[str, Any]
    """The right-hand side of the normal system, keyed as its unknowns: the map RHS over the full
    sky under `'map'`, replicated on every device, and for each kind of amplitudes present (see
    [`AMPLITUDE_STRIDES`][furax.mapmaking._tile.AMPLITUDE_STRIDES]) a list with the right-hand
    side of each bucket, `(tile, ...)`, or `(segment, ...)` for the amplitudes the detectors of an
    observation share; that of the amplitudes every observation shares is summed over the
    buckets, as the map RHS is."""
    modes: list[Array] | None = None
    r"""Under a low-rank noise model, the amplitudes $\sum_t T_t^\top D_t^{-1} M_t d_t$ of the
    correlated modes of the data of each bucket, `(segment, frequency, mode, 2)` (see
    [`Tile.modes`][furax.mapmaking._tile.Tile.modes]); `None` otherwise."""
    whole_grams: dict[str, tuple[AbstractLinearOperator, AbstractLinearOperator | None]] = field(
        default_factory=dict
    )
    """For each kind of amplitudes every bucket reads whole, those every observation shares, the
    inverse of the diagonal of their Gram, summed over every tile, and the ridge added to it."""


def pass_kernel(
    config: MapMakingConfig,
    landscape: StokesLandscape,
    gaps: GapSplines | None = None,
) -> Callable[[Any, StokesType | None], dict[str, Any]]:
    r"""The kernel of the pass over a bucket: the contribution of one detector tile.

    The member is one tile's read, the triple `(data, padding, use)` restricted to the tile's
    detectors; under a low-rank noise model, the pair of it and the model of the tile's observation
    (see [`fit_low_rank`][]) with the tile's first detector, of which the tile takes its rows. The
    sky to inject, if any, is a shared input that every member receives whole.
    The kernel returns a dictionary, keyed as [`PASS_REDUCTIONS`][]:

    - `'hit_map'`: the hit map of the tile's good samples;
    - `'rhs'`: its right-hand side $J^\top W' M d$ (see
      [`Tile.rhs`][furax.mapmaking._tile.Tile.rhs]), keyed as the unknowns: the `'map'` and
      each kind of `AMPLITUDE_STRIDES`, `None` for a kind absent;
    - `'shared_gram'`: the templates its observation's detectors share, under `'gram'` its share
      of their Gram and under `'coarse'` their coarse amplitudes, both `None` without them. The
      Gram is factored once summed over the tiles;
    - `'global_diagonal'`: its share of the diagonal of the Gram of the templates every
      observation shares, `None` without them;
    - `'tile'`: the [`Tile`][furax.mapmaking._tile.Tile], the Gram of the templates its
      observation's detectors share not yet factored;
    - `'modes'`: under a low-rank noise model, the amplitudes $T^\top D^{-1} M d$ of the
      correlated modes of its data, `None` otherwise.

    Args:
        config: The mapmaking configuration.
        landscape: The landscape of the maps.
        gaps: The spline model of the explicit gaps of the bucket, if any.
    """
    low_rank = config.weighting.mode == WeightingMode.LOW_RANK
    fill_gaps = config.gap_treatment == GapTreatment.FILL
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE

    def kernel(member, sky):
        (data, padding, use), fitted = member if low_rank else (member, None)
        data = _prepared(data, padding, sky, config, sky_landscape, replaces)
        noise_model = None if fitted is None else _tile_model(*fitted, data)
        tod = raw = data[ReaderField.SAMPLE_DATA]
        if fill_gaps:
            # Gap filling needs whole observations, so it only runs with one tile per observation
            # (see `MapMakingConfig`). The RHS uses the filled TOD; the models are still built
            # from the raw one.
            # TODO: `_filled` and `_contribution` each build the tile (pointing, mask, noise fit)
            # from the same data; build it once before the `cond`, minding `restrict_streams`.
            tod = jax.lax.cond(use, lambda: _filled(data, padding, config, landscape), lambda: raw)
        return _contribution(data, tod, padding, use, config, landscape, gaps, noise_model)

    return kernel


def residual_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, diagnostics: DiagnosticsConfig
) -> Callable[[tuple[Any, tuple[Tile, dict[str, Any]]], StokesType | None, dict[str, Any]], Any]:
    """The kernel of the residual pass: the statistics of one detector tile.

    The member is one tile's read, as for [`pass_kernel`][], paired with the
    [`Tile`][furax.mapmaking._tile.Tile] the pass over the data built from it and the amplitudes
    of each kind it reads, solved for. The sky to inject, if any, and the unknowns every tile
    reads whole, the map solved for on the whole sky, under `'map'`, and the amplitudes every
    observation shares, are shared inputs. The kernel returns the statistics of
    [`tile_statistics`][furax.mapmaking._diagnostics.tile_statistics].

    Args:
        config: The mapmaking configuration.
        landscape: The landscape of the maps.
        diagnostics: The statistics to compute.
    """
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE

    def kernel(member, sky, solution):
        (data, padding, _), (tile, amplitudes) = member
        data, _, _ = _model_fields(data, padding, sky, config, landscape, sky_landscape, replaces)
        tod = data[ReaderField.SAMPLE_DATA]
        return tile_statistics(tile, tod, solution | amplitudes, diagnostics)

    return kernel


def fit_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, capacity: int, sample_rate: float
) -> Callable[[tuple[Any, Array], StokesType | None], tuple[Array, Array, Array]]:
    """The kernel of the pass that fits the low-rank noise model of each observation.

    The member is one tile's read, as for [`pass_kernel`][], and the tuple of its first detector
    and, when the detectors share templates, their amplitudes in its observation, fitted over it by
    the pass of [`shared_fit_kernel`][]. The spectrum
    and white level of each channel of the tile (see
    [`low_rank_tile_spectra`][furax.mapmaking._tile.low_rank_tile_spectra]) go to its rows of
    arrays `(stream, capacity, ...)`, zero elsewhere, so that their sum over the tiles of an
    observation holds all its channels; then comes the number of samples of the observation.

    Args:
        config: The mapmaking configuration.
        landscape: The landscape of the maps.
        capacity: The detectors of the arrays of an observation, a whole number of tiles.
        sample_rate: The sample rate at which the bins of the model are placed, in Hz.
    """
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE

    def kernel(member, sky):
        (data, padding, use), (start, *shared) = member
        shared_amplitudes = shared[0] if shared else None
        data, tod_format, temperature = _model_fields(
            data, padding, sky, config, landscape, sky_landscape, replaces
        )
        spectra, level, n_real = low_rank_tile_spectra(
            data, config, tod_format, padding, sample_rate, temperature, shared_amplitudes
        )
        n_streams = 1 if tod_format.stokes is None else tod_format.n_streams
        zero = jnp.zeros((), jnp.int32)
        spectra = spectra.reshape(n_streams, -1, spectra.shape[-1]) * use
        level = level.reshape(n_streams, -1) * use
        placed = (
            jax.lax.dynamic_update_slice(
                jnp.zeros((n_streams, capacity, *x.shape[2:]), x.dtype),
                x,
                (zero, start.astype(zero.dtype), *(zero for _ in x.shape[2:])),
            )
            for x in (spectra, level)
        )
        return (*placed, n_real)

    return kernel


def shared_fit_kernel(
    config: MapMakingConfig, landscape: StokesLandscape
) -> Callable[[Any, StokesType | None], dict[str, Any]]:
    """The kernel of the pass that fits the templates the detectors of an observation share.

    It precedes the pass of [`fit_kernel`][], whose TOD it rids of those templates, fitted over
    the whole observation rather than over each tile. The member is one tile's read, as for
    [`pass_kernel`][]; the kernel returns its share of their Gram and right-hand side (see
    [`shared_deprojection`][furax.mapmaking._tile.shared_deprojection]), keyed as
    [`SHARED_FIT_REDUCTIONS`][].

    Args:
        config: The mapmaking configuration.
        landscape: The landscape of the maps.
    """
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE

    def kernel(member, sky):
        data, padding, use = member
        data, tod_format, temperature = _model_fields(
            data, padding, sky, config, landscape, sky_landscape, replaces
        )
        terms = shared_deprojection(data, config, tod_format, padding, temperature)
        gram = terms.pop('gram')
        coarse = None if gram is None else gram.coarse
        terms['gram'] = None if gram is None else replace(gram, coarse=None)
        # padding tiles and failed observations contribute nothing
        return jax.tree.map(lambda x: x * use, terms) | {'coarse': coarse}

    return kernel


SHARED_FIT_REDUCTIONS = {
    'gram': 'segment_sum',
    'coarse': 'segment',
    'rhs': 'segment_sum',
    'global_diagonal': 'segment_sum',
    'global_rhs': 'segment_sum',
}
"""How the engine combines the outputs of [`shared_fit_kernel`][] over the tiles of a bucket: the
Grams and the right-hand sides summed over the tiles of each observation, the coarse amplitudes of
the Gram, which follow from the bases alone, kept per observation."""


def shared_fit_amplitudes(
    outputs: dict[str, Any], n_segments: int, regularization: float, axis: str
) -> PyTree[Array]:
    """The amplitudes of the templates the detectors of each observation share, `(segment, ...)`.

    The outputs of [`shared_fit_kernel`][], combined over the tiles of each observation, are
    solved for on the device that holds it.
    """
    segments = StackedEnsemble(n_segments, outputs, policy=Policy(axis=axis))

    def solve(member):
        amplitudes = {}
        if member['gram'] is not None:
            inverse, _ = _factored(member['gram'], member['coarse'], regularization)
            amplitudes |= inverse(member['rhs'])
        if member['global_diagonal'] is not None:
            inverse, _ = factor_diagonal(member['global_diagonal'], regularization=regularization)
            amplitudes |= inverse(member['global_rhs'])
        return amplitudes

    amplitudes, _ = map_reduce(solve, segments, 1)
    return amplitudes


def _model_fields(
    data: dict[str, Any],
    padding: Any,
    sky: StokesType | None,
    config: MapMakingConfig,
    landscape: StokesLandscape,
    sky_landscape: StokesLandscape,
    replaces: bool,
) -> tuple[dict[str, Any], TODFormat, Array | None]:
    """The fields of a tile for its model, its TOD format, and the I stream read for T2P.

    The I stream, read for the T2P template only, stays out of the model; `None` if not read.
    """
    data = _prepared(data, padding, sky, config, sky_landscape, replaces)
    tod = data[ReaderField.SAMPLE_DATA]
    temperature = None
    if config.read_stokes != landscape.stokes:
        temperature = tod.i
        data, tod = restrict_streams(data, tod, landscape.stokes)
    return data, _tod_format_of(tod), temperature


def fit_low_rank(
    spectra: Array,
    level: Array,
    n_real: Array,
    config: MapMakingConfig,
    landscape: StokesLandscape,
    capacity: int,
    n_samples: int,
    sample_rate: float,
    axis: str,
) -> tuple[LowRankNoiseModel, ModesCore]:
    """The low-rank noise model of each observation of a bucket, and its core $K$.

    The inputs are the outputs of [`fit_kernel`][] summed over the tiles of each observation,
    `(segment, ...)`, which is fitted on the device that holds it. The detector axis of the models
    is padded to `capacity`: the rows past the detectors of an observation are dead channels.

    Returns:
        The models and their cores (see
        [`LowRankNoiseModel.modes_core`][furax.mapmaking.lowrank.LowRankNoiseModel.modes_core]),
        their leaves `(segment, ...)`.
    """
    lead = (len(landscape.stokes), capacity) if config.demodulated else (capacity,)

    def fit(member):
        spectra, level, n_real = member
        model = fit_low_rank_spectra(
            spectra.reshape(-1, spectra.shape[-1]),
            level.reshape(-1),
            n_samples,
            n_real,
            sample_rate,
            lead=lead,
            config=config.weighting.low_rank,
        )
        return model, model.modes_core(n_samples)

    segments = StackedEnsemble(spectra.shape[0], (spectra, level, n_real), policy=Policy(axis=axis))
    (models, cores), _ = map_reduce(fit, segments, 1)
    return models, cores


def bucket_tiles(
    outputs: dict[str, Any],
    resolved: dict[str, Any],
    bucket: TileBucket,
    placement: TilePlacement,
    config: MapMakingConfig,
    cores: ModesCore | None = None,
) -> StackedEnsemble:
    """The tiles of a bucket, from the outputs of [`pass_kernel`][] combined over the pass.

    Each leaf of the tiles is held as the engine combined it: per tile, per segment, or once for
    the bucket. The Grams of the templates the detectors of an observation share are factored
    here, once summed over its tiles, on the device that holds the observation, and the templates
    joined to its tiles.

    Args:
        outputs: The outputs of the kernel, combined over the tiles.
        resolved: How each output of the kernel was combined, as `map_reduce` resolves it.
        bucket: The bucket.
        placement: The placement of its tiles on the devices.
        config: The mapmaking configuration.
        cores: Under a low-rank noise model, the core $K$ of the correlated modes of each
            observation, its leaves `(segment, ...)` (see [`fit_low_rank`][]).
    """
    tiles = StackedEnsemble(
        bucket.n_tiles,
        outputs['tile'],
        resolved['tile'],  # the kinds the leaves resolved to are their strides
        policy=Policy(axis=placement.axis),
        owner=placement.shard(bucket.owner),
        segments=bucket.n_segments,
    )
    if cores is not None:
        # structural: the tiles keep their leaves and strides, the cores one per segment
        joined = lift(_with_modes_core, tiles, _per_segment(cores, tiles))
        assert isinstance(joined, StackedEnsemble)
        tiles = joined
    shared = outputs['shared_gram']
    if shared['gram'] is None:
        return tiles
    inverses = _factor_shared_gram(shared['gram'], shared['coarse'], tiles, config)
    # structural: each ensemble keeps its leaves and strides, the inverses one per segment
    joined = lift(_with_shared_gram_inverse, tiles, inverses)
    assert isinstance(joined, StackedEnsemble)
    return joined


def _contribution(
    data: dict[str, Any],
    tod: PyTree[Array],
    padding: Any,
    use: Array,
    config: MapMakingConfig,
    landscape: StokesLandscape,
    gaps: GapSplines | None,
    noise_model: PyTree[NoiseModel] | None = None,
) -> dict[str, Any]:
    """The contribution of one tile, from the fields read for it and its TOD."""
    temperature = None
    if config.read_stokes != landscape.stokes:
        # The extra streams read, the I one for the T2P template, stay out of the model
        temperature = data[ReaderField.SAMPLE_DATA].i
        data, tod = restrict_streams(data, tod, landscape.stokes)
    # Padding/failed observations contribute nothing
    tile = Tile.create(data, padding, config, landscape, use, gaps, noise_model)

    # Hit map = nearest-neighbour coverage of the sample mask
    hit_pointing = PointingOperator.create(
        landscape,
        Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
        Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
    ).as_stokes_i(interpolate=False)
    # Read the mask directly: M(ones) = M.to_boolean_mask()
    # the mask as stored: (det, samp) even on a demodulated TOD, whose streams share it
    masked = jax.tree.leaves(tile.M.to_boolean_mask())[0]
    hits = jnp.int64(hit_pointing.T(StokesI(masked)).i)

    gram = global_diagonal = None
    if config.use_templates:
        implicit, explicit, gram, global_diagonal = assemble_templates(
            data, config, tile, temperature
        )
        tile = replace(tile, implicit=implicit, explicit=explicit)

    modes = None
    if config.gap_treatment == GapTreatment.FILL and not config.use_templates:
        # skip the inner mask of W, so that the fill survives N⁻¹
        assert isinstance(tile.W, WeightOperator)
        rhs = {'map': (tile.H.T @ tile.M @ tile.W.weight @ tile.F).reduce()(tod)}
    elif isinstance(tile.noise_model, LowRankNoiseModel):
        both = tile.rhs_and_modes()(tod)
        rhs, modes = both['unknowns'], both['modes']
    else:
        rhs = tile.rhs()(tod)
    return {
        'hit_map': hits,
        # absent kinds stay `None`, as `PASS_REDUCTIONS` declares every kind
        'rhs': dict.fromkeys(PASS_REDUCTIONS['rhs']) | rhs,
        'shared_gram': {
            'gram': None if gram is None else replace(gram, coarse=None),
            'coarse': None if gram is None else gram.coarse,
        },
        'global_diagonal': global_diagonal,
        'tile': tile,
        'modes': modes,
    }


def _prepared(
    data: dict[str, Any],
    padding: Any,
    sky: StokesType | None,
    config: MapMakingConfig,
    sky_landscape: StokesLandscape,
    replaces: bool,
) -> dict[str, Any]:
    """The fields read for a tile, its TOD in the pipeline's dtype, with the sky injected, if any."""
    # The TOD is read in float32, whatever the pipeline's dtype
    tod = jax.tree.map(lambda x: x.astype(config.dtype), data[ReaderField.SAMPLE_DATA])
    data = {**data, ReaderField.SAMPLE_DATA: tod}
    if sky is not None:  # the samples to inject into, per detector
        shape = jax.tree.leaves(tod)[0].shape
        H = acquisition_operator(data, config, sky_landscape)
        data = inject(H, sky, data, unpadded(padding, shape), replaces)
    return data


def _tile_model(
    model: LowRankNoiseModel, start: Int[Array, ''], data: dict[str, Any]
) -> LowRankNoiseModel:
    """The low-rank noise model of a tile: its rows of that of its observation, from `start`."""
    n_detectors = jax.tree.leaves(data[ReaderField.SAMPLE_DATA])[0].shape[-2]
    n_streams, capacity = model.lead if len(model.lead) > 1 else (1, model.lead[0])
    detectors = start + jnp.arange(n_detectors)
    channels = (jnp.arange(n_streams)[:, None] * capacity + detectors).ravel()
    lead = (n_detectors,) if len(model.lead) == 1 else (n_streams, n_detectors)
    return model.select(channels, lead)


def _tod_format_of(tod: PyTree[Array]) -> TODFormat:
    """The format of a TOD."""
    if isinstance(tod, Stokes):
        return TODFormat(tod.shape, tod.dtype, tod.stokes)
    return TODFormat(tod.shape, tod.dtype)


def _filled(
    data: dict[str, Any], padding: Any, config: MapMakingConfig, landscape: StokesLandscape
) -> PyTree[Array]:
    """The TOD of a tile, its gaps filled by a constrained noise realization."""
    tile = Tile.create(data, padding, config, landscape)
    # Only reached under GapTreatment.FILL, where W is the plain inner-mask weight.
    assert isinstance(tile.W, WeightOperator)
    # Optional M_b N M_b preconditioner (covariance from the noise model).
    preconditioner = None
    if config.gaps.fill_options.precondition:
        cov = tile.noise_operator(config.weighting.correlation_length, inverse=False)
        m_bad = tile.M.complement()
        preconditioner = (m_bad @ cov @ m_bad).reduce()
    return gap_fill(
        jax.random.key(config.gaps.fill_options.seed),
        data[ReaderField.SAMPLE_DATA],
        tile.W.weight,
        tile.M,
        rate=tile.sample_rate,
        max_cg_steps=config.gaps.fill_options.max_steps,
        rtol=config.gaps.fill_options.rtol,
        preconditioner=preconditioner,
        metadata=data[ReaderField.METADATA],
    )


def _factor_shared_gram(
    gram: Any, coarse: Any, tiles: StackedEnsemble, config: MapMakingConfig
) -> StackedEnsemble:
    """The inverse of the Gram of the shared templates of each observation, and its ridge.

    The Gram is summed over the tiles of the observation, paired with its coarse amplitudes, and
    factored on the device that holds the observation, one observation per round. The inverse is
    returned as an ensemble of `tiles`, each tile reading that of its observation.
    """
    assert config.templates is not None  # a shared Gram comes with templates
    assert tiles.segments is not None  # the tiles are grouped by observation
    regularization = config.templates.regularization
    ensemble = StackedEnsemble(
        tiles.segments, (gram, coarse), policy=Policy(axis=tiles.policy.axis)
    )

    inverse, kinds = map_reduce(lambda member: _factored(*member, regularization), ensemble, 'auto')
    return _per_segment(inverse, tiles, kinds)


def _factored(
    gram: BlockGram, coarse: Any, regularization: float
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """The inverse and ridge of the Gram of an observation's shared templates.

    The Gram is summed over the tiles of the observation, and factored by
    [`factor`][furax.mapmaking.templates.gram.factor] with its coarse amplitudes.
    """
    return factor(replace(gram, coarse=coarse), regularization=regularization)


def _per_segment(stacked: Any, tiles: StackedEnsemble, kinds: Any = None) -> StackedEnsemble:
    """An ensemble of `tiles` whose members read the entries of `stacked`, `(segment, ...)`.

    `kinds` are those `map_reduce` resolved the leaves of `stacked` to, by default `stacked`
    itself: the leaves of kind None, shared, stay None.
    """
    if kinds is None:
        kinds = stacked
    return StackedEnsemble(
        tiles.size,
        stacked,
        # stacked over the segments, an entry is one per observation among the tiles; the
        # shared leaves, of kind None, stay None
        jax.tree.map(lambda _: 'segment', kinds),
        policy=tiles.policy,
        owner=tiles.owner,
        segments=tiles.segments,
    )


def _with_modes_core(tile: Tile, core: ModesCore) -> Tile:
    """`tile` with the core of the correlated modes of its observation."""
    return replace(tile, modes_core=core)


def _with_shared_gram_inverse(tile: Tile, inverse: tuple[AbstractLinearOperator, Any]) -> Tile:
    """`tile` with the Gram of the templates its observation's detectors share factored."""
    shared = TemplateBundle(tile.explicit[SHARED_TEMPLATES].operator, *inverse)
    return replace(tile, explicit=tile.explicit | {SHARED_TEMPLATES: shared})
