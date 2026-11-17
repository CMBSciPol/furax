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
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jax.tree_util import register_dataclass
from jaxtyping import Array, Int64, PyTree

from furax import AbstractLinearOperator
from furax.core import Policy, StackedEnsemble, lift, map_reduce
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import StokesI, StokesType

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
    restrict_streams,
)
from .config import GapTreatment, InjectionMode, MapMakingConfig
from .gap_filling import gap_fill
from .layout import TileBucket, TilePlacement
from .templates.gram import factor
from .weight import WeightOperator

PASS_REDUCTIONS = {
    'hit_map': 'sum',
    # keyed as the unknowns: the map RHS summed, that of each kind of amplitudes held as the
    # tiles hold them
    'rhs': {'map': 'sum'}
    | {
        kind: 'segment_sum' if stride == 'segment' else stride
        for kind, stride in AMPLITUDE_STRIDES.items()
    },
    # the Gram of the templates an observation's detectors share, summed over its tiles, paired
    # with its coarse amplitudes, which follow from the bases alone, the same in every tile
    'shared_gram': {'gram': 'segment_sum', 'coarse': 'segment'},
    # what the tiles of an observation share (boresight pointing, bases of the shared templates,
    # ...) is kept once per segment, the rest per tile
    'tile': 'auto',
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
    observation share."""


def pass_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, gaps: GapSplines | None = None
) -> Callable[[tuple[Any, Any, Array], StokesType | None], dict[str, Any]]:
    r"""The kernel of the pass over a bucket: the contribution of one detector tile.

    The member is one tile's read, the triple `(data, padding, use)` restricted to the tile's
    detectors; the sky to inject, if any, is a shared input that every member receives whole.
    The kernel returns a dictionary, keyed as [`PASS_REDUCTIONS`][]:

    - `'hit_map'`: the hit map of the tile's good samples;
    - `'rhs'`: its right-hand side $J^\top W' M d$ (see
      [`Tile.rhs`][furax.mapmaking._tile.Tile.rhs]), keyed as the unknowns: the `'map'` and
      each kind of `AMPLITUDE_STRIDES`, `None` for a kind absent;
    - `'shared_gram'`: the templates its observation's detectors share, under `'gram'` its share
      of their Gram and under `'coarse'` their coarse amplitudes, both `None` without them. The
      Gram is factored once summed over the tiles;
    - `'tile'`: the [`Tile`][furax.mapmaking._tile.Tile], the Gram of the templates its
      observation's detectors share not yet factored.

    Args:
        config: The mapmaking configuration.
        landscape: The landscape of the maps.
        gaps: The spline model of the explicit gaps of the bucket, if any.
    """
    fill_gaps = config.gap_treatment == GapTreatment.FILL
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE

    def kernel(member, sky):
        data, padding, use = member
        # The TOD is read in float32, whatever the pipeline's dtype
        tod = jax.tree.map(lambda x: x.astype(config.dtype), data[ReaderField.SAMPLE_DATA])
        data = {**data, ReaderField.SAMPLE_DATA: tod}
        if sky is not None:  # the samples to inject into, per detector
            shape = jax.tree.leaves(tod)[0].shape
            H = acquisition_operator(data, config, sky_landscape)
            data = inject(H, sky, data, unpadded(padding, shape), replaces)
        tod = raw = data[ReaderField.SAMPLE_DATA]
        if fill_gaps:
            # Gap filling needs whole observations, so it only runs with one tile per observation
            # (see `MapMakingConfig`). The RHS uses the filled TOD; the models are still built
            # from the raw one.
            # TODO: `_filled` and `_contribution` each build the tile (pointing, mask, noise fit)
            # from the same data; build it once before the `cond`, minding `restrict_streams`.
            tod = jax.lax.cond(use, lambda: _filled(data, padding, config, landscape), lambda: raw)
        return _contribution(data, tod, padding, use, config, landscape, gaps)

    return kernel


def bucket_tiles(
    outputs: dict[str, Any],
    resolved: dict[str, Any],
    bucket: TileBucket,
    placement: TilePlacement,
    config: MapMakingConfig,
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
    """
    tiles = StackedEnsemble(
        bucket.n_tiles,
        outputs['tile'],
        resolved['tile'],  # the kinds the leaves resolved to are their strides
        policy=Policy(axis=placement.axis),
        owner=placement.shard(bucket.owner),
        segments=bucket.n_segments,
    )
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
) -> dict[str, Any]:
    """The contribution of one tile, from the fields read for it and its TOD."""
    temperature = None
    if config.read_stokes != landscape.stokes:
        # The extra streams read, the I one for the T2P template, stay out of the model
        temperature = data[ReaderField.SAMPLE_DATA].i
        data, tod = restrict_streams(data, tod, landscape.stokes)
    # Padding/failed observations contribute nothing
    tile = Tile.create(data, padding, config, landscape, use, gaps)

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

    gram = None
    if config.use_templates:
        implicit, explicit, gram = assemble_templates(data, config, tile, temperature)
        tile = replace(tile, implicit=implicit, explicit=explicit)

    if config.gap_treatment == GapTreatment.FILL and not config.use_templates:
        # skip the inner mask of W, so that the fill survives N⁻¹
        assert isinstance(tile.W, WeightOperator)
        rhs = {'map': (tile.H.T @ tile.M @ tile.W.weight @ tile.F).reduce()(tod)}
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
        'tile': tile,
    }


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

    def factor_one(member):
        gram, coarse = member
        return factor(replace(gram, coarse=coarse), regularization=regularization)

    inverse, kinds = map_reduce(factor_one, ensemble, 'auto')
    return StackedEnsemble(
        tiles.size,
        inverse,
        # stacked over the segments, an inverse is one per observation among the tiles; the
        # shared leaves, of kind None, stay None
        jax.tree.map(lambda _: 'segment', kinds),
        policy=tiles.policy,
        owner=tiles.owner,
        segments=tiles.segments,
    )


def _with_shared_gram_inverse(tile: Tile, inverse: tuple[AbstractLinearOperator, Any]) -> Tile:
    """`tile` with the Gram of the templates its observation's detectors share factored."""
    shared = TemplateBundle(tile.explicit[SHARED_TEMPLATES].operator, *inverse)
    return replace(tile, explicit=tile.explicit | {SHARED_TEMPLATES: shared})
