"""The pass over the data: what each detector tile contributes to the system, and its buckets.

The mapmaker reads its observations once, a bucket at a time. Each detector tile of a bucket goes
through [`tile_kernel`][], which builds the tile's model and returns its hit map and map RHS, summed
over the tiles, the right-hand side of its amplitudes, its share of the Gram of the templates its
observation's detectors share, and the [`Tile`][furax.mapmaking._bucket.Tile] the normal system
keeps. The engine combines them over the tiles as [`TILE_REDUCTIONS`][] says, and
[`bucket_tiles`][] makes the tiles of the bucket a stacked ensemble.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, PyTree

from furax import AbstractLinearOperator, DiagonalOperator
from furax.core import Policy, StackedEnsemble, lift, map_reduce
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType

from ._bucket import AMPLITUDE_KINDS, GAPS, AmplitudeKind, Tile
from ._gaps import GapSplines
from ._injection import inject, unpadded
from ._model import (
    SHARED_TEMPLATES,
    TEMPLATES,
    TemplateBundle,
    TemplateKind,
    TileModel,
    acquisition_operator,
    assemble_templates,
    restrict_streams,
)
from ._observation import ReaderField
from .config import GapTreatment, InjectionMode, MapMakingConfig
from .gap_filling import gap_fill
from .layout import TileBucket, TilePlacement
from .templates import TemplateOperator
from .templates.gram import factor
from .weight import WeightOperator

TILE_REDUCTIONS = (
    'sum',  # the hit map
    'sum',  # the map RHS
    # the right-hand side of each kind of amplitudes, held where they are
    {
        kind: 'segment_sum' if holder == 'observation' else 1
        for kind, holder in AMPLITUDE_KINDS.items()
    },
    # the templates an observation's detectors share: a Gram summed over its tiles, paired with
    # its coarse amplitudes, which follow from the bases alone, the same in every tile, and the
    # operator, kept as the tile's own
    ('segment_sum', 'segment', 'auto'),
    # what the tiles of an observation share (boresight pointing, bases of the shared templates,
    # ...) is kept once per segment, the rest per tile
    'auto',
)
"""How the engine combines the outputs of [`tile_kernel`][] over the tiles of a bucket (see
`map_reduce`): the hit map and the map RHS are summed, and the rest is kept where it lives, per
tile or per observation."""


def tile_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, gaps: GapSplines | None = None
) -> Callable[[tuple[Any, Any, Array], StokesType | None], tuple[Any, ...]]:
    r"""The kernel of the pass over a bucket: the contribution of one detector tile.

    The member is one tile's read, the triple `(data, padding, use)` restricted to the tile's
    detectors; the sky to inject, if any, is a shared input that every member receives whole.
    The kernel returns, in the order of [`TILE_REDUCTIONS`][]:

    - the hit map of the tile's good samples;
    - its share of the map RHS, $H^\top W' d$;
    - the right-hand side of its amplitudes, keyed by each kind of `AMPLITUDE_KINDS`, `None` for a
      kind absent: its template RHS $T^\top W' d$, its share of that of the templates its
      observation's detectors share, and its gap RHS $C^\top N^{-1} M d$;
    - the templates its observation's detectors share: its share of their Gram, their coarse
      amplitudes and their operator, `(None, None, None)` without them. The Gram is factored once
      summed over the tiles;
    - the [`Tile`][furax.mapmaking._bucket.Tile], without the templates its observation's
      detectors share.

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
            tod = jax.lax.cond(use, lambda: _filled(data, padding, config, landscape), lambda: raw)
        return _contribution(data, tod, padding, use, config, landscape, gaps)

    return kernel


def bucket_tiles(
    shared: tuple[Any, Any, Any],
    tile: Tile,
    resolved: tuple[Any, ...],
    bucket: TileBucket,
    placement: TilePlacement,
    config: MapMakingConfig,
) -> StackedEnsemble:
    """The tiles of a bucket, from the outputs of [`tile_kernel`][] combined over the pass.

    Each leaf of the tiles is held as the engine combined it: per tile, per segment, or once for
    the bucket. The Grams of the templates the detectors of an observation share are factored
    here, once summed over its tiles, on the device that holds the observation, and the templates
    joined to its tiles.

    Args:
        shared: The templates the detectors of each observation share, as the kernel returns them.
        tile: The tiles.
        resolved: How each output of the kernel was combined, as `map_reduce` resolves it.
        bucket: The bucket.
        placement: The placement of its tiles on the devices.
        config: The mapmaking configuration.
    """
    *_, (_, _, operator_kinds), tile_kinds = resolved
    tiles = StackedEnsemble(
        bucket.n_tiles,
        tile,
        tile_kinds,  # the kinds the leaves resolved to are their strides
        policy=Policy(axis=placement.axis),
        owner=placement.shard(bucket.owner),
        segments=bucket.n_segments,
    )
    gram, coarse, operator = shared
    if gram is None:
        return tiles
    operators = replace(tiles, stacked=operator, strides=operator_kinds, axes=0)
    inverses = _factor_shared_gram(gram, coarse, tiles, config)
    # structural: each ensemble keeps its leaves and strides, the inverses one per segment
    joined = lift(_with_shared_templates, tiles, operators, inverses)
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
) -> tuple[Any, ...]:
    """The contribution of one tile, from the fields read for it and its TOD."""
    temperature = None
    if config.read_stokes != landscape.stokes:
        # The extra streams read, the I one for the T2P template, stay out of the model
        temperature = data[ReaderField.SAMPLE_DATA].i
        data, tod = restrict_streams(data, tod, landscape.stokes)
    # Padding/failed observations contribute nothing
    model = TileModel.create(data, padding, config, landscape, use, gaps)

    # Hit map = nearest-neighbour coverage of the sample mask
    hit_pointing = PointingOperator.create(
        landscape,
        Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
        Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
    ).as_stokes_i(interpolate=False)
    # Read the mask directly: M(ones) = M.to_boolean_mask()
    masked_tod = model.M.to_boolean_mask()
    # The mask is (ndet, nsamp) even in the demodulated case (all streams share the same)
    masked = masked_tod.data if isinstance(masked_tod, Stokes) else masked_tod
    hits = jnp.int64(hit_pointing.T(StokesI(masked)).i)

    implicit = templates = shared = None
    if config.use_templates:  # templates are rejected with gap filling (see `MapMakingConfig`)
        implicit, templates, shared = assemble_templates(data, config, model, temperature)
        wd = model.deprojected_weight(implicit)(tod)
        map_rhs = model.H.T(wd)
    else:
        # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
        filled = config.gap_treatment == GapTreatment.FILL
        map_rhs = (model.rhs_operator_prefilled if filled else model.rhs_operator)(tod)
        wd = None
    rhs: dict[AmplitudeKind, Any] = dict.fromkeys(AMPLITUDE_KINDS)
    gap_preconditioner = None
    shared_outputs = (None, None, None)
    if templates is not None:
        rhs[TEMPLATES] = templates.operator.T(wd)
    if shared is not None:
        operator, gram = shared
        rhs[SHARED_TEMPLATES] = operator.T(wd)
        shared_outputs = (replace(gram, coarse=None), gram.coarse, operator)
    if model.gaps is not None:
        rhs[GAPS], gap_preconditioner = _gap_system(model, tod)
    # the shared templates join the tiles once their Gram is factored (see `bucket_tiles`)
    explicit: dict[TemplateKind, TemplateBundle | None] = {
        TEMPLATES: templates,
        SHARED_TEMPLATES: None,
    }
    tile = Tile(model, implicit, explicit, gap_preconditioner)
    return hits, map_rhs, rhs, shared_outputs, tile


def _filled(
    data: dict[str, Any], padding: Any, config: MapMakingConfig, landscape: StokesLandscape
) -> PyTree[Array]:
    """The TOD of a tile, its gaps filled by a constrained noise realization."""
    model = TileModel.create(data, padding, config, landscape)
    # Only reached under GapTreatment.FILL, where W is the plain inner-mask weight.
    assert isinstance(model.W, WeightOperator)
    # Optional M_b N M_b preconditioner (covariance from the noise model).
    preconditioner = None
    if config.gaps.fill_options.precondition:
        cov = model.noise_operator(config.weighting.correlation_length, inverse=False)
        m_bad = model.M.complement()
        preconditioner = (m_bad @ cov @ m_bad).reduce()
    return gap_fill(
        jax.random.key(config.gaps.fill_options.seed),
        data[ReaderField.SAMPLE_DATA],
        model.W.weight,
        model.M,
        rate=model.sample_rate,
        max_cg_steps=config.gaps.fill_options.max_steps,
        rtol=config.gaps.fill_options.rtol,
        preconditioner=preconditioner,
        metadata=data[ReaderField.METADATA],
    )


def _gap_system(model: TileModel, tod: PyTree[Array]) -> tuple[Array, DiagonalOperator]:
    r"""The gap RHS $C^\top N^{-1} M d$ of a tile, and the preconditioner of its gap amplitudes.

    The preconditioner is the inverse of the diagonal of their white-noise Gram $C^\top D C$.
    """
    assert isinstance(model.W, WeightOperator) and model.gaps is not None
    C = model.gap_basis()
    diagonal = C.squared().T(model.diag_W().weight(model.tod_format.ones()))
    usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
    rhs = C.T(model.W.weight(model.M(tod)))
    inverse = jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)
    return rhs, DiagonalOperator(inverse, in_structure=C.in_structure)


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


def _with_shared_templates(
    tile: Tile, operator: TemplateOperator, inverse: tuple[AbstractLinearOperator, Any]
) -> Tile:
    """`tile` with the templates its observation's detectors share, their Gram factored."""
    shared = TemplateBundle(operator, *inverse)
    return replace(tile, explicit=tile.explicit | {SHARED_TEMPLATES: shared})
