"""The pass over the data: what each detector tile contributes to the system, and its buckets.

The mapmaker reads its observations once, a bucket at a time. Each detector tile of a bucket goes
through [`tile_kernel`][], which builds the tile's model and returns its
[`TileContribution`][]: its hit map and map RHS, summed over the tiles, and what the normal system
keeps of it, its model, templates and the right-hand sides of its amplitudes. The engine combines
the contributions of the tiles as [`TILE_REDUCTIONS`][] says, and [`bucket_model`][] assembles
what it keeps into the [`BucketModel`][furax.mapmaking._bucket.BucketModel] of the bucket.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Int64, PyTree

from furax.core import Policy, StackedEnsemble, map_reduce
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType

from ._bucket import BucketModel, Tile
from ._gaps import GapSplines
from ._injection import inject, unpadded
from ._model import (
    SHARED_TEMPLATES,
    TEMPLATE_KINDS,
    TEMPLATES,
    TemplateBundle,
    TemplateKind,
    TileModel,
    TileTemplates,
    acquisition_operator,
    restrict_streams,
)
from ._observation import ReaderField
from .config import GapTreatment, InjectionMode, MapMakingConfig
from .gap_filling import gap_fill
from .layout import TileBucket, TilePlacement
from .templates import TemplateOperator
from .templates.gram import factor
from .weight import WeightOperator


class TileContribution(NamedTuple):
    """What one detector tile contributes to the system, from the pass over the data."""

    hits: Int64[Array, '...']
    """The hit map of the tile's good samples."""
    map_rhs: StokesType
    r"""The tile's share of the map RHS, $H^\top W' d$."""
    model: TileModel
    """The tile's model."""
    implicit: TemplateBundle | None
    """The tile's implicit templates, factored: its weight marginalizes their amplitudes."""
    template_operators: dict[TemplateKind, TemplateOperator | None]
    """The operator of the explicit templates of each kind of `TEMPLATE_KINDS`, `None` if
    absent."""
    template_gram_inverse: tuple[Any, Any] | None
    """The inverse of the Gram of the templates of each detector, and its ridge, factored in the
    tile (see [`factor`][furax.mapmaking.templates.gram.factor])."""
    shared_template_gram: tuple[Any, Any]
    """The tile's share of the Gram of the templates the observation's detectors share, and their
    coarse amplitudes, `(None, None)` without them: the Gram is factored once summed over the
    tiles."""
    template_rhs: dict[TemplateKind, PyTree[Array] | None]
    """The tile's share of the template RHS of each kind of template amplitudes."""
    gap_rhs: Array | None
    """The gap RHS (see [`Tile.gap_rhs`][]), if the gaps are explicit."""
    gap_preconditioner: Array | None
    """The preconditioner of the gap amplitudes (see [`Tile.gap_preconditioner`][])."""


_REDUCTIONS: dict[str, Any] = {
    'hits': 'sum',
    'map_rhs': 'sum',
    # what the tiles of an observation share (boresight pointing, bases of the shared templates,
    # ...) is kept once per segment, the rest per tile
    'model': 'auto',
    'implicit': 'auto',
    'template_operators': 'auto',
    'template_gram_inverse': 1,
    # a Gram summed over the tiles of an observation is paired with its coarse amplitudes, which
    # follow from the bases alone, the same in every tile
    'shared_template_gram': ('segment_sum', 'segment'),
    'template_rhs': {
        kind: 'segment_sum' if holder == 'observation' else 1
        for kind, holder in TEMPLATE_KINDS.items()
    },
    'gap_rhs': 1,
    'gap_preconditioner': 1,
}

TILE_REDUCTIONS = TileContribution(**_REDUCTIONS)
"""How the engine combines the contributions of the tiles of a bucket (see `map_reduce`): the
hit map and the map RHS are summed, and the rest is kept where its amplitudes live, per tile or
per observation."""


def tile_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, gaps: GapSplines | None = None
) -> Callable[[tuple[Any, Any, Array], StokesType | None], TileContribution]:
    """The kernel of the pass over a bucket: the contribution of one detector tile.

    The member is one tile's read, the triple `(data, padding, use)` restricted to the tile's
    detectors; the sky to inject, if any, is a shared input that every member receives whole.

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


def bucket_model(
    out: TileContribution,
    resolved: TileContribution,
    bucket: TileBucket,
    placement: TilePlacement,
    config: MapMakingConfig,
) -> BucketModel:
    """The model of a bucket, from the contributions of its tiles combined over the pass.

    The Grams of the templates the detectors of an observation share are factored here, once
    summed over its tiles, on the device that holds the observation.

    Args:
        out: The contributions of the tiles, combined by [`TILE_REDUCTIONS`][].
        resolved: How each of their leaves was combined, as `map_reduce` resolves it.
        bucket: The bucket.
        placement: The placement of its tiles on the devices.
        config: The mapmaking configuration.
    """
    strides = _strides(out, resolved)
    templates = template_strides = None
    if config.use_templates:
        shared, shared_strides = _factor_shared_gram(
            out.shared_template_gram, placement, _gram_factor(config)
        )
        templates = TileTemplates.from_inverses(
            out.implicit,
            out.template_operators,
            {TEMPLATES: out.template_gram_inverse, SHARED_TEMPLATES: shared},
        )
        template_strides = TileTemplates.from_inverses(
            strides.implicit,
            strides.template_operators,
            {TEMPLATES: strides.template_gram_inverse, SHARED_TEMPLATES: shared_strides},
        )
    tile = Tile(
        out.model,
        templates,
        out.template_rhs[TEMPLATES],
        out.gap_rhs,
        out.gap_preconditioner,
    )
    tile_strides = Tile(
        strides.model,
        template_strides,
        strides.template_rhs[TEMPLATES],
        strides.gap_rhs,
        strides.gap_preconditioner,
    )
    tiles = StackedEnsemble(
        bucket.n_tiles,
        tile,
        tuple(jax.tree.structure(tile).flatten_up_to(tile_strides)),
        policy=Policy(axis=placement.axis),
        owner=placement.shard(bucket.owner),
        segments=bucket.n_segments,
    )
    return BucketModel(tiles, out.template_rhs[SHARED_TEMPLATES])


def _contribution(
    data: dict[str, Any],
    tod: PyTree[Array],
    padding: Any,
    use: Array,
    config: MapMakingConfig,
    landscape: StokesLandscape,
    gaps: GapSplines | None,
) -> TileContribution:
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

    if config.use_templates:  # templates are rejected with gap filling (see `MapMakingConfig`)
        built = TileTemplates.assemble(data, config, model, tod, temperature)
        wd = built.weighted_tod
        map_rhs, implicit = model.H.T(wd), built.implicit
        operators, grams = built.operators, built.grams
    else:
        # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
        filled = config.gap_treatment == GapTreatment.FILL
        map_rhs = (model.rhs_operator_prefilled if filled else model.rhs_operator)(tod)
        implicit, wd = None, None
        operators = grams = dict.fromkeys(TEMPLATE_KINDS)
    gram, shared_gram = grams[TEMPLATES], grams[SHARED_TEMPLATES]
    gap_rhs = gap_preconditioner = None
    if model.gaps is not None:
        gap_rhs, gap_preconditioner = _gap_system(model, tod)
    return TileContribution(
        hits=hits,
        map_rhs=map_rhs,
        model=model,
        implicit=implicit,
        template_operators=operators,
        template_gram_inverse=None if gram is None else _gram_factor(config)(gram),
        shared_template_gram=(None, None)
        if shared_gram is None
        else (replace(shared_gram, coarse=None), shared_gram.coarse),
        template_rhs={kind: None if op is None else op.T(wd) for kind, op in operators.items()},
        gap_rhs=gap_rhs,
        gap_preconditioner=gap_preconditioner,
    )


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


def _gap_system(model: TileModel, tod: PyTree[Array]) -> tuple[Array, Array]:
    r"""The gap RHS $C^\top N^{-1} M d$ of a tile, and the preconditioner of its gap amplitudes.

    The preconditioner is the inverse of the diagonal of their white-noise Gram $C^\top D C$.
    """
    assert isinstance(model.W, WeightOperator) and model.gaps is not None
    C = model.gap_basis()
    diagonal = C.squared().T(model.diag_W().weight(model.tod_format.ones()))
    usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
    rhs = C.T(model.W.weight(model.M(tod)))
    return rhs, jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1)


def _gram_factor(config: MapMakingConfig) -> Callable[[Any], Any]:
    """`factor` of a Gram, with the run's template regularization."""
    templates = config.templates
    return partial(factor, regularization=0.0 if templates is None else templates.regularization)


def _factor_shared_gram(
    gram: tuple[Any, Any], placement: TilePlacement, factor_gram: Callable[[Any], Any]
) -> tuple[Any, Any]:
    """The inverse of the Gram of the shared templates of each observation, and its strides.

    The Gram is summed over the tiles of the observation, paired with its coarse amplitudes, and
    factored on the device that holds the observation, one observation per round. Both are `None`
    without shared templates.
    """
    gram, coarse = gram
    if gram is None:
        return None, None
    n_segments = jax.tree.leaves(gram)[0].shape[0]
    ensemble = StackedEnsemble(n_segments, (gram, coarse), policy=Policy(axis=placement.axis))

    def factor_one(member):
        gram, coarse = member
        return factor_gram(replace(gram, coarse=coarse))

    inverse, kinds = map_reduce(factor_one, ensemble, 'auto')
    # one per observation: per segment among the tiles
    return inverse, _strides(inverse, kinds, segment=True)


def _strides(outputs: PyTree[Any], resolved: PyTree[Any], segment: bool = False) -> PyTree[Any]:
    """The stride of each output leaf of `map_reduce` in an ensemble of tiles.

    An output stacked per tile has stride 1, one kept or summed per segment the stride
    `'segment'`, and one shared by the bucket None. With `segment`, the outputs are those of an
    ensemble of segments, whose stacked outputs are per segment among the tiles.

    The resolved spec puts `None` on the shared leaves, which `jax.tree.map` over the outputs
    would read as a missing subtree; the returned tree has the structure of the outputs, so it
    maps with them.
    """
    treedef = jax.tree.structure(outputs)
    kinds = treedef.flatten_up_to(resolved)
    per_member = 'segment' if segment else 1
    strides = [
        None if kind is None else 'segment' if kind in ('segment', 'segment_sum') else per_member
        for kind in kinds
    ]
    return jax.tree.unflatten(treedef, strides)
