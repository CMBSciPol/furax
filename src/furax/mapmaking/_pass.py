"""The pass over the data: what each detector tile contributes, and the model of a bucket.

`tile_kernel` accumulates one detector tile, and `TILE_REDUCTIONS` says how the engine combines
its outputs over the tiles of a bucket; `bucket_model` assembles the model of the bucket from them.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, PyTree

from furax.core import Policy, StackedEnsemble, map_reduce
from furax.obs.landscapes import StokesLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, StokesI, StokesType

from ._bucket import BucketModel, Tile
from ._gaps import GapSplines
from ._injection import inject, unpadded
from ._model import (
    FACTORED_KINDS,
    ObservationModel,
    ObservationTemplates,
    acquisition_operator,
    restrict_streams,
)
from ._observation import ReaderField
from .config import GapTreatment, InjectionMode, MapMakingConfig
from .gap_filling import gap_fill
from .layout import TileBucket, TilePlacement
from .templates.gram import factor
from .weight import WeightOperator

# The sums run over the tiles; of the rest, what the tiles of an observation share
# (boresight pointing, common template bases, ...) is kept once per segment. The Gram
# inverses and right-hand sides of the templates go where their amplitudes live. A Gram
# summed over the tiles of an observation is paired with its coarse amplitudes, which
# follow from the bases alone, the same in every tile.
TILE_REDUCTIONS = {
    **dict.fromkeys(('hits', 'map'), 'sum'),
    **dict.fromkeys(('model', 'implicit', 'operators'), 'auto'),
    'inverses': 1,
    'grams': {
        kind: ('segment_sum', 'segment') for kind, held in FACTORED_KINDS.items() if held == 'sum'
    },
    'rhs': {kind: 'segment_sum' if held == 'sum' else 1 for kind, held in FACTORED_KINDS.items()},
    'gaps': 1,
}


def tile_kernel(
    config: MapMakingConfig, landscape: StokesLandscape, gaps: GapSplines | None = None
) -> Callable[[tuple[Any, Any, Array], StokesType | None], dict[str, Any]]:
    """Accumulate one detector tile: its hit map, RHS and model.

    The member is one tile's read, the triple `(observation, padding, valid)` restricted to
    the tile's detectors; the sky to inject, if any, is a shared input that every member
    receives whole. `gaps` is the spline model of the explicit gaps of the bucket, if any.
    """
    fill_gaps = config.gaps.treatment == GapTreatment.FILL and not config.binned
    build_templates = config.use_templates
    sky_landscape = landscape.with_stokes(config.read_stokes)
    replaces = config.injection is not None and config.injection.mode == InjectionMode.REPLACE
    factored = _gram_factor(config)

    def cast(tod):
        # The TOD is read in float32, whatever the pipeline's dtype
        return jax.tree.map(lambda x: x.astype(config.dtype), tod)

    def cast_tod(data):
        return {**data, ReaderField.SAMPLE_DATA: cast(data[ReaderField.SAMPLE_DATA])}

    def kernel(member, sky):
        # Gap filling needs whole observations, so it only runs with one tile per observation
        # (see `MapMakingConfig`).
        data, padding, use = member
        data = cast_tod(data)
        if sky is not None:  # the samples to inject into, per detector
            shape = jax.tree.leaves(data[ReaderField.SAMPLE_DATA])[0].shape
            H = acquisition_operator(data, config, sky_landscape)
            data = inject(H, sky, data, unpadded(padding, shape), replaces)
        tod = raw = data[ReaderField.SAMPLE_DATA]
        if fill_gaps:
            # The RHS uses the filled TOD; the models are still built from the raw one.
            tod = jax.lax.cond(
                use,
                lambda: fill_observation_gaps(data, padding),
                lambda: raw,  # nothing to fill
            )
        return accumulate(data, tod, padding, use)

    def fill_observation_gaps(data, padding):
        obs = ObservationModel.create(data, padding, config, landscape)
        # Only reached under GapTreatment.FILL, where W is the plain inner-mask weight.
        assert isinstance(obs.W, WeightOperator)
        # Optional M_b N M_b preconditioner (covariance from the noise model).
        preconditioner = None
        if config.gaps.fill_options.precondition:
            cov = obs.noise_operator(config.weighting.correlation_length, inverse=False)
            m_bad = obs.M.complement()
            preconditioner = (m_bad @ cov @ m_bad).reduce()
        return gap_fill(
            jax.random.key(config.gaps.fill_options.seed),
            data[ReaderField.SAMPLE_DATA],
            obs.W.weight,
            obs.M,
            rate=obs.sample_rate,
            max_cg_steps=config.gaps.fill_options.max_steps,
            rtol=config.gaps.fill_options.rtol,
            preconditioner=preconditioner,
            metadata=data[ReaderField.METADATA],
        )

    def accumulate(data, tod, padding, use):
        temperature = None
        if config.read_stokes != landscape.stokes:
            # The extra streams read, the I one for the T2P template, stay out of the model
            temperature = data[ReaderField.SAMPLE_DATA].i
            data, tod = restrict_streams(data, tod, landscape.stokes)
        # Padding/failed observations contribute nothing
        obs = ObservationModel.create(data, padding, config, landscape, use, gaps)

        # Hit map = nearest-neighbour coverage of the sample mask
        hit_pointing = PointingOperator.create(
            landscape,
            Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
            Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
        ).as_stokes_i(interpolate=False)
        # Read the mask directly: M(ones) = M.to_boolean_mask()
        masked_tod = obs.M.to_boolean_mask()
        # The mask is (ndet, nsamp) even in the demodulated case (all streams share the same)
        masked = masked_tod.data if isinstance(masked_tod, Stokes) else masked_tod
        hits_i = jnp.int64(hit_pointing.T(StokesI(masked)).i)

        if build_templates:  # templates are rejected with gap filling (see `MapMakingConfig`)
            built = ObservationTemplates.assemble(data, config, obs, tod, temperature)
            wd = built.weighted_tod
            rhs_i, implicit = obs.H.T(wd), built.implicit
            operators, grams = built.operators, built.grams
        else:
            # Gaps filled: skip the data-side mask so the fill survives N⁻¹.
            rhs_i = (obs.rhs_operator_prefilled if fill_gaps else obs.rhs_operator)(tod)
            implicit, wd = None, None
            operators = grams = dict.fromkeys(FACTORED_KINDS)
        return {
            'hits': hits_i,
            'map': rhs_i,
            'gaps': {} if obs.gaps is None else gap_system(obs, tod),
            'model': obs,
            'implicit': implicit,
            'operators': operators,
            'inverses': {
                kind: None if gram is None else factored(gram)
                for kind, gram in grams.items()
                if FACTORED_KINDS[kind] == 'stack'
            },
            'grams': {
                kind: (None, None) if gram is None else (replace(gram, coarse=None), gram.coarse)
                for kind, gram in grams.items()
                if FACTORED_KINDS[kind] == 'sum'
            },
            # the tile's share of the right-hand side of each kind of amplitudes
            'rhs': {kind: None if op is None else op.T(wd) for kind, op in operators.items()},
        }

    def gap_system(obs, tod):
        # the right-hand side Cᵀ N⁻¹ M d of the gap amplitudes, and the inverse of the diagonal
        # of their white-noise Gram Cᵀ D C, which preconditions them
        assert isinstance(obs.W, WeightOperator) and obs.gaps is not None
        C = obs.gap_basis()
        diagonal = C.squared().T(obs.diag_W().weight(obs.tod_format.ones()))
        usable = diagonal > 0  # the slots past the amplitudes of a detector are unused
        return {
            'gap_rhs': C.T(obs.W.weight(obs.M(tod))),
            'gap_preconditioner': jnp.where(usable, 1 / jnp.where(usable, diagonal, 1), 1),
        }

    return kernel


def bucket_model(
    bucket: TileBucket,
    out: dict[str, Any],
    resolved: dict[str, Any],
    placement: TilePlacement,
    config: MapMakingConfig,
) -> BucketModel:
    """The model of a bucket, from the outputs of its pass over the data and their kinds."""
    strides = _strides(out, resolved)
    templates = template_strides = None
    if config.use_templates:
        inverses, inverse_strides = _factor_common_grams(out['grams'], placement, config)
        templates = ObservationTemplates.from_inverses(
            out['implicit'], out['operators'], out['inverses'] | inverses
        )
        template_strides = ObservationTemplates.from_inverses(
            strides['implicit'], strides['operators'], strides['inverses'] | inverse_strides
        )
    tile = Tile(out['model'], templates, out['rhs']['explicit'], **out['gaps'])
    tile_strides = Tile(
        strides['model'], template_strides, strides['rhs']['explicit'], **strides['gaps']
    )
    tiles = StackedEnsemble(
        bucket.n_tiles,
        tile,
        tuple(jax.tree.structure(tile).flatten_up_to(tile_strides)),
        policy=Policy(axis=placement.axis),
        owner=placement.shard(bucket.owner),
        segments=bucket.n_segments,
    )
    return BucketModel(tiles, out['rhs']['common'])


def _factor_common_grams(
    grams: dict[str, tuple[Any, Any]], placement: TilePlacement, config: MapMakingConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The inverses of the Grams summed over the tiles of each observation, and their strides.

    Each observation's Gram is factored on the device that holds it, one per round.
    """
    factored_gram = _gram_factor(config)
    inverses, strides = {}, {}
    for kind, (gram, coarse) in grams.items():
        if gram is None:
            inverses[kind] = strides[kind] = None
            continue
        n_segments = jax.tree.leaves(gram)[0].shape[0]
        ensemble = StackedEnsemble(n_segments, (gram, coarse), policy=Policy(axis=placement.axis))

        def factored(member):
            gram, coarse = member
            return factored_gram(replace(gram, coarse=coarse))

        inverse, kinds = map_reduce(factored, ensemble, 'auto')
        inverses[kind] = inverse
        # one per observation: per segment among the tiles
        strides[kind] = _strides(inverse, kinds, segment=True)
    return inverses, strides


def _gram_factor(config: MapMakingConfig) -> Callable[[Any], Any]:
    """`factor` of a Gram, with the run's template regularization."""
    templates = config.templates
    return partial(factor, regularization=0.0 if templates is None else templates.regularization)


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
