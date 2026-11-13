r"""The normal system of the multi-observation mapmaker, evaluated one detector tile at a time.

The mapmaker solves $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$ over the detector
tiles $t$ of every observation. The unknowns $x$ are a dictionary of the map of the selected sky
pixels, under `'map'`, and of the amplitudes solved for with it (see [`AMPLITUDE_KINDS`][]): the
template amplitudes of each detector, those the detectors of an observation share, and the gap
amplitudes. $J_t = [H_t S^\top, T_t, T_{s,t}, C_t]$ maps them to the tile's TOD, and $W'_t$ is its
weight (see
[`TileModel.deprojected_weight`][furax.mapmaking._model.TileModel.deprojected_weight]).
The right-hand side $\sum_t J_t^\top W'_t d_t$ has the same keys: the map RHS, the template RHS,
and the gap RHS.

With explicit gaps, the flagged samples are modelled by the spline amplitudes of the tile's gaps,
$C_t$: $J_t = [M_t H_t S^\top, M_t T_t, M_t T_{s,t}, C_t]$, with $M_t$ the mask of the good
samples, under the inverse noise $N_t^{-1}$ unmasked. The map block is that of the masked weight,
$H^\top M N^{-1} M H$; marginalizing the gap amplitudes would give the nested weight on the span of
the splines.

A bucket keeps what its tiles hold as a `BucketModel`: a stacked ensemble of its tiles, split
over the devices, grouped in segments by observation. Each leaf is stacked per tile, per segment
when the tiles of an observation share it (the boresight pointing, the bases and Gram inverse of
the shared templates, ...), or held once for the bucket. The operators of this module evaluate the
sums over it with the ensemble engine.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Final, Literal

import jax
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

import furax.tree
from furax import AbstractLinearOperator, DiagonalOperator, symmetric
from furax.core import BlockRowOperator, Policy, StackedEnsemble, lift, map_reduce
from furax.obs.stokes import StokesType

from ._model import SHARED_TEMPLATES, TEMPLATE_KINDS, Holder, TemplateKind, TileModel, TileTemplates
from .weight import WeightOperator

GAPS: Final = 'gaps'
"""The gap amplitudes of each detector, among the unknowns of the system."""

AmplitudeKind = TemplateKind | Literal['gaps']
"""A kind of amplitudes solved for with the map: a key of the unknowns of the system, besides
the `'map'`."""

AMPLITUDE_KINDS: dict[AmplitudeKind, Holder] = {**TEMPLATE_KINDS, GAPS: 'tile'}
"""The amplitudes solved for with the map, each a key of the unknowns, and where they are held."""


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """What the normal system needs of one detector tile of an observation."""

    model: TileModel
    """The tile's acquisition operator, weight and noise model."""
    templates: TileTemplates | None
    """The tile's templates, with amplitudes per detector or shared by the observation."""
    template_rhs: PyTree[Array] | None
    r"""The template RHS $T_t^\top W'_t d_t$ of the template amplitudes of the tile's detectors
    (None without them)."""
    gap_rhs: Float[Array, '*channels budget'] | None = None
    r"""The gap RHS $C_t^\top N_t^{-1} M_t d_t$ (None without explicit gaps), laid out as the TOD,
    the budget of amplitudes of each channel in place of its samples."""
    gap_preconditioner: Float[Array, '*channels budget'] | None = None
    """The inverse of the diagonal of the white-noise Gram of the gap amplitudes."""

    def operator(self, kind: AmplitudeKind) -> AbstractLinearOperator:
        """The operator from the amplitudes of `kind` to the TOD: $T_t$, $T_{s,t}$ or $C_t$."""
        if kind == GAPS:
            return self.model.gap_basis()
        bundle = None if self.templates is None else self.templates.explicit[kind]
        assert bundle is not None
        return bundle.operator


@register_dataclass
@dataclass(frozen=True)
class BucketModel:
    """The detector tiles of every observation of a bucket."""

    tiles: StackedEnsemble
    """The [`Tile`][]s, one member per tile, grouped in segments by observation and laid out
    over the devices; the right-hand sides of their amplitudes are stacked per tile."""
    shared_template_rhs: PyTree[Array] | None
    """The template RHS of the template amplitudes each observation's detectors share,
    `(segment, ...)` (None without them)."""

    @property
    def templates(self) -> TileTemplates | None:
        """The templates of a tile of the bucket, for their structure.

        It tells which kinds of templates are present, and their static fields.
        """
        return self.tiles.template.templates

    @property
    def amplitude_kinds(self) -> tuple[AmplitudeKind, ...]:
        """The kinds of amplitudes solved for with the map, each a key of the unknowns."""
        templates = self.templates
        kinds = ()
        if templates is not None:
            kinds = tuple(kind for kind in TEMPLATE_KINDS if templates.explicit[kind] is not None)
        if self.tiles.template.gap_rhs is not None:
            kinds += (GAPS,)
        return kinds

    def rhs(self, kind: AmplitudeKind) -> PyTree[Array]:
        """The right-hand side of the amplitudes of `kind`.

        It is `(tile, ...)`, or `(segment, ...)` for the template amplitudes the detectors of an
        observation share.
        """
        if kind == SHARED_TEMPLATES:
            return self.shared_template_rhs
        return self.tiles.stacked.gap_rhs if kind == GAPS else self.tiles.stacked.template_rhs

    def preconditioner(self, kind: AmplitudeKind) -> AbstractLinearOperator:
        """The preconditioner of the amplitudes of `kind`.

        The template amplitudes are preconditioned by the inverse of their Gram, each tile's for
        those of its detectors and each observation's for those they share, the gap amplitudes by
        the inverse of the diagonal of their white-noise Gram.
        """
        if kind == GAPS:
            structure = furax.tree.as_structure(self.rhs(kind))
            return DiagonalOperator(self.tiles.stacked.gap_preconditioner, in_structure=structure)
        inverse = self._template_operator(kind, 'gram_inverse')
        assert inverse is not None
        return inverse

    def ridge(self, kind: AmplitudeKind) -> AbstractLinearOperator | None:
        """The ridge the Gram of the amplitudes of `kind` was factored with, a term of the system.

        `None` without regularization, and for the gap amplitudes.
        """
        return None if kind == GAPS else self._template_operator(kind, 'ridge')

    def _template_operator(self, kind: TemplateKind, part: str) -> AbstractLinearOperator | None:
        """An operator of the templates of `kind`, on their amplitudes; `None` if absent.

        `part` is a field of [`TemplateBundle`][furax.mapmaking._model.TemplateBundle], whose
        operator is held per tile, or per observation for the shared templates.
        """

        def bundle(tile: Tile) -> Any:
            assert tile.templates is not None
            return tile.templates.explicit[kind]

        if getattr(bundle(self.tiles.template), part) is None:
            return None
        # structural: the operators keep the leaves of the tiles, and their strides
        operators = lift(lambda tile: getattr(bundle(tile), part), self.tiles)
        assert isinstance(operators, StackedEnsemble)
        if AMPLITUDE_KINDS[kind] == 'observation':
            operators = segment_ensemble(operators)
        return MemberOperator(operators, in_structure=furax.tree.as_structure(self.rhs(kind)))


def bucket_apply(
    model: BucketModel,
    apply: Callable[
        [Tile, PyTree[Array], PyTree[Array] | None, PyTree[Array] | None], tuple[Any, ...]
    ],
    whole: PyTree[Array],
    per_tile: PyTree[Array] | None,
    per_observation: PyTree[Array] | None,
    policy: Policy | None = None,
) -> tuple[Any, Any, Any]:
    """`apply` on every detector tile of a bucket.

    Args:
        model: The bucket.
        apply: From a tile, `whole`, its amplitudes and those of its observation, to `(map,
            per_tile, per_observation)`: a map, the tile's amplitudes and its observation's, any
            of them `None`.
        whole: What every tile takes whole, such as a map, held by every device. Its leaves must
            be arrays; anything else (an operator, whose index may hold an `Ellipsis`) is closed
            over `apply` instead.
        per_tile: The amplitudes held per tile, `(tile, ...)`, if any.
        per_observation: The amplitudes held per observation, `(segment, ...)`, if any.
        policy: How the tiles are walked; by default that of `model.tiles`. A round of `batch`
            tiles holds the temporaries of that many tiles.

    Returns:
        The maps summed over the bucket, the amplitudes of each tile `(tile, ...)`, and those of
        each observation summed over its tiles, `(segment, ...)`.
    """
    tiles = model.tiles
    strides = (
        *tiles.strides,
        *(1 for _ in jax.tree.leaves(per_tile)),
        *('segment' for _ in jax.tree.leaves(per_observation)),
    )
    ensemble = StackedEnsemble(
        tiles.size,
        (tiles.stacked, per_tile, per_observation),
        strides,
        owner=tiles.owner,
        segments=tiles.segments,
        policy=policy if policy is not None else tiles.policy,
    )

    def tile_fn(member: tuple[Any, ...], whole: PyTree[Array]) -> tuple[Any, Any, Any]:
        tile, tile_amplitudes, observation_amplitudes = member
        return apply(tile, whole, tile_amplitudes, observation_amplitudes)

    out, _ = map_reduce(tile_fn, ensemble, ('sum', 1, 'segment_sum'), whole)
    return out


def tile_system(
    tile: Tile,
    selection: AbstractLinearOperator | None,
    kinds: tuple[AmplitudeKind, ...],
    weight: Literal['system', 'plain', 'white'] = 'system',
) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
    r"""The joint operator $J_t$ of a tile, and its weight $W'_t$.

    $J_t$ maps the unknowns, a dictionary of the `'map'` and of the amplitudes of each kind of
    `kinds`, to the tile's TOD. With the gap amplitudes among them, which stand for the flagged
    samples, the other blocks are masked and the weight is the inverse noise unmasked.

    Args:
        tile: The tile.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        kinds: The kinds of amplitudes among the unknowns (see [`AMPLITUDE_KINDS`][]).
        weight: The weight (see [`BucketNormalOperator.weight`][]).
    """
    model = tile.model
    H = model.H if selection is None else model.H @ selection.T
    blocks = {'map': H} | {kind: tile.operator(kind) for kind in kinds if kind != GAPS}
    W = model.diag_W() if weight == 'white' else model.W
    if GAPS in kinds:
        assert isinstance(W, WeightOperator)
        blocks = {kind: model.M @ block for kind, block in blocks.items()}
        return BlockRowOperator(blocks | {GAPS: tile.operator(GAPS)}), W.weight
    if weight == 'system':
        implicit = None if tile.templates is None else tile.templates.implicit
        return BlockRowOperator(blocks), model.deprojected_weight(implicit)
    return BlockRowOperator(blocks), (W @ model.F).reduce()


def segment_ensemble(ensemble: StackedEnsemble) -> StackedEnsemble:
    """An ensemble of tiles holding only segment and shared leaves, as one of its segments.

    Raises:
        ValueError: If a leaf varies with the tile.
    """
    if any(isinstance(s, int) for s in ensemble.strides):
        raise ValueError('a leaf varies with the tile, not with its segment')
    assert ensemble.segments is not None
    strides = tuple(None if s is None else 1 for s in ensemble.strides)
    return StackedEnsemble(ensemble.segments, ensemble.stacked, strides, policy=ensemble.policy)


@symmetric
class BucketNormalOperator(AbstractLinearOperator):
    r"""$\sum_t J_t^\top W'_t J_t$ over the detector tiles of a bucket.

    The unknowns are a dictionary: the `'map'`, and the amplitudes of each kind the bucket has
    (see [`BucketModel.amplitude_kinds`][]) if `amplitudes` is set. Without `selection`, the map
    is the whole sky.
    """

    model: BucketModel
    selection: AbstractLinearOperator | None
    """The selection of the estimated pixels, if any."""
    weight: Literal['system', 'plain', 'white'] = field(default='system', metadata={'static': True})
    """$W'_t$: the weight of the system, or $W F$ without the implicit templates, or the same with
    the white part of the noise only."""
    amplitudes: bool = field(default=True, metadata={'static': True})
    """Whether the unknowns include the template and gap amplitudes."""
    policy: Policy | None = field(default=None, metadata={'static': True})
    """How the tiles are walked (see `bucket_apply`); by default that of `model.tiles`."""

    def mv(self, x: Any) -> Any:
        m, per_tile, per_observation = _by_holder(x)
        # closed over, not engine arguments: the selection may hold non-array leaves
        selection, kinds, weight = self.selection, self._kinds, self.weight

        def apply(tile, m, tile_amplitudes, observation_amplitudes):
            J, W = tile_system(tile, selection, kinds, weight)
            u = J({'map': m, **tile_amplitudes, **observation_amplitudes})
            return _by_holder(J.T(W(u)))

        m, per_tile, per_observation = bucket_apply(
            self.model, apply, m, per_tile, per_observation, self.policy
        )
        return {'map': m, **per_tile, **per_observation}

    def response(self, sky: StokesType) -> Any:
        r"""$\sum_t J_t^\top W'_t H_t \, s$: the right-hand side of the TOD of a sky `sky`."""
        selection, kinds, weight = self.selection, self._kinds, self.weight

        def apply(tile, sky, _tile_amplitudes, _observation_amplitudes):
            J, W = tile_system(tile, selection, kinds, weight)
            # masked as the data are: with the gap amplitudes, the weight is unmasked
            return _by_holder(J.T(W(tile.model.M(tile.model.H(sky)))))

        m, per_tile, per_observation = bucket_apply(self.model, apply, sky, {}, {}, self.policy)
        return {'map': m, **per_tile, **per_observation}

    @property
    def _kinds(self) -> tuple[AmplitudeKind, ...]:
        """The kinds of amplitudes among the unknowns."""
        return self.model.amplitude_kinds if self.amplitudes else ()


@symmetric
class MemberOperator(AbstractLinearOperator):
    """The operator of each member of an ensemble, on its own amplitudes: `x[i] -> operators[i](x[i])`.

    The operators are one per tile of a bucket, such as the inverses of the Grams of the template
    amplitudes of its detectors, which precondition them, or one per segment, for the template
    amplitudes the detectors of an observation share (see [`segment_ensemble`][]).
    """

    operators: StackedEnsemble
    """The operators, one member per entry of the amplitudes."""

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        # small computations: every member of a device is evaluated at once, vectorized
        ensemble = self.operators.zip(x)
        policy = replace(ensemble.policy, batch=ensemble.size_per_device)

        def apply(member: tuple[Any, ...]) -> PyTree[Array]:
            operator, v = member
            return operator(v)

        out, _ = map_reduce(apply, replace(ensemble, policy=policy), 1)
        return out


def _by_holder(x: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The map of the unknowns `x`, and their amplitudes held per tile and per observation.

    The engine strides the two differently: per tile, and per segment.
    """

    def held(holder):
        return {kind: x[kind] for kind, h in AMPLITUDE_KINDS.items() if h == holder and kind in x}

    return x['map'], held('tile'), held('observation')
