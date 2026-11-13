r"""The normal system of the multi-observation mapmaker, evaluated one detector tile at a time.

The mapmaker solves $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$ over the detector
tiles $t$ of every observation, with $J_t = [H_t S^\top, T_{e,t}, T_{g,t}]$ mapping the selected
sky pixels, the tile's own template amplitudes and those its observation's detectors share to the
tile's TOD, and $W'_t$ its weight (see
[`ObservationModel.deprojected_weight`][furax.mapmaking._model.ObservationModel.deprojected_weight]).

With explicit gaps, the flagged samples are modelled by the spline amplitudes of the tile's gaps,
$C_t$: $J_t = [M_t H_t S^\top, M_t T_{e,t}, M_t T_{g,t}, C_t]$, with $M_t$ the mask of the good
samples, under the inverse noise $N_t^{-1}$ unmasked. The map block is that of the masked weight,
$H^\top M N^{-1} M H$; marginalizing the gap amplitudes would give the nested weight on the span of
the splines.

A bucket keeps what its tiles hold as a `BucketModel`: a stacked ensemble of its tiles, split
over the devices, grouped in segments by observation. Each leaf is stacked per tile, per segment
when the tiles of an observation share it (the boresight pointing, the bases and Gram inverse of
the templates with shared amplitudes, ...), or held once for the bucket. The operators of this
module evaluate the sums over it with the ensemble engine.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.core import BlockRowOperator, Policy, StackedEnsemble, map_reduce
from furax.obs.stokes import StokesType

from ._model import FACTORED_KINDS, ObservationModel, ObservationTemplates
from .weight import WeightOperator

GAPS = 'gaps'
"""The kind of the amplitudes of the explicit gaps among the unknowns, per tile."""


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """What the normal system needs of one detector tile of an observation."""

    model: ObservationModel
    """The tile's acquisition operator, weight and noise model."""
    templates: ObservationTemplates | None
    """The tile's templates, with amplitudes per detector or shared by the observation."""
    amplitude_rhs: PyTree[Array] | None
    """The right-hand side of the tile's own explicit amplitudes (None without them)."""
    gap_rhs: Float[Array, '*channels budget'] | None = None
    """The right-hand side of the amplitudes of the tile's explicit gaps (None without them), laid
    out as its TOD, the budget of amplitudes of each channel in place of its samples."""
    gap_preconditioner: Float[Array, '*channels budget'] | None = None
    """The inverse of the diagonal of the white-noise Gram of those amplitudes."""


@register_dataclass
@dataclass(frozen=True)
class BucketModel:
    """The detector tiles of every observation of a bucket."""

    tiles: StackedEnsemble
    """The [`Tile`][]s, one member per tile, grouped in segments by observation and laid out
    over the devices; the right-hand sides of their own amplitudes are stacked per tile."""
    common_rhs: PyTree[Array] | None
    """Right-hand side of each observation's shared amplitudes, `(segment, ...)` (None without)."""

    @property
    def templates(self) -> ObservationTemplates | None:
        """The kinds of templates of the bucket, from the structure of its tiles."""
        return self.tiles.template.templates

    @property
    def amplitude_kinds(self) -> tuple[str, ...]:
        """The kinds of amplitudes solved for with the map, each a key of the unknowns."""
        templates = self.templates
        kinds = ()
        if templates is not None:
            kinds = tuple(kind for kind in FACTORED_KINDS if getattr(templates, kind) is not None)
        if self.tiles.template.gap_rhs is not None:
            kinds += (GAPS,)
        return kinds


def bucket_apply(
    model: BucketModel,
    apply: Callable[
        [Tile, PyTree[Array], PyTree[Array] | None, PyTree[Array] | None], tuple[Any, ...]
    ],
    whole: PyTree[Array],
    own: PyTree[Array] | None,
    common: PyTree[Array] | None,
    policy: Policy | None = None,
) -> tuple[Any, Any, Any]:
    """`apply` on every detector tile of a bucket.

    Args:
        model: The bucket.
        apply: From a tile, `whole`, its own amplitudes and those of its observation, to `(map,
            own, common)`: a map, the tile's own amplitudes and its observation's, any of them
            `None`.
        whole: What every tile takes whole, such as a map, held by every device. Its leaves must
            be arrays; anything else (an operator, whose index may hold an `Ellipsis`) is closed
            over `apply` instead.
        own: The amplitudes of each tile alone, `(tile, ...)`, if any.
        common: The amplitudes of each observation, `(segment, ...)`, if any.
        policy: How the tiles are walked; by default that of `model.tiles`. A round of `batch`
            tiles holds the temporaries of that many tiles.

    Returns:
        The maps summed over the bucket, the tiles' own amplitudes `(tile, ...)`, and those of
        each observation summed over its tiles, `(segment, ...)`.
    """
    tiles = model.tiles
    strides = (
        *tiles.strides,
        *(1 for _ in jax.tree.leaves(own)),
        *('segment' for _ in jax.tree.leaves(common)),
    )
    ensemble = StackedEnsemble(
        tiles.size,
        (tiles.stacked, own, common),
        strides,
        owner=tiles.owner,
        segments=tiles.segments,
        policy=policy if policy is not None else tiles.policy,
    )

    def tile_fn(member: tuple[Any, ...], whole: PyTree[Array]) -> tuple[Any, Any, Any]:
        tile, own_t, common_o = member
        return apply(tile, whole, own_t, common_o)

    out, _ = map_reduce(tile_fn, ensemble, ('sum', 1, 'segment_sum'), whole)
    return out


def joint_operator(
    tile: Tile, selection: AbstractLinearOperator | None, kinds: tuple[str, ...]
) -> AbstractLinearOperator:
    r"""$J_t = [H_t S^\top, T_{e,t}, T_{g,t}, C_t]$ of a tile, with the amplitudes asked for.

    Its input is a dictionary: the `'map'`, and the amplitudes of each kind of `kinds`. With the
    gaps among them, the other blocks are masked: the gap amplitudes stand for the flagged samples.

    Args:
        tile: The tile.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        kinds: The kinds of amplitudes to include (see [`BucketModel.amplitude_kinds`][]).
    """
    H = tile.model.H if selection is None else tile.model.H @ selection.T
    blocks = {'map': H}
    for kind in kinds:
        if kind == GAPS:
            continue
        bundle = None if tile.templates is None else getattr(tile.templates, kind)
        assert bundle is not None
        blocks[kind] = bundle.operator
    if GAPS in kinds:
        blocks = {kind: tile.model.M @ block for kind, block in blocks.items()}
        blocks[GAPS] = tile.model.gap_basis()
    return BlockRowOperator(blocks)


def system_weight(tile: Tile) -> AbstractLinearOperator:
    """$W_t$ of a tile, its implicit templates marginalized."""
    implicit = None if tile.templates is None else tile.templates.implicit
    return tile.model.deprojected_weight(implicit)


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
    (see [`BucketModel.amplitude_kinds`][]), if `amplitudes` is set. Otherwise, or without such
    amplitudes, they are the map alone. Without `selection`, the map is the whole sky.
    """

    model: BucketModel
    selection: AbstractLinearOperator | None
    """The selection of the estimated pixels, if any."""
    weight: Literal['system', 'plain', 'white'] = field(default='system', metadata={'static': True})
    """$W'_t$: the weight of the system, or $W F$ without the implicit templates, or the same with
    the white part of the noise only."""
    amplitudes: bool = field(default=True, metadata={'static': True})
    """Whether the unknowns include the amplitudes of the explicit templates and gaps."""
    policy: Policy | None = field(default=None, metadata={'static': True})
    """How the tiles are walked (see `bucket_apply`); by default that of `model.tiles`."""

    def mv(self, x: Any) -> Any:
        m, own, common = self._split(x)
        # closed over, not an engine argument: the selection may hold non-array leaves
        selection = self.selection

        def apply(tile, m, own_t, common_o):
            J = joint_operator(tile, selection, self._kinds)
            return _by_owner(J.T(self._weight(tile)(J({'map': m, **own_t, **common_o}))))

        return self._join(*bucket_apply(self.model, apply, m, own, common, self.policy))

    def response(self, sky: StokesType) -> Any:
        r"""$\sum_t J_t^\top W'_t H_t \, s$: the right-hand side of the TOD of a sky `sky`."""
        gaps = GAPS in self._kinds

        def apply(tile, sky, _own, _common):
            J = joint_operator(tile, self.selection, self._kinds)
            tod = tile.model.H(sky)
            if gaps:  # the weight is unmasked
                tod = tile.model.M(tod)
            return _by_owner(J.T(self._weight(tile)(tod)))

        return self._join(*bucket_apply(self.model, apply, sky, {}, {}, self.policy))

    def _weight(self, tile: Tile) -> AbstractLinearOperator:
        model = tile.model
        W = model.diag_W() if self.weight == 'white' else model.W
        if GAPS in self._kinds:  # N⁻¹ unmasked: the gap amplitudes stand for the flagged samples
            assert isinstance(W, WeightOperator)
            return W.weight
        if self.weight == 'system':
            return system_weight(tile)
        return (W @ model.F).reduce()

    @property
    def _kinds(self) -> tuple[str, ...]:
        """The kinds of amplitudes among the unknowns."""
        return self.model.amplitude_kinds if self.amplitudes else ()

    def _split(self, x: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        """The map and the amplitudes of the unknowns `x`, in parts (see [`_by_owner`][])."""
        if not self._kinds:
            return x, {}, {}
        return _by_owner(x)

    def _join(self, m: Any, own: dict[str, Any], common: dict[str, Any]) -> Any:
        """The unknowns from their parts."""
        if not self._kinds:
            return m
        return {'map': m, **own, **common}


@symmetric
class MemberOperator(AbstractLinearOperator):
    """The operator of each member of an ensemble, on its own amplitudes: `x[i] -> operators[i](x[i])`.

    The members are the tiles of a bucket, such as the inverses of the Grams of their own explicit
    amplitudes, which precondition them, or its segments, for the amplitudes the detectors of an
    observation share (see [`segment_ensemble`][]).
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


def _by_owner(x: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The map of the unknowns `x`, and their amplitudes of each tile and of each observation.

    The engine strides the two differently: per tile, and per segment. The amplitudes of the
    explicit gaps are per tile.
    """
    common = {kind: v for kind, v in x.items() if FACTORED_KINDS.get(kind) == 'sum'}
    own = {kind: v for kind, v in x.items() if kind != 'map' and kind not in common}
    return x['map'], own, common
