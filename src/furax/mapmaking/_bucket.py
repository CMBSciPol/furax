r"""The normal system of the multi-observation mapmaker, evaluated one detector tile at a time.

The mapmaker solves $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$ over the detector
tiles $t$ of every observation, with $J_t = [H_t S^\top, T_{e,t}, T_{g,t}]$ mapping the selected
sky pixels, the tile's own template amplitudes and those its observation's detectors share to the
tile's TOD, and $W'_t$ its weight (see
[`ObservationModel.deprojected_weight`][furax.mapmaking._model.ObservationModel.deprojected_weight]).

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
from jaxtyping import Array, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.core import BlockRowOperator, Policy, StackedEnsemble, map_reduce
from furax.obs.stokes import StokesType

from ._model import ObservationModel, ObservationTemplates


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
    tile: Tile, selection: AbstractLinearOperator | None, own: bool, common: bool
) -> AbstractLinearOperator:
    r"""$J_t = [H_t S^\top, T_{e,t}, T_{g,t}]$ of a tile, with the amplitudes asked for.

    Args:
        tile: The tile.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        own: Whether to include the tile's own explicit amplitudes.
        common: Whether to include those its observation's detectors have in common.
    """
    H = tile.model.H if selection is None else tile.model.H @ selection.T
    blocks = [H]
    if own:
        assert tile.templates is not None and tile.templates.explicit is not None
        blocks.append(tile.templates.explicit.operator)
    if common:
        assert tile.templates is not None and tile.templates.common is not None
        blocks.append(tile.templates.common.operator)
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

    The unknowns are a list: the map, then the tiles' own amplitudes and their observations', if
    the bucket has such templates and `amplitudes` is set. Otherwise they are the map alone. Without
    `selection`, the map is the whole sky.
    """

    model: BucketModel
    selection: AbstractLinearOperator | None
    """The selection of the estimated pixels, if any."""
    weight: Literal['system', 'plain', 'white'] = field(default='system', metadata={'static': True})
    """$W'_t$: the weight of the system, or $W F$ without the implicit templates, or the same with
    the white part of the noise only."""
    amplitudes: bool = field(default=True, metadata={'static': True})
    """Whether the unknowns include the amplitudes of the explicit templates."""
    policy: Policy | None = field(default=None, metadata={'static': True})
    """How the tiles are walked (see `bucket_apply`); by default that of `model.tiles`."""

    def mv(self, x: Any) -> Any:
        m, own, common = self._split(x)
        # closed over, not an engine argument: the selection may hold non-array leaves
        selection = self.selection

        def apply(tile, m, own_t, common_o):
            J = joint_operator(tile, selection, *self._amplitudes_present)
            inputs = [v for v in (own_t, common_o) if v is not None]
            return self._project(tile, self._weight(tile)(J([m, *inputs])), selection)

        return self._join(*bucket_apply(self.model, apply, m, own, common, self.policy))

    def response(self, sky: StokesType) -> Any:
        r"""$\sum_t J_t^\top W'_t H_t \, s$: the right-hand side of the TOD of a sky `sky`."""
        selection = self.selection

        def apply(tile, sky, _own, _common):
            return self._project(tile, self._weight(tile)(tile.model.H(sky)), selection)

        return self._join(*bucket_apply(self.model, apply, sky, None, None, self.policy))

    def _project(
        self, tile: Tile, u: PyTree[Array], selection: AbstractLinearOperator | None
    ) -> tuple[Any, Any, Any]:
        r"""$J_t^\top u$, in parts."""
        own, common = self._amplitudes_present
        return _unpack(joint_operator(tile, selection, own, common).T(u), own, common)

    def _weight(self, tile: Tile) -> AbstractLinearOperator:
        if self.weight == 'system':
            return system_weight(tile)
        model = tile.model
        W = model.diag_W() if self.weight == 'white' else model.W
        return (W @ model.F).reduce()

    @property
    def _amplitudes_present(self) -> tuple[bool, bool]:
        """Whether the unknowns have the tiles' own amplitudes, and their observations'."""
        templates = self.model.templates
        if not self.amplitudes or templates is None:
            return False, False
        return templates.explicit is not None, templates.common is not None

    def _split(self, x: Any) -> tuple[Any, Any, Any]:
        """The map and the amplitudes of each kind, `None` for those absent."""
        own, common = self._amplitudes_present
        if not own and not common:
            return x, None, None
        return _unpack(x, own, common)

    def _join(self, m: Any, own: Any, common: Any) -> Any:
        present_own, present_common = self._amplitudes_present
        if not present_own and not present_common:
            return m
        return [m, *([own] if present_own else []), *([common] if present_common else [])]


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


def _unpack(parts: list[PyTree[Array]], own: bool, common: bool) -> tuple[Any, Any, Any]:
    """The map and the amplitudes of each kind of a list of parts, `None` for those absent."""
    m, *rest = parts
    return m, rest.pop(0) if own else None, rest.pop(0) if common else None
