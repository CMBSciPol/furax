r"""The normal system of the multi-observation mapmaker, evaluated one detector tile at a time.

The mapmaker solves $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$ over the detector
tiles $t$ of every observation. The unknowns $x$ are a dictionary of the map of the selected sky
pixels, under `'map'`, and of the amplitudes solved for with it (see [`AMPLITUDE_KINDS`][]): the
template amplitudes of each detector, those the detectors of an observation share, and the gap
amplitudes. $J_t = [H_t S^\top, T_t, T_{s,t}, C_t]$ maps them to the tile's TOD, and $W'_t$ is its
weight (see
[`TileModel.deprojected_weight`][furax.mapmaking._model.TileModel.deprojected_weight]).
The right-hand side $\sum_t J_t^\top W'_t d_t$ has the same keys: the map RHS, the template RHS,
and the gap RHS. It is computed once, in the pass over the data, and held apart from the tiles
(see [`AccumulatedModel.rhs`][furax.mapmaking._accumulated.AccumulatedModel.rhs]).

With explicit gaps, the flagged samples are modelled by the spline amplitudes of the tile's gaps,
$C_t$: $J_t = [M_t H_t S^\top, M_t T_t, M_t T_{s,t}, C_t]$, with $M_t$ the mask of the good
samples, under the inverse noise $N_t^{-1}$ unmasked. The map block is that of the masked weight,
$H^\top M N^{-1} M H$; marginalizing the gap amplitudes would give the nested weight on the span of
the splines.

A bucket holds its [`Tile`][]s as a [`StackedEnsemble`][furax.core.StackedEnsemble], split over
the devices, grouped in segments by observation. Each leaf is stacked per tile, per segment when
the tiles of an observation share it (the boresight pointing, the bases and Gram inverse of the
shared templates, ...), or held once for the bucket. The functions and operators of this module
evaluate the sums over it with the ensemble engine.
"""

from dataclasses import dataclass, replace
from typing import Final, Literal

from jax.tree_util import register_dataclass

from furax import AbstractLinearOperator
from furax.core import (
    AdditionOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    DeferredOperator,
    StackedEnsemble,
    Stride,
    lift,
)

from ._gaps import GapBasisOperator
from ._model import TEMPLATE_KINDS, Holder, TemplateBundle, TemplateKind, TileModel
from .weight import WeightOperator

GAPS: Final = 'gaps'
"""The gap amplitudes of each detector, among the unknowns of the system."""

AmplitudeKind = TemplateKind | Literal['gaps']
"""A kind of amplitudes solved for with the map: a key of the unknowns of the system, besides
the `'map'`."""

AMPLITUDE_KINDS: dict[AmplitudeKind, Holder] = {**TEMPLATE_KINDS, GAPS: 'tile'}
"""The amplitudes solved for with the map, each a key of the unknowns, and where they are held."""

type Weight = Literal['system', 'plain', 'white']
"""$W'_t$ in the normal system of a tile: the weight of the system, or $W F$ without the implicit
templates, or the same with the white part of the noise only."""


@register_dataclass
@dataclass(frozen=True)
class Tile:
    """What the normal system needs of one detector tile of an observation.

    Its templates are partitioned by their `explicit` config flag, because the two enter the
    system in different places: the amplitudes of the implicit ones are marginalized in the
    weight, those of the explicit ones solved for with the map. The explicit ones further split by
    who sees their amplitudes: each detector alone, or the detectors of the observation together.
    """

    model: TileModel
    """The tile's acquisition operator, weight and noise model."""
    implicit: TemplateBundle | None
    """The implicit templates, whose amplitudes are each detector's own, marginalized."""
    explicit: dict[TemplateKind, TemplateBundle | None]
    """The explicit templates, solved for with the map, by the kind of their amplitudes (see
    `TEMPLATE_KINDS`); `None` for a kind absent."""
    gap_preconditioner: AbstractLinearOperator | None = None
    """The inverse of the diagonal of the white-noise Gram of the gap amplitudes (None without
    explicit gaps)."""

    def amplitudes(self, kind: AmplitudeKind) -> TemplateBundle | None:
        r"""The amplitudes of `kind`; `None` if the tile has none.

        Their bundle holds the operator from them to the TOD, $T_t$, $T_{s,t}$ or $C_t$, the
        inverse of their Gram, and its ridge. The gap amplitudes are preconditioned by the inverse
        of the diagonal of their white-noise Gram, with no ridge, and their basis is built from
        the mask when applied.
        """
        if kind != GAPS:
            return self.explicit[kind]
        model = self.model
        if model.gaps is None:
            return None
        assert self.gap_preconditioner is not None  # set with the gaps
        basis = GapBasisOperator.create(model.gaps, model.M, model.n_samples)
        return TemplateBundle(basis, self.gap_preconditioner, None)

    def operator(self, kind: AmplitudeKind) -> AbstractLinearOperator:
        """The operator from the amplitudes of `kind` to the TOD: $T_t$, $T_{s,t}$ or $C_t$."""
        bundle = self.amplitudes(kind)
        assert bundle is not None
        return bundle.operator


def amplitude_kinds(tiles: StackedEnsemble) -> tuple[AmplitudeKind, ...]:
    """The kinds of amplitudes solved for with the map in a bucket, each a key of the unknowns.

    Args:
        tiles: The tiles of the bucket.
    """
    return tuple(kind for kind in AMPLITUDE_KINDS if tiles.template.amplitudes(kind) is not None)


def preconditioner(tiles: StackedEnsemble, kind: AmplitudeKind) -> AbstractLinearOperator:
    """The preconditioner of the amplitudes of `kind` of a bucket.

    The amplitudes are preconditioned by the inverse of their Gram (see [`Tile.amplitudes`][]),
    each tile's for those of its detectors and each observation's for those they share.

    Args:
        tiles: The tiles of the bucket.
        kind: The kind of amplitudes.
    """
    inverse = _amplitude_operator(tiles, kind, 'gram_inverse')
    assert inverse is not None
    return inverse


def ridge(tiles: StackedEnsemble, kind: AmplitudeKind) -> AbstractLinearOperator | None:
    """The ridge the Gram of the amplitudes of `kind` was factored with, a term of the system.

    `None` without regularization, and for the gap amplitudes. The arguments are those of
    [`preconditioner`][].
    """
    return _amplitude_operator(tiles, kind, 'ridge')


def _amplitude_operator(
    tiles: StackedEnsemble, kind: AmplitudeKind, part: str
) -> AbstractLinearOperator | None:
    """An operator of the amplitudes of `kind`, on them; `None` if absent.

    `part` is a field of [`TemplateBundle`][furax.mapmaking._model.TemplateBundle], whose
    operator is held per tile, or per observation for the shared templates. The amplitudes are
    `(tile, ...)`, or `(segment, ...)` for those the detectors of an observation share.
    """
    if getattr(tiles.template.amplitudes(kind), part) is None:
        return None
    # structural: the operators keep the leaves of the tiles, and their strides
    operators = lift(lambda tile: getattr(tile.amplitudes(kind), part), tiles)
    assert isinstance(operators, StackedEnsemble)
    if AMPLITUDE_KINDS[kind] == 'observation':
        operators = segment_ensemble(operators)
    # small computations: every member of a device is evaluated at once, vectorized
    policy = replace(operators.policy, batch=operators.size_per_device)
    return BlockDiagonalOperator(replace(operators, policy=policy))


_STRIDES: dict[Holder, Literal[1, 'segment']] = {'tile': 1, 'observation': 'segment'}
"""The stride of the amplitudes held per tile, and per observation, in the sums over the tiles."""


def tile_joint(
    tile: Tile, selection: AbstractLinearOperator | None, kinds: tuple[AmplitudeKind, ...]
) -> AbstractLinearOperator:
    r"""The joint operator $J_t$ of a tile, from the unknowns to its TOD.

    The unknowns are a dictionary of the `'map'` and of the amplitudes of each kind of `kinds`.
    With the gap amplitudes among them, which stand for the flagged samples, the other blocks are
    masked.

    Args:
        tile: The tile.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        kinds: The kinds of amplitudes among the unknowns (see [`AMPLITUDE_KINDS`][]).
    """
    model = tile.model
    H = model.H if selection is None else model.H @ selection.T
    blocks = {'map': H} | {kind: tile.operator(kind) for kind in kinds if kind != GAPS}
    if GAPS in kinds:
        blocks = {kind: model.M @ block for kind, block in blocks.items()}
        blocks[GAPS] = tile.operator(GAPS)
    return BlockRowOperator(blocks)


def tile_weight(
    tile: Tile, kinds: tuple[AmplitudeKind, ...], weight: Weight = 'system'
) -> AbstractLinearOperator:
    """The weight $W'_t$ of a tile in the normal system.

    With the gap amplitudes among the unknowns, which stand for the flagged samples, it is the
    inverse noise unmasked.

    Args:
        tile: The tile.
        kinds: The kinds of amplitudes among the unknowns.
        weight: The weight (see [`Weight`][]).
    """
    model = tile.model
    W = model.diag_W() if weight == 'white' else model.W
    if GAPS in kinds:
        assert isinstance(W, WeightOperator)
        return W.weight
    if weight == 'system':
        return model.deprojected_weight(tile.implicit)
    return (W @ model.F).reduce()


@dataclass(frozen=True)
class TileNormal:
    r"""Builds $J_t^\top W'_t J_t$ of a tile from `(tile, selection)`.

    The builder of a [`DeferredOperator`][furax.core.DeferredOperator], see [`tile_joint`][] and
    [`tile_weight`][].
    """

    kinds: tuple[AmplitudeKind, ...]
    """The kinds of amplitudes among the unknowns."""
    weight: Weight = 'system'

    def __call__(
        self, member: tuple[Tile, AbstractLinearOperator | None]
    ) -> AbstractLinearOperator:
        tile, selection = member
        J = tile_joint(tile, selection, self.kinds)
        return J.T @ tile_weight(tile, self.kinds, self.weight) @ J


@dataclass(frozen=True)
class TileResponse:
    r"""Builds $J_t^\top W'_t M_t H_t$ of a tile from `(tile, selection)`.

    It maps a sky to the right-hand side of the tile's TOD, masked as the data are.
    """

    kinds: tuple[AmplitudeKind, ...]
    """The kinds of amplitudes among the unknowns."""

    def __call__(
        self, member: tuple[Tile, AbstractLinearOperator | None]
    ) -> AbstractLinearOperator:
        tile, selection = member
        J = tile_joint(tile, selection, self.kinds)
        # with the gap amplitudes, the weight is unmasked
        return J.T @ tile_weight(tile, self.kinds) @ tile.model.M @ tile.model.H


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


def bucket_normal(
    tiles: StackedEnsemble,
    selection: AbstractLinearOperator | None,
    weight: Weight = 'system',
    amplitudes: bool = True,
) -> AdditionOperator:
    r"""$\sum_t J_t^\top W'_t J_t$ over the detector tiles of a bucket.

    The unknowns are a dictionary: the `'map'`, which every tile reads whole, and the amplitudes
    of each kind the bucket has (see [`amplitude_kinds`][]) if `amplitudes` is set, of which each
    tile reads its own entry, or its observation's.

    Args:
        tiles: The tiles of the bucket.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
        weight: $W'_t$ (see [`Weight`][]).
        amplitudes: Whether the unknowns include the template and gap amplitudes.
    """
    kinds = amplitude_kinds(tiles) if amplitudes else ()
    build = TileNormal(kinds, weight)
    members = lift(lambda tile: DeferredOperator(build, (tile, selection)), tiles)
    return AdditionOperator(members, in_strides=_unknown_strides(kinds))


def bucket_response(
    tiles: StackedEnsemble, selection: AbstractLinearOperator | None
) -> AdditionOperator:
    r"""$\sum_t J_t^\top W'_t M_t H_t$ over the detector tiles of a bucket.

    It maps a sky on the full grid, which every tile reads whole, to the right-hand side of its
    TOD, keyed as the unknowns of [`bucket_normal`][].
    """
    kinds = amplitude_kinds(tiles)
    build = TileResponse(kinds)
    members = lift(lambda tile: DeferredOperator(build, (tile, selection)), tiles)
    return AdditionOperator(members, in_strides=None, out_strides=_unknown_strides(kinds))


def _unknown_strides(kinds: tuple[AmplitudeKind, ...]) -> dict[str, Stride]:
    """How each tile reads the unknowns: the map whole, the amplitudes per tile or segment."""
    return {'map': None} | {kind: _STRIDES[AMPLITUDE_KINDS[kind]] for kind in kinds}
