r"""The normal system of the multi-observation mapmaker, summed over the detector tiles of a bucket.

The system is $\sum_t J_t^\top W'_t J_t \, x = \sum_t J_t^\top W'_t d_t$, over the tiles $t$ (see
[`Tile.system`][furax.mapmaking._tile.Tile.system]). The unknowns $x$ are a dictionary of the map
of the selected sky pixels, under `'map'`, and of the amplitudes of each kind solved for with it.

A bucket holds its tiles as a [`StackedEnsemble`][furax.core.StackedEnsemble], split over the
devices and grouped in segments by observation: each leaf is stacked per tile, per segment, or
held once for the bucket.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from furax import AbstractLinearOperator
from furax.core import (
    AdditionOperator,
    BlockDiagonalOperator,
    DeferredOperator,
    StackedEnsemble,
    Stride,
    lift,
)

from ._tile import AMPLITUDE_STRIDES, AmplitudeKind, TemplateBundle, Tile
from .preconditioner import BJPreconditioner


def bucket_normal(
    tiles: StackedEnsemble, selection: AbstractLinearOperator | None
) -> AdditionOperator:
    r"""$\sum_t J_t^\top W'_t J_t$ over the detector tiles of a bucket.

    The unknowns are a dictionary: the `'map'`, which every tile reads whole, and the amplitudes
    of each kind the tiles have (see [`Tile.kinds`][furax.mapmaking._tile.Tile.kinds]), of which
    each tile reads its own entry, or its observation's.

    Args:
        tiles: The tiles of the bucket.
        selection: The selection $S$ of the estimated pixels; the whole sky without it.
    """
    return _summed(Tile.normal, tiles, selection, in_strides=_unknown_strides(tiles))


def bucket_response(
    tiles: StackedEnsemble, selection: AbstractLinearOperator | None
) -> AdditionOperator:
    r"""$\sum_t J_t^\top W'_t M_t H_t$ over the detector tiles of a bucket.

    It maps a sky on the full grid, which every tile reads whole, to the right-hand side of the
    TOD it gives, keyed as the unknowns of [`bucket_normal`][]. The arguments are those of
    [`bucket_normal`][].
    """
    strides = _unknown_strides(tiles)
    return _summed(Tile.response, tiles, selection, in_strides=None, out_strides=strides)


def block_jacobi(buckets: list[StackedEnsemble], binned: bool) -> BJPreconditioner:
    """The block-Jacobi preconditioner, from the map system summed over the buckets.

    The system is under the weight without the noise correlations, nor the implicit templates
    (see [`bucket_map_normal`][]).
    """
    terms = [bucket_map_normal(tiles, binned) for tiles in buckets]
    return BJPreconditioner.create(AdditionOperator(terms, sequential=True))


def bucket_map_normal(tiles: StackedEnsemble, binned: bool) -> AdditionOperator:
    r"""$\sum_t H_t^\top W_t F_t H_t$ over the detector tiles of a bucket, on the full sky.

    Args:
        tiles: The tiles of the bucket.
        binned: Whether $W_t$ is the weight without the implicit templates; otherwise it is its
            white part, $\operatorname{diag}(W_t)$.
    """
    return _summed(Tile.binned_normal if binned else Tile.white_normal, tiles)


def preconditioner(tiles: StackedEnsemble, kind: AmplitudeKind) -> AbstractLinearOperator:
    """The preconditioner of the amplitudes of `kind` of a bucket.

    The amplitudes are preconditioned by the inverse of their Gram (see
    [`Tile.amplitudes`][furax.mapmaking._tile.Tile.amplitudes]), each tile's for those of its
    detectors and each observation's for those they share.

    Args:
        tiles: The tiles of the bucket.
        kind: The kind of amplitudes.
    """
    inverse = _amplitude_operator(tiles, kind, lambda bundle: bundle.gram_inverse)
    assert inverse is not None
    return inverse


def ridge(tiles: StackedEnsemble, kind: AmplitudeKind) -> AbstractLinearOperator | None:
    """The ridge the Gram of the amplitudes of `kind` was factored with, a term of the system.

    `None` without regularization, and for the gap amplitudes. The arguments are those of
    [`preconditioner`][].
    """
    return _amplitude_operator(tiles, kind, lambda bundle: bundle.ridge)


def _summed(
    build: Callable[..., AbstractLinearOperator], tiles: StackedEnsemble, *args: Any, **strides: Any
) -> AdditionOperator:
    r"""$\sum_t$ `build(tile_t, *args)` over the tiles, each operator built when applied."""
    members = lift(lambda tile: DeferredOperator(build, tile, *args), tiles)
    return AdditionOperator(members, **strides)


def _unknown_strides(tiles: StackedEnsemble) -> dict[str, Stride]:
    """How each tile reads the unknowns: the map whole, the amplitudes per tile or segment."""
    return {'map': None} | {kind: AMPLITUDE_STRIDES[kind] for kind in tiles.template.kinds}


def _amplitude_operator(
    tiles: StackedEnsemble,
    kind: AmplitudeKind,
    part: Callable[[TemplateBundle], AbstractLinearOperator | None],
) -> AbstractLinearOperator | None:
    """An operator of the amplitudes of `kind`, on them; `None` if absent.

    `part` picks it out of their [`TemplateBundle`][furax.mapmaking._tile.TemplateBundle], held
    per tile, or per observation for the shared templates. The amplitudes are `(tile, ...)`, or
    `(segment, ...)` for those the detectors of an observation share.
    """
    if part(tiles.template.amplitudes(kind)) is None:
        return None
    # structural: the operators keep the leaves of the tiles, and their strides
    operators = lift(lambda tile: part(tile.amplitudes(kind)), tiles)
    assert isinstance(operators, StackedEnsemble)
    if AMPLITUDE_STRIDES[kind] == 'segment':
        operators = operators.by_segment()
    # small computations: every member of a device is evaluated at once, vectorized
    policy = replace(operators.policy, batch=operators.size_per_device)
    return BlockDiagonalOperator(replace(operators, policy=policy))
