r"""The normal system of the multi-observation mapmaker, over the detector tiles of a bucket.

The system is $J^\top W J \, x = J^\top W M d$. The unknowns $x$ are a dictionary of the map of
the selected sky pixels, under `'map'`, and of the amplitudes of each kind solved for with it. The
joint operator $J$ maps them to the TOD of every tile of the bucket, each tile applying its own
$J_t$ (see [`Tile.joint`][furax.mapmaking._tile.Tile.joint]), whose pointing reads the selected
pixels alone once they are selected (see [`selected`][]). The weight $W$ acts on the TOD of the
bucket, each tile's by its [`system_weight`][furax.mapmaking._tile.Tile.system_weight] $W'_t$ (see
[`weight`][]).

$J$ and $W$ are assemblies over the tiles, which `reduce` fuses: each tile applies its own
$J_t^\top W'_t J_t$, in one sweep over the tiles.

A bucket holds its tiles as a [`StackedEnsemble`][furax.core.StackedEnsemble], split over the
devices and grouped in segments by observation: each leaf is stacked per tile, per segment, or
held once for the bucket.
"""

from collections.abc import Callable
from dataclasses import replace

from furax import AbstractLinearOperator
from furax.core import (
    AdditionOperator,
    AssemblyOperator,
    BlockDiagonalOperator,
    DeferredOperator,
    StackedEnsemble,
    Stride,
    lift,
)
from furax.obs.landscapes import LocalStokesLandscape

from ._tile import AMPLITUDE_STRIDES, AmplitudeKind, TemplateBundle, Tile
from .preconditioner import BJPreconditioner


def selected(tiles: StackedEnsemble, pixels: LocalStokesLandscape) -> StackedEnsemble:
    """The tiles of a bucket, whose map is that of the selected `pixels`.

    See [`Tile.selected`][furax.mapmaking._tile.Tile.selected].
    """
    # structural with the pointing on the fly: the tiles keep their leaves, and share the pixels
    return lift(lambda tile: tile.selected(pixels), tiles)


def bucket_normal(tiles: StackedEnsemble) -> AbstractLinearOperator:
    r"""$J^\top W J$ over the detector tiles of a bucket.

    The unknowns are a dictionary: the `'map'`, which every tile reads whole, and the amplitudes
    of each kind the tiles have (see [`Tile.kinds`][furax.mapmaking._tile.Tile.kinds]), of which
    each tile reads its own entry, or its observation's.
    """
    J = joint(tiles)
    return (J.T @ weight(tiles) @ J).reduce()


def bucket_response(tiles: StackedEnsemble) -> AbstractLinearOperator:
    r"""$J^\top W M H$ over the detector tiles of a bucket.

    It maps a sky on the full grid, which every tile reads whole, to the right-hand side of the
    TOD it gives, keyed as the unknowns of [`bucket_normal`][].
    """
    scanned = AssemblyOperator(_deferred(Tile.masked_acquisition, tiles), None, 1)
    return (joint(tiles).T @ weight(tiles) @ scanned).reduce()


def joint(tiles: StackedEnsemble) -> AssemblyOperator:
    r"""The joint operator $J$, from the unknowns to the TOD of each tile, `(tile, ...)`."""
    return AssemblyOperator(_deferred(Tile.joint, tiles), _unknown_strides(tiles), 1)


def weight(tiles: StackedEnsemble) -> AssemblyOperator:
    r"""The weight $W$ on the TOD of the tiles of a bucket, `(tile, ...)`.

    Each tile weights its own TOD by its
    [`system_weight`][furax.mapmaking._tile.Tile.system_weight] $W'_t$.
    """
    return AssemblyOperator(_deferred(Tile.system_weight, tiles), 1)


def block_jacobi(buckets: list[StackedEnsemble], binned: bool) -> BJPreconditioner:
    """The block-Jacobi preconditioner, from the map system summed over the buckets.

    The system is under the weight without the noise correlations, nor the implicit templates
    (see [`bucket_map_block`][]).
    """
    terms = [bucket_map_block(tiles, binned) for tiles in buckets]
    return BJPreconditioner.create(AdditionOperator(terms, sequential=True))


def bucket_map_block(tiles: StackedEnsemble, binned: bool) -> AdditionOperator:
    r"""$\sum_t H_t^\top W_t F_t H_t$ over the detector tiles of a bucket, on the full sky.

    Args:
        tiles: The tiles of the bucket.
        binned: Whether $W_t$ is the weight without the implicit templates; otherwise it is its
            white part, $\operatorname{diag}(W_t)$.
    """
    return AdditionOperator(_deferred(Tile.map_block if binned else Tile.white_map_block, tiles))


def preconditioner(tiles: StackedEnsemble, kind: AmplitudeKind) -> AbstractLinearOperator:
    """The preconditioner of the amplitudes of `kind` of a bucket.

    The amplitudes are preconditioned by the inverse of their Gram (see
    [`Tile.templates`][furax.mapmaking._tile.Tile.templates]), each tile's for those of its
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


def _deferred(
    build: Callable[[Tile], AbstractLinearOperator], tiles: StackedEnsemble
) -> StackedEnsemble:
    """The operators `build(tile)` of the tiles, each built when applied."""
    return lift(lambda tile: DeferredOperator(build, tile), tiles)


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
    if part(tiles.template.templates(kind)) is None:
        return None
    # structural: the operators keep the leaves of the tiles, and their strides
    operators = lift(lambda tile: part(tile.templates(kind)), tiles)
    if AMPLITUDE_STRIDES[kind] == 'segment':
        operators = operators.by_segment()
    # the operators are small: every member of a device is evaluated at once, vectorized
    policy = replace(operators.policy, batch=operators.size_per_device)
    return BlockDiagonalOperator(replace(operators, policy=policy))
