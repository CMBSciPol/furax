r"""Symmetric matrices with a block-banded core and a dense border, and their solvers.

A [`BorderedBandedMatrix`][] stores a batch of symmetric matrices

$$ M = \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix} $$

over unknowns split in two: a core $x_a$ of `n_blocks` blocks of `k`, on which $A$ is
block-banded, and a border $x_c$ of `m`, on which $C$ is dense, coupled by the dense $B$. The
core and the border are each optional. [`probe_bordered_banded`][] recovers such a matrix from
the operator it represents, and [`ArrowheadCholeskyOperator`][] solves a batch of independent
blocks coupled to one dense block.
"""

import functools
from collections.abc import Callable
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import DTypeLike, Float, Int, PyTree

from furax import AbstractLinearOperator, symmetric, tree

from .cholesky import BandedCholeskyOperator, BorderedBandedCholeskyOperator, _einsum

__all__ = [
    'ArrowheadCholeskyOperator',
    'BorderedBandedMatrix',
    'probe_bordered_banded',
]


@symmetric
class BorderedBandedMatrix(AbstractLinearOperator):
    r"""A batch of symmetric matrices with a block-banded core and a dense border.

    The matrix acts on a pair `(x_a, x_c)`: the core `(*batch, n_blocks, k)` and the border
    `(*batch, m)`, either `None` when the matrix has no such part. The core block $A$ is stored as
    its upper bands (see [`banded_cholesky`][furax.linalg.banded_cholesky]), the border $B$ with
    its rows in the flattened `(n_blocks, k)` order.

    Examples:
        A tridiagonal core of two scalar blocks, bordered by one unknown:

        >>> bands = jnp.array([[[[4.0]], [[1.0]]], [[[4.0]], [[0.0]]]])  # A = [[4, 1], [1, 4]]
        >>> matrix = BorderedBandedMatrix(bands, jnp.ones((2, 1)), jnp.array([[3.0]]))
        >>> matrix((jnp.ones((2, 1)), jnp.ones(1)))
        (Array([[6.],
               [6.]], dtype=float32), Array([5.], dtype=float32))
        >>> x_a, x_c = matrix.cholesky()((jnp.array([[6.0], [6.0]]), jnp.array([5.0])))
        >>> x_a.ravel(), x_c
        (Array([1., 1.], dtype=float32), Array([1.], dtype=float32))
    """

    bands: Float[Array, '*batch n w1 k k'] | None = None
    """The upper bands of the core block $A$."""
    border: Float[Array, '*batch nk m'] | None = None
    """The border $B$, present with both a core and a corner."""
    corner: Float[Array, '*batch m m'] | None = None
    """The corner block $C$."""

    def __post_init__(self) -> None:
        structure = self._structure()
        if self.in_structure is None:
            object.__setattr__(self, 'in_structure', structure)
        super().__post_init__()

    def mv(self, x: tuple[Array | None, Array | None]) -> tuple[Array | None, Array | None]:
        x_a, x_c = x
        y_a = y_c = None
        if self.bands is not None:
            assert x_a is not None
            n, w1 = self.bands.shape[-4], self.bands.shape[-3]
            y_a = jnp.zeros_like(x_a)
            for d in range(min(w1, n)):  # the upper band `d`, and its transpose below the diagonal
                band = self.bands[..., : n - d, d, :, :]
                y_a = y_a.at[..., : n - d, :].add(
                    _einsum('...iab,...ib->...ia', band, x_a[..., d:, :])
                )
                if d:
                    lower = _einsum('...iab,...ia->...ib', band, x_a[..., : n - d, :])
                    y_a = y_a.at[..., d:, :].add(lower)
        if self.corner is not None:
            y_c = _einsum('...ab,...b->...a', self.corner, x_c)
        if self.border is not None:
            assert x_a is not None and y_a is not None and y_c is not None
            flat_a = x_a.reshape(*x_a.shape[:-2], -1)
            y_a = y_a + _einsum('...am,...m->...a', self.border, x_c).reshape(y_a.shape)
            y_c = y_c + _einsum('...am,...a->...m', self.border, flat_a)
        return y_a, y_c

    def cholesky(self) -> AbstractLinearOperator:
        """The inverse, by Cholesky factorization, on the same pairs `(x_a, x_c)`."""
        if self.bands is not None and self.corner is not None:
            assert self.border is not None
            return BorderedBandedCholeskyOperator.from_blocks(self.bands, self.border, self.corner)
        if self.bands is not None:
            return BandedCholeskyOperator.from_bands(self.bands, self.in_structure)
        assert self.corner is not None
        return BandedCholeskyOperator.from_dense(self.corner, self.in_structure)

    def _structure(self) -> tuple[jax.ShapeDtypeStruct | None, jax.ShapeDtypeStruct | None]:
        core = corner = None
        if self.bands is not None:
            *batch, n, _, k, _ = self.bands.shape
            core = jax.ShapeDtypeStruct((*batch, n, k), self.bands.dtype)
        if self.corner is not None:
            corner = jax.ShapeDtypeStruct(self.corner.shape[:-1], self.corner.dtype)
        if core is None and corner is None:
            raise ValueError('a bordered banded matrix needs a core or a corner')
        if (self.border is not None) != (core is not None and corner is not None):
            raise ValueError('the border is present exactly when both the core and corner are')
        if self.border is not None and core is not None and corner is not None:
            *batch, n, k = core.shape
            m = corner.shape[-1]
            if self.border.shape != (*batch, n * k, m) or corner.shape != (*batch, m):
                raise ValueError(
                    f'unknowns {core.shape} and {corner.shape} take a border of shape '
                    f'{(*batch, n * k, m)}, not {self.border.shape}'
                )
        return core, corner


def probe_bordered_banded(
    apply: Callable[
        [tuple[Array | None, Array | None]], tuple[tuple[Array | None, Array | None], Any]
    ],
    *,
    batch_shape: tuple[int, ...],
    n_blocks: int,
    k: int,
    n_border: int,
    half_width: int,
    dtype: DTypeLike,
    batch_size: int = 1,
    n_colors: int | None = None,
    all_extras: bool = False,
) -> tuple[BorderedBandedMatrix, PyTree[Array]]:
    """Recover a bordered banded matrix from the responses of its operator to a few probes.

    `apply` maps a pair `(x_a, x_c)` to the pair `(y_a, y_c)` of the matrix times it, and to any
    other output, `None` for nothing. Every element of the batch is probed alike, so the operator
    must not couple them. A border unknown is probed alone, its response the matrix column itself.
    The core is probed one of its `k` functions at a time, on every block of one color: the blocks
    `n_colors` apart. A core block couples to those at most `half_width` away, so with
    `n_colors = 2·half_width + 1` the response on a block within reach of a colored one holds that
    block's coupling alone. More colors spread the probed blocks further apart, for other outputs
    that reach further than the band. There are `n_colors · k + n_border` probes. The other outputs
    are collected for every probe, and only then restricted to the border ones, unless
    `all_extras`.

    Args:
        apply: The operator, and the other outputs to collect.
        batch_shape: The batch of matrices.
        n_blocks: The number of core blocks, 0 without a core.
        k: The size of a core block.
        n_border: The size of the border, 0 without one.
        half_width: The number of blocks a core block couples to on either side.
        dtype: The dtype of the probes.
        batch_size: The number of probes applied at once, bounding the memory.
        n_colors: The number of colors of the core probes, at least `2·half_width + 1` or
            `n_blocks`, whichever is fewer; by default, that many.
        all_extras: Whether to return the other outputs of every probe, rather than of the border
            ones only.

    Returns:
        The matrix, and the other outputs for the probes of the border unknowns, or of all the
        probes, core ones first, color by color, then function by function of a block, stacked on
        a leading axis.

    Raises:
        ValueError: If `n_colors` is less than both `2·half_width + 1` and `n_blocks`.
    """
    if n_colors is not None and n_colors < min(n_blocks, 2 * half_width + 1):
        raise ValueError(f'{n_colors} colors cannot resolve a band of half width {half_width}')
    n_colors = min(n_blocks, 2 * half_width + 1 if n_colors is None else n_colors)
    n_core = n_colors * k

    def one(p: Int[Array, '']) -> tuple[Array | None, Array | None, Any]:
        x_a = x_c = None
        if n_blocks:
            color, a = p // k, p % k
            blocks = jnp.arange(n_blocks) % n_colors == color
            pattern = (blocks[:, None] & (jnp.arange(k) == a)).astype(dtype)
            x_a = jnp.broadcast_to(pattern, (*batch_shape, n_blocks, k))
        if n_border:
            unit = jax.nn.one_hot(p - n_core, n_border, dtype=dtype)  # zero for the core probes
            x_c = jnp.broadcast_to(unit, (*batch_shape, n_border))
        (y_a, y_c), extra = apply((x_a, x_c))
        return y_a, y_c, extra

    y_a, y_c, extra = jax.lax.map(one, jnp.arange(n_core + n_border), batch_size=batch_size)

    bands = border = corner = None
    if n_blocks:
        assert y_a is not None
        bands = _core_bands(y_a[:n_core], n_colors, k, half_width)
    if n_border:
        assert y_c is not None
        corner = jnp.moveaxis(y_c[n_core:], 0, -1)  # (*batch, row, column)
        corner = (corner + jnp.swapaxes(corner, -1, -2)) / 2
        if n_blocks:
            assert y_a is not None
            columns = y_a[n_core:].reshape(n_border, *batch_shape, n_blocks * k)
            border = jnp.moveaxis(columns, 0, -1)
    if not all_extras:
        extra = tree.stacked_get(extra, slice(n_core, None))
    return BorderedBandedMatrix(bands, border, corner), extra


@symmetric
class ArrowheadCholeskyOperator(AbstractLinearOperator):
    r"""Inverse of a symmetric positive-definite arrowhead matrix, by Schur complement.

    The matrix couples independent blocks $A$, such as one per detector, to one dense block $C$:

    $$ M = \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix}. $$

    With $Z = A^{-1} B$, the unknowns of $C$ are eliminated into $S = C - B^\top Z$, and solving
    $M x = r$ is

    $$ x_c = S^{-1} (r_c - Z^\top r_a), \qquad x_a = A^{-1} r_a - Z x_c. $$

    Calling the operator on a pair `(r_a, r_c)` solves for `(x_a, x_c)`: `r_a` a pytree on which
    $A^{-1}$ acts, `r_c` a vector of the `m` unknowns of $C$. Unlike
    [`BorderedBandedCholeskyOperator`][furax.linalg.BorderedBandedCholeskyOperator], a batch of
    bordered matrices, it is one matrix, whose blocks $A$ share the one dense block $C$.

    Examples:
        >>> blocks = BandedCholeskyOperator.from_dense(jnp.array([[[4.0]], [[4.0]]]))  # A = 4 I
        >>> cross = jnp.array([[[1.0], [1.0]]])  # B, one column per unknown of C
        >>> op = ArrowheadCholeskyOperator.from_blocks(blocks, cross, jnp.array([[3.0]]))
        >>> x_a, x_c = op((jnp.array([[5.0], [5.0]]), jnp.array([5.0])))  # M x = r for x = 1
        >>> x_a.ravel(), x_c
        (Array([1., 1.], dtype=float32), Array([1.], dtype=float32))
    """

    blocks: AbstractLinearOperator
    """$A^{-1}$."""
    coupling: PyTree[Array]
    """$Z$, shaped like the unknowns of $A$ with a leading axis of those of $C$."""
    schur: BandedCholeskyOperator
    """$S^{-1}$."""

    @classmethod
    def from_blocks(
        cls,
        blocks: AbstractLinearOperator,
        cross: PyTree[Array],
        corner: Float[Array, 'm m'],
        batch_size: int = 1,
    ) -> Self:
        """Eliminate the unknowns of the dense block.

        Args:
            blocks: $A^{-1}$.
            cross: $B$, shaped like the unknowns of $A$ with a leading axis of those of $C$.
            corner: $C$.
            batch_size: The number of columns of $B$ solved at once, bounding the memory.
        """
        coupling = tree.stacked_map(blocks, cross, batch_size=batch_size)
        schur = corner - _dots(cross, coupling)
        schur = (schur + schur.T) / 2  # symmetric up to rounding
        structure = (blocks.in_structure, jax.ShapeDtypeStruct(corner.shape[:1], corner.dtype))
        return cls(
            blocks, coupling, BandedCholeskyOperator.from_dense(schur), in_structure=structure
        )

    def mv(self, x: tuple[PyTree[Array], Array]) -> tuple[PyTree[Array], Array]:
        r_a, r_c = x
        x_c = self.schur(r_c - _dots(self.coupling, jax.tree.map(lambda a: a[None], r_a))[:, 0])
        correction = jax.tree.map(lambda z: _einsum('m...,m->...', z, x_c), self.coupling)
        x_a = tree.sub(self.blocks(r_a), correction)
        return x_a, x_c


def _dots(a: PyTree[Array], b: PyTree[Array]) -> Float[Array, 'p q']:
    """The dot products of the leading-axis slices of two pytrees of the same structure.

    A matrix product at full precision, many times faster than the reduction of
    [`stacked_gram`][furax.tree.stacked_gram] on large slices.
    """
    products = jax.tree.map(
        lambda u, v: _einsum('pi,qi->pq', u.reshape(u.shape[0], -1), v.reshape(v.shape[0], -1)),
        a,
        b,
    )
    return functools.reduce(jnp.add, jax.tree.leaves(products))


def _core_bands(
    responses: Float[Array, 'probe *batch n k'], n_colors: int, k: int, half_width: int
) -> Float[Array, '*batch n w1 k k']:
    """The core's bands: `bands[..., i, d, a, b]` couples `(i, a)` to `(i + d, b)`."""
    n = responses.shape[-2]
    batch = responses.shape[1:-2]
    y = responses.reshape(n_colors, k, -1, n, k)  # (color, b, batch, i, a)
    y = y.transpose(2, 3, 0, 4, 1)  # (batch, i, color, a, b)
    i, d = jnp.arange(n)[:, None], jnp.arange(min(half_width, n - 1) + 1)[None, :]
    # block `i + d` has color `(i + d) % n_colors`, the only one of that color within reach
    bands = jnp.where((i + d < n)[..., None, None], y[:, i, (i + d) % n_colors], 0)
    # the diagonal blocks are symmetric, up to the rounding of the responses
    diagonal = bands[:, :, 0]
    bands = bands.at[:, :, 0].set((diagonal + jnp.swapaxes(diagonal, -1, -2)) / 2)
    return bands.reshape(*batch, *bands.shape[1:])
