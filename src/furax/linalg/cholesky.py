"""Block-banded Cholesky factorization for symmetric positive-(semi)definite matrices."""

from functools import partial
from math import prod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, symmetric

__all__ = [
    'BandedCholeskyOperator',
    'BorderedBandedCholeskyOperator',
    'banded_cholesky',
    'banded_cholesky_solve',
]

# Full float32 precision for the block products to avoid TF32 rounding
_matmul = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


@symmetric
class BandedCholeskyOperator(AbstractLinearOperator):
    """Inverse of a symmetric positive-(semi)definite matrix, by Cholesky solve.

    This operator can be used with an ordinary dense matrix ([`from_dense`][]), or from
    a banded matrix (zero away from the diagonal, [`from_bands`][]) and so is stored
    compactly instead of as the full matrix.

    Calling ``A(b)`` solves ``A x = b``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.linalg import BandedCholeskyOperator
        >>> A = jnp.array([[4., 2.], [2., 3.]])
        >>> op = BandedCholeskyOperator.from_dense(A)
        >>> op(jnp.array([1., 0.]))  # solves A x = [1, 0]
        Array([ 0.375, -0.25 ], dtype=float32)
    """

    lb: Float[Array, '*batch n w1 k k']
    """The lower-triangular Cholesky factor in compact band-layout."""

    @classmethod
    def from_bands(
        cls,
        bands: Float[Array, '*batch n w1 k k'],
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
        regularization: float = 0.0,
    ) -> Self:
        """Factor a block-banded matrix (see [`banded_cholesky`][]) and wrap it as an operator.

        Args:
            bands: Upper-band representation of the block-banded matrix, shape
                ``(*batch, n_blocks, w+1, k, k)``.
            in_structure: Structure of the values the operator is called with.
                Defaults to a plain array of shape ``(*batch, n_blocks, k)``.
                Pass a PyTree explicitly to solve for several arrays at once instead.
            regularization: Relative ridge added to each diagonal block before factoring.
        """
        if in_structure is None:
            n, k = bands.shape[-4], bands.shape[-1]
            in_structure = jax.ShapeDtypeStruct((*bands.shape[:-4], n, k), bands.dtype)
        return cls(banded_cholesky(bands, regularization), in_structure=in_structure)

    @classmethod
    def from_dense(
        cls,
        matrix: Float[Array, '*batch k k'],
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
        regularization: float = 0.0,
    ) -> Self:
        """Factor a fully dense matrix (the degenerate ``n_blocks=1, w=0`` band case).

        Args:
            matrix: The dense matrix, shape ``(*batch, k, k)``.
            in_structure: Structure of the values the operator is called with.
                Defaults to a plain array of shape ``(*batch, k)``.
                Pass a PyTree explicitly to solve for several arrays at once instead.
            regularization: Relative ridge added to the diagonal before factoring.
        """
        if in_structure is None:
            in_structure = jax.ShapeDtypeStruct(matrix.shape[:-1], matrix.dtype)
        return cls.from_bands(matrix[..., None, None, :, :], in_structure, regularization)

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        batch_ndim = self.lb.ndim - 4
        n_blocks, k = self.lb.shape[-4], self.lb.shape[-1]
        leaves, treedef = jax.tree.flatten(x)
        # Flatten every leaf's non-batch axes and concatenate them into one length-K vector per
        # batch element, matching how `lb` was built from `bands`/`matrix` (K = n_blocks * k).
        flats = [leaf.reshape(*leaf.shape[:batch_ndim], -1) for leaf in leaves]
        xf = jnp.concatenate(flats, axis=-1)  # (*batch, K)
        xb = xf.reshape(*xf.shape[:-1], n_blocks, k)  # split K back into the (n_blocks, k) layout
        yb = banded_cholesky_solve(self.lb, xb)
        yf = yb.reshape(*yb.shape[:-2], -1)  # (*batch, K)
        # Undo the flatten/concatenate: split the solution back into per-leaf chunks (in the same
        # order they were concatenated) and reshape each to its original leaf shape.
        sizes = [prod(leaf.shape[batch_ndim:]) for leaf in leaves]
        chunks = jnp.split(yf, np.cumsum(sizes)[:-1], axis=-1)
        out = [c.reshape(leaf.shape) for c, leaf in zip(chunks, leaves, strict=True)]
        return treedef.unflatten(out)


@symmetric
class BorderedBandedCholeskyOperator(AbstractLinearOperator):
    r"""Inverse of a symmetric positive-definite matrix with a block-banded core and a dense border.

    The matrix is split into a block-banded part $A$ over the first unknowns $x_a$ and a small
    dense part $C$ over the last ones $x_c$, coupled by a dense border $B$:

    $$ M = \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix}. $$

    Storing it whole would lose the band. Instead, $x_c$ is eliminated through the Schur
    complement $S = C - B^\top A^{-1} B$: with $Z = A^{-1} B$, solving $M x = r$ is

    $$ x_c = S^{-1} (r_c - Z^\top r_a), \qquad x_a = A^{-1} r_a - Z x_c, $$

    which keeps only the band factor of $A$, $Z$ and the factor of $S$.

    Calling `op((r_a, r_c))` solves $M x = r$ and returns `(x_a, x_c)`, with `r_a` of shape
    `(*batch, n_blocks, k)` and `r_c` of shape `(*batch, k_c)`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.linalg import BorderedBandedCholeskyOperator
        >>> bands = jnp.array([[[[4.]], [[1.]]], [[[4.]], [[0.]]]])  # A = [[4, 1], [1, 4]]
        >>> border = jnp.array([[1.], [1.]])  # B
        >>> corner = jnp.array([[3.]])  # C
        >>> op = BorderedBandedCholeskyOperator.from_blocks(bands, border, corner)
        >>> x_a, x_c = op((jnp.array([[6.], [6.]]), jnp.array([5.])))  # M x = r for x = 1
        >>> x_a.ravel(), x_c
        (Array([1., 1.], dtype=float32), Array([1.], dtype=float32))
    """

    banded: BandedCholeskyOperator
    """$A^{-1}$, on `(*batch, n_blocks, k)`."""
    coupling: Float[Array, '*batch ka kc']
    """$Z = A^{-1} B$, with `ka = n_blocks * k`."""
    schur: BandedCholeskyOperator
    """$S^{-1}$, on `(*batch, k_c)`."""

    @classmethod
    def from_blocks(
        cls,
        bands: Float[Array, '*batch n w1 k k'],
        border: Float[Array, '*batch ka kc'],
        corner: Float[Array, '*batch kc kc'],
        regularization: float = 0.0,
    ) -> Self:
        """Factor the matrix from its banded core, border and corner.

        Args:
            bands: Upper-band representation of $A$ (see [`banded_cholesky`][]), shape
                `(*batch, n_blocks, w+1, k, k)`.
            border: $B$, shape `(*batch, n_blocks * k, k_c)`, rows in the flattened
                `(n_blocks, k)` order.
            corner: $C$, shape `(*batch, k_c, k_c)`.
            regularization: Relative ridge added to each diagonal block of $A$ and to $C$,
                each scaled by its own mean diagonal, before factoring.
        """
        banded = BandedCholeskyOperator.from_bands(bands, regularization=regularization)
        n_blocks, k, k_c = bands.shape[-4], bands.shape[-1], border.shape[-1]
        batch = border.shape[:-2]

        # Z = A⁻¹ B: one banded solve per column of B
        columns = jnp.moveaxis(border.reshape(*batch, n_blocks, k, k_c), -1, -3)  # (.., kc, n, k)
        coupling = banded_cholesky_solve(banded.lb[..., None, :, :, :, :], columns)
        coupling = jnp.moveaxis(coupling, -3, -1).reshape(*batch, n_blocks * k, k_c)

        if regularization:
            scale = jnp.mean(jnp.diagonal(corner, axis1=-2, axis2=-1), axis=-1)
            corner = corner + regularization * scale[..., None, None] * jnp.eye(k_c)
        schur = corner - _einsum('...ac,...ab->...cb', border, coupling)
        schur = 0.5 * (schur + jnp.swapaxes(schur, -1, -2))  # symmetric up to rounding

        in_structure = (
            jax.ShapeDtypeStruct((*batch, n_blocks, k), bands.dtype),
            jax.ShapeDtypeStruct((*batch, k_c), bands.dtype),
        )
        return cls(
            banded,
            coupling,
            BandedCholeskyOperator.from_dense(schur),
            in_structure=in_structure,
        )

    def mv(self, x: tuple[Array, Array]) -> tuple[Array, Array]:
        r_a, r_c = x
        flat_a = r_a.reshape(*r_c.shape[:-1], -1)
        x_c = self.schur(r_c - _einsum('...ac,...a->...c', self.coupling, flat_a))
        x_a = self.banded(r_a) - _einsum('...ac,...c->...a', self.coupling, x_c).reshape(r_a.shape)
        return x_a, x_c


def banded_cholesky(
    bands: Float[Array, '*batch n w1 k k'], regularization: float = 0.0
) -> Float[Array, '*batch n w1 k k']:
    """Block Cholesky factor of a symmetric positive-(semi)definite block-banded matrix.

    "Block-banded" means ``A`` is made of ``k×k`` blocks, and only the blocks within ``w``
    of the diagonal can be non-zero. E.g., for ``n_blocks = 4`` and bandwidth ``w = 1``:

        A = [ A00  A01   0    0  ]
            [ A01ᵀ A11  A12   0  ]
            [  0   A12ᵀ A22  A23 ]
            [  0    0   A23ᵀ A33 ]

    Only the diagonal and upper blocks are stored (symmetry gives the rest) packed as
    ``bands[..., j, d, :, :] = A[j, j+d]`` for ``d = 0..w`` (``d=0`` the diagonal blocks,
    ``d=w`` the furthest off-diagonal ones still inside the band)::

        bands[:, 0] = [A00, A11, A22, A33]  (d=0: the diagonal blocks)
        bands[:, 1] = [A01, A12, A23,  · ]  (d=1: one block off the diagonal; the trailing
                                               entry is unused padding, since block 3 has no
                                               block 4 to pair with)

    Returns the lower factor in the same band layout, ``lb[..., i, d, :, :] = L[i, i-d]`` with
    ``L Lᵀ = A``. Batched over arbitrary leading dims. Band ``w = 0`` is the block-diagonal case
    (every off-diagonal block is zero); ``n_blocks = 1`` is the fully dense case.

    ``regularization`` adds a relative ridge to each diagonal block before factoring, scaled by
    that block's own mean diagonal. This is purely intended as a numerical safeguard.

    Args:
        bands: Upper-band representation of the block-banded matrix, shape
            ``(*batch, n_blocks, w+1, k, k)``.
        regularization: Relative ridge added to each diagonal block before factoring.

    Returns:
        The lower band factor, same shape as ``bands``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.linalg import banded_cholesky
        >>> # tridiagonal SPD A = [[4,1,0,0],[1,4,1,0],[0,1,4,1],[0,0,1,4]], scalar (k=1) blocks
        >>> diag = jnp.array([4., 4., 4., 4.]).reshape(4, 1, 1, 1)
        >>> off = jnp.array([1., 1., 1., 0.]).reshape(4, 1, 1, 1)  # trailing entry is padding
        >>> bands = jnp.concatenate([diag, off], axis=1)  # (n_blocks=4, w1=2, k=1, k=1)
        >>> lb = banded_cholesky(bands)
        >>> lb[:, :, 0, 0]  # columns are (L[i,i], L[i,i-1])
        Array([[2.        , 0.        ],
               [1.9364916 , 0.5       ],
               [1.9321835 , 0.5163978 ],
               [1.9318755 , 0.51754916]], dtype=float32)
    """
    if regularization:
        k = bands.shape[-1]
        diag = jnp.diagonal(bands[..., 0, :, :], axis1=-2, axis2=-1)  # (*batch, n, k)
        scale = jnp.mean(diag, axis=-1)[..., None, None]  # (*batch, n, 1, 1)
        ridge = regularization * scale * jnp.eye(k, dtype=bands.dtype)
        bands = bands.at[..., 0, :, :].add(ridge)
    if bands.shape[-3] == 1:  # block diagonal: the blocks are independent, factor them all at once
        return jnp.linalg.cholesky(bands)
    return _block_banded_cholesky(bands)


@partial(jnp.vectorize, signature='(n,w1,k,k)->(n,w1,k,k)')
def _block_banded_cholesky(bands: Float[Array, 'n w1 k k']) -> Float[Array, 'n w1 k k']:
    """Block Cholesky of a symmetric positive-definite block-banded matrix.

    Batched over arbitrary leading dims via the ``jnp.vectorize`` signature.

    ``bands[j, d, :, :]`` is the upper block ``A[j, j+d]`` (``d = 0..w``, ``w = w1-1`` the block
    bandwidth; ``d=0`` the symmetric diagonal block). Returns the lower factor in the same band
    layout, ``lb[i, d, :, :] = L[i, i-d]`` with ``L Lᵀ = A``.

    Band ``w = 0`` degenerates to an independent Cholesky of each diagonal ``k×k`` block (the
    block-diagonal case); ``n_blocks = 1`` further degenerates to a single plain Cholesky.
    """
    n, w1, k, _ = bands.shape
    w = w1 - 1

    def read(arr: Array, idx: Array):
        return jax.lax.dynamic_index_in_dim(arr, jnp.clip(idx, 0, n - 1), axis=0, keepdims=False)

    def row(i: Array, lb: Array):
        cur = jnp.zeros((w1, k, k), bands.dtype)  # this row's blocks lb[i, :]
        # columns j = i - d0, processed high d0 (far) to low (diagonal last).
        for d0 in range(w, -1, -1):
            j = i - d0
            s = jnp.swapaxes(read(bands, j)[d0], -1, -2)  # A[i, j] = A[j, i]ᵀ = bands[j, d0]ᵀ
            for a in range(d0 + 1, w + 1):  # Schur update: − Σ L[i, i-a] L[j, i-a]ᵀ
                # L[j, i-a]: for the diagonal block (d0=0, j=i) it is this row's block ``cur[a]``,
                # not yet written to ``lb``; for d0>0 (j<i) it is a finished earlier row.
                l_ja = cur[a] if d0 == 0 else read(lb, j)[a - d0]
                s = s - _matmul(cur[a], jnp.swapaxes(l_ja, -1, -2))
            if d0 == 0:
                cur = cur.at[0].set(jnp.linalg.cholesky(s))  # L[i,i] = chol(S)
            else:
                ljj = read(lb, j)[0]  # L[j,j] (lower), computed in an earlier row
                # j < 0 aliases into an unwritten (zero) pivot: fine forward (masked below), but
                # 0/0 gives nan gradient even through the masked branch, so swap in the identity.
                ljj = jnp.where(j >= 0, ljj, jnp.eye(k, dtype=bands.dtype))
                x = jax.scipy.linalg.solve_triangular(ljj, s.T, lower=True).T  # X L[j,j]ᵀ = S
                cur = cur.at[d0].set(jnp.where(j >= 0, x, 0.0))
        return jax.lax.dynamic_update_index_in_dim(lb, cur, i, axis=0)

    factor: Array = jax.lax.fori_loop(0, n, row, jnp.zeros((n, w1, k, k), bands.dtype))
    return factor


@partial(jnp.vectorize, signature='(n,w1,k,k),(n,k)->(n,k)')
def banded_cholesky_solve(
    lb: Float[Array, '*batch n w1 k k'], b: Float[Array, '*batch n k']
) -> Float[Array, '*batch n k']:
    """Solve ``L Lᵀ x = b`` given the lower band factor ``lb`` from [`banded_cholesky`][].

    Batched over arbitrary leading dims (matching ``lb``'s, excluding its trailing
    ``(n, w1, k, k)`` core).

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.linalg import banded_cholesky, banded_cholesky_solve
        >>> a = jnp.array([[4., 2.], [2., 3.]])
        >>> lb = banded_cholesky(a[None, None, None])
        >>> x = banded_cholesky_solve(lb, jnp.array([[[1., 0.]]]))
        >>> a @ x[0, 0]
        Array([1., 0.], dtype=float32)
    """
    n, w1, k, _ = lb.shape
    w = w1 - 1
    if w == 0:  # block diagonal: the blocks are independent, solve them all at once
        diagonal, rhs = lb[:, 0], b[..., None]
        y = jax.lax.linalg.triangular_solve(diagonal, rhs, left_side=True, lower=True)
        x = jax.lax.linalg.triangular_solve(
            diagonal, y, left_side=True, lower=True, transpose_a=True
        )
        return x[..., 0]

    def read(arr: Array, idx: Array):
        return jax.lax.dynamic_index_in_dim(arr, jnp.clip(idx, 0, n - 1), axis=0, keepdims=False)

    def fwd(i: Array, y: Array):  # forward: L y = b
        rhs = read(b, i)
        lb_i = read(lb, i)
        for d in range(1, w + 1):
            rhs = rhs - jnp.where(i - d >= 0, _matmul(lb_i[d], read(y, i - d)), 0.0)
        yi = jax.scipy.linalg.solve_triangular(lb_i[0], rhs, lower=True)
        return jax.lax.dynamic_update_index_in_dim(y, yi, i, axis=0)

    y = jax.lax.fori_loop(0, n, fwd, jnp.zeros((n, k), b.dtype))

    def bwd(step: Array, x: Array):  # backward: Lᵀ x = y
        i = n - 1 - step
        rhs = read(y, i)
        for d in range(1, w + 1):  # L[i+d, i] = lb[i+d, d]
            rhs = rhs - jnp.where(
                i + d <= n - 1,
                _matmul(jnp.swapaxes(read(lb, i + d)[d], -1, -2), read(x, i + d)),
                0.0,
            )
        xi = jax.scipy.linalg.solve_triangular(read(lb, i)[0].T, rhs, lower=False)
        return jax.lax.dynamic_update_index_in_dim(x, xi, i, axis=0)

    solution: Array = jax.lax.fori_loop(0, n, bwd, jnp.zeros((n, k), b.dtype))
    return solution
