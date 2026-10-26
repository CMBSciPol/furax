"""The pseudo-inverse of symmetric positive semi-definite dense matrices."""

from typing import Self

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, symmetric, tree

from .cholesky import _einsum

__all__ = ['DensePseudoInverseOperator']


@symmetric
class DensePseudoInverseOperator(AbstractLinearOperator):
    r"""Pseudo-inverse of a batch of symmetric positive semi-definite dense matrices.

    The matrix $M$ is normalized as $D^{-1/2} M D^{-1/2}$, with $D$ the diagonal `scale`, and
    inverted on the eigenvectors of the normalized matrix whose eigenvalue exceeds `rtol`; the
    others are left out. Without a scale, $D$ is the largest eigenvalue, as for
    `numpy.linalg.pinv`.

    A scale is the reference against which an eigenvalue is negligible. When $M = A - B$ is formed
    by subtracting two matrices that cancel on some directions, those come out at the rounding
    error of $A$ and $B$, possibly amplified by how $B$ was computed: normalized by the diagonal
    of $A$, a direction is left out when less than `rtol` of its weight in $A$ remains, however
    small the largest eigenvalue of $M$.

    With `clamp`, the directions below `rtol` are inverted at `rtol` instead of left out, which
    makes a positive definite approximate inverse, such as a preconditioner, of a matrix only
    positive semi-definite up to rounding.

    Calling `op(b)` returns $M^+ b$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax.linalg import DensePseudoInverseOperator
        >>> M = jnp.array([[1., 1.], [1., 1.]])  # singular: the direction [1, -1] is left out
        >>> op = DensePseudoInverseOperator.from_dense(M)
        >>> op(jnp.array([2., 0.]))
        Array([0.5, 0.5], dtype=float32)
    """

    eigenvectors: Float[Array, '*batch k k']
    """The eigenvectors of the normalized matrix, scaled back by $D^{-1/2}$."""
    inverse_eigenvalues: Float[Array, '*batch k']
    """The inverse eigenvalues of the normalized matrix, zero for those left out."""

    @classmethod
    def from_dense(
        cls,
        matrix: Float[Array, '*batch k k'],
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
        *,
        scale: Float[Array, '*batch k'] | None = None,
        rtol: float | None = None,
        clamp: bool = False,
    ) -> Self:
        """Decompose a dense matrix.

        Args:
            matrix: The symmetric matrix, shape `(*batch, k, k)`.
            in_structure: Structure of the values the operator is called with.
                Defaults to a plain array of shape `(*batch, k)`.
                Pass a PyTree explicitly to apply it to several arrays at once instead.
            scale: The positive diagonal normalizing the matrix; by default, its largest
                eigenvalue.
            rtol: The normalized eigenvalue below which a direction is left out; by default,
                `k` times the machine epsilon of the matrix dtype.
            clamp: Whether to invert the directions below the cutoff at the cutoff, rather than
                leave them out.
        """
        if in_structure is None:
            in_structure = jax.ShapeDtypeStruct(matrix.shape[:-1], matrix.dtype)
        if rtol is None:
            rtol = matrix.shape[-1] * float(jnp.finfo(matrix.dtype).eps)
        if scale is None:
            eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
            cutoff = rtol * jnp.max(jnp.abs(eigenvalues), axis=-1, keepdims=True)
        else:
            root = 1 / jnp.sqrt(scale)
            eigenvalues, eigenvectors = jnp.linalg.eigh(
                root[..., :, None] * matrix * root[..., None, :]
            )
            eigenvectors = root[..., :, None] * eigenvectors
            cutoff = rtol
        if clamp:
            inverse = 1 / jnp.maximum(eigenvalues, cutoff)
        else:
            kept = eigenvalues > cutoff
            inverse = jnp.where(kept, 1 / jnp.where(kept, eigenvalues, 1), 0)
        return cls(eigenvectors, inverse, in_structure=in_structure)

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        def apply(xf):
            projected = _einsum('...ji,...j->...i', self.eigenvectors, xf)
            return _einsum(
                '...ij,...j->...i', self.eigenvectors, self.inverse_eigenvalues * projected
            )

        batch_ndim = self.eigenvectors.ndim - 2
        return tree.unravel(x, apply(tree.ravel(x, batch_ndim)), batch_ndim)

    def as_matrix(self) -> Float[Array, 'a a']:
        blocks = _einsum(
            '...ik,...k,...jk->...ij',
            self.eigenvectors,
            self.inverse_eigenvalues,
            self.eigenvectors,
        )
        # where each entry of a batch element's vector lies among all the entries
        size = self.in_size
        positions = tree.unravel(self.in_structure, jnp.arange(size))
        index = tree.ravel(positions, self.eigenvectors.ndim - 2)
        matrix = jnp.zeros((size, size), blocks.dtype)
        return matrix.at[index[..., :, None], index[..., None, :]].set(blocks)
