from typing import NamedTuple

import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num, PRNGKeyArray, PyTree

from furax.core import AbstractLinearOperator
from furax.core._base import symmetric
from furax.tree_block import (
    apply_operator_block,
    batched_dot,
    block_normal_like,
    qr_pytree,
)


class NystromResult(NamedTuple):
    """Result of randomized Nyström approximation.

    Attributes:
        eigenvalues: The k approximate eigenvalues.
        eigenvectors: Block PyTree containing k eigenvectors.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, 'k ...']]


def randomized_nystrom(
    A: AbstractLinearOperator,
    k: int,
    *,
    oversampling: int = 10,
    key: PRNGKeyArray,
) -> NystromResult:
    """Compute a low-rank approximation using randomized Nyström.

    Args:
        A: A symmetric positive semidefinite linear operator.
        k: Target rank for the approximation.
        oversampling: Extra columns for better accuracy (default: 10).
        key: Random key for generating test matrix.

    Returns:
        NystromResult with eigenvalues and eigenvectors.
    """
    total_k = k + oversampling

    # Generate random Gaussian test matrix as block PyTree
    Omega = block_normal_like(A.in_structure(), total_k, key)

    # Orthogonalize for numerical stability
    Q, _ = qr_pytree(Omega)

    # Sketch the matrix: Y = A @ Q
    Y = apply_operator_block(A, Q)

    # Compute Q^T Y for Cholesky factorization
    QtY = batched_dot(Q, Y)  # (total_k, total_k)

    # Shift for numerical stability
    nu = jnp.spacing(jnp.linalg.norm(QtY, ord='fro'))
    QtY_shifted = QtY + nu * jnp.eye(total_k)

    # Cholesky factorization: C such that C^T C = Q^T Y
    C = jnp.linalg.cholesky(QtY_shifted)  # lower triangular

    # Solve C @ B^T = Y^T for B, i.e., B = Y @ C^{-T}
    # First flatten Y to matrix form, solve, then unflatten
    Y_leaves, treedef = jax.tree.flatten(Y)
    Y_flat = jnp.concatenate(
        [leaf.reshape(total_k, -1) for leaf in Y_leaves], axis=1
    )  # (total_k, n)
    B_flat = jnp.linalg.solve(C, Y_flat)  # (total_k, n)

    # SVD of B^T: B^T = U @ S @ V^T
    U_flat, S, Vt = jnp.linalg.svd(B_flat.T, full_matrices=False)

    # Truncate to rank k
    U_flat = U_flat[:, :k]  # (n, k)
    S = S[:k]  # (k,)

    # Eigenvalues are S^2 (with shift removed)
    eigenvalues = jnp.maximum(0, S**2 - nu)

    # Convert U back to block PyTree format
    # U_flat is (n, k), we need (k, ...) for each leaf
    eigenvectors_leaves = []
    start = 0
    for leaf in Y_leaves:
        leaf_size = leaf[0].size  # size of single vector in this leaf
        end = start + leaf_size
        # U_flat[:, :k] -> transpose to (k, n) then reshape
        U_leaf = U_flat[start:end, :].T.reshape((k,) + leaf.shape[1:])
        eigenvectors_leaves.append(U_leaf)
        start = end

    eigenvectors = treedef.unflatten(eigenvectors_leaves)

    return NystromResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


@symmetric
class NystromPreconditioner(AbstractLinearOperator):
    """Nyström preconditioner: M^{-1} = λ_k * U Λ^{-1} U^T + (I - U U^T)

    This approximates the inverse of A using the Nyström low-rank approximation.
    """

    nystrom: NystromResult
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        nystrom: NystromResult,
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
    ):
        self.nystrom = nystrom
        if in_structure is None:
            self._in_structure = jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype),
                nystrom.eigenvectors,
            )
        else:
            self._in_structure = in_structure

    def mv(self, x: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
        S = self.nystrom.eigenvalues
        U = self.nystrom.eigenvectors
        lambda_k = S[-1]  # smallest eigenvalue (sorted descending from SVD)

        # Compute coeffs = U^T @ x
        def leaf_dots(u_leaf: Array, x_leaf: Array) -> Array:
            k = u_leaf.shape[0]
            u_flat = u_leaf.reshape(k, -1)
            x_flat = x_leaf.ravel()
            return u_flat @ jnp.conj(x_flat)

        leaf_dots_list = jax.tree.leaves(jax.tree.map(leaf_dots, U, x))
        coeffs = leaf_dots_list[0]
        for leaf in leaf_dots_list[1:]:
            coeffs = coeffs + leaf

        # Compute: λ_k * (Λ^{-1} @ coeffs) + (coeffs - coeffs)
        # = λ_k * (1/S) * coeffs for the low-rank part
        # Plus (I - U U^T) @ x = x - U @ coeffs

        # First term: λ_k * U @ diag(1/S) @ U^T @ x
        scaled_coeffs_inv = lambda_k * coeffs / S

        # Second term: (I - U U^T) @ x = x - U @ coeffs
        # Combined: λ_k * U @ (1/S) @ coeffs + x - U @ coeffs
        #         = x + U @ (λ_k/S - 1) @ coeffs
        combined_coeffs = scaled_coeffs_inv - coeffs  # (λ_k/S - 1) * coeffs

        def compute_result(u_leaf: Array, x_leaf: Array) -> Array:
            correction = jnp.tensordot(combined_coeffs, u_leaf, axes=(0, 0))
            return x_leaf + correction

        return jax.tree.map(compute_result, U, x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
