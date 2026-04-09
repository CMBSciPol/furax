from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num, PRNGKeyArray, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.core._base import symmetric
from furax.tree_block import block_from_array, block_normal_like, block_to_array, gram, qr


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
    rank: int,
    *,
    oversampling: int = 10,
    key: PRNGKeyArray,
) -> NystromResult:
    """Compute a low-rank approximation using randomized Nyström.

    Args:
        A: A symmetric positive semidefinite linear operator.
        rank: Target rank for the approximation.
        oversampling: Extra columns for better accuracy (default: 10).
        key: Random key for generating test matrix.

    Returns:
        NystromResult with eigenvalues and eigenvectors.
    """
    size = sum(leaf.size for leaf in jax.tree.leaves(A.in_structure))
    k = min(rank + oversampling, size)

    # Generate random Gaussian test matrix as block PyTree
    Omega = block_normal_like(A.in_structure, k, key)

    # Orthogonalize for numerical stability
    Q, _ = qr(Omega)

    # Sketch the matrix: Y = A @ Q
    Y = jax.vmap(A.mv)(Q)

    # Compute Q^T Y for Cholesky factorization
    QtY = gram(Q, Y)  # (k, k)

    # Shift for numerical stability
    nu = jnp.spacing(jnp.linalg.norm(QtY, ord='fro'))
    QtY_shifted = QtY + nu * jnp.eye(k)

    # Cholesky factorization: C such that C^T C = Q^T Y
    C = jnp.linalg.cholesky(QtY_shifted)  # lower triangular

    # Solve C @ B^T = Y^T for B, i.e., B = Y @ C^{-T}
    Y_flat, treedef, shapes = block_to_array(Y)  # (k, n)
    B_flat = jnp.linalg.solve(C, Y_flat)  # (k, n)

    # SVD of B^T: B^T = U @ S @ V^T
    U_flat, S, Vt = jnp.linalg.svd(B_flat.T, full_matrices=False)

    # Truncate
    # U_flat is (n, k), transpose to (rank, n) for block_from_array
    S = S[:rank]  # (rank,)
    eigenvalues = jnp.maximum(0, S**2 - nu)
    eigenvectors = block_from_array(U_flat[:, :rank].T, treedef, shapes)

    return NystromResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


@symmetric
class NystromPreconditioner(AbstractLinearOperator):
    """Nyström preconditioner: M^{-1} = λ_k * U Λ^{-1} U^T + (I - U U^T)

    This approximates the inverse of A using the Nyström low-rank approximation.
    Reference: https://arxiv.org/pdf/2110.02820
    """

    nystrom: NystromResult

    def __init__(
        self,
        nystrom: NystromResult,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
    ):
        if in_structure is None:
            # Infer structure from eigenvectors by removing leading dimension
            in_structure = jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype),
                nystrom.eigenvectors,
            )
        object.__setattr__(self, 'nystrom', nystrom)
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
        S = self.nystrom.eigenvalues
        U = self.nystrom.eigenvectors
        lambda_k = S[-1]  # smallest eigenvalue (sorted descending from SVD)

        # M^{-1} x = x + U (λ_k Λ^{-1} - I) U^T x
        UTx = jax.vmap(lambda u: tree.dot(x, u))(U)  # (k,)
        scale = lambda_k / S - 1  # (k,)
        correction = jax.tree.map(lambda u: jnp.einsum('k,k...->...', scale * UTx, u), U)
        return tree.add(x, correction)
