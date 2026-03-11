"""LOBPCG eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.tree_block import (
    block_norm,
    gram,
    orthonormalize,
    vecmat,
)


class LOBPCGResult(NamedTuple):
    """Result of LOBPCG eigenvalue computation.

    Attributes:
        eigenvalues: The k computed eigenvalues, sorted ascending if largest=False.
        eigenvectors: A block PyTree containing k eigenvectors.
        iterations: The number of iterations performed.
        converged: A boolean array indicating which eigenpairs have converged.
        residual_norms: The norm of the residual for each eigenpair.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, ' k ...']]
    iterations: int
    converged: Bool[Array, ' k']
    residual_norms: Float[Array, ' k']


def _compute_residuals(
    AX: PyTree[Num[Array, 'k ...']],
    X: PyTree[Num[Array, 'k ...']],
    eigenvalues: Float[Array, ' k'],
) -> PyTree[Num[Array, 'k ...']]:
    """Compute residuals R = AX - eigenvalues * X."""
    # double transpose for correct broadcasting
    scaled_eigenvectors = jax.tree.map(lambda x: (x.T * eigenvalues).T, X)
    return tree.sub(AX, scaled_eigenvectors)


def _check_convergence(
    R: PyTree[Num[Array, 'k ...']], tol: float
) -> tuple[Float[Array, ' k'], Bool[Array, ' k']]:
    """Compute residual norms and check convergence."""
    residual_norms = block_norm(R)
    converged = residual_norms < tol
    return residual_norms, converged


def _rayleigh_ritz(
    S: PyTree[Num[Array, 'm ...']], AS: PyTree[Num[Array, 'm ...']], k: int, largest: bool = False
) -> tuple[Float[Array, ' k'], PyTree[Num[Array, ' k ...']], PyTree[Num[Array, ' k ...']]]:
    """Perform Rayleigh-Ritz procedure to extract k Ritz pairs."""
    G = gram(S, AS)
    G = (G + jnp.conj(G.T)) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(G)

    if largest:
        idx = jnp.arange(eigenvalues.shape[0] - k, eigenvalues.shape[0])
    else:
        idx = jnp.arange(k)

    selected_eigenvalues = eigenvalues[idx]
    selected_eigenvectors = eigenvectors[:, idx]
    ritz_vectors = vecmat(S, selected_eigenvectors)
    ritz_vectors_A = vecmat(AS, selected_eigenvectors)

    return selected_eigenvalues, ritz_vectors, ritz_vectors_A


def lobpcg_standard(
    A: AbstractLinearOperator,
    X: PyTree[Num[Array, 'k ...']],
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
    largest: bool = False,
    preconditioner: AbstractLinearOperator | None = None,
) -> LOBPCGResult:
    """LOBPCG for standard eigenvalue problem A x = lambda x.

    Computes the k smallest (or largest) eigenvalues and corresponding eigenvectors
    of a Hermitian linear operator A using the Locally Optimal Block Preconditioned
    Conjugate Gradient method.

    Args:
        A: A Hermitian linear operator.
        X: Initial guess as a block PyTree with k vectors (leading dimension k).
        maxiter: Maximum iteration count (size of maximal Krylov subspace).
        tol: Convergence tolerance for residual norms.
        largest: If True, compute largest eigenvalues; otherwise smallest.
        preconditioner: Optional preconditioner operator M approximating A^{-1}.
            When provided, the search direction is M R instead of R, where R
            is the residual. This can significantly accelerate convergence.

    Returns:
        LOBPCGResult containing eigenvalues, eigenvectors, iteration count,
        convergence status, and residual norms.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure, block_normal_like
        >>> # Create operator with known eigenvalues [1, 2, 3, 4, 5]
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(0))
        >>> result = lobpcg_standard(A, X)
        >>> result.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
    """
    leaves = jax.tree.leaves(X)
    k = leaves[0].shape[0]
    # batched version of operator A to operator on blocks
    A = jax.vmap(A.mv)  # type: ignore[assignment]
    M = jax.vmap(preconditioner.mv) if preconditioner is not None else None

    # Step 1: Orthonormalize initial vectors
    X = orthonormalize(X)

    # Step 2: Compute AX and initial Rayleigh quotient
    AX = A(X)

    # Initial eigenvalue estimates from Rayleigh quotient
    eigenvalues, X, AX = _rayleigh_ritz(X, AX, k, largest)

    # First iteration (no P): search direction R (preconditioned if M is given)
    R = _compute_residuals(AX, X, eigenvalues)
    residual_norms, converged = _check_convergence(R, tol)
    if M is not None:
        R = M(R)

    S = orthonormalize(tree.concatenate([X, R]))
    eigenvalues, X, AX = _rayleigh_ritz(S, A(S), k, largest)
    P = R
    iteration = 1

    def cond(carry):  # type: ignore[no-untyped-def]
        _, _, _, _, iteration, converged, _ = carry
        return jnp.logical_and(iteration < maxiter, ~jnp.all(converged))

    # Continue with while loop for remaining iterations
    def bond(carry):  # type: ignore[no-untyped-def]
        X, AX, P, eigenvalues, iteration, converged, residual_norms = carry

        # Compute residuals and search direction (preconditioned if M is given)
        R = _compute_residuals(AX, X, eigenvalues)
        residual_norms, converged = _check_convergence(R, tol)
        if M is not None:
            R = M(R)

        # Build search space S = [X, R, P] and orthonormalize
        S = orthonormalize(tree.concatenate([X, R, P]))
        AS = A(S)

        # Rayleigh-Ritz
        new_eigenvalues, new_X, new_AX = _rayleigh_ritz(S, AS, k, largest)

        return (
            new_X,
            new_AX,
            R,
            new_eigenvalues,
            iteration + 1,
            converged,
            residual_norms,
        )

    carry = (X, AX, P, eigenvalues, iteration, converged, residual_norms)

    final_carry = jax.lax.while_loop(cond, bond, carry)
    X, AX, P, eigenvalues, iteration, converged, residual_norms = final_carry

    # Final residual computation
    R = _compute_residuals(AX, X, eigenvalues)
    residual_norms, converged = _check_convergence(R, tol)

    return LOBPCGResult(
        eigenvalues=eigenvalues,
        eigenvectors=X,
        iterations=iteration,
        converged=converged,
        residual_norms=residual_norms,
    )
