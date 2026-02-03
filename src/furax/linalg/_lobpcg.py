"""LOBPCG eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Num, PRNGKeyArray, PyTree

from furax.core import AbstractLinearOperator
from furax.tree_block import (
    apply_operator_block,
    apply_rotation,
    batched_dot,
    block_normal_like,
    block_norms,
    orthonormalize,
)


def _rayleigh_ritz(
    S: PyTree[Num[Array, 'm ...']], AS: PyTree[Num[Array, 'm ...']], k: int, largest: bool = False
) -> tuple[Float[Array, ' k'], PyTree[Num[Array, ' k ...']]]:
    """Perform Rayleigh-Ritz procedure to extract k Ritz pairs."""
    G = batched_dot(S, AS)
    G = (G + jnp.conj(G.T)) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(G)

    if largest:
        idx = jnp.arange(eigenvalues.shape[0] - k, eigenvalues.shape[0])
    else:
        idx = jnp.arange(k)

    selected_eigenvalues = eigenvalues[idx]
    selected_eigenvectors = eigenvectors[:, idx]
    ritz_vectors = apply_rotation(S, selected_eigenvectors)

    return selected_eigenvalues, ritz_vectors


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


def lobpcg_standard(
    A: AbstractLinearOperator,
    X: PyTree[Num[Array, 'k ...']] | None = None,
    *,
    k: int | None = None,
    max_iters: int = 100,
    tol: float = 1e-6,
    largest: bool = False,
    key: PRNGKeyArray | None = None,
) -> LOBPCGResult:
    """LOBPCG for standard eigenvalue problem A x = lambda x.

    Computes the k smallest (or largest) eigenvalues and corresponding eigenvectors
    of a Hermitian linear operator A using the Locally Optimal Block Preconditioned
    Conjugate Gradient method.

    Args:
        A: A Hermitian linear operator.
        X: Initial guess as a block PyTree with k vectors. If None, random initialization
           is used (requires key parameter).
        k: Number of eigenvalues to compute. Required if X is None.
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance for residual norms.
        largest: If True, compute largest eigenvalues; otherwise smallest.
        key: Random key for initialization when X is None.

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
        >>> result = lobpcg_standard(A, k=2, key=jax.random.PRNGKey(0))
        >>> result.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
    """
    # Handle initialization
    if X is None:
        if k is None:
            raise ValueError('k must be specified when X is None')
        if key is None:
            raise ValueError('key must be specified when X is None')
        X = block_normal_like(A.in_structure(), k, key)
    else:
        leaves = jax.tree.leaves(X)
        k = leaves[0].shape[0]

    # Step 1: Orthonormalize initial vectors
    X = orthonormalize(X)

    # Step 2: Compute AX and initial Rayleigh quotient
    AX = apply_operator_block(A, X)

    # Initial eigenvalue estimates from Rayleigh quotient
    eigenvalues, X = _rayleigh_ritz(X, AX, k, largest)
    AX = apply_operator_block(A, X)

    # First iteration (no P)
    R = jax.tree.map(
        lambda ax_leaf, x_leaf: ax_leaf
        - eigenvalues.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf,
        AX,
        X,
    )
    residual_norms = block_norms(R)
    converged = residual_norms < tol

    def concat_blocks(*blocks: PyTree) -> PyTree:
        return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *blocks)

    AR = apply_operator_block(A, R)
    S = concat_blocks(X, R)
    AS = concat_blocks(AX, AR)
    S = orthonormalize(S)
    AS = apply_operator_block(A, S)
    eigenvalues, X = _rayleigh_ritz(S, AS, k, largest)
    AX = apply_operator_block(A, X)
    P = R
    AP = AR
    iteration = 1

    def cond_fn(carry):  # type: ignore[no-untyped-def]
        _, _, _, _, _, iteration, converged, _ = carry
        return jnp.logical_and(iteration < max_iters, ~jnp.all(converged))

    # Continue with while loop for remaining iterations
    def body_fn_with_p(carry):  # type: ignore[no-untyped-def]
        X, AX, P, AP, eigenvalues, iteration, converged, residual_norms = carry

        # Compute residuals
        R = jax.tree.map(
            lambda ax_leaf, x_leaf: ax_leaf
            - eigenvalues.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf,
            AX,
            X,
        )
        residual_norms = block_norms(R)
        converged = residual_norms < tol

        # Build search space S = [X, R, P]
        AR = apply_operator_block(A, R)
        S = concat_blocks(X, R, P)
        AS = concat_blocks(AX, AR, AP)

        # Orthonormalize S
        S = orthonormalize(S)
        AS = apply_operator_block(A, S)

        # Rayleigh-Ritz
        new_eigenvalues, new_X = _rayleigh_ritz(S, AS, k, largest)
        new_AX = apply_operator_block(A, new_X)

        # Update P
        new_P = R
        new_AP = AR

        return (
            new_X,
            new_AX,
            new_P,
            new_AP,
            new_eigenvalues,
            iteration + 1,
            converged,
            residual_norms,
        )

    carry = (X, AX, P, AP, eigenvalues, iteration, converged, residual_norms)

    final_carry = jax.lax.while_loop(cond_fn, body_fn_with_p, carry)
    X, AX, P, AP, eigenvalues, iteration, converged, residual_norms = final_carry

    # Final residual computation
    R = jax.tree.map(
        lambda ax_leaf, x_leaf: ax_leaf
        - eigenvalues.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf,
        AX,
        X,
    )
    residual_norms = block_norms(R)
    converged = residual_norms < tol

    return LOBPCGResult(
        eigenvalues=eigenvalues,
        eigenvectors=X,
        iterations=iteration,
        converged=converged,
        residual_norms=residual_norms,
    )
