"""Lanczos eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Key, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.tree_block import (
    apply_operator_block,
    apply_rotation,
    block_normal_like,
    block_norms,
    block_zeros_like,
)


class LanczosResult(NamedTuple):
    """Result of Lanczos eigenvalue computation.

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


def lanczos_tridiag(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    m: int,
) -> tuple[Float[Array, ' m'], Float[Array, ' m-1'], PyTree[Num[Array, 'm ...']]]:
    """Run m iterations of the Lanczos algorithm to build a tridiagonal matrix.

    The Lanczos algorithm generates an orthonormal basis {v_0, v_1, ..., v_{m-1}}
    for the Krylov subspace K_m(A, v0) = span{v0, Av0, A^2 v0, ..., A^{m-1} v0}.

    The matrix A restricted to this basis is tridiagonal with diagonal alpha
    and off-diagonal beta.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector (will be normalized).
        m: Number of Lanczos iterations (size of Krylov subspace).

    Returns:
        alpha: Diagonal of the tridiagonal matrix (m,).
        beta: Off-diagonal of the tridiagonal matrix (m-1,).
        V: Orthonormal Lanczos vectors as a block PyTree with shape (m, ...).
    """
    # Normalize v0
    norm_v0 = tree.norm(v0)
    v = tree.mul(1.0 / norm_v0, v0)

    # Pre-allocate storage for Lanczos vectors, tridiagonal elements
    V = block_zeros_like(v0, m)
    V = jax.tree.map(lambda V_leaf, v_leaf: V_leaf.at[0].set(v_leaf), V, v)
    alpha = jnp.zeros(m)
    beta = jnp.zeros(m - 1) if m > 1 else jnp.array([])

    # v_prev = 0 (conceptually)
    v_prev = tree.zeros_like(v)
    beta_prev = jnp.array(0.0)

    def body_fn(j, carry):  # type: ignore[no-untyped-def]
        V, alpha, beta, v, v_prev, beta_prev = carry

        # w = A @ v_j
        w = A(v)

        # alpha_j = v_j^H @ w
        alpha_j = tree.dot(v, w)
        alpha = alpha.at[j].set(alpha_j)

        # w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        w = tree.add(tree.mul(-alpha_j, v), w)
        w = tree.add(tree.mul(-beta_prev, v_prev), w)

        # Reorthogonalization (full) against all previously computed vectors
        # Scan over all m indices, but only apply for k <= j
        def reorth_step(carry, k):  # type: ignore[no-untyped-def]
            w = carry
            v_k = jax.tree.map(lambda V_leaf: V_leaf[k], V)
            coeff = tree.dot(v_k, w)
            # Only subtract if k <= j (mask out future vectors)
            coeff = jnp.where(k <= j, coeff, 0.0)
            w = tree.add(tree.mul(-coeff, v_k), w)
            return w, None

        w, _ = jax.lax.scan(reorth_step, w, jnp.arange(m))

        # beta_j = ||w||
        beta_j = tree.norm(w)

        # v_{j+1} = w / beta_j (if beta_j != 0)
        safe_beta = jnp.maximum(beta_j, 1e-14)
        v_next = tree.mul(1.0 / safe_beta, w)

        # Only update beta and V[j+1] if j < m - 1
        beta = jnp.where(j < m - 1, beta.at[j].set(beta_j), beta)
        V = jax.lax.cond(
            j < m - 1,
            lambda: jax.tree.map(lambda V_leaf, v_leaf: V_leaf.at[j + 1].set(v_leaf), V, v_next),
            lambda: V,
        )

        # Update for next iteration
        v_prev = v
        v = v_next
        beta_prev = beta_j

        return V, alpha, beta, v, v_prev, beta_prev

    init_carry = (V, alpha, beta, v, v_prev, beta_prev)
    V, alpha, beta, _, _, _ = jax.lax.fori_loop(0, m, body_fn, init_carry)

    return alpha, beta, V


def _tridiag_eigh(
    alpha: Float[Array, ' m'], beta: Float[Array, ' m-1']
) -> tuple[Float[Array, ' m'], Float[Array, 'm m']]:
    """Compute eigenvalues and eigenvectors of a symmetric tridiagonal matrix.

    Args:
        alpha: Diagonal elements (m,).
        beta: Off-diagonal elements (m-1,).

    Returns:
        eigenvalues: Eigenvalues sorted in ascending order (m,).
        eigenvectors: Eigenvectors as columns (m, m).
    """
    # `jax.scipy.linalg.eigh_tridiag` does not compute eigenvectors...
    # For now we have to build the full tridiagonal matrix
    T = jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    # Compute eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(T)
    return eigenvalues, eigenvectors


def _compute_residual_norms(
    A: AbstractLinearOperator,
    eigenvectors: PyTree[Num[Array, 'k ...']],
    eigenvalues: Float[Array, ' k'],
) -> Float[Array, ' k']:
    """Compute residual norms ||A @ x - lambda * x|| for each eigenpair.

    Args:
        A: The linear operator.
        eigenvectors: Block PyTree with k eigenvectors.
        eigenvalues: The k eigenvalues.

    Returns:
        Residual norms for each eigenpair.
    """
    A_eigenvectors = apply_operator_block(A, eigenvectors)
    k = eigenvalues.shape[0]

    def compute_residual(Ax_leaf: Array, x_leaf: Array) -> Array:
        return Ax_leaf - eigenvalues.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf

    residuals = jax.tree.map(compute_residual, A_eigenvectors, eigenvectors)
    return block_norms(residuals)


def _lanczos_step(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    m: int,
    k: int,
    largest: bool,
) -> tuple[Float[Array, ' k'], PyTree[Num[Array, 'k ...']], Float[Array, ' k']]:
    """Perform one Lanczos iteration and extract the k best eigenpairs.

    Args:
        A: A Hermitian linear operator.
        v0: Starting vector (will be normalized).
        m: Krylov subspace dimension.
        k: Number of eigenvalues to compute.
        largest: If True, compute largest eigenvalues; otherwise smallest.

    Returns:
        eigenvalues: The k eigenvalues.
        eigenvectors: Block PyTree with k eigenvectors.
        residual_norms: Residual norms for each eigenpair.
    """
    alpha, beta, V = lanczos_tridiag(A, v0, m)
    ritz_values, ritz_vectors = _tridiag_eigh(alpha, beta)

    # Select k smallest or largest Ritz pairs
    idx = jnp.arange(m - k, m) if largest else jnp.arange(k)
    eigenvalues = ritz_values[idx]
    selected_ritz_vectors = ritz_vectors[:, idx]  # (m, k)

    # Compute eigenvectors: linear combination of Lanczos vectors
    eigenvectors = apply_rotation(V, selected_ritz_vectors)

    residual_norms = _compute_residual_norms(A, eigenvectors, eigenvalues)
    return eigenvalues, eigenvectors, residual_norms


def lanczos_eigh(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']] | None = None,
    *,
    k: int = 1,
    m: int | None = None,
    max_restarts: int = 10,
    tol: float = 1e-6,
    largest: bool = False,
    key: Key[Array, ''] | None = None,
) -> LanczosResult:
    """Lanczos algorithm for computing k smallest/largest eigenvalues.

    Uses the Lanczos algorithm with implicit restarts to compute the k smallest
    (or largest) eigenvalues and corresponding eigenvectors of a Hermitian
    linear operator A.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector. If None, random initialization is used (requires key).
        k: Number of eigenvalues to compute.
        m: Dimension of Krylov subspace. If None, defaults to min(2*k + 1, n) where
           n is the dimension of the operator.
        max_restarts: Maximum number of implicit restarts.
        tol: Convergence tolerance for residual norms.
        largest: If True, compute largest eigenvalues; otherwise smallest.
        key: Random key for initialization when v0 is None.

    Returns:
        LanczosResult containing eigenvalues, eigenvectors, iteration count,
        convergence status, and residual norms.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> # Create operator with known eigenvalues [1, 2, 3, 4, 5]
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> result = lanczos_eigh(A, k=2, key=jax.random.PRNGKey(0))
        >>> result.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
    """
    # Handle initialization
    if v0 is None:
        if key is None:
            raise ValueError('key must be specified when v0 is None')
        v0_block = block_normal_like(A.in_structure(), 1, key)
        v0 = jax.tree.map(lambda leaf: leaf[0], v0_block)

    # Determine Krylov subspace dimension
    if m is None:
        leaves = jax.tree.leaves(A.in_structure())
        n = sum(leaf.size for leaf in leaves)
        m = min(2 * k + 1, n)

    if m < k:
        raise ValueError(f'm ({m}) must be >= k ({k})')

    # Initial Lanczos step
    eigenvalues, eigenvectors, residual_norms = _lanczos_step(A, v0, m, k, largest)
    converged = residual_norms < tol

    # Restart loop: refine using the best eigenvector as new starting vector
    def restart_cond(carry):  # type: ignore[no-untyped-def]
        _, _, iteration, converged, _ = carry
        return jnp.logical_and(iteration < max_restarts, ~jnp.all(converged))

    def restart_body(carry):  # type: ignore[no-untyped-def]
        _, eigenvectors, iteration, _, _ = carry
        v0_restart = jax.tree.map(lambda leaf: leaf[0], eigenvectors)
        new_eigenvalues, new_eigenvectors, new_residual_norms = _lanczos_step(
            A, v0_restart, m, k, largest
        )
        new_converged = new_residual_norms < tol
        return (new_eigenvalues, new_eigenvectors, iteration + 1, new_converged, new_residual_norms)

    initial_carry = (eigenvalues, eigenvectors, 1, converged, residual_norms)
    (
        final_eigenvalues,
        final_eigenvectors,
        final_iteration,
        final_converged,
        final_residual_norms,
    ) = jax.lax.while_loop(restart_cond, restart_body, initial_carry)

    return LanczosResult(
        eigenvalues=final_eigenvalues,
        eigenvectors=final_eigenvectors,
        iterations=final_iteration,
        converged=final_converged,
        residual_norms=final_residual_norms,
    )
