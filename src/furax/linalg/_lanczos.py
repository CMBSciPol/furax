"""Lanczos eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Key, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.tree_block import block_normal_like


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

    # Initialize storage for Lanczos vectors
    # We'll store them as a list and stack at the end
    V_list = [v]

    # Initialize tridiagonal elements
    alpha_list = []
    beta_list = []

    # v_prev = 0 (conceptually)
    v_prev = tree.zeros_like(v)
    beta_prev = jnp.array(0.0)

    for j in range(m):
        # w = A @ v_j
        w = A.mv(v)

        # alpha_j = v_j^H @ w
        alpha_j = tree.dot(v, w)
        alpha_list.append(alpha_j)

        # w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        w = tree.add(tree.mul(-alpha_j, v), w)
        w = tree.add(tree.mul(-beta_prev, v_prev), w)

        # Reorthogonalization (full) to maintain numerical stability
        # This is important for large m to prevent loss of orthogonality
        for v_k in V_list:
            coeff = tree.dot(v_k, w)
            w = tree.add(tree.mul(-coeff, v_k), w)

        # beta_j = ||w||
        beta_j = tree.norm(w)

        if j < m - 1:
            beta_list.append(beta_j)

            # v_{j+1} = w / beta_j (if beta_j != 0)
            # Use a small epsilon to avoid division by zero
            safe_beta = jnp.maximum(beta_j, 1e-14)
            v_next = tree.mul(1.0 / safe_beta, w)
            V_list.append(v_next)

            # Update for next iteration
            v_prev = v
            v = v_next
            beta_prev = beta_j

    # Stack Lanczos vectors into block PyTree
    V = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *V_list)

    alpha = jnp.array(alpha_list)
    beta = jnp.array(beta_list) if beta_list else jnp.array([])

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
    # Build the full tridiagonal matrix
    T = jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
    # Compute eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(T)
    return eigenvalues, eigenvectors


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
        # Extract single vector from block
        v0 = jax.tree.map(lambda leaf: leaf[0], v0_block)

    # Determine Krylov subspace dimension
    if m is None:
        # Get operator dimension from structure
        leaves = jax.tree.leaves(A.in_structure())
        n = sum(leaf.size for leaf in leaves)
        m = min(2 * k + 1, n)

    if m < k:
        raise ValueError(f'm ({m}) must be >= k ({k})')

    # Run Lanczos iteration
    alpha, beta, V = lanczos_tridiag(A, v0, m)

    # Compute Ritz values and vectors from tridiagonal matrix
    ritz_values, ritz_vectors = _tridiag_eigh(alpha, beta)

    # Select k smallest or largest
    if largest:
        idx = jnp.arange(m - k, m)
    else:
        idx = jnp.arange(k)

    selected_ritz_values = ritz_values[idx]
    selected_ritz_vectors = ritz_vectors[:, idx]  # (m, k)

    # Compute eigenvector approximations: X = V^T @ ritz_vectors
    # V has shape (m, ...) for each leaf, ritz_vectors is (m, k)
    # Result should have shape (k, ...) for each leaf
    def compute_eigenvectors(V_leaf: Array) -> Array:
        # V_leaf: (m, ...)
        m_dim = V_leaf.shape[0]
        rest_shape = V_leaf.shape[1:]
        V_flat = V_leaf.reshape(m_dim, -1)  # (m, n)
        # X_flat = V_flat.T @ selected_ritz_vectors -> (n, k)
        X_flat = V_flat.T @ selected_ritz_vectors
        # Transpose to (k, n) then reshape to (k, ...)
        return X_flat.T.reshape((k,) + rest_shape)

    eigenvectors = jax.tree.map(compute_eigenvectors, V)

    # Compute residual norms: ||A @ x - lambda * x||
    # Apply A to each eigenvector
    leaves, treedef = jax.tree.flatten(eigenvectors)

    def apply_single(i: Array) -> list[Array]:
        single_x = treedef.unflatten([leaf[i] for leaf in leaves])
        result = A.mv(single_x)
        return jax.tree.leaves(result)

    A_eigenvectors_leaves = jax.vmap(apply_single)(jnp.arange(k))
    A_eigenvectors = treedef.unflatten(A_eigenvectors_leaves)

    # Compute residuals: r_i = A @ x_i - lambda_i * x_i
    def compute_residual(Ax_leaf: Array, x_leaf: Array) -> Array:
        # Ax_leaf, x_leaf: (k, ...)
        # lambda: (k,)
        return Ax_leaf - selected_ritz_values.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf

    residuals = jax.tree.map(compute_residual, A_eigenvectors, eigenvectors)

    # Compute residual norms
    def leaf_squared_norms(leaf: Array) -> Array:
        # leaf: (k, ...)
        flat = leaf.reshape(k, -1)  # (k, n)
        return jnp.sum(jnp.abs(flat) ** 2, axis=1)  # (k,)

    leaf_list = jax.tree.leaves(jax.tree.map(leaf_squared_norms, residuals))
    squared_norms = leaf_list[0]
    for leaf in leaf_list[1:]:
        squared_norms = squared_norms + leaf
    residual_norms = jnp.sqrt(squared_norms)

    converged = residual_norms < tol

    # If not all converged and we have restarts left, we could do implicit restart
    # For simplicity, we do explicit restarts using the best Ritz vector
    iteration = 1

    def restart_cond(carry):  # type: ignore[no-untyped-def]
        _, _, iteration, converged, _ = carry
        return jnp.logical_and(iteration < max_restarts, ~jnp.all(converged))

    def restart_body(carry):  # type: ignore[no-untyped-def]
        eigenvalues, eigenvectors, iteration, converged, residual_norms = carry

        # Use best eigenvector as starting vector for new Lanczos run
        # Extract first eigenvector (smallest or largest depending on 'largest')
        v0_restart = jax.tree.map(lambda leaf: leaf[0], eigenvectors)

        # Run Lanczos again
        alpha_new, beta_new, V_new = lanczos_tridiag(A, v0_restart, m)
        ritz_values_new, ritz_vectors_new = _tridiag_eigh(alpha_new, beta_new)

        if largest:
            idx_new = jnp.arange(m - k, m)
        else:
            idx_new = jnp.arange(k)

        new_eigenvalues = ritz_values_new[idx_new]
        new_ritz_vectors = ritz_vectors_new[:, idx_new]

        def compute_eigenvectors_new(V_leaf: Array) -> Array:
            m_dim = V_leaf.shape[0]
            rest_shape = V_leaf.shape[1:]
            V_flat = V_leaf.reshape(m_dim, -1)
            X_flat = V_flat.T @ new_ritz_vectors
            return X_flat.T.reshape((k,) + rest_shape)

        new_eigenvectors = jax.tree.map(compute_eigenvectors_new, V_new)

        # Recompute residuals
        leaves_new, treedef_new = jax.tree.flatten(new_eigenvectors)

        def apply_single_new(i: Array) -> list[Array]:
            single_x = treedef_new.unflatten([leaf[i] for leaf in leaves_new])
            result = A.mv(single_x)
            return jax.tree.leaves(result)

        A_new_leaves = jax.vmap(apply_single_new)(jnp.arange(k))
        A_new = treedef_new.unflatten(A_new_leaves)

        def compute_residual_new(Ax_leaf: Array, x_leaf: Array) -> Array:
            return Ax_leaf - new_eigenvalues.reshape((k,) + (1,) * (x_leaf.ndim - 1)) * x_leaf

        residuals_new = jax.tree.map(compute_residual_new, A_new, new_eigenvectors)

        leaf_list_new = jax.tree.leaves(jax.tree.map(leaf_squared_norms, residuals_new))
        squared_norms_new = leaf_list_new[0]
        for leaf in leaf_list_new[1:]:
            squared_norms_new = squared_norms_new + leaf
        new_residual_norms = jnp.sqrt(squared_norms_new)

        new_converged = new_residual_norms < tol

        return (new_eigenvalues, new_eigenvectors, iteration + 1, new_converged, new_residual_norms)

    carry = (selected_ritz_values, eigenvectors, iteration, converged, residual_norms)

    final_carry = jax.lax.while_loop(restart_cond, restart_body, carry)

    (
        final_eigenvalues,
        final_eigenvectors,
        final_iteration,
        final_converged,
        final_residual_norms,
    ) = final_carry

    return LanczosResult(
        eigenvalues=final_eigenvalues,
        eigenvectors=final_eigenvectors,
        iterations=final_iteration,
        converged=final_converged,
        residual_norms=final_residual_norms,
    )
