"""Lanczos eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.tree_block import (
    block_norm,
    block_zeros_like,
    vecmat,
)


class LanczosResult(NamedTuple):
    """Result of Lanczos eigenvalue computation.

    Attributes:
        eigenvalues: The k computed eigenvalues, sorted ascending if largest=False.
        eigenvectors: A block PyTree containing k eigenvectors.
        residual_norms: The norm of the residual for each eigenpair.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, ' k ...']]
    residual_norms: Float[Array, ' k']


# =============================================================================
# Basic Lanczos
# =============================================================================


def lanczos_tridiag(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    m: int,
) -> tuple[
    Float[Array, ' m'],
    Float[Array, ' m-1'],
    PyTree[Num[Array, 'm ...']],
    Float[Array, ''],
    PyTree[Num[Array, '...']],
]:
    """Run m iterations of the Lanczos algorithm to build a tridiagonal matrix.

    The Lanczos algorithm generates an orthonormal basis {v_0, v_1, ..., v_{m-1}}
    for the Krylov subspace K_m(A, v0) = span{v0, Av0, A^2 v0, ..., A^{m-1} v0}.

    The matrix A restricted to this basis is tridiagonal with diagonal alpha
    and off-diagonal beta.  The full m-step Lanczos factorization is:
        A V = V T + beta_last * v_last * e_{m-1}^T

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector (will be normalized).
        m: Number of Lanczos iterations (size of Krylov subspace).

    Returns:
        alpha: Diagonal of the tridiagonal matrix (m,).
        beta: Off-diagonal of the tridiagonal matrix (m-1,).
        V: Orthonormal Lanczos vectors as a block PyTree with shape (m, ...).
        beta_last: Norm of the residual after m steps (the m-th beta).
        v_last: Residual direction after m steps (the (m+1)-th Lanczos vector).
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

        # Full reorthogonalization against all previously computed vectors
        def reorth_step(k, w):  # type: ignore[no-untyped-def]
            v_k = jax.tree.map(lambda V_leaf: V_leaf[k], V)
            coeff = tree.dot(v_k, w)
            return tree.add(tree.mul(-coeff, v_k), w)

        w = jax.lax.fori_loop(0, j + 1, reorth_step, w)

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
    V, alpha, beta, v_last, _, beta_last = jax.lax.fori_loop(0, m, body_fn, init_carry)

    return alpha, beta, V, beta_last, v_last


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
    A_eigenvectors = jax.vmap(A.mv)(eigenvectors)
    # double transpose for correct broadcasting
    scaled_eigenvectors = jax.tree.map(lambda x: (x.T * eigenvalues).T, eigenvectors)
    residuals = tree.sub(A_eigenvectors, scaled_eigenvectors)
    return block_norm(residuals)


def _default_m(A: AbstractLinearOperator, k: int) -> int:
    """Default Krylov subspace size: min(2k, n)."""
    leaves = jax.tree.leaves(A.in_structure)
    n = sum(leaf.size for leaf in leaves)
    return min(2 * k, n)  # type: ignore[no-any-return]


def lanczos_eigh(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    *,
    k: int = 20,
    m: int | None = None,
) -> LanczosResult:
    """Lanczos algorithm for computing k eigenvalues via an m-dimensional Krylov subspace.

    Builds an m-dimensional Krylov subspace (m >= k) and computes all m Ritz pairs.
    When m == k the method returns all m Ritz pairs sorted ascending.  When m > k,
    only the k Ritz pairs with the smallest residual norms are returned.  Selecting
    by residual norm (rather than by eigenvalue magnitude) picks the pairs that have
    converged most reliably within the subspace, which need not be the extremal ones.

    The cheap Lanczos residual bound is used:
        ||A y_i - θ_i y_i|| ≈ |β_m| |s_i[m-1]|
    where s_i is the i-th eigenvector of the m×m tridiagonal T_m.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector for the Krylov subspace.
        k: Number of eigenpairs to return.
        m: Size of the Krylov subspace.  Must be >= k.  Defaults to min(2k, n).
            Larger m builds a richer subspace and can yield more accurate Ritz
            pairs, at the cost of m matrix-vector products and O(m) vector storage.

    Returns:
        LanczosResult containing the k best eigenvalues, eigenvectors, and their
        residual norms, sorted by eigenvalue ascending.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> from furax.tree_block import block_normal_like
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(0))[0]
        >>> result = lanczos_eigh(A, v0, k=5)
        >>> result.eigenvalues
        Array([1., 2., 3., 4., 5.], dtype=float32)
    """
    m = m or _default_m(A, k)
    if m < k:
        raise ValueError(f'm ({m}) must be >= k ({k})')

    # Run Lanczos to build tridiagonal matrix in m-dimensional Krylov subspace
    alpha, beta, V, beta_last, _ = lanczos_tridiag(A, v0, m)
    ritz_values, ritz_vectors = _tridiag_eigh(alpha, beta)

    # Compute eigenvectors: linear combination of Lanczos vectors
    eigenvectors = vecmat(V, ritz_vectors)

    # Cheap residual norms via Lanczos relation: ||r_i|| = |beta_last| * |s_i[-1]|
    residual_norms = jnp.abs(beta_last) * jnp.abs(ritz_vectors[-1, :])

    # Select the k best Ritz pairs by residual norm, then sort by eigenvalue ascending
    best_idx = jnp.argsort(residual_norms)[:k]
    best_idx = best_idx[jnp.argsort(ritz_values[best_idx])]
    return LanczosResult(
        eigenvalues=ritz_values[best_idx],
        eigenvectors=jax.tree.map(lambda leaf: leaf[best_idx], eigenvectors),
        residual_norms=residual_norms[best_idx],
    )


# =============================================================================
# Krylov-Schur Method (Thick Restart Lanczos)
# =============================================================================


def _build_ks_matrix(
    theta_k: Float[Array, ' k'],
    h: Float[Array, ' k'],
    alpha_ext: Float[Array, ' p'],
    beta_ext: Float[Array, ' p-1'],
    k: int,
    m: int,
) -> Float[Array, 'm m']:
    """Build the bordered tridiagonal m×m inner matrix H for Krylov-Schur.

    The matrix has the structure::

        H[:k, :k] = diag(theta_k)          (k Ritz values)
        H[k,  :k] = h                       (coupling row)
        H[:k,  k] = h                       (coupling col, symmetry)
        H[k:,  k:] = tridiag(alpha_ext, beta_ext)

    Args:
        theta_k: Ritz values retained from the KS restart (k,).
        h: Coupling vector ``beta_last * S[m-1, wanted_idx]`` (k,).
        alpha_ext: Diagonal of the p×p extension block (p,).
        beta_ext: Off-diagonal of the p×p extension block (p-1,).
        k: Number of retained Ritz pairs.
        m: Total Krylov size (k + p).

    Returns:
        H: Symmetric bordered tridiagonal matrix (m, m).
    """
    H = jnp.zeros((m, m), dtype=theta_k.dtype)
    H = H.at[:k, :k].set(jnp.diag(theta_k))
    H = H.at[k, :k].set(h)
    H = H.at[:k, k].set(h)
    H = H.at[k:, k:].set(jnp.diag(alpha_ext) + jnp.diag(beta_ext, 1) + jnp.diag(beta_ext, -1))
    return H


def _ks_extend(
    A: AbstractLinearOperator,
    V_k: PyTree[Num[Array, 'k ...']],
    v_start: PyTree[Num[Array, '...']],
    k: int,
    m: int,
) -> tuple[
    Float[Array, ' p'],
    Float[Array, ' p-1'],
    PyTree[Num[Array, 'm ...']],
    Float[Array, ''],
    PyTree[Num[Array, '...']],
]:
    """Extend a k-step Krylov-Schur factorization to m steps.

    Runs p = m - k Lanczos iterations starting from v_start with full
    reorthogonalization against all accumulated vectors.  Unlike
    :func:`_extend_lanczos`, there is no single β connecting V_k to v_start;
    the full k-dimensional coupling is handled by reorthogonalization at the
    first step.

    Args:
        A: A Hermitian linear operator.
        V_k: k Ritz vectors from the KS restart, block PyTree with shape (k, ...).
        v_start: Starting vector for the extension (the residual direction from
            the previous Lanczos run, already unit norm).
        k: Number of existing Ritz pairs.
        m: Target number of Lanczos vectors.

    Returns:
        alpha_ext: Diagonal of the p×p extension block (p,).
        beta_ext: Off-diagonal of the p×p extension block (p-1,).
        V_m: Full m-vector basis [V_k | Lanczos extension] as a block PyTree.
        beta_last: Residual norm after m steps.
        v_last: Residual direction after m steps (unit norm).
    """
    p = m - k
    dtype = jax.tree.leaves(v_start)[0].dtype

    # Pre-allocate m-vector basis; fill first k slots with Ritz vectors
    V_m = block_zeros_like(v_start, m)
    V_m = jax.tree.map(lambda Vm_l, Vk_l: Vm_l.at[:k].set(Vk_l), V_m, V_k)
    V_m = jax.tree.map(lambda Vm_l, vn_l: Vm_l.at[k].set(vn_l), V_m, v_start)

    alpha_ext = jnp.zeros(p, dtype=dtype)
    beta_store = jnp.zeros(p, dtype=dtype)  # beta_store[p-1] is beta_last

    # v_prev at j=0 is the last Ritz vector; beta_prev=0 so it cancels out,
    # and full reorthogonalization handles the coupling to all Ritz vectors.
    v_prev = jax.tree.map(lambda leaf: leaf[k - 1], V_m)

    def body_fn(j, carry):  # type: ignore[no-untyped-def]
        V_m, alpha_ext, beta_store, v, v_prev, beta_prev = carry
        j_abs = j + k  # absolute position in the m-step factorization

        w = A(v)
        alpha_j = tree.dot(v, w)
        alpha_ext = alpha_ext.at[j].set(alpha_j)

        w = tree.add(tree.mul(-alpha_j, v), w)
        w = tree.add(tree.mul(-beta_prev, v_prev), w)

        # Full reorthogonalization against all accumulated vectors
        def reorth_step(k_idx, w):  # type: ignore[no-untyped-def]
            v_k = jax.tree.map(lambda V_leaf: V_leaf[k_idx], V_m)
            coeff = tree.dot(v_k, w)
            return tree.add(tree.mul(-coeff, v_k), w)

        w = jax.lax.fori_loop(0, j_abs + 1, reorth_step, w)

        beta_j = tree.norm(w)
        v_new = tree.mul(1.0 / jnp.maximum(beta_j, 1e-14), w)

        beta_store = beta_store.at[j].set(beta_j)
        V_m = jax.lax.cond(
            j_abs < m - 1,
            lambda: jax.tree.map(lambda Vm_l, vn_l: Vm_l.at[j_abs + 1].set(vn_l), V_m, v_new),
            lambda: V_m,
        )

        return V_m, alpha_ext, beta_store, v_new, v, beta_j

    init_carry = (V_m, alpha_ext, beta_store, v_start, v_prev, jnp.array(0.0, dtype=dtype))
    V_m, alpha_ext, beta_store, v_last, _, beta_last = jax.lax.fori_loop(0, p, body_fn, init_carry)

    return alpha_ext, beta_store[: p - 1], V_m, beta_last, v_last


def lanczos_ks(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    *,
    k: int = 20,
    m: int | None = None,
    which: str = 'smallest',
    max_restarts: int = 20,
    tol: float = 1e-6,
) -> LanczosResult:
    """Krylov-Schur method (thick restart Lanczos) for computing k eigenpairs.

    The m-step Krylov-Schur factorization after restart + extension satisfies::

        A V_m = V_m H + beta_last * v_last * e_{m-1}^T

    where H is the bordered tridiagonal inner matrix.  The residual bound is:
    ``||A y_i - theta_i y_i|| ≤ |beta_last| * |S[m-1, i]|``.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector for the Krylov subspace.
        k: Number of eigenpairs to compute.
        m: Number of Lanczos vectors.  Must be > k.
            Defaults to min(2*k, n).
        which: Which k eigenpairs to target.  One of:
            - 'smallest' (default): k smallest eigenvalues.
            - 'largest': k largest eigenvalues.
            - 'best': k Ritz pairs with the smallest residual norms.
        max_restarts: Maximum number of restart cycles.
        tol: Convergence tolerance for Lanczos residual bounds.

    Returns:
        LanczosResult containing eigenvalues, eigenvectors, and residual norms,
        sorted by eigenvalue ascending.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> from furax.tree_block import block_normal_like
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(0))[0]
        >>> result = lanczos_ks(A, v0, k=2, which='smallest')
        >>> result.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
    """
    if which not in ('smallest', 'largest', 'best'):
        raise ValueError(f"which must be 'smallest', 'largest', or 'best', got {which!r}")
    m = m or _default_m(A, k)
    if m <= k:
        raise ValueError(f'm ({m}) must be > k ({k})')

    def _select_wanted(theta, beta_last, S):  # type: ignore[no-untyped-def]
        if which == 'smallest':
            sorted_idx = jnp.argsort(theta)
        elif which == 'largest':
            sorted_idx = jnp.argsort(-theta)
        else:  # 'best'
            ritz_res = jnp.abs(beta_last) * jnp.abs(S[-1, :])
            sorted_idx = jnp.argsort(ritz_res)
        return sorted_idx[:k]

    def _check_converged(beta_last, S, wanted_idx):  # type: ignore[no-untyped-def]
        ritz_res = jnp.abs(beta_last) * jnp.abs(S[m - 1, wanted_idx])
        return jnp.all(ritz_res < tol)

    # Initial m-step factorization
    alpha, beta, V, beta_last, v_last = lanczos_tridiag(A, v0, m)
    theta, S = _tridiag_eigh(alpha, beta)
    wanted_idx = _select_wanted(theta, beta_last, S)
    init_converged = _check_converged(beta_last, S, wanted_idx)

    def cond_fn(state):  # type: ignore[no-untyped-def]
        *_, iteration, converged, _theta, _S = state
        return jnp.logical_and(iteration < max_restarts, ~converged)

    def body_fn(state):  # type: ignore[no-untyped-def]
        V, beta_last, v_last, iteration, _converged, theta, S = state

        wanted_idx = _select_wanted(theta, beta_last, S)

        # KS restart: rotate basis to Ritz vectors
        V_k = vecmat(V, S[:, wanted_idx])
        theta_k = theta[wanted_idx]
        h = beta_last * S[m - 1, wanted_idx]

        # Extend k-step KS factorization to m steps
        alpha_ext, beta_ext, V, beta_last, v_last = _ks_extend(A, V_k, v_last, k, m)

        # Build bordered tridiagonal H and solve dense eigenvalue problem
        H = _build_ks_matrix(theta_k, h, alpha_ext, beta_ext, k, m)
        theta, S = jnp.linalg.eigh(H)

        wanted_idx = _select_wanted(theta, beta_last, S)
        converged = _check_converged(beta_last, S, wanted_idx)

        return V, beta_last, v_last, iteration + 1, converged, theta, S

    init_state = (V, beta_last, v_last, jnp.array(0), init_converged, theta, S)
    V, beta_last, _v_last, _iters, _conv, theta, S = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    # Reuse (theta, S) from final state — no additional eigendecomposition needed
    wanted_idx = _select_wanted(theta, beta_last, S)
    # Sort selected pairs by eigenvalue ascending
    wanted_idx = wanted_idx[jnp.argsort(theta[wanted_idx])]
    eigenvalues = theta[wanted_idx]
    eigenvectors = vecmat(V, S[:, wanted_idx])
    residual_norms = _compute_residual_norms(A, eigenvectors, eigenvalues)

    return LanczosResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residual_norms=residual_norms,
    )
