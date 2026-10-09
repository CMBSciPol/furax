"""Lanczos eigenvalue solver for PyTree-aware linear operators."""

from collections.abc import Callable
from functools import reduce
from typing import Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Key, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator

LanczosWhich = Literal['LM', 'SM', 'LA', 'SA', 'BE']


def _initial_vector(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']] | None,
    key: Key[Array, ''] | None,
) -> PyTree[Num[Array, '...']]:
    """Return v0, or a standard normal vector drawn from key if v0 is not given."""
    if v0 is not None:
        return v0
    if key is None:
        raise ValueError('Either a starting vector v0 or a random key must be given')
    return tree.normal_like(A.in_structure, key)


def _restart_key(key: Key[Array, ''] | None) -> Key[Array, '']:
    """Key for the vectors that continue the iteration after an invariant subspace is found.

    It is derived from the caller's key, independently of the draw of v0, or is fixed when
    no key is given so that a call with only v0 stays deterministic.
    """
    return jax.random.fold_in(jax.random.key(0) if key is None else key, 1)


def _orthogonalize(
    w: PyTree, V: PyTree, n: int | Array, P: PyTree | None = None
) -> tuple[PyTree, Float[Array, '']]:
    """Remove from w its components along the first n vectors of block PyTree V.

    Without P, V is orthonormal. With P, w is a dual vector (an image under M⁻¹) and P = M⁻¹ V
    is the dual of the M⁻¹-orthonormal V: w loses its components along P, so that M w is
    M⁻¹-orthogonal to V.

    Also returns the sum of the squared components removed, which is the squared norm lost
    by w (by M w in the M⁻¹ norm, with P).
    """
    P = V if P is None else P

    def step(i, carry):
        w, removed = carry
        v_i, p_i = tree.stacked_get((V, P), i)
        c_i = tree.dot(v_i, w)
        w = tree.add(tree.mul(-c_i, p_i), w)  # w -= <v_i, w> p_i
        return w, removed + jnp.abs(c_i) ** 2

    real_dtype = jnp.finfo(jax.tree.leaves(w)[0].dtype).dtype
    return jax.lax.fori_loop(0, n, step, (w, jnp.zeros((), real_dtype)))


class LanczosResult(NamedTuple):
    """Result of Lanczos eigenvalue computation.

    Attributes:
        eigenvalues: The k computed eigenvalues, sorted ascending.
        eigenvectors: A block PyTree containing k eigenvectors.
        residual_norms: The norm of the residual for each eigenpair.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, ' k ...']]
    residual_norms: Float[Array, ' k']


# =============================================================================
# Basic Lanczos
# =============================================================================


class _Basis(NamedTuple):
    """Lanczos vectors V, with their duals P = M⁻¹ V when preconditioned (None otherwise)."""

    V: PyTree[Num[Array, 'n ...']]
    P: PyTree[Num[Array, 'n ...']] | None


def _normalize(
    w: PyTree, M: AbstractLinearOperator | None
) -> tuple[Float[Array, ''], Callable[[], _Basis]]:
    r"""Norm of the vector whose dual is w, and a thunk returning that vector, unit norm.

    Without M, the vector is w itself, in the Euclidean norm. With M, it is M w, in the M⁻¹
    norm: $\|M w\|_{M^{-1}} = \sqrt{w^T M w}$. A zero w gives a zero vector, not NaN.
    """
    if M is None:
        norm = jnp.real(tree.norm(w))
        inv = 1.0 / jnp.where(norm > 0, norm, 1.0)
        return norm, lambda: _Basis(tree.mul(inv, w), None)
    Mw = M(w)
    norm = jnp.sqrt(jnp.real(tree.dot(w, Mw)))
    inv = 1.0 / jnp.where(norm > 0, norm, 1.0)
    return norm, lambda: _Basis(tree.mul(inv, Mw), tree.mul(inv, w))


def _lanczos_loop(
    A: AbstractLinearOperator,
    M: AbstractLinearOperator | None,
    basis: _Basis,
    alpha: Float[Array, ' m'],
    beta: Float[Array, ' m-1'],
    v_start: _Basis,
    v_prev: _Basis,
    beta_prev: Float[Array, ''],
    j_start: int,
    m: int,
    key: Key[Array, ''],
) -> tuple[_Basis, Float[Array, ' m'], Float[Array, ' m-1'], Float[Array, ''], _Basis]:
    """Run Lanczos iterations from absolute position j_start to m-1.

    basis[j_start] must already be set to v_start before calling. With a preconditioner M, the
    iteration is on M A, which is self-adjoint in the M⁻¹ inner product: the vectors are
    M⁻¹-orthonormal, and their duals M⁻¹ v are carried along so that M⁻¹ is never applied.

    Args:
        A: A Hermitian linear operator.
        M: Optional Hermitian positive definite preconditioner.
        basis: Pre-allocated m-vector basis with basis[j_start] = v_start.
        alpha: Diagonal array (m,), may be pre-filled for j < j_start.
        beta: Off-diagonal array (m-1,), may be pre-filled for j < j_start.
        v_start: Starting vector for the first iteration.
        v_prev: Previous Lanczos vector (zeros for a fresh start).
        beta_prev: Previous beta (0 for a fresh start).
        j_start: Absolute index of the first iteration.
        m: Total number of Lanczos vectors.
        key: Random key for the restart vectors drawn on breakdown.

    Returns:
        basis: Updated m-vector basis.
        alpha: Updated diagonal (m,).
        beta: Updated off-diagonal (m-1,).
        beta_last: Residual norm after the final step.
        v_last: Residual direction after the final step, or zero if beta_last is 0.
    """
    # DGKS criterion: an orthogonalization pass that keeps less than this fraction of the
    # norm has cancelled enough to lose orthogonality, and is repeated.
    eta = 1 / jnp.sqrt(2)

    def dual(v: _Basis) -> PyTree:
        return v.V if v.P is None else v.P

    def body_fn(j, carry):
        basis, alpha, beta, v, v_prev, beta_prev = carry

        Av = A(v.V)
        alpha_j = jnp.real(tree.dot(v.V, Av))  # α_j = <v_j, A v_j>
        alpha = alpha.at[j].set(alpha_j)

        # The residual is formed in the dual space: w = A v_j - α_j v_j - β_{j-1} v_{j-1} without
        # M, w = A v_j - α_j M⁻¹ v_j - β_{j-1} M⁻¹ v_{j-1} with M (the next vector is then M w).
        w = tree.add(tree.mul(-alpha_j, dual(v)), Av)
        w = tree.add(tree.mul(-beta_prev, dual(v_prev)), w)
        w, removed = _orthogonalize(w, basis.V, j + 1, basis.P)  # full reorthogonalization
        beta_j, normalized = _normalize(w, M)  # β_j = ||w|| (||M w||_M⁻¹ with M)
        # Norm before the pass, from the components it removed; this needs no extra M.
        w_norm = jnp.sqrt(beta_j**2 + removed)

        def keep():
            return normalized(), beta_j

        def refine():
            w2, _ = _orthogonalize(w, basis.V, j + 1, basis.P)
            beta2, normalized2 = _normalize(w2, M)
            # A second pass that keeps more than eta of the norm leaves w orthogonal to
            # V[:j+1] to working precision, even when w is mere rounding noise.
            # Breakdown: the second pass cancels again (or w is exactly zero), so w lies in
            # span(V[:j+1]) up to rounding and that span is invariant. Normalizing w would not
            # give a vector orthogonal to the basis, and keeping a zero vector would add a
            # spurious Ritz value 0 that looks converged. Instead set β_j = 0, which decouples
            # T into exact blocks, and continue from a random vector orthogonal to V[:j+1].
            # At the last step the residual term vanishes and v_last is left zero.
            breakdown = beta2 <= eta * beta_j
            restart = breakdown & (j < m - 1)
            r = tree.normal_like(w, jax.random.fold_in(key, j))
            # The second pass restores orthogonality lost when r is nearly in span(V[:j+1]).
            for _ in range(2):
                r, _ = _orthogonalize(r, basis.V, j + 1, basis.P)
            r = _normalize(r, M)[1]()
            x = jax.tree.map(
                lambda r_leaf, w_leaf: jnp.where(
                    restart, r_leaf, jnp.where(breakdown, jnp.zeros_like(w_leaf), w_leaf)
                ),
                r,
                normalized2(),
            )
            return x, jnp.where(breakdown, 0.0, beta2).astype(beta_j.dtype)

        v_next, beta_j = jax.lax.cond(beta_j <= eta * w_norm, refine, keep)

        beta = jnp.where(j < m - 1, beta.at[j].set(beta_j), beta)
        # The last step has no slot for v_next (j + 1 = m), which is returned as v_last instead.
        basis = tree.stacked_set(basis, j + 1, v_next, mode='drop')

        return basis, alpha, beta, v_next, v, beta_j

    init_carry = (basis, alpha, beta, v_start, v_prev, beta_prev)
    basis, alpha, beta, v_last, _, beta_last = jax.lax.fori_loop(j_start, m, body_fn, init_carry)
    return basis, alpha, beta, beta_last, v_last


def lanczos_tridiag(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']],
    m: int,
    key: Key[Array, ''] | None = None,
    *,
    preconditioner: AbstractLinearOperator | None = None,
) -> tuple[
    Float[Array, ' m'],
    Float[Array, ' m-1'],
    PyTree[Num[Array, 'm ...']],
    Float[Array, ''],
    PyTree[Num[Array, '...']],
]:
    r"""Run m iterations of the Lanczos algorithm to build a tridiagonal matrix.

    The Lanczos algorithm generates an orthonormal basis {v_0, v_1, ..., v_{m-1}}
    for the Krylov subspace K_m(A, v0) = span{v0, Av0, A^2 v0, ..., A^{m-1} v0}.

    The matrix A restricted to this basis is tridiagonal with diagonal alpha
    and off-diagonal beta.  The full m-step Lanczos factorization is:

    $$
    A V = V T + \beta_\text{last}\, v_\text{last}\, e_{m-1}^T
    $$

    If the Krylov subspace becomes invariant under A after j < m steps, beta[j-1] is 0 up
    to rounding and the iteration continues from a unit vector orthogonal to the basis
    built so far (a random one if the residual vanishes numerically), so that V stays
    orthonormal and the factorization above holds up to rounding.

    With a `preconditioner` $M$, the same holds with $M A$ in place of $A$, which is
    self-adjoint in the $M^{-1}$ inner product: $V$ is $M^{-1}$-orthonormal and spans the
    Krylov subspace of $M A$ from $M v_0$. Each step applies $M$ once; $M^{-1}$ is never
    applied.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector (will be normalized).
        m: Number of Lanczos iterations (size of Krylov subspace), at most the operator size.
        key: Random key for the vectors that continue the iteration after an invariant
            subspace is found. Defaults to `jax.random.key(0)`.
        preconditioner: Optional Hermitian positive definite operator $M$.

    Returns:
        alpha: Diagonal of the tridiagonal matrix (m,).
        beta: Off-diagonal of the tridiagonal matrix (m-1,).
        V: Orthonormal Lanczos vectors as a block PyTree with shape (m, ...).
        beta_last: Norm of the residual after m steps (the m-th beta).
        v_last: Residual direction after m steps (the (m+1)-th Lanczos vector), or zero
            if beta_last is 0.
    """
    if key is None:
        key = jax.random.key(0)
    alpha, beta, basis, beta_last, v_last = _lanczos_tridiag(A, preconditioner, v0, m, key)
    return alpha, beta, basis.V, beta_last, v_last.V


def _lanczos_tridiag(
    A: AbstractLinearOperator,
    M: AbstractLinearOperator | None,
    v0: PyTree,
    m: int,
    key: Key[Array, ''],
) -> tuple[Float[Array, ' m'], Float[Array, ' m-1'], _Basis, Float[Array, ''], _Basis]:
    """[`lanczos_tridiag`][], returning the bases with their duals."""
    n = A.in_size
    if m > n:
        raise ValueError(f'm ({m}) must be <= the operator size ({n})')

    _, normalized = _normalize(v0, M)
    v = normalized()

    basis = tree.stacked_set(tree.stacked_zeros_like(v, m), 0, v)
    alpha = jnp.zeros(m)
    beta = jnp.zeros(m - 1)

    basis, alpha, beta, beta_last, v_last = _lanczos_loop(
        A, M, basis, alpha, beta, v, tree.zeros_like(v), jnp.array(0.0), 0, m, key
    )
    return alpha, beta, basis, beta_last, v_last


def _default_m(A: AbstractLinearOperator, k: int) -> int:
    """Default Krylov subspace size: min(2k, n)."""
    return min(2 * k, A.in_size)


def lanczos_eigh(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']] | None = None,
    *,
    key: Key[Array, ''] | None = None,
    k: int = 20,
    m: int | None = None,
) -> LanczosResult:
    r"""Lanczos algorithm for computing k eigenvalues via an m-dimensional Krylov subspace.

    Builds an $m$-dimensional Krylov subspace ($m \ge k$) and computes all $m$ Ritz
    pairs.  When $m = k$ the method returns all $m$ Ritz pairs sorted ascending.  When
    $m > k$, only the $k$ Ritz pairs with the smallest residual norms are returned.  Selecting
    by residual norm (rather than by eigenvalue magnitude) picks the pairs that have
    converged most reliably within the subspace, which need not be the extremal ones.

    Note:
        There is no `which` parameter (unlike [`lanczos_tr`][]).  Targeted selection
        of smallest/largest pairs would require either restarts or post-filtering of
        unconverged Ritz values, neither of which is meaningful for a single-shot
        m-step factorization.  Use [`lanczos_tr`][] if you need extremal eigenpairs.

    Note:
        If the Krylov subspace becomes invariant under $A$ before $m$ steps (e.g. when
        $v_0$ misses some eigenvectors, or $A$ has repeated eigenvalues), the iteration
        continues from a random vector orthogonal to the current basis.  The pairs
        computed so far are then exact, and later pairs can include further copies of
        repeated eigenvalues.

    The cheap Lanczos residual bound is used:

    $$
    \|A y_i - \theta_i y_i\| \approx |\beta_m| \cdot |s_i[m-1]|
    $$

    where $s_i$ is the $i$-th eigenvector of the $m \times m$ tridiagonal $T_m$.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector for the Krylov subspace. If not given, it is drawn from `key`.
        key: Random key used to draw a standard normal `v0` when `v0` is not given, and
            the vectors that continue the iteration after an invariant subspace is found.
            If not given, those vectors are drawn from a fixed key.
        k: Number of eigenpairs to return.
        m: Size of the Krylov subspace.  Must be at least `k` and at most n.  Defaults to
            `min(2*k, n)`, where n is the size of the operator input.  Larger m
            builds a richer subspace and can yield more accurate Ritz pairs, at the
            cost of m matrix-vector products and storage for m vectors.

    Returns:
        [`LanczosResult`][] containing the k best eigenvalues, eigenvectors, and their
        residual norms, sorted by eigenvalue ascending.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> result = lanczos_eigh(A, key=jax.random.key(0), k=5)
        >>> result.eigenvalues
        Array([1., 2., 3., 4., 5.], dtype=float32)
    """
    v0 = _initial_vector(A, v0, key)
    m = m or _default_m(A, k)
    if m < k:
        raise ValueError(f'm ({m}) must be >= k ({k})')

    # Run Lanczos to build tridiagonal matrix in m-dimensional Krylov subspace
    alpha, beta, V, beta_last, _ = lanczos_tridiag(A, v0, m, _restart_key(key))
    ritz_values, ritz_vectors = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=False)

    eigenvectors = tree.stacked_combine(V, ritz_vectors)  # y_i = V s_i

    # ||A y_i - θ_i y_i|| ≈ |β_m| |s_i[-1]|
    residual_norms = jnp.abs(beta_last) * jnp.abs(ritz_vectors[-1, :])

    # Select the k best Ritz pairs by residual norm, then sort by eigenvalue ascending
    best_idx = jnp.argsort(residual_norms)[:k]
    best_idx = best_idx[jnp.argsort(ritz_values[best_idx])]
    return LanczosResult(
        eigenvalues=ritz_values[best_idx],
        eigenvectors=tree.stacked_get(eigenvectors, best_idx),
        residual_norms=residual_norms[best_idx],
    )


# =============================================================================
# Thick-Restart Lanczos (= symmetric Krylov-Schur)
# =============================================================================


def _build_bordered_tridiag(
    theta_k: Float[Array, ' k'],
    h: Float[Array, ' k'],
    alpha_ext: Float[Array, ' p'],
    beta_ext: Float[Array, ' p-1'],
    k: int,
    m: int,
) -> Float[Array, 'm m']:
    """Build the bordered tridiagonal m×m inner matrix H for thick-restart Lanczos.

    The matrix has the structure:

        H[:k, :k] = diag(theta_k)  # k Ritz values
        H[k,  :k] = h              # coupling row
        H[:k,  k] = h              # coupling col
        H[k:,  k:] = tridiag(alpha_ext, beta_ext)

    Args:
        theta_k: Ritz values retained from the thick-restart (k,).
        h: Coupling vector `beta_last * S[m-1, wanted_idx]` (k,).
        alpha_ext: Diagonal of the p×p extension block (p,).
        beta_ext: Off-diagonal of the p×p extension block (p-1,).
        k: Number of retained Ritz pairs.
        m: Total Krylov size (k + p).

    Returns:
        H: Symmetric bordered tridiagonal matrix (m, m).
    """
    dtype = reduce(jnp.promote_types, (theta_k.dtype, h.dtype, alpha_ext.dtype, beta_ext.dtype))
    H = jnp.zeros((m, m), dtype=dtype)
    H = H.at[:k, :k].set(jnp.diag(theta_k))
    H = H.at[k, :k].set(h)
    H = H.at[:k, k].set(h)
    H = H.at[k:, k:].set(jnp.diag(alpha_ext) + jnp.diag(beta_ext, 1) + jnp.diag(beta_ext, -1))
    return H


def _tr_extend(
    A: AbstractLinearOperator,
    M: AbstractLinearOperator | None,
    V_k: _Basis,
    v_start: _Basis,
    k: int,
    m: int,
    key: Key[Array, ''],
) -> tuple[Float[Array, ' p'], Float[Array, ' p-1'], _Basis, Float[Array, ''], _Basis]:
    """Extend a k-step thick-restart factorization to m steps.

    Runs p = m - k Lanczos iterations starting from v_start with full
    reorthogonalization against all accumulated vectors.

    Args:
        A: A Hermitian linear operator.
        M: Optional Hermitian positive definite preconditioner.
        V_k: k Ritz vectors from the thick-restart, block PyTree with shape (k, ...).
        v_start: Starting vector for the extension (the residual direction from
            the previous Lanczos run, already unit norm).
        k: Number of existing Ritz pairs.
        m: Target number of Lanczos vectors.
        key: Random key for the restart vectors drawn on breakdown.

    Returns:
        alpha_ext: Diagonal of the p×p extension block (p,).
        beta_ext: Off-diagonal of the p×p extension block (p-1,).
        V_m: Full m-vector basis [V_k | Lanczos extension] as a block PyTree.
        beta_last: Residual norm after m steps.
        v_last: Residual direction after m steps (unit norm, or zero if beta_last is 0).
    """
    dtype = jax.tree.leaves(v_start)[0].dtype
    real_dtype = jnp.empty((), dtype=dtype).real.dtype

    # Pre-allocate m-vector basis; fill first k slots with Ritz vectors
    V_m = tree.stacked_zeros_like(v_start, m)
    V_m = tree.stacked_set(V_m, slice(0, k), V_k)
    V_m = tree.stacked_set(V_m, k, v_start)

    # Set beta_prev=0 so the explicit `-beta_prev * v_prev` term in _lanczos_loop
    # vanishes; v_prev itself is unused (any vector would do). Coupling between the
    # new Lanczos vector and the k retained Ritz vectors is instead handled by the
    # full reorthogonalization loop, which projects against V_m[:j+1] (= all Ritz
    # vectors plus the current extension vectors).
    v_prev = tree.stacked_get(V_m, k - 1)

    alpha = jnp.zeros(m, dtype=real_dtype)
    beta = jnp.zeros(m - 1, dtype=real_dtype)

    V_m, alpha, beta, beta_last, v_last = _lanczos_loop(
        A, M, V_m, alpha, beta, v_start, v_prev, jnp.array(0.0, dtype=real_dtype), k, m, key
    )
    return alpha[k:], beta[k:], V_m, beta_last, v_last


def lanczos_tr(
    A: AbstractLinearOperator,
    v0: PyTree[Num[Array, '...']] | None = None,
    *,
    key: Key[Array, ''] | None = None,
    k: int = 20,
    m: int | None = None,
    which: LanczosWhich = 'LM',
    max_restarts: int = 300,
    tol: float = 1e-10,
    preconditioner: AbstractLinearOperator | None = None,
) -> LanczosResult:
    r"""Thick-restart Lanczos for computing k eigenpairs of a Hermitian operator.

    Each restart cycle maintains an m-step Lanczos factorization:

    $$
    A V_m = V_m H + \beta_\text{last}\, v_\text{last}\, e_{m-1}^T
    $$

    where $H$ is a bordered tridiagonal inner matrix.

    A pair is converged when the cheap Lanczos residual bound (ARPACK criterion)

    $$
    |\beta_\text{last}| \cdot |S[m-1, i]| \le \text{tol} \cdot \max(|\theta_i|, \epsilon \|A\|)
    $$

    where $S$ is the eigenvector matrix of the $m \times m$ inner matrix $H$, so
    $S[m-1, i]$ is the last component of the $i$-th eigenvector of $H$, and
    $\max_i |\theta_i|$ is used as an estimate of $\|A\|$.  $\epsilon$ is the machine
    epsilon of the eigenvalue dtype.

    Uses full reorthogonalization throughout.  No locking: all k pairs are
    recomputed at every restart regardless of convergence status.

    With a `preconditioner` $M$, computes eigenpairs of $M A$ instead, which is
    self-adjoint in the $M^{-1}$ inner product.  The eigenvectors $Z$ are then
    $M^{-1}$-orthonormal ($Z^T M^{-1} Z = I$, hence $Z^T A Z = \operatorname{diag}(\theta)$),
    and the criterion above holds with $M A$ in place of $A$.  Each step applies $M$ once;
    $M^{-1}$ is never applied.

    Note:
        `residual_norms` is the cheap bound, not the true residual.  Exactly
        `0` means "converged below the detectable coupling", not zero error.

    Note:
        Lanczos converges fastest to *extremal* eigenvalues; interior ones
        (near the middle of the spectrum) converge slowly.  This makes `which`
        targets that select interior pairs hard: `'SM'` for an indefinite
        operator picks eigenvalues closest to zero, which are interior, and
        restarts alone will not converge them unless `m` is a large fraction
        of `n` (no shift-invert is implemented).  `'LM'`, `'LA'`, `'SA'`
        and the two ends of `'BE'` are extremal and converge with small `m`.

    Note:
        A Krylov subspace built from a single vector contains at most one eigenvector
        per distinct eigenvalue.  Further copies of a repeated eigenvalue are only
        found when the iteration restarts from a random vector after the
        subspace becomes invariant, so they can be missed: for `diag(1, 1, 2, 2, 3, 3)`,
        `which='LA'` with `k=2` may return the exact pairs for 2 and 3, not 3 twice.

    Args:
        A: A Hermitian linear operator.
        v0: Initial vector for the Krylov subspace. If not given, it is drawn from `key`.
        key: Random key used to draw a standard normal `v0` when `v0` is not given, and
            the vectors that continue the iteration after an invariant subspace is found.
            If not given, those vectors are drawn from a fixed key.
        k: Number of eigenpairs to compute.
        m: Size of the Krylov subspace.  Must be larger than `k` and at most n.
            Defaults to `min(2*k, n)`, where n is the size of the operator input.
        which: Which k eigenpairs to target.  One of:

            - `'LM'` (default): k largest magnitude ($|\lambda|$).
            - `'SM'`: k smallest magnitude ($|\lambda|$).
            - `'LA'`: k largest algebraic.
            - `'SA'`: k smallest algebraic.
            - `'BE'`: half (`k//2`) from each end of the spectrum; for odd k the
              extra pair comes from the high (largest algebraic) end.
        max_restarts: Maximum number of restart cycles.
        tol: Convergence tolerance; see criterion above.
        preconditioner: Optional Hermitian positive definite operator $M$.

    Returns:
        [`LanczosResult`][] containing eigenvalues, eigenvectors, and residual norms,
        sorted by eigenvalue ascending.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> result = lanczos_tr(A, key=jax.random.key(0), k=2, which='SA')
        >>> result.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
    """
    if which not in get_args(LanczosWhich):
        raise ValueError(f'which must be one of {get_args(LanczosWhich)}, got {which!r}')
    v0 = _initial_vector(A, v0, key)
    m = m or _default_m(A, k)
    if m <= k:
        raise ValueError(f'm ({m}) must be > k ({k})')

    def _select_wanted(theta):
        if which == 'LM':  # largest magnitude
            sorted_idx = jnp.argsort(-jnp.abs(theta))
        elif which == 'SM':  # smallest magnitude
            sorted_idx = jnp.argsort(jnp.abs(theta))
        elif which == 'LA':  # largest algebraic
            sorted_idx = jnp.argsort(-theta)
        elif which == 'SA':  # smallest algebraic
            sorted_idx = jnp.argsort(theta)
        else:  # 'BE': half from each end of the spectrum
            sorted_idx = jnp.argsort(theta)
            n_low = k // 2
            n_high = k - n_low
            return jnp.concatenate([sorted_idx[:n_low], sorted_idx[-n_high:]])
        return sorted_idx[:k]

    def _check_converged(theta, beta_last, S, wanted_idx):
        # ARPACK criterion: |β_m| |s_i[-1]| ≤ tol * max(|θ_i|, eps*||A||)
        # Floor prevents stall when θ_i ≈ 0; max|θ| estimates ||A||.
        ritz_res = jnp.abs(beta_last) * jnp.abs(S[-1, wanted_idx])  # |β_m| |s_i[-1]|
        eps = jnp.finfo(theta.dtype).eps
        scale = jnp.maximum(jnp.abs(theta[wanted_idx]), eps * jnp.max(jnp.abs(theta)))
        return jnp.all(ritz_res <= tol * scale)

    # Initial m-step factorization
    # Each cycle draws its restart vectors from its own key, so that a breakdown at the same
    # step in two cycles does not retry a direction already in the retained Ritz vectors.
    restart_key = _restart_key(key)
    M = preconditioner
    alpha, beta, V, beta_last, v_last = _lanczos_tridiag(
        A, M, v0, m, jax.random.fold_in(restart_key, 0)
    )
    theta, S = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=False)
    wanted_idx = _select_wanted(theta)
    init_converged = _check_converged(theta, beta_last, S, wanted_idx)

    def cond_fn(state):
        *_, iteration, converged, _theta, _S, _wanted_idx = state
        return jnp.logical_and(iteration < max_restarts, ~converged)

    def body_fn(state):
        V, beta_last, v_last, iteration, _converged, theta, S, wanted_idx = state

        V_k = tree.stacked_combine(V, S[:, wanted_idx])  # U_k = V S[:,wanted]  (Ritz vectors)
        theta_k = theta[wanted_idx]  # θ_k
        h = beta_last * S[-1, wanted_idx]  # h_i = β_m s_i[-1]  (coupling)

        alpha_ext, beta_ext, V, beta_last, v_last = _tr_extend(
            A, M, V_k, v_last, k, m, jax.random.fold_in(restart_key, iteration + 1)
        )

        H = _build_bordered_tridiag(theta_k, h, alpha_ext, beta_ext, k, m)
        theta, S = jnp.linalg.eigh(H)  # H S = S diag(θ)

        wanted_idx = _select_wanted(theta)
        converged = _check_converged(theta, beta_last, S, wanted_idx)

        return V, beta_last, v_last, iteration + 1, converged, theta, S, wanted_idx

    init_state = (V, beta_last, v_last, jnp.array(0), init_converged, theta, S, wanted_idx)
    V, beta_last, _v_last, _iters, _conv, theta, S, wanted_idx = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    # Sort selected pairs by eigenvalue ascending
    wanted_idx = wanted_idx[jnp.argsort(theta[wanted_idx])]
    eigenvalues = theta[wanted_idx]
    eigenvectors = tree.stacked_combine(V.V, S[:, wanted_idx])  # y_i = V s_i
    residual_norms = jnp.abs(beta_last) * jnp.abs(S[-1, wanted_idx])  # |β_m| |s_i[-1]|

    return LanczosResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residual_norms=residual_norms,
    )
