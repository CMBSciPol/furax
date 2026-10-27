"""LOBPCG eigenvalue solver for PyTree-aware linear operators."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, Int, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator
from furax.tree import _svqb_coefficients

from ._dense import _eigh_live, _hermitian_part, _matmul


class LOBPCGResult(NamedTuple):
    r"""Result of LOBPCG eigenvalue computation.

    Attributes:
        eigenvalues: The k computed eigenvalues, sorted ascending.
        eigenvectors: A block PyTree containing the k orthonormal eigenvectors.
        residual_norms: The norm of the residual $A x_i - \theta_i x_i$ of each eigenpair.
        converged: Whether each eigenpair meets the convergence criterion.
        iterations: The number of iterations performed.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, ' k ...']]
    residual_norms: Float[Array, ' k']
    converged: Bool[Array, ' k']
    iterations: Int[Array, '']


def lobpcg_standard(
    A: AbstractLinearOperator,
    X: PyTree[Num[Array, 'k ...']],
    *,
    maxiter: int = 100,
    tol: float = 1e-6,
    largest: bool = False,
    preconditioner: AbstractLinearOperator | None = None,
) -> LOBPCGResult:
    r"""LOBPCG for the standard eigenvalue problem $A x = \lambda x$.

    Computes the k smallest (or largest) eigenpairs of a Hermitian operator with the Locally
    Optimal Block Preconditioned Conjugate Gradient method. Each iteration applies $A$ to k
    vectors and finds the best k Ritz pairs in the span of the current eigenvector estimates
    $X$, their residuals $R$ (preconditioned by $M$ if given) and the previous search
    directions $P$.

    An eigenpair is converged when its residual satisfies

    $$
    \|A x_i - \theta_i x_i\| \le \max(\text{tol} \cdot |\theta_i|, 10 \sqrt{n}\, \epsilon \|A\|),
    $$

    where n is the operator size and $\epsilon$ the machine epsilon of the eigenvalue dtype; the
    iteration stops when all of them are. The second term is the accuracy to which residuals
    are computed in floating point, so that eigenvalues close to zero can converge. $\|A\|$ is
    estimated by the largest Ritz value in magnitude found so far.

    Note:
        The method is that of Knyazev (2001), with the search directions $P$ of Hetmaniuk &
        Lehoucq (2006): the part of the previous $X$ that the new $X$ does not span. It differs
        from the reference algorithm as follows.

        - The search space is not orthonormalized explicitly, which takes several passes over
          the vectors (as in Stathopoulos & Wu, 2002, or Duersch et al., 2018): the residuals are
          projected once against $X$ and $P$, and the Rayleigh-Ritz problem is solved in an
          orthonormal frame of the search space derived from its actual Gram matrix (SVQB).
          Only Gram matrices and linear combinations of the vectors are computed, so they may
          be sharded. Directions that become dependent near convergence are dropped from the
          frame, which keeps the Rayleigh-Ritz problem well conditioned. As $X$ and $P$ are only
          ever built as combinations of the search space, deriving the frame from its actual
          Gram matrix also keeps the rounding errors in their orthonormality from accumulating
          over the iterations, notably in single precision.
        - A dropped direction would give a spurious eigenvalue 0 in the projected problem,
          which could be selected for the smallest eigenpairs: it is placed past the end of
          the projected spectrum instead.
        - $A X$ and $A P$ are updated as the same combinations instead of being recomputed, so
          that an iteration applies $A$ to k vectors only.

    Args:
        A: A Hermitian linear operator.
        X: Initial guess as a block PyTree with k vectors (leading dimension k), linearly
            independent. A random block, e.g. from [`stacked_normal_like`][furax.tree.stacked_normal_like],
            will do.
        maxiter: Maximum number of iterations.
        tol: Convergence tolerance; see the criterion above.
        largest: If True, compute the largest eigenvalues; otherwise the smallest.
        preconditioner: Optional Hermitian positive definite operator $M$ approximating
            $A^{-1}$ (or $(A - \sigma I)^{-1}$ for a shift $\sigma$ near the wanted
            eigenvalues). The residuals are replaced by $M R$, which can accelerate convergence
            considerably. The eigenpairs computed are those of $A$ whatever $M$, unlike the
            `preconditioner` of [`lanczos_tr`][furax.linalg.lanczos_tr], which computes those of
            $M A$.

    Returns:
        [`LOBPCGResult`][furax.linalg.LOBPCGResult] containing the eigenvalues sorted
        ascending, eigenvectors, residual norms, convergence flags and iteration count.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure, stacked_normal_like
        >>> d = jnp.array([1., 2., 3., 4., 5., 6., 7., 8.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> X = stacked_normal_like(as_structure(d), 2, jax.random.key(0))
        >>> result = lobpcg_standard(A, X)
        >>> result.eigenvalues.round(4)
        Array([1., 2.], dtype=float32)
    """
    k = jax.tree.leaves(X)[0].shape[0]
    if k > A.in_size:
        raise ValueError(f'The number of vectors ({k}) must be <= the operator size ({A.in_size})')

    X = tree.stacked_orthonormalize(X)
    AX = tree.stacked_map(A, X)
    theta, C = jnp.linalg.eigh(_hermitian_part(tree.stacked_gram(X, AX)))
    X = tree.stacked_combine(X, C)
    AX = tree.stacked_combine(AX, C)
    norm_estimate = jnp.max(jnp.abs(theta))
    residual_norms, converged = _check_convergence(
        _residuals(X, AX, theta), theta, norm_estimate, A.in_size, tol
    )
    # Zero search directions are dropped from the Rayleigh-Ritz problem like dependent ones.
    P = tree.zeros_like(X)

    def cond_fn(state):
        *_, converged, iteration = state
        return (iteration < maxiter) & ~jnp.all(converged)

    def body_fn(state):
        X, AX, P, AP, theta, norm_estimate, _residual_norms, _converged, iteration = state

        R = _residuals(X, AX, theta)
        W = R if preconditioner is None else tree.stacked_map(preconditioner, R)
        # One projection against X and P: the search space S = [X, P, W] is nearly orthogonal,
        # and what remains of its dependence, and the scale of W, are handled on its Gram matrix.
        W = tree.sub(
            W,
            tree.add(
                tree.stacked_combine(X, tree.stacked_gram(X, W)),
                tree.stacked_combine(P, tree.stacked_gram(P, W)),
            ),
        )
        S = tree.stacked_concatenate([X, P, W])
        AS = tree.stacked_concatenate([AX, AP, tree.stacked_map(A, W)])

        # Orthonormal frame S T of the well-conditioned part of the span of S; the directions
        # dropped are zero columns of T.
        G_S = _hermitian_part(tree.stacked_gram(S, S))
        T, live = _svqb_coefficients(G_S, rtol=float(jnp.finfo(theta.dtype).eps) ** 0.5)
        K = _hermitian_part(_matmul(T.conj().T, _matmul(tree.stacked_gram(S, AS), T)))
        w, Q = _eigh_live(K, live, largest)

        if largest:
            wanted, others = slice(2 * k, 3 * k), slice(0, 2 * k)
        else:
            wanted, others = slice(0, k), slice(k, 3 * k)
        # Next search directions: the projection of the previous X, of coordinates Tᴴ G_S[:, :k]
        # in the frame, onto the Ritz vectors that are not selected, orthonormalized: P = S T E is
        # then orthonormal or zero, and orthogonal to the new X, even when the projection has
        # rank below k.
        coordinates = _matmul(T.conj().T, G_S[:, :k])
        E = _matmul(Q[:, others], _matmul(Q[:, others].conj().T, coordinates))
        E = tree.stacked_orthonormalize(jnp.where(live[:, None], E, 0).T).T
        B, D = _matmul(T, Q[:, wanted]), _matmul(T, E)

        X = tree.stacked_combine(S, B)
        AX = tree.stacked_combine(AS, B)
        P = tree.stacked_combine(S, D)
        AP = tree.stacked_combine(AS, D)
        theta = w[wanted]
        # Near convergence, the search space may span little more than X: keep the largest
        # Ritz value found so far, still a lower bound of ||A||.
        norm_estimate = jnp.maximum(norm_estimate, jnp.max(jnp.where(live, jnp.abs(w), 0)))
        residual_norms, converged = _check_convergence(
            _residuals(X, AX, theta), theta, norm_estimate, A.in_size, tol
        )
        return X, AX, P, AP, theta, norm_estimate, residual_norms, converged, iteration + 1

    init_state = (X, AX, P, P, theta, norm_estimate, residual_norms, converged, jnp.array(0))
    X, _, _, _, theta, _, residual_norms, converged, iterations = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )
    return LOBPCGResult(
        eigenvalues=theta,
        eigenvectors=X,
        residual_norms=residual_norms,
        converged=converged,
        iterations=iterations,
    )


def _residuals(
    X: PyTree[Num[Array, 'k ...']], AX: PyTree[Num[Array, 'k ...']], theta: Float[Array, ' k']
) -> PyTree[Num[Array, 'k ...']]:
    """Residuals A x_i - θ_i x_i."""
    return tree.sub(AX, _scale(X, theta))


def _check_convergence(
    R: PyTree[Num[Array, 'k ...']],
    theta: Float[Array, ' k'],
    norm_estimate: Float[Array, ''],
    n: int,
    tol: float,
) -> tuple[Float[Array, ' k'], Bool[Array, ' k']]:
    """Residual norms, and whether they meet the criterion of `lobpcg_standard`.

    The norm of A is estimated from the Ritz values of the whole search space, not only of the
    wanted pairs, which may all be close to zero.
    """
    residual_norms = tree.stacked_norm(R)
    eps = jnp.finfo(theta.dtype).eps
    threshold = jnp.maximum(tol * jnp.abs(theta), 10 * n**0.5 * eps * norm_estimate)
    return residual_norms, residual_norms <= threshold


def _scale(X: PyTree[Num[Array, 'k ...']], s: Num[Array, ' k']) -> PyTree[Num[Array, 'k ...']]:
    """Multiplies the i-th pytree of a stacked pytree by s[i]."""
    return jax.tree.map(lambda leaf: s.reshape(-1, *(1,) * (leaf.ndim - 1)) * leaf, X)
