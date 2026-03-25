from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Num, PyTree

from furax import tree
from furax.core import AbstractLinearOperator


class CGResult(NamedTuple):
    """Result of the Conjugate Gradient solver.

    Attributes:
        solution: The approximate solution x to Ax = b.
        residuals: Norm of the residual at each iteration, shape (max_iter,).
        iterations: Number of CG steps taken.
    """

    solution: PyTree[Num[Array, '...']]
    residuals: Float[Array, ' max_iter']
    iterations: Array


def cg(
    A: AbstractLinearOperator,
    b: PyTree[Num[Array, '...']],
    x0: PyTree[Num[Array, '...']] | None = None,
    *,
    preconditioner: AbstractLinearOperator | None = None,
    max_iter: int = 500,
    atol: float = 0.0,
    rtol: float = 1e-5,
    stabilise_every: int = 10,
) -> CGResult:
    """Conjugate Gradient solver for symmetric positive definite systems Ax = b.

    Unlike lineax's CG solver, this implementation records the residual norm at
    every iteration so you can monitor convergence.

    Convergence is declared when ``||r|| <= atol + rtol * ||b||``.

    Args:
        A: A symmetric positive definite linear operator.
        b: Right-hand side of the system.
        x0: Initial guess. Defaults to zeros.
        preconditioner: Optional preconditioner M such that MA is better
            conditioned. M must be symmetric positive definite.
        max_iter: Maximum number of iterations.
        atol: Absolute tolerance on the residual norm.
        rtol: Relative tolerance on the residual norm (scaled by ``||b||``).
            When both ``atol`` and ``rtol`` are 0, convergence is never
            declared and the solver runs for exactly ``max_iter`` steps.
        stabilise_every: If set, replace the recursively-updated residual with
            the true residual ``b - A x`` every N iterations (after steps
            N, 2N, 3N, ...). This counters floating-point drift at the cost of
            one extra matvec per stabilisation.

    Returns:
        CGResult with the solution, per-iteration residual norms, and iteration count.

    Example:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> from furax.linalg import cg
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> b = jnp.ones(5)
        >>> result = cg(A, b, max_iter=20)
        >>> result.solution
        Array([1.  , 0.5 , 0.333..., 0.25 , 0.2 ], dtype=float32)
    """
    if x0 is None:
        x0 = tree.zeros_like(b)

    has_scale = atol > 0 or rtol > 0
    norm_b = tree.norm(b)
    abs_tol = atol + rtol * norm_b

    def _converged(r_norm: Array) -> Array:
        return has_scale & (r_norm <= abs_tol)

    def _stable_r(x: PyTree) -> PyTree:
        return tree.sub(b, A(x))

    def _cheap_r(r: PyTree, alpha: Array, Ap: PyTree) -> PyTree:
        return tree.sub(r, tree.mul(alpha, Ap))

    # Initial residual r = b - A @ x0
    r = _stable_r(x0)

    # Apply preconditioner: z = M r
    M = preconditioner or (lambda _: _)
    z = M(r)

    p = z
    rz = tree.dot(r, z)  # r^T z (or r^T M^{-1} r)

    r0_norm = tree.norm(r)
    # residuals[0] = initial residual; residuals[i+1] = residual after step i.
    # If i+1 >= max_iter (loop ran to completion), the last write is dropped by JAX.
    residuals = jnp.zeros(max_iter).at[0].set(r0_norm)

    def cond_fn(carry):  # type: ignore[no-untyped-def]
        *_, converged, i, _ = carry
        return ~converged & (i < max_iter)

    def body_fn(carry):  # type: ignore[no-untyped-def]
        x, r, z, p, rz, _, i, residuals = carry

        Ap = A(p)
        pAp = tree.dot(p, Ap)

        safe_pAp = jnp.where(pAp == 0, 1.0, pAp)
        alpha = rz / safe_pAp

        x = tree.add(x, tree.mul(alpha, p))

        # Choose between stable (true) and cheap (recursive) residual
        if stabilise_every > 0:
            r = jax.lax.cond(
                i % stabilise_every == stabilise_every - 1,
                lambda: _stable_r(x),
                lambda: _cheap_r(r, alpha, Ap),
            )
        else:
            r = _cheap_r(r, alpha, Ap)

        r_norm = tree.norm(r)

        z = M(r)
        rz_new = tree.dot(r, z)
        safe_rz = jnp.where(rz == 0, 1.0, rz)
        beta = rz_new / safe_rz

        p = tree.add(z, tree.mul(beta, p))

        converged = _converged(r_norm)
        residuals = residuals.at[i + 1].set(r_norm)

        return x, r, z, p, rz_new, converged, i + 1, residuals

    already_converged = _converged(r0_norm)
    init_carry = (x0, r, z, p, rz, already_converged, jnp.int32(0), residuals)
    x, _, _, _, _, _, iterations, residuals = jax.lax.while_loop(cond_fn, body_fn, init_carry)

    return CGResult(solution=x, residuals=residuals, iterations=iterations)
