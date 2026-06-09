from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Num, PyTree

from furax import AbstractLinearOperator, tree


class CGResult(NamedTuple):
    """Result of the Conjugate Gradient solver.

    Attributes:
        solution: The approximate solution x to Ax = b.
        residuals: Norm of the residual at each iteration, shape (max_steps,).
        num_steps: Number of CG steps taken.
    """

    solution: PyTree[Num[Array, '...']]
    residuals: Float[Array, ' max_steps']
    num_steps: Array


def cg(
    A: AbstractLinearOperator,
    b: PyTree[Num[Array, '...']],
    x0: PyTree[Num[Array, '...']] | None = None,
    *,
    preconditioner: AbstractLinearOperator | None = None,
    max_steps: int = 500,
    atol: float = 0.0,
    rtol: float = 1e-5,
    stabilise_every: int = 10,
    truncate: bool = False,
    iteration_callback: Callable[[Array, Array], None] | None = None,
) -> CGResult:
    """Conjugate Gradient solver for symmetric positive definite systems Ax = b.

    The residual norm is recorded at every iteration (see ``CGResult.residuals``)
    so convergence can be monitored. Inputs may be sharded along their contracting
    dimensions, and the solve is both forward- and reverse-mode differentiable.

    Convergence is declared when ``||r|| <= atol + rtol * ||b||``. With ``atol``
    and ``rtol`` both 0 the criterion is never met and the solver runs for exactly
    ``max_steps`` steps; otherwise it stops early once the residual is small enough,
    and ``max_steps`` is only a ceiling. Negative curvature (a non positive definite
    ``A``) is handled per ``truncate``.

    Args:
        A: A symmetric positive definite linear operator.
        b: Right-hand side of the system.
        x0: Initial guess. Defaults to zeros.
        preconditioner: Optional preconditioner M such that MA is better
            conditioned. M must be symmetric positive definite.
        max_steps: Maximum number of iteration steps.
        atol: Absolute tolerance on the residual norm.
        rtol: Relative tolerance on the residual norm (scaled by ``||b||``).
        stabilise_every: If set, replace the recursively-updated residual with
            the true residual ``b - A x`` every N iterations (after steps
            N, 2N, 3N, ...). This counters floating-point drift at the cost of
            one extra matvec per stabilisation.
        truncate: How to handle negative curvature ``p^T A p < 0``, which a
            positive definite ``A`` never produces. If False (default), raise a
            runtime error (configurable via Equinox's ``EQX_ON_ERROR``; see
            ``equinox.error_if``). If True, stop iterating and return the last
            iterate before the bad direction (truncated CG).
        iteration_callback: Optional host callback called after each step with
            ``(step, r_norm)`` as 0-d JAX arrays.  Runs via
            ``jax.debug.callback`` so it is JIT-compatible and ordered.

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
        >>> result = cg(A, b, max_steps=20)
        >>> expected = jnp.array([1.0, 0.5, 1 / 3, 0.25, 0.2])
        >>> bool(jnp.allclose(result.solution, expected, atol=1e-4))
        True
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
    # If i+1 >= max_steps (loop ran to completion), the last write is dropped by JAX.
    residuals = jnp.zeros(max_steps).at[0].set(r0_norm)

    def cond_fn(carry):  # type: ignore[no-untyped-def]
        _, _, _, _, converged, truncated, i, _ = carry
        return ~converged & ~truncated & (i < max_steps)

    def body_fn(carry):  # type: ignore[no-untyped-def]
        x, r, p, rz, _, _, i, residuals = carry

        Ap = A(p)
        pAp = tree.dot(p, Ap)

        # `pAp == 0` only at convergence (p -> 0), handled by `safe_pAp`
        # `pAp < 0` is genuine negative curvature, which a positive definite A should never produce
        if truncate:
            # Negative curvature: take no step this iteration and flag the loop to stop.
            truncated = pAp < 0
            safe_pAp = jnp.where(pAp == 0, 1.0, pAp)
            alpha = jnp.where(truncated, 0.0, rz / safe_pAp)
        else:
            pAp = eqx.error_if(pAp, pAp < 0, 'cg: negative curvature detected p^T A p < 0')
            truncated = jnp.array(False)
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

        if iteration_callback is not None:
            # `ordered = True` would error in distributed mode
            jax.debug.callback(iteration_callback, i, r_norm)

        z = M(r)
        rz_new = tree.dot(r, z)
        safe_rz = jnp.where(rz == 0, 1.0, rz)
        beta = rz_new / safe_rz

        p = tree.add(z, tree.mul(beta, p))

        converged = _converged(r_norm)
        residuals = residuals.at[i + 1].set(r_norm)

        return x, r, p, rz_new, converged, truncated, i + 1, residuals

    already_converged = _converged(r0_norm)
    init_carry = (x0, r, p, rz, already_converged, jnp.array(False), jnp.int32(0), residuals)
    # `residuals` is only ever written (`.at[i].set`) inside the loop, never read, so mark it a
    # buffer. `kind='bounded'` makes the loop both forward- and reverse-mode differentiable (lax's
    # while_loop is forward-only); it unrolls up to `max_steps` with logarithmic checkpointing.
    x, _, _, _, _, _, num_steps, residuals = eqxi.while_loop(
        cond_fn,
        body_fn,
        init_carry,
        max_steps=max_steps,
        buffers=lambda carry: carry[-1],
        kind='bounded',
    )

    return CGResult(solution=x, residuals=residuals, num_steps=num_steps)
