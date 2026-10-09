from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Literal, NamedTuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array, Float, Num, PyTree

from furax import AbstractLinearOperator, tree


class CGResult(NamedTuple):
    """Result of the Conjugate Gradient solver.

    Attributes:
        solution: The approximate solution x to Ax = b.
        residuals: Norm of the residual at each iteration, shape (max_steps,).
        num_steps: Number of CG steps taken.
        converged: Whether the residual norm met the tolerance `atol + rtol * ||b||`. False
            when the solve ran out of steps or stopped on negative curvature.
    """

    solution: PyTree[Num[Array, '...']]
    residuals: Float[Array, ' max_steps']
    num_steps: Array
    converged: Array


class _CGCarry(NamedTuple):
    """Loop-carried state for the CG iteration.

    ``residuals`` is kept last and write-only so it can be passed as the ``eqxi.while_loop``
    buffer. ``truncated`` ends the loop early on negative curvature (``negative_curvature='truncate'``).
    """

    x: PyTree[Num[Array, '...']]
    r: PyTree[Num[Array, '...']]
    p: PyTree[Num[Array, '...']]
    rz: Array
    converged: Array
    truncated: Array
    step: Array
    residuals: Float[Array, ' max_steps']


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
    negative_curvature: Literal['ignore', 'error', 'truncate'] = 'ignore',
    loop_kind: Literal['lax', 'checkpointed', 'bounded'] = 'lax',
    differentiation: Literal['implicit', 'unrolled'] = 'implicit',
    iteration_callback: Callable[[Array, Array], None] | None = None,
) -> CGResult:
    r"""Conjugate Gradient solver for symmetric positive definite systems Ax = b.

    The residual norm is recorded at every iteration (see ``CGResult.residuals``)
    so convergence can be monitored. Inputs may be sharded along their contracting
    dimensions. The solution is differentiable in forward and reverse mode with
    respect to ``A`` and ``b``; see ``differentiation``.

    Convergence is declared when ``||r|| <= atol + rtol * ||b||``. With ``atol``
    and ``rtol`` both 0 the criterion is never met and the solver runs for exactly
    ``max_steps`` steps; otherwise it stops early once the residual is small enough,
    and ``max_steps`` is only a ceiling. ``A`` is assumed positive definite; negative
    curvature is handled per ``negative_curvature``.

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
        negative_curvature: How to handle negative curvature ``p^T A p < 0``, which a
            positive definite ``A`` never produces. One of:

            - ``'ignore'`` (default): assume ``A`` is positive definite and do not
                check. Fastest; the other modes add a per-iteration check.
            - ``'error'``: raise as soon as negative curvature is encountered (a
                debugging aid for verifying ``A``; configurable via Equinox's
                ``EQX_ON_ERROR``, see ``equinox.error_if``).
            - ``'truncate'``: stop and return the last iterate before the bad
                direction (truncated CG, as used by Newton-CG on indefinite Hessians).
        loop_kind: Lowering for the iteration loop (``equinox.internal.while_loop``).
            Only matters with ``differentiation='unrolled'``, which differentiates the loop.
            ``'lax'`` (default) is fastest and forward-mode differentiable only.
            ``'bounded'`` adds reverse-mode AD but its cost scales with ``max_steps``
            (the checkpoint structure runs to the ceiling regardless of early exit),
            so keep ``max_steps`` tight. ``'checkpointed'`` is reverse-mode only.
        differentiation: How derivatives of the solution are computed. One of:

            - ``'implicit'`` (default): differentiate the exact solution $x = A^{-1} b$ at
                the computed one, $\dot{x} = A^{-1} (\dot{b} - \dot{A} x)$. Each derivative
                costs one more CG solve with the same settings, in constant memory, and is
                as accurate as the solves are converged. The derivatives with respect to
                ``x0`` and ``preconditioner`` are zero, and so are those of
                ``CGResult.residuals``. A solve stopped by ``negative_curvature='truncate'``
                does not return $A^{-1} b$, so its implicit derivatives are not those of
                the returned solution.
            - ``'unrolled'``: differentiate through the CG iterations, which gives the exact
                derivative of the returned iterate, converged or not, including its
                dependence on ``x0`` and ``preconditioner``. Reverse mode requires
                ``loop_kind='bounded'`` or ``'checkpointed'``.
        iteration_callback: Optional host callback called after each step with
            ``(step, r_norm)`` as 0-d JAX arrays.  Runs via
            ``jax.debug.callback`` so it is JIT-compatible and ordered.

    Returns:
        CGResult with the solution, per-iteration residual norms, and iteration count.

    Examples:
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
    if negative_curvature not in ('ignore', 'error', 'truncate'):
        raise ValueError(
            f'negative_curvature must be ignore/error/truncate, got {negative_curvature!r}'
        )
    if differentiation not in ('implicit', 'unrolled'):
        raise ValueError(f'differentiation must be implicit/unrolled, got {differentiation!r}')
    solve = partial(
        _cg_loop,
        max_steps=max_steps,
        atol=atol,
        rtol=rtol,
        stabilise_every=stabilise_every,
        negative_curvature=negative_curvature,
        loop_kind=loop_kind,
    )
    if differentiation == 'unrolled':
        return solve(A, b, x0, preconditioner, iteration_callback)
    return _implicit_cg(solve, A, b, x0, preconditioner, iteration_callback)


def _implicit_cg(
    solve: Callable[..., CGResult],
    A: AbstractLinearOperator,
    b: PyTree[Num[Array, '...']],
    x0: PyTree[Num[Array, '...']] | None,
    preconditioner: AbstractLinearOperator | None,
    iteration_callback: Callable[[Array, Array], None] | None,
) -> CGResult:
    r"""Run `solve` with the derivative of the exact solution $A^{-1} b$.

    The tangent $\dot{x} = A^{-1} (\dot{b} - \dot{A} x)$ is solved through
    `jax.lax.custom_linear_solve`, which JAX knows how to transpose (into a solve with the
    symmetric `A`). The same rule therefore serves forward and reverse mode, and since it calls
    the differentiable solve again, higher-order derivatives too.
    """
    # `custom_jvp` accepts only JAX values, but operators may hold other leaves, such as the
    # `Ellipsis` in the indices of an `IndexOperator`: pass the arrays and close over the rest.
    dynamic, static = eqx.partition((A, b, x0, preconditioner), eqx.is_array)
    A_static = static[0]

    @jax.custom_jvp
    def implicit_solve(dynamic: PyTree[Array]) -> CGResult:
        A, b, x0, preconditioner = eqx.combine(dynamic, static)
        return solve(A, b, x0, preconditioner, iteration_callback)

    @partial(implicit_solve.defjvp, symbolic_zeros=True)
    def implicit_solve_jvp(
        primals: tuple[Any, ...], tangents: tuple[Any, ...]
    ) -> tuple[CGResult, CGResult]:
        (dynamic,) = primals
        A, _, _, preconditioner = eqx.combine(dynamic, static)
        A_dynamic, b_dynamic, _, _ = dynamic
        # `A^{-1} b` does not depend on the initial guess or the preconditioner.
        A_dot, b_dot, _, _ = tangents[0]
        result = implicit_solve(dynamic)
        x = result.solution

        # Inputs that are not differentiated have symbolic zero tangents: skip their terms, in
        # particular the matvec `A_dot x` when only `b` is differentiated.
        b_tangents, A_tangents = _differentiated_leaves(b_dot), _differentiated_leaves(A_dot)
        if not b_tangents and not A_tangents:
            # Only `x0` or `preconditioner` is differentiated.
            return result, jax.tree.map(_zero_tangent, result)
        rhs = _instantiate_zeros(b_dynamic, b_dot) if b_tangents else None
        if A_tangents:
            _, A_dot_x = jax.jvp(
                lambda op: eqx.combine(op, A_static)(x),
                (A_dynamic,),
                (_instantiate_zeros(A_dynamic, A_dot),),
            )
            rhs = tree.mul(-1, A_dot_x) if rhs is None else tree.sub(rhs, A_dot_x)

        # `custom_linear_solve` transposes only if every leaf of its right-hand side depends on
        # the input tangents. A leaf fed only by undifferentiated inputs, e.g. one block of a
        # block-diagonal `A`, is a constant zero, so tie every leaf to a differentiated tangent.
        anchor = 0 * jnp.sum((b_tangents + A_tangents)[0]).real
        rhs = jax.tree.map(lambda leaf: leaf + anchor.astype(leaf.dtype), rhs)

        def tangent_solve(_: Any, rhs: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
            return solve(A, rhs, None, preconditioner, None).solution

        x_dot = jax.lax.custom_linear_solve(A, rhs, tangent_solve, symmetric=True)
        # The convergence diagnostics are not differentiated, and integer and boolean outputs
        # take `float0` tangents.
        result_dot = jax.tree.map(_zero_tangent, result)._replace(solution=x_dot)
        return result, result_dot

    return implicit_solve(dynamic)


def _differentiated_leaves(tangent: PyTree[Any]) -> list[Array]:
    """Return the leaves of a `custom_jvp` tangent that are neither symbolic nor `float0` zeros."""
    return [
        leaf
        for leaf in jax.tree.leaves(tangent)
        if not isinstance(leaf, SymbolicZero) and leaf.dtype != jax.dtypes.float0
    ]


def _instantiate_zeros[P](primal: P, tangent: PyTree[Any]) -> P:
    """Replace the symbolic zeros in `tangent` by zero tangents shaped and sharded like `primal`."""
    return jax.tree.map(
        lambda p, t: _zero_tangent(p) if isinstance(t, SymbolicZero) else t, primal, tangent
    )


def _zero_tangent(x: Array) -> Array | np.ndarray:
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, jax.dtypes.float0)


def _cg_loop(
    A: AbstractLinearOperator,
    b: PyTree[Num[Array, '...']],
    x0: PyTree[Num[Array, '...']] | None,
    preconditioner: AbstractLinearOperator | None,
    iteration_callback: Callable[[Array, Array], None] | None,
    *,
    max_steps: int,
    atol: float,
    rtol: float,
    stabilise_every: int,
    negative_curvature: Literal['ignore', 'error', 'truncate'],
    loop_kind: Literal['lax', 'checkpointed', 'bounded'],
) -> CGResult:
    """The CG iteration of [`cg`][], differentiable only by unrolling."""
    truncate = negative_curvature == 'truncate'
    check_curvature = negative_curvature == 'error'

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
    eps = jnp.finfo(rz.real.dtype).eps
    # `rz` is a residual energy, so square the relative norm floor. The 100*eps
    # margin catches roundoff-level residuals before stabilisation can amplify them.
    rz_floor = (100 * eps) ** 2 * jnp.abs(rz)

    r0_norm = tree.norm(r)
    # residuals[0] = initial residual; residuals[i+1] = residual after step i.
    # If i+1 >= max_steps (loop ran to completion), the last write is dropped by JAX.
    residuals = jnp.zeros(max_steps).at[0].set(r0_norm)

    def cond_fn(c: _CGCarry) -> Array:
        return ~c.converged & ~c.truncated & (c.step < max_steps)

    def body_fn(c: _CGCarry) -> _CGCarry:
        x, r, p, rz, i = c.x, c.r, c.p, c.rz, c.step

        Ap = A(p)
        pAp = tree.dot(p, Ap)
        active = jnp.abs(rz) > rz_floor

        # `pAp == 0` only at convergence (p -> 0), handled by `safe_pAp`
        # `pAp < 0` is genuine negative curvature, which a positive definite A should never produce
        if truncate:
            # Negative curvature: take no step this iteration and flag the loop to stop.
            truncated = active & (pAp < 0)
            safe_pAp = jnp.where(pAp == 0, 1.0, pAp)
            alpha = jnp.where(active & ~truncated, rz / safe_pAp, 0.0)
        else:
            truncated = c.truncated
            if check_curvature:
                pAp = eqx.error_if(
                    pAp, active & (pAp < 0), 'cg: negative curvature detected p^T A p < 0'
                )
            safe_pAp = jnp.where(pAp == 0, 1.0, pAp)
            alpha = jnp.where(active, rz / safe_pAp, 0.0)

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
        # Once the preconditioned residual energy reaches the floating-point floor,
        # fixed-iteration CG should become inert. This avoids dividing true-residual
        # roundoff by true-residual roundoff after stabilisation.
        safe_rz = jnp.where(active, rz, 1.0)
        beta = jnp.where(jnp.abs(rz_new) > rz_floor, rz_new / safe_rz, 0.0)

        p = tree.add(z, tree.mul(beta, p))

        converged = _converged(r_norm)
        residuals = c.residuals.at[i + 1].set(r_norm)

        return _CGCarry(x, r, p, rz_new, converged, truncated, i + 1, residuals)

    init_carry = _CGCarry(
        x=x0,
        r=r,
        p=p,
        rz=rz,
        converged=_converged(r0_norm),
        truncated=jnp.array(False),
        step=jnp.int32(0),
        residuals=residuals,
    )
    # `residuals` is only ever written (`.at[i].set`) inside the loop, never read, so mark it a
    # buffer for an efficient in-place scatter. `loop_kind` selects the lowering: 'lax' is fastest
    # (forward-mode AD only), 'bounded'/'checkpointed' add reverse-mode AD (see `loop_kind` arg).
    out = eqxi.while_loop(
        cond_fn,
        body_fn,
        init_carry,
        max_steps=max_steps,
        buffers=lambda c: c.residuals,
        kind=loop_kind,
    )

    return CGResult(
        solution=out.x, residuals=out.residuals, num_steps=out.step, converged=out.converged
    )


@dataclass(frozen=True)
class CGSolver:
    """Conjugate Gradient settings, usable as the solver of an operator inverse.

    Pass it where a lineax solver is accepted, e.g. `A.I(solver=CGSolver(rtol=1e-8))` or
    `Config(solver=CGSolver())`, to solve with [`cg`][] instead of `lineax.CG`. The fields are the
    keyword arguments of [`cg`][]; see there for their meaning.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> from furax.linalg import CGSolver
        >>> d = jnp.array([1., 2., 4.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> A.I(solver=CGSolver(max_steps=10))(jnp.ones(3)).tolist()
        [1.0, 0.5, 0.25]
    """

    max_steps: int = 500
    atol: float = 0.0
    rtol: float = 1e-5
    stabilise_every: int = 10
    negative_curvature: Literal['ignore', 'error', 'truncate'] = 'ignore'
    loop_kind: Literal['lax', 'checkpointed', 'bounded'] = 'lax'
    differentiation: Literal['implicit', 'unrolled'] = 'implicit'
    iteration_callback: Callable[[Array, Array], None] | None = None

    def __call__(
        self,
        A: AbstractLinearOperator,
        b: PyTree[Num[Array, '...']],
        x0: PyTree[Num[Array, '...']] | None = None,
        *,
        preconditioner: AbstractLinearOperator | None = None,
    ) -> CGResult:
        """Solve `A x = b` with [`cg`][] using these settings."""
        options = {f.name: getattr(self, f.name) for f in fields(self)}
        return cg(A, b, x0, preconditioner=preconditioner, **options)
