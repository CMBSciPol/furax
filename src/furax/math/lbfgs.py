from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array
from optax import tree_utils as otu


def run_lbfgs(
    init_params: Array,
    fun: Callable[[Array], Array],
    *,
    max_iter: int,
    tol: float,
    lower_bound: Array | None = None,
    upper_bound: Array | None = None,
) -> tuple[Array, optax.OptState]:
    """Minimize a function using L-BFGS with optional box constraints.

    Box constraints are enforced by projecting onto the box after each step
    (projected L-BFGS, not true L-BFGS-B). The curvature history may be
    corrupted near active bounds, and convergence there may be slower, but the
    difference should be small when bounds are rarely active.

    Based on https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html
    """
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):  # type: ignore[no-untyped-def]
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        if lower_bound is not None or upper_bound is not None:
            lo = lower_bound if lower_bound is not None else jnp.full_like(params, -jnp.inf)
            up = upper_bound if upper_bound is not None else jnp.full_like(params, jnp.inf)
            params = optax.projections.projection_box(params, lo, up)
        return params, state

    def continuing_criterion(carry):  # type: ignore[no-untyped-def]
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
    return final_params, final_state
