import jax
import jax.numpy as jnp
import optax
import jax.tree_util as jtu
import optax.tree_utils as otu
from functools import partial
from typing import NamedTuple
from optax import GradientTransformation

import lineax as lx  # For conjugate gradient solver


@partial(jax.jit, static_argnums=(1, 2, 3, 5))
def optimize(init_params, fun, opt, max_iter, tol, verbose=False, **kwargs):
    # Define a function that computes both value and gradient of the objective.
    value_and_grad_fun = jax.value_and_grad(fun)

    # Single optimization step.
    def step(carry):
        params, state, _, _ = carry
        value, grad = value_and_grad_fun(params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun, **kwargs
        )  # Perform update
        params = optax.apply_updates(params, updates)  # Update params
        return (params, state, updates, value)

    # Stopping condition.
    def continuing_criterion(carry):
        _, state, updates, value = carry
        iter_num = otu.tree_get(state, 'count')  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = otu.tree_l2_norm(updates)  # Compute update norm
        if verbose:
            jax.debug.print(
                'update norm {a} at iter {b} value {c}', a=update_norm, b=iter_num, c=value
            )
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_carry = (init_params, opt.init(init_params), init_params, jnp.inf)

    # Run the while loop.
    final_params, final_state, _, _ = jax.lax.while_loop(continuing_criterion, step, init_carry)

    return final_params, final_state


def _get_size_of_params(params):
    def add(x, y):
        if jnp.isscalar(y):
            return x + 1
        elif isinstance(y, jnp.ndarray):
            return x + y.size
        else:
            raise ValueError('Unkown type')

    if isinstance(params, jnp.ndarray):
        return params.size
    return jax.tree.reduce(add, params, initializer=0)


def _outer(tree1, tree2):
    def leaf_fn(x):
        return jtu.tree_map(lambda leaf: jnp.tensordot(x, leaf, axes=0), tree2)

    return jtu.tree_map(leaf_fn, tree1)


class ScaleByNewtonCGState(NamedTuple):
    """State for Newton-CG optimizer."""

    count: int  # Number of iterations
    params: jnp.ndarray  # Previous parameters
    grads: jnp.ndarray  # Current gradients
    value: float  # Current objective value


def scale_by_newton_cg(f, **kwargs) -> GradientTransformation:
    objective_f = lambda params, args: f(params, **kwargs)
    df = jax.grad(objective_f)
    solver = lx.NormalCG(rtol=1e-6, atol=1e-6)

    def init_fn(params):
        return ScaleByNewtonCGState(
            count=jnp.asarray(0, jnp.int32),
            params=otu.tree_zeros_like(params),
            grads=otu.tree_zeros_like(params),
            value=0.0,
        )

    def update_fn(updates, state, params, value=None, grad=None, value_fn=None, **kwargs):
        # Compute gradients and Hessian-vector products
        grads = updates
        operator = lx.JacobianLinearOperator(df, params, args=None)

        solution = lx.linear_solve(operator, grads, solver)
        direction = jax.tree.map(lambda x: -x, solution.value)
        # Update state
        new_state = ScaleByNewtonCGState(
            count=state.count + 1, params=params, grads=grads, value=0.0
        )

        return direction, new_state

    return GradientTransformation(init_fn, update_fn)


def newton_cg(f, **kwargs) -> GradientTransformation:
    """
    Newton-CG optimizer for unconstrained optimization.

    Args:
        f (Callable): Objective function.

    Returns:
        GradientTransformation: Optax-compatible Newton-CG optimizer.
    """
    return optax.chain(
        scale_by_newton_cg(f, **kwargs),
        optax.scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy='one'),
    )
