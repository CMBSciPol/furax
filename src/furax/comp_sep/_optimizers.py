import jax
import jax.numpy as jnp
import optax
import jax.tree_util as jtu
import optax.tree_utils as otu
from functools import partial
from typing import NamedTuple
from optax import GradientTransformation

import lineax as lx  # For conjugate gradient solver

from jaxtyping import PyTree , ScalarLike

class OptimizerState(NamedTuple):
    """State for Newton-CG optimizer."""
    params : PyTree
    state : PyTree
    updates : PyTree
    value : ScalarLike
    best_val : ScalarLike
    best_params : PyTree

def _debug_callback(update_norm , iter_num , value , max_iters , log_interval):
    if iter_num == 0 or iter_num % int(max_iters * log_interval) == 0 :
        print(f"update norm {update_norm} at iter {iter_num} value {value}")


@partial(jax.jit, static_argnums=(1, 2, 3, 5 , 6))
def optimize(init_params, fun, opt, max_iter, tol, verbose=False,log_interval=0.1, **kwargs):
    # Define a function that computes both value and gradient of the objective.
    value_and_grad_fun = jax.value_and_grad(fun)

    # Single optimization step.
    def step(carry):
        value, grad = value_and_grad_fun(carry.params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(
            grad, carry.state, carry.params, value=carry.value, grad=grad, value_fn=fun, **kwargs
        )  # Perform update
        params = optax.apply_updates(carry.params, updates)  # Update params

        best_params = jax.tree.map(lambda x, y: jnp.where(carry.best_val < value, x, y), carry.best_params, carry.params)
        best_val = jnp.where(carry.best_val < value, carry.best_val, value)
        return carry._replace(params=params, state=state, updates=updates, value=value, best_val=best_val, best_params=best_params)

    # Stopping condition.
    def continuing_criterion(carry):
        iter_num = otu.tree_get(carry.state, 'count')  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = otu.tree_l2_norm(carry.updates)  # Compute update norm
        if verbose:
            jax.debug.callback(_debug_callback, update_norm , iter_num , carry.value , max_iter , log_interval)
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_state = OptimizerState(init_params , opt.init(init_params) ,
                                init_params , jnp.inf , jnp.inf , init_params)

    # Run the while loop.
    final_opt_state = jax.lax.while_loop(continuing_criterion, step, init_state)

    # was last evaluation better than the best
    best_params = jax.tree.map(lambda x, y: jnp.where(final_opt_state.best_val < final_opt_state.value, x, y), final_opt_state.best_params, final_opt_state.params)
    best_value = jnp.where(final_opt_state.best_val < final_opt_state.value, final_opt_state.best_val, final_opt_state.value)
    final_opt_state = final_opt_state._replace(best_params=best_params, best_val=best_value)

    return final_opt_state.best_params , final_opt_state


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
