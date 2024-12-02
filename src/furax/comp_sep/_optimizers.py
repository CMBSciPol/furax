import jax
import jax.numpy as jnp
import optax
import jax.tree_util as jtu
import optax.tree_utils as otu
from functools import partial
from typing import NamedTuple
from optax import GradientTransformation

import lineax as lx  # For conjugate gradient solver


@partial(jax.jit, static_argnums=(1, 2, 3))
def optimize(init_params, fun, opt, max_iter, tol, **kwargs):
    # Define a function that computes both value and gradient of the objective.
    value_and_grad_fun = jax.value_and_grad(fun)

    # Single optimization step.
    def step(carry):
        params, state, _ = carry
        value, grad = value_and_grad_fun(params, **kwargs)  # Compute value and gradient
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun, **kwargs
        )  # Perform update
        params = optax.apply_updates(params, updates)  # Update params
        return (params, state, updates)

    # Stopping condition.
    def continuing_criterion(carry):
        _, state, updates = carry
        iter_num = otu.tree_get(state, 'count')  # Get iteration count from optimizer state
        iter_num = 0 if iter_num is None else iter_num
        update_norm = otu.tree_l2_norm(updates)  # Compute update norm
        return (iter_num == 0) | ((iter_num < max_iter) & (update_norm >= tol))

    # Initialize optimizer state.
    init_carry = (init_params, opt.init(init_params), init_params)

    # Run the while loop.
    final_params, final_state, _ = jax.lax.while_loop(continuing_criterion, step, init_carry)

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


class ScaleByBFGSState(NamedTuple):
    """State for BFGS solver."""

    count: int  # Number of iterations
    params: jnp.ndarray  # Previous parameters
    grads: jnp.ndarray  # Current gradient
    value: float  # Current objective value
    inv_hessian: jnp.ndarray  # Current inverse Hessian estimate


def scale_by_bfgs(init_inv_hessian=None) -> GradientTransformation:
    """
    BFGS optimizer for unconstrained optimization.

    Args:
        init_inv_hessian (jnp.ndarray): Initial inverse Hessian approximation (default: identity).

    Returns:
        GradientTransformation: Optax-compatible BFGS optimizer.
    """

    def init_fn(params):
        n = _get_size_of_params(params)
        if init_inv_hessian is None:
            inv_hessian = jnp.eye(n)  # Default to identity
        else:
            assert init_inv_hessian.shape == (
                n,
                n,
            ), 'Hessian shape must by n_param X n_param'
            inv_hessian = init_inv_hessian

        return ScaleByBFGSState(
            count=0,
            params=otu.tree_zeros_like(params),
            grads=otu.tree_zeros_like(params),
            value=0.0,
            inv_hessian=inv_hessian,
        )

    def update_fn(updates, state, params=None, value=None, grad=None, value_fn=None, **kwargs):
        """
        Update the parameters using the BFGS algorithm.

        Args:
            updates: The gradients (assumed to be the updates).
            state: The current optimizer state.
            params: Current parameters (optional).
            value: Current objective value (optional).
            grad: Current gradients (optional).

        Returns:
            Tuple of (updates, new_state).
        """
        grads = updates if grad is None else grad

        # Compute differences
        s_k = otu.tree_sub(params, state.params)
        y_k = otu.tree_sub(updates, state.grads)
        inv_rho_k = otu.tree_vdot(s_k, y_k)
        rho_k = jnp.where(inv_rho_k == 0.0, 0.0, 1.0 / inv_rho_k)
        # Update inverse Hessian using BFGS formula
        H_k = state.inv_hessian

        s_k_flat = jnp.stack(jax.tree.leaves(s_k)).flatten()
        y_k_flat = jnp.stack(jax.tree.leaves(y_k)).flatten()
        outer_sk_yk = jnp.outer(s_k_flat, y_k_flat)
        outer_sk_sk = jnp.outer(s_k_flat, s_k_flat)

        H_k_next = (
            H_k
            + ((rho_k + jnp.vdot(y_k_flat, H_k @ y_k_flat)) * outer_sk_sk / rho_k**2)
            - ((H_k @ outer_sk_yk.T + outer_sk_yk @ H_k) / rho_k)
        )

        # Compute search direction
        grad_flat, struct = jax.tree.flatten(grads)
        direction = -H_k @ jnp.asarray(grad_flat)
        direction = jax.tree.unflatten(struct, direction)

        # Update state
        new_state = ScaleByBFGSState(
            count=state.count + 1,
            params=params,
            grads=grads,
            value=value if value is not None else state.value,
            inv_hessian=H_k_next,
        )

        return direction, new_state

    return GradientTransformation(init_fn, update_fn)


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
