import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax.math.lbfgs import run_lbfgs

# Minimise the Rosenbrock function: f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
# Global minimum at (1, 1) with f=0.
# Standard benchmark for gradient-based optimisers.


def rosenbrock(params):
    x, y = params[0], params[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def test_convergence():
    params, _ = run_lbfgs(jnp.array([-1.0, 1.0]), rosenbrock, max_iter=100, tol=1e-6)
    assert_allclose(params, jnp.ones(2), atol=1e-5)


def test_bounds_inactive():
    """When the unconstrained minimum is inside the box, bounds have no effect."""
    lo = jnp.array([-5.0, -5.0])
    up = jnp.array([5.0, 5.0])
    params, _ = run_lbfgs(
        jnp.array([-1.0, 1.0]), rosenbrock, max_iter=100, tol=1e-6, lower_bound=lo, upper_bound=up
    )
    assert_allclose(params, jnp.ones(2), atol=1e-5)


def test_bounds_active_quadratic():
    """When the unconstrained minimum is outside the box, solution is at the boundary."""
    # f(x) = sum((x - center)^2), minimum at center = (-3, -3), outside [1, 5]^2
    center = jnp.array([-3.0, -3.0])
    fun = lambda x: jnp.sum((x - center) ** 2)
    lo = jnp.array([1.0, 1.0])
    up = jnp.array([5.0, 5.0])
    params, _ = run_lbfgs(
        jnp.array([2.0, 2.0]), fun, max_iter=50, tol=1e-6, lower_bound=lo, upper_bound=up
    )
    # Constrained minimum is at the lower corner (1, 1).
    assert_allclose(params, lo, atol=1e-5)
    assert jnp.all(params >= lo)


@pytest.mark.xfail(reason='projected L-BFGS degrades near active bounds; true L-BFGS-B needed')
def test_bounds_active_rosenbrock():
    """Constrained minimum of Rosenbrock on [2,5]^2 is at (2, 4).
    Projected L-BFGS stalls before converging due to corrupted curvature history near x=2.
    """
    lo = jnp.array([2.0, 2.0])
    up = jnp.array([5.0, 5.0])
    params, _ = run_lbfgs(
        jnp.array([3.0, 3.0]), rosenbrock, max_iter=200, tol=1e-6, lower_bound=lo, upper_bound=up
    )
    assert_allclose(params, jnp.array([2.0, 4.0]), atol=1e-4)
