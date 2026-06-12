import jax.numpy as jnp
from jax import Array

from furax.math import bspline


def test_cubic_bspline_support() -> None:
    u: Array = jnp.linspace(-1, 5, 100)
    y: Array = bspline.cubic_bspline(u)
    # support is [0,4)
    assert jnp.all(y[u < 0] == 0.0)
    assert jnp.all(y[u >= 4] == 0.0)
    assert jnp.all(y[(u >= 0) & (u < 4)] >= 0.0)


def test_spline_basis_shape() -> None:
    t: Array = jnp.linspace(0, 10, 200)
    n_knots = 5
    B: Array = bspline.spline_basis(t, n_knots=n_knots)
    # K = n_knots + 2
    assert B.shape == (n_knots + 2, t.size)
