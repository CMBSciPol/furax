import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax.math import bspline


def test_cubic_bspline_support() -> None:
    u = jnp.linspace(-1, 5, 100)
    y = bspline.cubic_bspline(u)
    # support is [0,4)
    assert jnp.all(y[u < 0] == 0.0)
    assert jnp.all(y[u >= 4] == 0.0)
    assert jnp.all(y[(u >= 0) & (u < 4)] >= 0.0)


def test_spline_basis_shape() -> None:
    t = jnp.linspace(0, 10, 200)
    n_knots = 5
    B = bspline.spline_basis(t, n_knots=n_knots)
    # K = n_knots + 2
    assert B.shape == (n_knots + 2, t.size)


@pytest.mark.parametrize('n_knots', [3, 5, 20])
def test_spline_window_rebuilds_dense_basis(n_knots: int) -> None:
    # the 4-knot window, scattered back to (K, N), must equal the dense spline_basis.
    t = jnp.linspace(0, 10, 200)
    dense = bspline.spline_basis(t, n_knots)  # (K, N)
    offset, weights = bspline.spline_window(t, n_knots)  # (N,), (N, 4)

    K, n = dense.shape
    rebuilt = jnp.zeros((K, n))
    rows = offset[:, None] + jnp.arange(4)  # (N, 4) knot indices
    cols = jnp.broadcast_to(jnp.arange(n)[:, None], (n, 4))
    rebuilt = rebuilt.at[rows, cols].add(weights)
    assert_allclose(rebuilt, dense, atol=1e-12)
