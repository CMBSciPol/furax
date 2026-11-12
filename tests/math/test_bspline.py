import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline

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


# Between the knots: scipy closes the support at its right end, the cardinal B-spline does not.
U = np.arange(-40, 200) / 40 + 1 / 80


def _scipy_cardinal_bspline(u: np.ndarray, degree: int) -> np.ndarray:
    """The cardinal B-spline of `degree`, on the knots 0, 1, ..., degree + 1, zero outside."""
    element = BSpline.basis_element(np.arange(degree + 2), extrapolate=False)
    return np.nan_to_num(element(u))


@pytest.mark.parametrize('degree', [0, 1, 2, 3])
def test_cardinal_bspline_matches_scipy(degree: int) -> None:
    expected = _scipy_cardinal_bspline(U, degree)
    assert_allclose(bspline.cardinal_bspline(jnp.asarray(U), degree), expected, atol=1e-14)


def test_cardinal_bspline_takes_a_degree_per_element() -> None:
    degree = np.arange(U.size) % 4
    expected = np.choose(degree, [_scipy_cardinal_bspline(U, d) for d in range(4)])
    actual = bspline.cardinal_bspline(jnp.asarray(U), jnp.asarray(degree))
    assert_allclose(actual, expected, atol=1e-14)
