import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from numpy.testing import assert_allclose

from furax.math.zernike import zernike, zernike_vander

# NumPy, so that they stay double precision: the tests enable x64 only once collected
X = np.array([0.0, 0.3, -0.5, 0.6, 0.1])
Y = np.array([0.0, 0.4, 0.2, -0.8, -0.7])
R2 = X**2 + Y**2


@pytest.mark.parametrize(
    ('n', 'm', 'expected'),
    [
        (0, 0, np.ones_like(X)),
        (1, -1, Y),
        (1, 1, X),
        (2, -2, 2 * X * Y),
        (2, 0, 2 * R2 - 1),
        (2, 2, X**2 - Y**2),
        (3, -1, (3 * R2 - 2) * Y),
        (3, 3, X**3 - 3 * X * Y**2),
        (4, 0, 6 * R2**2 - 6 * R2 + 1),
    ],
)
def test_closed_forms(n, m, expected):
    assert_allclose(zernike(n, m, jnp.asarray(X), jnp.asarray(Y)), expected, rtol=1e-12, atol=1e-12)


def test_orthogonal_over_the_unit_disk():
    # ∫ Z_n^m Z_n'^m' dA = π ε_m / (2n + 2) δ_nn' δ_mm'; the quadrature is exact for these degrees
    x, y, area = _disk_quadrature()
    modes = _modes(6)
    values = np.stack([np.asarray(zernike(n, m, x, y)) for n, m in modes])
    gram = np.einsum('aij,bij,ij->ab', values, values, area)
    assert_allclose(gram, np.diag(_norms(6)), atol=1e-12)


@pytest.mark.parametrize(('n', 'm'), [(20, 0), (30, 0), (30, -10)])
def test_single_precision_holds_at_high_orders(n, m):
    # the explicit sum of powers of ρ loses every digit in float32 by these orders
    rng = np.random.default_rng(0)
    rho, phi = np.sqrt(rng.uniform(0, 1, 200)), rng.uniform(0, 2 * np.pi, 200)
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    reference = _scipy_zernike(n, m, rho, phi)
    actual = zernike(n, m, jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32))
    assert actual.dtype == jnp.float32
    assert_allclose(actual, reference, atol=1e-5)


@pytest.mark.parametrize(('n', 'm'), [(1, 0), (1, 2), (2, -3)])
def test_rejects_indices_without_a_zernike_polynomial(n, m):
    with pytest.raises(ValueError, match='no Zernike polynomial'):
        zernike(n, m, jnp.asarray(X), jnp.asarray(Y))


def _disk_quadrature():
    """Points `(rho, phi)` and weights exact for the products of Zernike polynomials up to order 6:
    Gauss-Legendre in ρ², uniform in φ, with dA = dρ² dφ / 2."""
    u, w = np.polynomial.legendre.leggauss(16)
    rho2, phi = (u + 1) / 2, np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rho = np.sqrt(rho2)[:, None]
    area = (w / 2)[:, None] * np.full(phi.size, np.pi / phi.size)
    return jnp.asarray(rho * np.cos(phi)), jnp.asarray(rho * np.sin(phi)), area


def _norms(order):
    """The norms ∫ (Z_n^m)² dA = π ε_m / (2n + 2), with ε_0 = 2 and ε_m = 1 otherwise."""
    return [np.pi * (2 if m == 0 else 1) / (2 * n + 2) for n, m in _modes(order)]


def _scipy_zernike(n, m, rho, phi):
    """Z_n^m in double precision, from the Jacobi polynomials of scipy."""
    a, k = abs(m), (n - abs(m)) // 2
    angular = np.cos(a * phi) if m >= 0 else np.sin(a * phi)
    return (-1) ** k * sp.eval_jacobi(k, a, 0, 1 - 2 * rho**2) * rho**a * angular


def _modes(order):
    return [(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)]


@pytest.mark.parametrize('order', [0, 1, 2, 3, 4, 5, 6, 12])
def test_vander_stacks_the_polynomials_in_ansi_order(order):
    expected = np.stack(
        [zernike(n, m, jnp.asarray(X), jnp.asarray(Y)) for n, m in _modes(order)], axis=-1
    )
    vander = zernike_vander(jnp.asarray(X), jnp.asarray(Y), order)
    assert vander.shape == (X.size, (order + 1) * (order + 2) // 2)
    assert_allclose(vander, expected, rtol=1e-12, atol=1e-12)


def test_vander_keeps_the_shape_of_the_points():
    rng = np.random.default_rng(1)
    rho, phi = np.sqrt(rng.uniform(0, 1, (4, 3))), rng.uniform(0, 2 * np.pi, (4, 3))
    x, y = jnp.asarray(rho * np.cos(phi)), jnp.asarray(rho * np.sin(phi))
    vander = zernike_vander(x, y, 4)
    assert vander.shape == (4, 3, 15)
    # every point has the row it would have alone
    flat = zernike_vander(x.ravel(), y.ravel(), 4)
    assert_allclose(vander.reshape(12, 15), flat, atol=1e-14)
    assert_allclose(vander[2, 1], zernike_vander(x[2, 1], y[2, 1], 4), atol=1e-14)


def test_vander_is_orthogonal_over_the_unit_disk():
    x, y, area = _disk_quadrature()
    values = np.asarray(zernike_vander(x, y, 6))  # (rho, phi, mode)
    gram = np.einsum('ijk,ijl,ij->kl', values, values, area)
    assert_allclose(gram, np.diag(_norms(6)), atol=1e-12)


@pytest.mark.parametrize('order', [20, 30])
def test_vander_is_accurate_in_single_precision_at_high_orders(order):
    # the explicit sum of powers of ρ loses every digit in float32 by these orders
    rng = np.random.default_rng(0)
    rho, phi = np.sqrt(rng.uniform(0, 1, 200)), rng.uniform(0, 2 * np.pi, 200)
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    reference = np.stack([_scipy_zernike(n, m, rho, phi) for n, m in _modes(order)], axis=-1)
    actual = zernike_vander(jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32), order)
    assert actual.dtype == jnp.float32
    assert_allclose(actual, reference, atol=1e-5)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_vander_keeps_the_dtype_of_the_points(dtype):
    x, y = jnp.asarray(X, dtype), jnp.asarray(Y, dtype)
    assert zernike_vander(x, y, 5).dtype == dtype


@pytest.mark.parametrize('order', [0, 1, 4, 7])
def test_vander_at_the_centre_is_finite_and_only_the_axisymmetric_modes_survive(order):
    zero = jnp.zeros(2)
    values = zernike_vander(zero, zero, order)
    assert jnp.all(jnp.isfinite(values))
    m = np.array([m for _, m in _modes(order)])
    assert_allclose(values[:, m != 0], 0.0, atol=1e-15)
    # at the centre, Z_n^0 = (-1)^(n/2) for even n and the odd ones do not exist
    assert_allclose(values[0, m == 0], [(-1.0) ** (n // 2) for n, mm in _modes(order) if mm == 0])


def test_vander_has_a_finite_gradient_at_the_centre():
    def total(x):
        return zernike_vander(x, jnp.zeros_like(x), 6).sum()

    assert jnp.all(jnp.isfinite(jax.grad(total)(jnp.zeros(2))))


def test_vander_rejects_a_negative_order():
    with pytest.raises(ValueError, match='non-negative'):
        zernike_vander(jnp.asarray(X), jnp.asarray(Y), -1)
