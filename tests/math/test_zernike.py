import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp
from numpy.testing import assert_allclose

from furax.math.zernike import zernike

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
    # ∫ Z_n^m Z_n'^m' dA = π ε_m / (2n + 2) δ_nn' δ_mm', with ε_0 = 2 and ε_m = 1 otherwise; the
    # polar quadrature (Gauss-Legendre in ρ², uniform in φ) is exact for these degrees
    u, w = np.polynomial.legendre.leggauss(16)
    rho2, phi = (u + 1) / 2, np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rho = np.sqrt(rho2)[:, None]
    x, y = jnp.asarray(rho * np.cos(phi)), jnp.asarray(rho * np.sin(phi))
    area = (w / 2)[:, None] * np.full(phi.size, np.pi / phi.size)  # dA = dρ² dφ / 2
    modes = [(n, m) for n in range(7) for m in range(-n, n + 1, 2)]
    values = np.stack([np.asarray(zernike(n, m, x, y)) for n, m in modes])
    gram = np.einsum('aij,bij,ij->ab', values, values, area)
    norms = [np.pi * (2 if m == 0 else 1) / (2 * n + 2) for n, m in modes]
    assert_allclose(gram, np.diag(norms), atol=1e-12)


@pytest.mark.parametrize(('n', 'm'), [(20, 0), (30, 0), (30, -10)])
def test_single_precision_holds_at_high_orders(n, m):
    # the explicit sum of powers of ρ loses every digit in float32 by these orders
    rng = np.random.default_rng(0)
    rho, phi = np.sqrt(rng.uniform(0, 1, 200)), rng.uniform(0, 2 * np.pi, 200)
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    a, k = abs(m), (n - abs(m)) // 2
    angular = np.cos(a * phi) if m >= 0 else np.sin(a * phi)
    reference = (-1) ** k * sp.eval_jacobi(k, a, 0, 1 - 2 * rho**2) * rho**a * angular
    actual = zernike(n, m, jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32))
    assert actual.dtype == jnp.float32
    assert_allclose(actual, reference, atol=1e-5)


@pytest.mark.parametrize(('n', 'm'), [(1, 0), (1, 2), (2, -3)])
def test_rejects_indices_without_a_zernike_polynomial(n, m):
    with pytest.raises(ValueError, match='no Zernike polynomial'):
        zernike(n, m, jnp.asarray(X), jnp.asarray(Y))
