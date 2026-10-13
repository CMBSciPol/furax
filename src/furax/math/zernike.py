r"""Zernike polynomials, orthogonal over the unit disk.

Each polynomial $Z_n^m$ is a radial polynomial of order $n$ times an azimuthal harmonic of frequency
$|m|$, so radial and azimuthal structure can be selected independently. They model smooth patterns
on a disk, such as the focal plane of a telescope.
"""

from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

__all__ = ['zernike']


def zernike(n: int, m: int, x: Float[Array, '*shape'], y: Float[Array, '*shape']) -> Array:
    r"""The Zernike polynomial $Z_n^m$ at the Cartesian points $(x, y)$ of the unit disk.

    In polar coordinates, $Z_n^m(\rho, \phi) = R_n^{|m|}(\rho) \cos(m\phi)$ for $m \ge 0$ and
    $R_n^{|m|}(\rho) \sin(|m|\phi)$ for $m < 0$. The radial polynomial is normalized to
    $R_n^{|m|}(1) = 1$, and written with a Jacobi polynomial of degree $k = (n - |m|) / 2$:

    $$ R_n^{|m|}(\rho) = (-1)^k \rho^{|m|} P_k^{(|m|, 0)}(1 - 2\rho^2). $$

    It is evaluated by the three-term recurrence of the Jacobi polynomials, which keeps single
    precision accurate to a few units in the last place at high orders, where the explicit
    alternating sum of powers of $\rho$ cancels catastrophically.

    Args:
        n: The radial order.
        m: The azimuthal frequency, with $|m| \le n$ and $n - |m|$ even; negative for a sine.
        x: The first Cartesian coordinate.
        y: The second Cartesian coordinate.

    Raises:
        ValueError: If no Zernike polynomial has the indices `(n, m)`.

    Examples:
        >>> x, y = jnp.array([0.0, 0.6]), jnp.array([0.0, 0.8])
        >>> zernike(2, 0, x, y)  # 2ρ² - 1
        Array([-1.,  1.], dtype=float32)
        >>> zernike(1, -1, x, y)  # ρ sin φ = y
        Array([0. , 0.8], dtype=float32)
    """
    a = abs(m)
    if a > n or (n - a) % 2:
        raise ValueError(f'no Zernike polynomial of order (n, m) = ({n}, {m})')
    k = (n - a) // 2
    radial = (-1) ** k * _jacobi(k, a, 1 - 2 * (x**2 + y**2))
    # ρ^|m| cos(|m|φ) and ρ^|m| sin(|m|φ): the real and imaginary parts of (x + iy)^|m|
    re, im = jnp.ones_like(x), jnp.zeros_like(x)
    for _ in range(a):
        re, im = re * x - im * y, re * y + im * x
    return radial * (re if m >= 0 else im)


def _jacobi(k: int, alpha: int, x: Float[Array, '*shape']) -> Float[Array, '*shape']:
    r"""The Jacobi polynomial $P_k^{(\alpha, 0)}(x)$, by its three-term recurrence."""
    previous, current = jnp.ones_like(x), (alpha + 1) + (alpha + 2) * (x - 1) / 2
    if k == 0:
        return previous
    for j in range(2, k + 1):
        c = 2 * j + alpha
        previous, current = (
            current,
            (
                (c - 1) * (c * (c - 2) * x + alpha**2) * current
                - 2 * (j + alpha - 1) * (j - 1) * c * previous
            )
            / (2 * j * (j + alpha) * (c - 2)),
        )
    return current
