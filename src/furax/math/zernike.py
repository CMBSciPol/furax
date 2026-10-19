r"""Zernike polynomials.

The Zernike polynomials $Z_n^m(\rho, \phi)$ form an orthogonal basis of the functions on the unit
disk, in polar coordinates $(\rho, \phi)$. Each is a radial polynomial of degree $n$ times an
azimuthal harmonic of frequency $|m|$, so the radial and the azimuthal structure are indexed
separately: $n \ge 0$, and $m$ with $|m| \le n$ and $n - |m|$ even. They are polynomials of
degree $n$ in the Cartesian coordinates $(x, y) = \rho (\cos\phi, \sin\phi)$, and orthogonal for the
area measure:

$$ \int_0^{2\pi} \int_0^1 Z_n^m Z_{n'}^{m'} \rho \, d\rho \, d\phi
   = \frac{\pi (1 + \delta_{m0})}{2n + 2} \, \delta_{nn'} \delta_{mm'}. $$
"""

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float

__all__ = [
    'zernike',
    'zernike_vander',
]


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

    Warning:
        The recurrence is a Python loop, unrolled when traced under `jax.jit`: the compilation
        time grows with the order, so this functions get slow to compile beyond order 10 or so.
        Use [`zernike_vander`][] to circumvent this limitation.

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
    z = (x + 1j * y) ** a
    return radial * (z.real if m >= 0 else z.imag)


def zernike_vander(
    x: Float[Array, ' *shape'], y: Float[Array, ' *shape'], order: int
) -> Float[Array, ' *shape {(order+1)*(order+2)//2}']:
    r"""The Zernike polynomials of every order up to `order`, in the layout of a Vandermonde matrix.

    The columns are the $Z_n^m$ of the standard (ANSI) order: $n = 0$ to `order`, and for each $n$,
    $m = -n, -n + 2, \dots, n$. There are $(\text{order} + 1)(\text{order} + 2) / 2$ of them.

    The polynomials of one azimuthal frequency $|m|$ share a recurrence, which gives all their
    radial degrees at once. The recurrences of all the frequencies run together in a single scan,
    so the traced program has the same size at any order.

    Args:
        x: The first Cartesian coordinate of the points, of any shape.
        y: The second Cartesian coordinate.
        order: The highest radial order.

    Returns:
        The polynomials, of shape `(*x.shape, n_modes)`.

    Raises:
        ValueError: If `order` is negative.

    Examples:
        >>> x, y = jnp.array([0.0, 0.6]), jnp.array([0.0, 0.8])
        >>> zernike_vander(x, y, 1)  # 1, y, x
        Array([[1. , 0. , 0. ],
               [1. , 0.8, 0.6]], dtype=float32)
    """
    if order < 0:
        raise ValueError(f'the order must be non-negative, got {order}')
    modes = np.array([(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)])
    n, m = modes.T
    frequency = np.abs(m)
    degree = (n - frequency) // 2
    radial = _jacobi_table(order, 1 - 2 * (x**2 + y**2))[..., degree, frequency]
    re, im = _powers(x, y, order)
    azimuth = jnp.where(m >= 0, re[..., frequency], im[..., frequency])
    return jnp.asarray((-1.0) ** degree, x.dtype) * radial * azimuth


def _jacobi_table(order: int, x: Float[Array, ' *shape']) -> Float[Array, ' *shape degree freq']:
    r"""The Jacobi polynomials $P_k^{(\alpha, 0)}(x)$ for every $\alpha \le$ `order` and $k \le$ `order / 2`."""
    alpha = jnp.arange(order + 1, dtype=x.dtype)  # one recurrence per frequency, along a new axis
    u = x[..., None]  # the points, with an axis to broadcast against the frequencies
    first, second = jnp.ones_like(u * alpha), _jacobi_second(alpha, u)

    def step(carry, j):
        previous, current = carry
        new = _jacobi_next(previous, current, j, alpha, u)
        return (current, new), new

    degrees = jnp.arange(2, order // 2 + 1, dtype=x.dtype)
    _, rest = jax.lax.scan(step, (first, second), degrees)
    table = jnp.concatenate([first[None], second[None], rest])[: order // 2 + 1]
    return jnp.moveaxis(table, 0, -2)


def _powers(
    x: Float[Array, ' *shape'], y: Float[Array, ' *shape'], order: int
) -> tuple[Array, Array]:
    r"""The real and imaginary parts of $(x + iy)^a$ for $a \le$ `order`, in the last axis."""
    z = (x + 1j * y)[..., None]
    ones = jnp.ones((*x.shape, 1), z.dtype)
    powers = jnp.cumprod(jnp.concatenate([ones, jnp.broadcast_to(z, (*x.shape, order))], -1), -1)
    return powers.real, powers.imag


def _jacobi(k: int, alpha: int, x: Float[Array, '*shape']) -> Float[Array, '*shape']:
    r"""The Jacobi polynomial $P_k^{(\alpha, 0)}(x)$, by its three-term recurrence."""
    previous, current = jnp.ones_like(x), _jacobi_second(alpha, x)
    if k == 0:
        return previous
    for j in range(2, k + 1):
        previous, current = current, _jacobi_next(previous, current, j, alpha, x)
    return current


def _jacobi_second(alpha: Array | int, x: Array) -> Array:
    r"""The Jacobi polynomial $P_1^{(\alpha, 0)}(x)$, which starts the recurrence after $P_0 = 1$."""
    return (alpha + 1) + (alpha + 2) * (x - 1) / 2


def _jacobi_next(
    previous: Array, current: Array, j: Array | int, alpha: Array | int, x: Array
) -> Array:
    r"""The Jacobi polynomial $P_j^{(\alpha, 0)}$ from $P_{j-2}$ and $P_{j-1}$.

    The three-term recurrence, with $c = 2j + \alpha$:

    $$ 2j(j + \alpha)(c - 2)\, P_j = (c - 1)\left[c(c - 2)\, x + \alpha^2\right] P_{j-1}
       - 2(j + \alpha - 1)(j - 1)\, c\, P_{j-2}. $$
    """
    c = 2 * j + alpha
    return (
        (c - 1) * (c * (c - 2) * x + alpha**2) * current
        - 2 * (j + alpha - 1) * (j - 1) * c * previous
    ) / (2 * j * (j + alpha) * (c - 2))
