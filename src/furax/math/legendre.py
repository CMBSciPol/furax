r"""Legendre polynomials, evaluated by recurrence.

The polynomial $P_n$ of degree $n$ is orthogonal to the lower degrees over $[-1, 1]$. The
three-term recurrence $(n + 1) P_{n+1} = (2n + 1)\, x P_n - n P_{n-1}$ evaluates all the degrees up to
$n$ at the cost of $n$ values per point, and is stable up to high degrees.
"""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

__all__ = ['legvander']


def legvander(x: Float[Array, ' *shape'], deg: int) -> Float[Array, ' *shape {deg+1}']:
    r"""The pseudo-Vandermonde matrix of the Legendre polynomials, as `numpy.polynomial.legendre.legvander`.

    Its columns are the polynomials $P_n(x)$ of the orders $n = 0$ to `deg`, evaluated at the
    points `x`, which must lie in $[-1, 1]$ for the recurrence to be stable.

    Args:
        x: The points, of any shape.
        deg: The highest order.

    Returns:
        The matrix, of shape `(*x.shape, deg + 1)`: one polynomial per entry of the last axis.

    Examples:
        >>> x = jnp.array([-1.0, 0.0, 0.5])
        >>> legvander(x, 2)  # 1, x, (3x² - 1) / 2
        Array([[ 1.   , -1.   ,  1.   ],
               [ 1.   ,  0.   , -0.5  ],
               [ 1.   ,  0.5  , -0.125]], dtype=float32)
    """

    def step(carry, n):
        previous, current = carry
        return (current, ((2 * n + 1) * x * current - n * previous) / (n + 1)), current

    init = (jnp.zeros_like(x), jnp.ones_like(x))
    _, values = jax.lax.scan(step, init, jnp.arange(deg + 1, dtype=x.dtype))
    return jnp.moveaxis(values, 0, -1)
