from jax import Array
from jax import numpy as jnp
from jaxtyping import Float


def cubic_bspline(u: Float[Array, ' samp']) -> Float[Array, ' samp']:
    """Cubic cardinal B-spline, support [0,4)."""
    u = jnp.asarray(u)
    out = jnp.zeros_like(u)

    out = jnp.where((u >= 0) & (u < 1), (1 / 6) * u**3, out)
    out = jnp.where((u >= 1) & (u < 2), (1 / 6) * (-3 * u**3 + 12 * u**2 - 12 * u + 4), out)
    out = jnp.where((u >= 2) & (u < 3), (1 / 6) * (3 * u**3 - 24 * u**2 + 60 * u - 44), out)
    out = jnp.where((u >= 3) & (u < 4), (1 / 6) * (4 - u) ** 3, out)

    return out


def spline_basis(
    times: Float[Array, ' samp'],
    n_knots: int,
) -> Float[Array, 'k samp']:
    """
    Returns:
        B: (K, N) spline basis matrix
    """
    t_min = jnp.min(times)
    t_max = jnp.max(times)
    t = (times - t_min) / (t_max - t_min + 1e-12)

    K = n_knots + 2
    spacing = 1.0 / (n_knots + 1)

    basis = []
    for j in range(K):
        center = (j + 1) * spacing
        u = (t - center) / spacing + 2.0
        basis.append(cubic_bspline(u))

    return jnp.stack(basis, axis=0)
