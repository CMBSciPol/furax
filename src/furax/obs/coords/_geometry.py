"""Directions in the horizon frame (North, West, Up) and their tangent planes."""

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

__all__ = [
    'azel_to_vec',
    'tangent_basis',
    'vec_to_azel',
]


def azel_to_vec(az: ArrayLike, el: ArrayLike) -> Float[Array, '... 3']:
    """The unit vectors in the horizon frame pointing at (az, el).

    Args:
        az: Azimuth, from North through East.
        el: Elevation, from the horizon.

    Returns:
        Array of shape (..., 3), the broadcast shape of the inputs followed by the vector axis.
    """
    az, el = jnp.asarray(az), jnp.asarray(el)
    cos_el = jnp.cos(el)
    return jnp.stack([cos_el * jnp.cos(az), -cos_el * jnp.sin(az), jnp.sin(el)], axis=-1)


def vec_to_azel(vec: ArrayLike) -> tuple[Float[Array, '...'], Float[Array, '...']]:
    r"""The azimuth, in $(-\pi, \pi]$, and elevation of vectors in the horizon frame.

    The vectors, of shape (..., 3), need not be normalized.
    """
    vec = jnp.asarray(vec)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return jnp.arctan2(-y, x), jnp.arctan2(z, jnp.hypot(x, y))


def tangent_basis(
    az: ArrayLike, el: ArrayLike
) -> tuple[Float[Array, '... 3'], Float[Array, '... 3']]:
    r"""The unit vectors of increasing azimuth (cross-elevation) and elevation at (az, el).

    Returns:
        A tuple `(e_xel, e_el)` of arrays of shape (..., 3), which together with
        `v = azel_to_vec(az, el)` form an orthonormal basis, with $e_\mathrm{el} \times
        e_\mathrm{xel} = v$.
    """
    az, el = jnp.broadcast_arrays(az, el)
    sin_az, cos_az = jnp.sin(az), jnp.cos(az)
    sin_el, cos_el = jnp.sin(el), jnp.cos(el)
    e_xel = jnp.stack([-sin_az, -cos_az, jnp.zeros_like(az)], axis=-1)
    e_el = jnp.stack([-sin_el * cos_az, sin_el * sin_az, cos_el], axis=-1)
    return e_xel, e_el
