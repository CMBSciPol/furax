"""Functions for quaternion operations using JAX.

This module provides vectorized functions for multiplying quaternions, and rotating
3D vectors by quaternions.

We use scalar-vector storage, i.e. (1,i,j,k) with the scalar part first.
"""

from functools import partial
from typing import TypeAlias

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float

__all__ = [
    'qmul',
    'qrot',
    'qrot_xaxis',
    'qrot_zaxis',
]

Quat: TypeAlias = Float[Array, '... 4']
Vec3: TypeAlias = Float[Array, '... 3']


@jit
@partial(jnp.vectorize, signature='(4),(4)->(4)')
def qmul(q1: Quat, q2: Quat) -> Quat:
    """Compute quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    # https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])


@jit
@partial(jnp.vectorize, signature='(4),(3)->(3)')
def qrot(q: Quat, vec: Vec3) -> Vec3:
    """Rotate vector by quaternion."""
    rot = to_rotation_matrix(q)
    return rot @ vec  # type: ignore[no-any-return]


@jit
@partial(jnp.vectorize, signature='(4)->(3,3)')
def to_rotation_matrix(q: Float[Array, '*dims 4']) -> Float[Array, '*dims 3 3']:
    """Return 3x3 rotation matrix associated with quaternion q."""
    w, x, y, z = q
    # ref: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    wx = w * x
    wy = w * y
    wz = w * z
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z
    mat = jnp.array(
        [
            [-yy - zz, xy - wz, xz + wy],
            [xy + wz, -xx - zz, yz - wx],
            [xz - wy, yz + wx, -xx - yy],
        ]
    )
    n = jnp.linalg.norm(q)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.eye(3, 3) + s * mat  # type: ignore[no-any-return]


@jit
@partial(jnp.vectorize, signature='(4)->(3)')
def qrot_zaxis(q: Quat) -> Vec3:
    """Rotate the Z axis [0,0,1] by a given quaternion."""
    w, x, y, z = q
    n = jnp.linalg.norm(q)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.array([0, 0, 1]) + s * jnp.array([x * z + w * y, y * z - w * x, -x * x - y * y])  # type: ignore[no-any-return]


@jit
@partial(jnp.vectorize, signature='(4)->(3)')
def qrot_xaxis(q: Quat) -> Vec3:
    """Rotate the X axis [1,0,0] by a given quaternion."""
    w, x, y, z = q
    n = jnp.linalg.norm(q)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.array([1, 0, 0]) + s * jnp.array([-y * y - z * z, x * y + w * z, x * z - w * y])  # type: ignore[no-any-return]
