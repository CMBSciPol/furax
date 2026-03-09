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
    'euler',
    'to_polarization_angle_cos_sin',
    'qmul',
    'qrot',
    'qrot_xaxis',
    'qrot_zaxis',
    'to_polarization_angle',
]

Quat: TypeAlias = Float[Array, '... 4']


Vec3: TypeAlias = Float[Array, '... 3']
Ang3: TypeAlias = Float[Array, '... 3']
Ang: TypeAlias = Float[Array, '...']


@partial(jit, static_argnums=(0,))
def euler(axis: int, angle: Float[Array, '...']) -> Quat:
    """The quaternion representing an Euler rotation.

    For example, if axis=2 the computed quaternion(s) will have
    components:

        q = (cos(angle/2), 0, 0, sin(angle/2))

    Args:
        axis: The index of the cartesian axis of the rotation (x, y, z).
            Must be 0, 1, or 2.
        angle: Angle of rotation, in radians.

    Returns:
        Quaternion array of shape (..., 4).
    """
    angle = jnp.asarray(angle)
    c = jnp.cos(angle / 2)
    s = jnp.sin(angle / 2)
    zeros = jnp.zeros_like(angle)
    components = [c, zeros, zeros, zeros]
    components[axis + 1] = s
    return jnp.stack(components, axis=-1)


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
@partial(jnp.vectorize, signature='(4)->()')
def qnorm(q: Quat) -> Quat:
    """Normalise quaternion."""
    norm2 = jnp.sum(q**2)
    return q / norm2**0.5 if norm2 > 0 else jnp.array([0.0])


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
    n = jnp.sum(q**2)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.eye(3, 3) + s * mat


@jit
@partial(jnp.vectorize, signature='(4)->(3)')
def qrot_zaxis(q: Quat) -> Vec3:
    """Rotate the Z axis [0,0,1] by a given quaternion."""
    w, x, y, z = q
    n = jnp.sum(q**2)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.stack([s * (x * z + w * y), s * (y * z - w * x), 1 - s * (x * x + y * y)])


@jit
@partial(jnp.vectorize, signature='(4)->(3)')
def qrot_xaxis(q: Quat) -> Vec3:
    """Rotate the X axis [1,0,0] by a given quaternion."""
    w, x, y, z = q
    n = jnp.sum(q**2)
    s = jnp.where(n > 1e-8, 2 / n, 0)
    return jnp.stack([1 - s * (y * y + z * z), s * (x * y + w * z), s * (x * z - w * y)])


def unstack_angles(angles: Ang3) -> tuple[Ang, Ang, Ang]:
    return angles[..., 0], angles[..., 1], angles[..., 2]


def stack_angles(angle1: Ang, angle2: Ang, angle3: Ang) -> Ang3:
    return jnp.stack([angle1, angle2, angle3], axis=-1)


@jit
@partial(jnp.vectorize, signature='(4)->(),(),()')
def to_iso_angles(q: Quat) -> tuple[Ang, Ang, Ang]:
    """Convert quaternions to the ISO polar coordinate system angles (theta, phi, psi)."""
    a, b, c, d = q
    theta = 2 * jnp.atan2((b**2 + c**2) ** 0.5, (a**2 + d**2) ** 0.5)
    phi = jnp.atan2(c * d - a * b, a * c + b * d)
    psi = jnp.atan2(c * d + a * b, a * c - b * d)
    return theta, phi, psi


@jit
@partial(jnp.vectorize, signature='(),(),()->(4)')
def from_iso_angles(theta: Ang, phi: Ang, psi: Ang) -> Quat:
    """Compute quaternions from the ISO polar coordinate system angles (theta, phi, psi)."""
    cos_th = jnp.cos(theta * 0.5)
    sin_th = jnp.sin(theta * 0.5)
    cos_pp = jnp.cos((psi + phi) * 0.5)
    sin_pp = jnp.sin((psi + phi) * 0.5)
    cos_pm = jnp.cos((psi - phi) * 0.5)
    sin_pm = jnp.sin((psi - phi) * 0.5)
    return jnp.array([cos_th * cos_pp, sin_th * sin_pm, sin_th * cos_pm, cos_th * sin_pp])


@jit
def to_lonlat_angles(q: Quat) -> tuple[Ang, Ang, Ang]:
    """Convert quaternions to the lonlat coordinate system angles (alpha, delta, psi).
    alpha (lon), delta (lat), psi = phi, pi/2-theta, psi
    """
    theta, phi, psi = to_iso_angles(q)
    return phi, jnp.pi / 2 - theta, psi


@jit
def from_lonlat_angles(alpha: Ang, delta: Ang, psi: Ang) -> Quat:
    """Compute quaternions from the lonlat coordinate system angles (alpha, delta, psi).
    theta, phi, psi = pi/2-delta, alpha, psi
    """
    return from_iso_angles(jnp.pi / 2 - delta, alpha, psi)  # type: ignore[no-any-return]


@jit
@partial(jnp.vectorize, signature='(4)->(),(),()')
def to_xieta_angles(q: Quat) -> tuple[Ang, Ang, Ang]:
    """Convert quaternions to the xieta coordinate system angles (xi, eta, gamma)."""
    a, b, c, d = q
    xi = 2 * (a * b - c * d)
    eta = 2 * (-c * a - d * b)
    gamma = jnp.atan2(2 * a * d, a**2 - d**2)
    return xi, eta, gamma


@jit
def from_xieta_angles(xi: Ang, eta: Ang, gamma: Ang) -> Quat:
    """Compute quaternions from the xieta coordinate system angles (xi, eta, gamma)."""
    theta = jnp.asin((xi**2 + eta**2) ** 0.5)
    phi = jnp.atan2(-xi, -eta)
    psi = gamma - phi
    return from_iso_angles(theta, phi, psi)  # type: ignore[no-any-return]


@jit
@partial(jnp.vectorize, signature='(4)->()')
def to_gamma_angles(q: Quat) -> Ang:
    """Convert quaternions to the xieta coordinate system angles (xi, eta, gamma),
    but only computes and returns the gamma angle."""
    a, b, c, d = q
    gamma = jnp.atan2(2 * a * d, a**2 - d**2)
    return gamma


@jit
@partial(jnp.vectorize, signature='(4)->(),()')
def to_polarization_angle_cos_sin(q: Quat) -> tuple[Ang, Ang]:
    """Compute cos and sin of the polarization angle from the rotation quaternion.

    Equivalent to ``(cos(pa), sin(pa))`` where ``pa = to_polarization_angle(q)``,
    but avoids transcendental functions by using quaternion algebra directly.

    See :func:`to_polarization_angle` for the definition and convention.
    """
    a, b, c, d = q
    cos_theta = a**2 - b**2 - c**2 + d**2
    # clip to avoid numerical issues giving cos_theta**2 > 1
    half_sin_theta = 0.5 * jnp.sqrt(jnp.clip(1 - cos_theta**2, 0.0, None))
    at_pole = half_sin_theta == 0
    safe = jnp.where(at_pole, 1.0, half_sin_theta)
    # angle undefined at the pole, use pa = 0
    # this matches the result of to_polarization_angle (atan2(0, 0) = 0)
    cos_pa = jnp.where(at_pole, 1.0, (a * c - b * d) / safe)
    sin_pa = jnp.where(at_pole, 0.0, (a * b + c * d) / safe)
    return cos_pa, sin_pa


@jit
@partial(jnp.vectorize, signature='(4)->()')
def to_polarization_angle(q: Quat) -> Ang:
    """Compute the polarization angle from the rotation quaternion using the COSMO convention.

    The polarization angle is measured from the South through the East.

    The rotation quaternion `q` transforms detector coordinates to celestial (equatorial) coordinates.
    In detector coordinates:
    - The detector points in the z direction.
    - The detector is sensitive to electric fields in the x direction.

    After applying the rotation:
    - The vector `v` identifies the point on the equatorial sphere where the detector is pointing.
    - The vector `u` defines the polarization-sensitive direction, tangent to the unit sphere at `v`.

    The unit vector toward the South in the tangent plane is:
    - `w = -z - (-z · v) v`

    The polarization angle `pa` between `w` and `u` is computed as:
    - `cos(pa) = w · u = -u_z` (since `u · v = 0`)
    - `sin(pa) = (w × u) · v = (u × v) · w = (v × u) · z = v_x u_y - v_y u_x`
    - Therefore, `pa = atan2(v_x u_y - v_y u_x, -u_z)`

    Args:
        q: Rotation quaternion array of shape (*dims, 4).

    Returns:
        Polarization angle array of shape (*dims).
    """
    v = qrot_zaxis(q)
    u = qrot_xaxis(q)
    return jnp.arctan2(v[0] * u[1] - v[1] * u[0], -u[2])
