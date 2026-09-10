"""Coordinate-system conversions built on `fastquat.Quaternion`.

Quaternions use scalar-vector storage, i.e. (1,i,j,k) with the scalar part first.
"""

import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import jit
from jaxtyping import Array, Float

__all__ = [
    'XAXIS',
    'YAXIS',
    'ZAXIS',
    'euler',
    'to_iso_angles',
    'from_iso_angles',
    'to_lonlat_angles',
    'from_lonlat_angles',
    'to_xieta_angles',
    'from_xieta_angles',
    'to_gamma_angles',
    'to_polarization_angle_cos_sin',
    'to_polarization_angle',
]

type Angle = Float[Array, '...']

XAXIS = np.array([1.0, 0.0, 0.0])
YAXIS = np.array([0.0, 1.0, 0.0])
ZAXIS = np.array([0.0, 0.0, 1.0])


@jit(static_argnums=(0,))
def euler(axis: int, angle: Angle) -> Quaternion:
    r"""The quaternion representing an Euler rotation.

    For example, if axis=2 the computed quaternion(s) will have components:

    $$
    q = (\cos(\theta/2), 0, 0, \sin(\theta/2))
    $$

    Args:
        axis: The index of the cartesian axis of the rotation (x, y, z).
            Must be 0, 1, or 2.
        angle: Angle of rotation, in radians.

    Returns:
        Quaternion of shape (...).
    """
    angle = jnp.asarray(angle)
    c = jnp.cos(angle / 2)
    s = jnp.sin(angle / 2)
    zeros = jnp.zeros_like(angle)
    components = [c, zeros, zeros, zeros]
    components[axis + 1] = s
    return Quaternion(*components)


@jit
def to_iso_angles(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    """Convert quaternions to the ISO polar coordinate system angles (theta, phi, psi)."""
    a, b, c, d = q.to_components()
    theta = 2 * jnp.atan2((b**2 + c**2) ** 0.5, (a**2 + d**2) ** 0.5)
    phi = jnp.atan2(c * d - a * b, a * c + b * d)
    psi = jnp.atan2(c * d + a * b, a * c - b * d)
    return theta, phi, psi


@jit
def from_iso_angles(theta: Angle, phi: Angle, psi: Angle) -> Quaternion:
    """Compute quaternions from the ISO polar coordinate system angles (theta, phi, psi)."""
    cos_th = jnp.cos(theta * 0.5)
    sin_th = jnp.sin(theta * 0.5)
    cos_pp = jnp.cos((psi + phi) * 0.5)
    sin_pp = jnp.sin((psi + phi) * 0.5)
    cos_pm = jnp.cos((psi - phi) * 0.5)
    sin_pm = jnp.sin((psi - phi) * 0.5)
    return Quaternion(cos_th * cos_pp, sin_th * sin_pm, sin_th * cos_pm, cos_th * sin_pp)


@jit
def to_lonlat_angles(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    """Convert quaternions to the lonlat coordinate system angles (alpha, delta, psi).

    alpha (lon), delta (lat), psi = phi, pi/2-theta, psi
    """
    theta, phi, psi = to_iso_angles(q)
    return phi, jnp.pi / 2 - theta, psi


@jit
def from_lonlat_angles(alpha: Angle, delta: Angle, psi: Angle) -> Quaternion:
    """Compute quaternions from the lonlat coordinate system angles (alpha, delta, psi).

    theta, phi, psi = pi/2-delta, alpha, psi
    """
    return from_iso_angles(jnp.pi / 2 - delta, alpha, psi)


@jit
def to_xieta_angles(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    """Convert quaternions to the xieta coordinate system angles (xi, eta, gamma)."""
    a, b, c, d = q.to_components()
    xi = 2 * (a * b - c * d)
    eta = 2 * (-c * a - d * b)
    gamma = jnp.atan2(2 * a * d, a**2 - d**2)
    return xi, eta, gamma


@jit
def from_xieta_angles(xi: Angle, eta: Angle, gamma: Angle) -> Quaternion:
    """Compute quaternions from the xieta coordinate system angles (xi, eta, gamma)."""
    theta = jnp.asin((xi**2 + eta**2) ** 0.5)
    phi = jnp.atan2(-xi, -eta)
    psi = gamma - phi
    return from_iso_angles(theta, phi, psi)


@jit
def to_gamma_angles(q: Quaternion) -> Angle:
    """Convert quaternions to the xieta coordinate system angles (xi, eta, gamma).

    Only the gamma angle is computed and returned.
    """
    a, _b, _c, d = q.to_components()
    return jnp.atan2(2 * a * d, a**2 - d**2)


@jit
def to_polarization_angle_cos_sin(q: Quaternion) -> tuple[Angle, Angle]:
    """Compute cos and sin of the polarization angle from the rotation quaternion.

    Equivalent to ``(cos(pa), sin(pa))`` where ``pa = to_polarization_angle(q)``,
    but avoids transcendental functions by using quaternion algebra directly.

    See [`to_polarization_angle`][] for the definition and convention.
    """
    a, b, c, d = q.to_components()
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
def to_polarization_angle(q: Quaternion) -> Angle:
    """Compute the polarization angle from the rotation quaternion using the COSMO convention.

    The polarization angle is measured from the South through the East.

    The rotation quaternion `q` transforms detector coordinates to celestial (equatorial)
    coordinates. In detector coordinates:

    - The detector points in the z direction.
    - The detector is sensitive to electric fields in the x direction.

    After applying the rotation:

    - The vector `v` identifies the point on the equatorial sphere where the detector is pointing.
    - The vector `u` defines the polarization-sensitive direction, tangent to the unit sphere
      at `v`.

    The unit vector toward the South in the tangent plane is:

    - `w = -z - (-z . v) v`

    The polarization angle `pa` between `w` and `u` is computed as:

    - `cos(pa) = w . u = -u_z` (since `u . v = 0`)
    - `sin(pa) = (w x u) . v = (u x v) . w = (v x u) . z = v_x u_y - v_y u_x`
    - Therefore, `pa = atan2(v_x u_y - v_y u_x, -u_z)`

    Args:
        q: Rotation quaternion.

    Returns:
        Polarization angle array of shape q.shape.
    """
    v = q.rotate_vector(ZAXIS)
    u = q.rotate_vector(XAXIS)
    return jnp.arctan2(v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0], -u[..., 2])
