r"""Angle parametrizations of rotations, built on `fastquat.Quaternion`.

Quaternions use scalar-vector storage, i.e. $(1, i, j, k)$ with the scalar part first. All angles
are in radians. The polarization angles assume a rotation $q$ of a local frame looking along its
$z$ axis and sensitive to polarization along its $x$ axis.
"""

from typing import NamedTuple, Self

import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jaxtyping import Array, ArrayLike, DTypeLike, Float

__all__ = [
    'XAXIS',
    'YAXIS',
    'ZAXIS',
    'ZSPhi',
    'azel_to_quat',
    'euler',
    'gamma_angle',
    'gamma_angle_cos_sin',
    'iso_to_quat',
    'lonlat_to_quat',
    'polarization_angle',
    'polarization_angle_cos_sin',
    'quat_to_azel',
    'quat_to_iso',
    'quat_to_lonlat',
    'quat_to_xieta',
    'xieta_to_quat',
]

type Angle = Float[Array, '...']

XAXIS = np.array([1.0, 0.0, 0.0])
YAXIS = np.array([0.0, 1.0, 0.0])
ZAXIS = np.array([0.0, 0.0, 1.0])


def euler(axis: int, angle: ArrayLike) -> Quaternion:
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


def iso_to_quat(theta: ArrayLike, phi: ArrayLike, psi: ArrayLike = 0.0) -> Quaternion:
    r"""The rotation of ZYZ Euler angles, $q = R_z(\phi) R_y(\theta) R_z(\psi)$.

    For a pointing, $(\theta, \phi)$ locate the direction the detector looks at, and $\psi$ is its
    polarization angle, the angle [`polarization_angle`][] returns.

    Args:
        theta: Co-latitude, from the north pole, in $[0, \pi]$.
        phi: Longitude.
        psi: Polarization angle: the angle from the south of the meridian to the direction of
            polarization sensitivity, measured through the east (COSMO convention).
    """
    theta, phi, psi = jnp.asarray(theta), jnp.asarray(phi), jnp.asarray(psi)
    cos_th, sin_th = jnp.cos(theta / 2), jnp.sin(theta / 2)
    cos_pp, sin_pp = jnp.cos((psi + phi) / 2), jnp.sin((psi + phi) / 2)
    cos_pm, sin_pm = jnp.cos((psi - phi) / 2), jnp.sin((psi - phi) / 2)
    return Quaternion(cos_th * cos_pp, sin_th * sin_pm, sin_th * cos_pm, cos_th * sin_pp)


def quat_to_iso(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    r"""The ZYZ Euler angles $(\theta, \phi, \psi)$ of a rotation, see [`iso_to_quat`][].

    At the poles ($\theta = 0$ or $\pi$) only $\phi + \psi$ or $\phi - \psi$ is defined: there,
    $\phi = 0$, the limit along that meridian, so that a rotation round-trips.
    """
    a, b, c, d = q.to_components()
    theta = 2 * jnp.atan2((b**2 + c**2) ** 0.5, (a**2 + d**2) ** 0.5)
    phi = jnp.atan2(c * d - a * b, a * c + b * d)
    # at a pole, phi is atan2(0, 0): take phi = 0, from whose meridian polarization_angle
    # measures psi there
    pole = ((b == 0) & (c == 0)) | ((a == 0) & (d == 0))
    return theta, jnp.where(pole, 0.0, phi), polarization_angle(q)


def lonlat_to_quat(lon: ArrayLike, lat: ArrayLike, psi: ArrayLike = 0.0) -> Quaternion:
    r"""The rotation of [`iso_to_quat`][], with a latitude instead of a co-latitude.

    Args:
        lon: Longitude $\phi$.
        lat: Latitude $\pi/2 - \theta$, in $[-\pi/2, \pi/2]$.
        psi: Polarization angle, as in [`iso_to_quat`][].
    """
    return iso_to_quat(jnp.pi / 2 - jnp.asarray(lat), lon, psi)


def quat_to_lonlat(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    """The longitude, latitude and polarization angle of a rotation, see [`lonlat_to_quat`][]."""
    theta, phi, psi = quat_to_iso(q)
    return phi, jnp.pi / 2 - theta, psi


def azel_to_quat(az: ArrayLike, el: ArrayLike, psi: ArrayLike = 0.0) -> Quaternion:
    r"""The rotation of [`lonlat_to_quat`][] in horizon coordinates.

    The azimuth increases from north through east, opposite to the longitude.

    Args:
        az: Azimuth $-\phi$.
        el: Elevation $\pi/2 - \theta$, in $[-\pi/2, \pi/2]$.
        psi: Polarization angle, as in [`iso_to_quat`][]. For the boresight of a telescope, its
            roll angle.
    """
    return lonlat_to_quat(-jnp.asarray(az), el, psi)


def quat_to_azel(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    """The azimuth, elevation and polarization angle of a rotation, see [`azel_to_quat`][]."""
    lon, lat, psi = quat_to_lonlat(q)
    return -lon, lat, psi


def xieta_to_quat(xi: ArrayLike, eta: ArrayLike, gamma: ArrayLike = 0.0) -> Quaternion:
    r"""The detector quaternion at a position and orientation in the focal plane.

    For a detector quaternion $q$, with $v = q\,\hat z$ the direction the detector looks at in the
    boresight frame, $\xi = -v_y$ and $\eta = -v_x$ are its orthographic coordinates, and
    $\gamma = \phi + \psi$ its angle about the boresight, see [`gamma_angle`][]. The quaternion is
    NaN where $\xi^2 + \eta^2 > 1$.

    Args:
        xi: First orthographic coordinate, $-v_y$.
        eta: Second orthographic coordinate, $-v_x$.
        gamma: Angle of the detector about the boresight.
    """
    # R_z(phi) R_y(theta) R_z(gamma - phi): the rotation by theta = asin(r), with
    # r^2 = xi^2 + eta^2, about the axis (xi, -eta, 0) / r, then by gamma about z. Written
    # without asin and atan2, whose derivatives are infinite at the boresight.
    xi, eta = jnp.broadcast_arrays(xi, eta)
    cos_theta = jnp.sqrt(1 - xi**2 - eta**2)
    k = 1 / jnp.sqrt(2 * (1 + cos_theta))  # sin(theta / 2) / r
    offset = Quaternion(jnp.sqrt((1 + cos_theta) / 2), xi * k, -eta * k, jnp.zeros_like(xi))
    return offset * euler(2, gamma)


def quat_to_xieta(q: Quaternion) -> tuple[Angle, Angle, Angle]:
    r"""The $(\xi, \eta, \gamma)$ of a detector quaternion, see [`xieta_to_quat`][].

    A direction 90 degrees or more away from the boresight has no such coordinates, so there a
    rotation does not round-trip.
    """
    a, b, c, d = q.to_components()
    return 2 * (a * b - c * d), -2 * (a * c + b * d), gamma_angle(q)


class ZSPhi(NamedTuple):
    r"""A direction on the unit sphere as $(\cos\theta, \sin\theta, \phi)$.

    The co-latitude $\theta$ and longitude $\phi$ of [`iso_to_quat`][], with the co-latitude given by
    its cosine $z$ and sine $s$. It is the form HEALPix pixel centres come in, and the spin-2
    transport takes, so that neither needs to evaluate $\theta$.

    Attributes:
        z: $\cos\theta$, the $z$ coordinate of the direction.
        s: $\sin\theta$.
        phi: Longitude $\phi$.
    """

    z: Angle
    s: Angle
    phi: Angle

    @classmethod
    def from_angles(cls, theta: Angle, phi: Angle) -> Self:
        """The direction at co-latitude `theta` and longitude `phi`."""
        return cls(jnp.cos(theta), jnp.sin(theta), phi)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Self:
        r"""The direction $q\,\hat z$ of a unit quaternion, with a single arctangent."""
        a, b, c, d = q.to_components()
        p, m = a**2 + d**2, b**2 + c**2
        return cls(p - m, 2 * jnp.sqrt(p * m), jnp.atan2(c * d - a * b, a * c + b * d))

    def astype(self, dtype: DTypeLike) -> Self:
        """The direction cast to the given floating-point type."""
        return type(self)(*(component.astype(dtype) for component in self))


def gamma_angle(q: Quaternion) -> Angle:
    r"""The angle $\gamma = \phi + \psi$ of a rotation about the $z$ axis.

    For a detector quaternion, the orientation of the detector about the boresight, see
    [`xieta_to_quat`][]. For a detector looking opposite the boresight, where $\phi$ is undefined,
    $\phi = 0$ as in [`iso_to_quat`][].
    """
    a, b, c, d = q.to_components()
    opposite = (a == 0) & (d == 0)
    return jnp.where(opposite, jnp.atan2(2 * b * c, c**2 - b**2), jnp.atan2(2 * a * d, a**2 - d**2))


def gamma_angle_cos_sin(q: Quaternion) -> tuple[Angle, Angle]:
    r"""$(\cos\gamma, \sin\gamma)$ of [`gamma_angle`][], without trigonometric functions."""
    a, b, c, d = q.to_components()
    opposite = (a == 0) & (d == 0)
    # (a, d) or, looking opposite the boresight, (c, b): the half-angle of gamma
    x, y = jnp.where(opposite, c, a), jnp.where(opposite, b, d)
    norm = x**2 + y**2
    safe = jnp.where(norm == 0, 1.0, norm)
    return jnp.where(norm == 0, 1.0, (x**2 - y**2) / safe), jnp.where(
        norm == 0, 0.0, 2 * x * y / safe
    )


def polarization_angle(q: Quaternion) -> Angle:
    r"""Compute the polarization angle from the rotation quaternion using the COSMO convention.

    The polarization angle is measured from the South through the East. It is the angle $\psi$ of
    [`iso_to_quat`][]. The South is that of the meridian of the frame `q` rotates into, so the same
    detector has different polarization angles in, e.g., equatorial and galactic coordinates. At
    the poles of that frame, which have no meridian, it is measured from that of $\phi = 0$, as in
    [`iso_to_quat`][]. For the orientation of a detector about the boresight, which sits at such a
    pole, see [`gamma_angle`][].

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

    The polarization angle `pa` between `w` and `u` is given by:

    - `cos(pa) = w . u = -u_z` (since `u . v = 0`)
    - `sin(pa) = (w x u) . v = (u x v) . w = (v x u) . z = v_x u_y - v_y u_x`
    - Therefore, `pa = atan2(v_x u_y - v_y u_x, -u_z)`

    Args:
        q: Rotation quaternion.

    Returns:
        Polarization angle array of shape q.shape.
    """
    cos_pa, sin_pa = polarization_angle_cos_sin(q)
    return jnp.atan2(sin_pa, cos_pa)


def polarization_angle_cos_sin(q: Quaternion) -> tuple[Angle, Angle]:
    """Compute cos and sin of the polarization angle from the rotation quaternion.

    Equivalent to `(cos(pa), sin(pa))` where `pa = polarization_angle(q)`, but avoids
    transcendental functions by using quaternion algebra directly.

    See [`polarization_angle`][] for the definition and convention.
    """
    a, b, c, d = q.to_components()
    # sin(theta) / 2, as in `ZSPhi.from_quaternion`: exactly zero at a pole, where b = c = 0 or
    # a = d = 0
    half_sin_theta = jnp.sqrt((a**2 + d**2) * (b**2 + c**2))
    at_pole = half_sin_theta == 0
    safe = jnp.where(at_pole, 1.0, half_sin_theta)
    # at the pole, the angle from the meridian phi = 0: that of (a, d) at the north pole, of (c, b)
    # at the south pole, where the other pair vanishes
    cos_pa = jnp.where(at_pole, a**2 - d**2 + c**2 - b**2, (a * c - b * d) / safe)
    sin_pa = jnp.where(at_pole, 2 * (a * d + b * c), (a * b + c * d) / safe)
    return cos_pa, sin_pa
