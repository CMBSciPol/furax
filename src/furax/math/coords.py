r"""Angle parametrizations of rotations, built on `fastquat.Quaternion`.

Quaternions use scalar-vector storage, i.e. $(1, i, j, k)$ with the scalar part first. All angles
are in radians. The polarization angles assume a rotation $q$ of a local frame looking along its
$z$ axis and sensitive to polarization along its $x$ axis.
"""

from typing import NamedTuple, Self

import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jaxtyping import Array, DTypeLike, Float

__all__ = [
    'XAXIS',
    'YAXIS',
    'ZAXIS',
    'AzElAngles',
    'IsoAngles',
    'LonLatAngles',
    'XiEtaAngles',
    'ZSPhi',
    'euler',
    'gamma_angle',
    'gamma_angle_cos_sin',
    'polarization_angle',
    'polarization_angle_cos_sin',
]

type Angle = Float[Array, '...']

XAXIS = np.array([1.0, 0.0, 0.0])
YAXIS = np.array([0.0, 1.0, 0.0])
ZAXIS = np.array([0.0, 0.0, 1.0])


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


class IsoAngles(NamedTuple):
    r"""The ZYZ Euler angles of a rotation, $q = R_z(\phi) R_y(\theta) R_z(\psi)$.

    For a pointing, $(\theta, \phi)$ locate the direction the detector looks at, and $\psi$ is its
    polarization angle, the angle [`polarization_angle`][] returns. At the poles ($\theta = 0$ or
    $\pi$) only $\phi + \psi$ or $\phi - \psi$ is defined: there, $\phi = 0$, the limit along
    that meridian, so that a rotation round-trips.

    Attributes:
        theta: Co-latitude, from the north pole, in $[0, \pi]$.
        phi: Longitude.
        psi: Polarization angle: the angle from the south of the meridian to the direction of
            polarization sensitivity, measured through the east (COSMO convention).
    """

    theta: Angle
    phi: Angle
    psi: Angle

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Self:
        """The ISO angles of a rotation."""
        a, b, c, d = q.to_components()
        theta = 2 * jnp.atan2((b**2 + c**2) ** 0.5, (a**2 + d**2) ** 0.5)
        phi = jnp.atan2(c * d - a * b, a * c + b * d)
        psi = jnp.atan2(c * d + a * b, a * c - b * d)
        # at a pole, both arctangents above are atan2(0, 0): take phi = 0, and psi = phi + psi
        # (north) or psi - phi (south), the only combination defined there
        north = (b == 0) & (c == 0)
        south = (a == 0) & (d == 0)
        psi = jnp.where(
            north,
            jnp.atan2(2 * a * d, a**2 - d**2),
            jnp.where(south, jnp.atan2(2 * b * c, c**2 - b**2), psi),
        )
        return cls(theta, jnp.where(north | south, 0.0, phi), psi)

    def to_quaternion(self) -> Quaternion:
        """The rotation of these ISO angles."""
        theta, phi, psi = self
        cos_th, sin_th = jnp.cos(theta / 2), jnp.sin(theta / 2)
        cos_pp, sin_pp = jnp.cos((psi + phi) / 2), jnp.sin((psi + phi) / 2)
        cos_pm, sin_pm = jnp.cos((psi - phi) / 2), jnp.sin((psi - phi) / 2)
        return Quaternion(cos_th * cos_pp, sin_th * sin_pm, sin_th * cos_pm, cos_th * sin_pp)


class LonLatAngles(NamedTuple):
    r"""The [`IsoAngles`][] of a rotation, with a latitude instead of a co-latitude.

    Attributes:
        lon: Longitude $\phi$.
        lat: Latitude $\pi/2 - \theta$, in $[-\pi/2, \pi/2]$.
        psi: Polarization angle, as in [`IsoAngles`][].
    """

    lon: Angle
    lat: Angle
    psi: Angle

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Self:
        """The longitude, latitude and polarization angle of a rotation."""
        theta, phi, psi = IsoAngles.from_quaternion(q)
        return cls(phi, jnp.pi / 2 - theta, psi)

    def to_quaternion(self) -> Quaternion:
        """The rotation of these angles."""
        return IsoAngles(jnp.pi / 2 - self.lat, self.lon, self.psi).to_quaternion()


class AzElAngles(NamedTuple):
    r"""The [`LonLatAngles`][] of a rotation in horizon coordinates.

    The azimuth increases from north through east, opposite to the longitude.

    Attributes:
        az: Azimuth $-\phi$.
        el: Elevation $\pi/2 - \theta$, in $[-\pi/2, \pi/2]$.
        psi: Polarization angle, as in [`IsoAngles`][].
    """

    az: Angle
    el: Angle
    psi: Angle

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Self:
        """The azimuth, elevation and polarization angle of a rotation."""
        lon, lat, psi = LonLatAngles.from_quaternion(q)
        return cls(-lon, lat, psi)

    def to_quaternion(self) -> Quaternion:
        """The rotation of these angles."""
        return LonLatAngles(-self.az, self.el, self.psi).to_quaternion()


class XiEtaAngles(NamedTuple):
    r"""The position and orientation of a detector in the focal plane, relative to the boresight.

    For a detector quaternion $q$, with $v = q\,\hat z$ the direction the
    detector looks at in the boresight frame, $\xi = -v_y$ and $\eta = -v_x$ are its orthographic
    coordinates, and $\gamma = \phi + \psi$ its angle about the boresight, see
    [`gamma_angle`][]. A direction 90 degrees or more away from the boresight has no such
    coordinates, so a rotation does not always round-trip.

    Attributes:
        xi: First orthographic coordinate, $-v_y$.
        eta: Second orthographic coordinate, $-v_x$.
        gamma: Angle of the detector about the boresight.
    """

    xi: Angle
    eta: Angle
    gamma: Angle

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Self:
        r"""The $(\xi, \eta, \gamma)$ of a detector quaternion."""
        a, b, c, d = q.to_components()
        return cls(2 * (a * b - c * d), -2 * (a * c + b * d), gamma_angle(q))

    def to_quaternion(self) -> Quaternion:
        """The detector quaternion of these angles."""
        theta = jnp.asin((self.xi**2 + self.eta**2) ** 0.5)
        phi = jnp.atan2(-self.xi, -self.eta)
        return IsoAngles(theta, phi, self.gamma - phi).to_quaternion()


class ZSPhi(NamedTuple):
    r"""A direction on the unit sphere as $(\cos\theta, \sin\theta, \phi)$.

    The co-latitude $\theta$ and longitude $\phi$ of [`IsoAngles`][], with the co-latitude given by
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
    [`XiEtaAngles`][]. For a detector looking opposite the boresight, where $\phi$ is undefined,
    $\phi = 0$ as in [`IsoAngles`][].
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
    [`IsoAngles`][]. The South is that of the meridian of the frame `q` rotates into, so the same
    detector has different polarization angles in, e.g., equatorial and galactic coordinates. At
    the poles of that frame, which have no meridian, it is measured from that of $\phi = 0$, as in
    [`IsoAngles`][]. For the orientation of a detector about the boresight, which sits at such a
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
    sin_pa = v[..., 0] * u[..., 1] - v[..., 1] * u[..., 0]
    cos_pa = -u[..., 2]
    # both vanish only at a pole (v = ±z), where w = 0: there, the south of the meridian phi = 0
    # is v_z x, and its east is y
    at_pole = (sin_pa == 0) & (cos_pa == 0)
    return jnp.arctan2(
        jnp.where(at_pole, u[..., 1], sin_pa), jnp.where(at_pole, v[..., 2] * u[..., 0], cos_pa)
    )


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
