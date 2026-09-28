"""Pointing models for alt-azimuthal mounts"""

import dataclasses
from abc import abstractmethod
from typing import Any, NamedTuple, Self

import equinox as eqx
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, ArrayLike, Float

from ..coords import ZAXIS, XiEtaAngles, euler

__all__ = [
    'AbstractPointingModel',
    'BasicPointingModel',
    'Observations',
    'SATV1PointingModel',
    'SATV2PointingModel',
]


class AbstractPointingModel(eqx.Module):
    @classmethod
    def names(cls) -> tuple[str, ...]:
        """The names of the parameters."""
        return tuple(
            field.name for field in dataclasses.fields(cls) if not field.metadata.get('static')
        )

    def to_vector(self) -> Float[Array, ' n_params']:
        """The parameters as a vector, in the order of `names()`."""
        return jnp.stack([getattr(self, name) for name in self.names()])

    @classmethod
    def from_vector(cls, vector: ArrayLike) -> Self:
        """The parameters from a vector, in the order of `names()`."""
        return cls(*jnp.asarray(vector))

    @abstractmethod
    def quaternion(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Quaternion:
        """The rotation from the telescope frame to the horizon frame.

        Args:
            az: Azimuth encoder readings.
            el: Elevation encoder readings.
            roll: Roll encoder readings.

        Returns:
            Quaternion of the broadcast shape of the inputs.
        """

    def direction(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Float[Array, '... 3']:
        """The unit vectors in the horizon frame along the optical axis.

        The arguments are those of `quaternion`.
        """
        return self.quaternion(az, el, roll).rotate_vector(ZAXIS)


def _eqx_array_field(default: float = 0.0) -> Any:
    """An equinox module field converted to a float array on construction."""
    return eqx.field(default=default, converter=lambda val: jnp.asarray(val, dtype=float))


class BasicPointingModel(AbstractPointingModel):
    r"""A basic telescope pointing model.

    This model only takes into account encoder zero points and collimation error. For encoder
    readings (az, el, roll), the rotation from the telescope frame to the horizon frame is

    $$
    q = R_z(-(\mathrm{az} + \mathrm{az_0}))\,R_y(\pi/2 - (\mathrm{el} + \mathrm{el_0}))\,
        X(\mathrm{col}_\xi, \mathrm{col}_\eta)\,R_z(\mathrm{roll} + \mathrm{roll_0}),
    $$

    where $X(\xi, \eta)$ takes the centre of the focal plane to $(\xi, \eta)$, see
    [`XiEtaAngles`][]. The collimation error is fixed to the mount: it does not turn with the
    boresight rotation. It is the `fp_offset_{xi,eta}0` of [`SATV1PointingModel`][].

    Attributes:
        az0: Azimuth encoder zero point.
        el0: Elevation encoder zero point.
        roll0: Roll (boresight rotation) encoder zero point.
        col_xi: Collimation: $\xi$ of the boresight, relative to the optical axis.
        col_eta: Collimation: $\eta$ of the boresight, relative to the optical axis.
    """

    az0: Float[Array, '...'] = _eqx_array_field()
    el0: Float[Array, '...'] = _eqx_array_field()
    roll0: Float[Array, '...'] = _eqx_array_field()
    col_xi: Float[Array, '...'] = _eqx_array_field()
    col_eta: Float[Array, '...'] = _eqx_array_field()

    def quaternion(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Quaternion:
        az, el, roll = jnp.asarray(az), jnp.asarray(el), jnp.asarray(roll)
        return (
            euler(2, -(az + self.az0))
            * euler(1, jnp.pi / 2 - (el + self.el0))
            * XiEtaAngles(self.col_xi, self.col_eta).to_quaternion()
            * euler(2, roll + self.roll0)
        )


def _base_tilt(c: Array, s: Array) -> Quaternion:
    """The base tilt of TPOINT parameters AN = `c` and AW = `s`: the rotation vector (-s, c, 0)."""
    return Quaternion.from_rotation_vector(jnp.stack([-s, c, jnp.zeros_like(c)], axis=-1))


class SATV1PointingModel(AbstractPointingModel):
    r"""The pointing model of the Simons Observatory small aperture telescopes, `sat_v1`.

    A port of `sotodlib.coords.pointing_model.model_sat_v1`, with the same parameters and sign
    conventions, so that SO parameters can be used as they are. The roll is that of sotodlib's
    SAT boresight, minus the boresight encoder reading. The encoder readings are corrected to

    $$
    \begin{aligned}
    \mathrm{az}' &= \mathrm{az} + \mathrm{enc\_offset\_az} + \mathrm{az\_rot}\,\mathrm{el}'
        + \mathrm{acec} \cos\mathrm{az} - \mathrm{aces} \sin\mathrm{az}, \\
    \mathrm{el}' &= \mathrm{el} + \mathrm{enc\_offset\_el}, \\
    \mathrm{roll}' &= \mathrm{roll} - \mathrm{enc\_offset\_boresight},
    \end{aligned}
    $$

    and the rotation from the telescope frame to the horizon frame is

    $$
    q = H(\mathrm{az}', \mathrm{el}')\,B\,R_z(-\mathrm{az}')\,R_y(\pi/2 - \mathrm{el}')\,
        X_\mathrm{offset}\,X_\mathrm{rot}\,R_z(\mathrm{roll}')\,X_\mathrm{rot}^{-1},
    $$

    where $X_\mathrm{offset}$ and $X_\mathrm{rot}$ take the centre of the focal plane to
    `fp_offset_{xi,eta}0` and `fp_rot_{xi,eta}0`, $B$ is the base tilt and $H$ its second-order
    azimuth harmonics.

    Attributes:
        enc_offset_az: Azimuth encoder offset.
        enc_offset_el: Elevation encoder offset.
        enc_offset_boresight: Boresight rotation encoder offset.
        fp_offset_xi0: Collimation: $\xi$ of the centre of the boresight rotation, relative to the
            optical axis. It is fixed to the mount.
        fp_offset_eta0: Collimation: $\eta$ of that offset.
        fp_rot_xi0: $\xi$ of the centre of the boresight rotation in the focal plane.
        fp_rot_eta0: $\eta$ of the centre of the boresight rotation in the focal plane.
        az_rot: Linear dependence of the azimuth on the elevation, dimensionless.
        base_tilt_cos: Base tilt, TPOINT AN.
        base_tilt_sin: Base tilt, TPOINT AW.
        harmonic_2el_sin: Second-order elevation harmonic, TPOINT HESA2.
        harmonic_2el_cos: Second-order elevation harmonic, TPOINT HECA2.
        harmonic_2az_sin: Second-order azimuth harmonic, TPOINT HASA2.
        harmonic_2az_cos: Second-order azimuth harmonic, TPOINT HACA2.
        acec: Azimuth centering error, cosine term.
        aces: Azimuth centering error, sine term.
    """

    enc_offset_az: Float[Array, '...'] = _eqx_array_field()
    enc_offset_el: Float[Array, '...'] = _eqx_array_field()
    enc_offset_boresight: Float[Array, '...'] = _eqx_array_field()
    fp_offset_xi0: Float[Array, '...'] = _eqx_array_field()
    fp_offset_eta0: Float[Array, '...'] = _eqx_array_field()
    fp_rot_xi0: Float[Array, '...'] = _eqx_array_field()
    fp_rot_eta0: Float[Array, '...'] = _eqx_array_field()
    az_rot: Float[Array, '...'] = _eqx_array_field()
    base_tilt_cos: Float[Array, '...'] = _eqx_array_field()
    base_tilt_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_2el_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_2el_cos: Float[Array, '...'] = _eqx_array_field()
    harmonic_2az_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_2az_cos: Float[Array, '...'] = _eqx_array_field()
    acec: Float[Array, '...'] = _eqx_array_field()
    aces: Float[Array, '...'] = _eqx_array_field()

    def quaternion(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Quaternion:
        az, el, roll = jnp.asarray(az), jnp.asarray(el), jnp.asarray(roll)
        centering = self.acec * jnp.cos(az) - self.aces * jnp.sin(az)
        el = el + self.enc_offset_el
        az = az + self.enc_offset_az + self.az_rot * el + centering
        roll = roll - self.enc_offset_boresight

        # second-order base tilt, at the corrected azimuth
        cos_2az, sin_2az = jnp.cos(2 * az), jnp.sin(2 * az)
        delta_az = self.harmonic_2az_cos * cos_2az + self.harmonic_2az_sin * sin_2az
        delta_el = self.harmonic_2el_cos * cos_2az + self.harmonic_2el_sin * sin_2az
        harmonics = euler(2, az) * euler(1, delta_el) * euler(2, -az) * euler(2, delta_az)

        fp_rot = XiEtaAngles(self.fp_rot_xi0, self.fp_rot_eta0).to_quaternion()
        return (
            harmonics
            * _base_tilt(self.base_tilt_cos, self.base_tilt_sin)
            * euler(2, -az)
            * euler(1, jnp.pi / 2 - el)
            * XiEtaAngles(self.fp_offset_xi0, self.fp_offset_eta0).to_quaternion()
            * fp_rot
            * euler(2, roll)
            * fp_rot.conj()
        )


class SATV2PointingModel(AbstractPointingModel):
    r"""The pointing model of the Simons Observatory small aperture telescopes, `sat_v2`.

    A port of `sotodlib.coords.pointing_model.model_sat_v2`, with the same parameters and sign
    conventions, so that SO parameters can be used as they are. The roll is that of sotodlib's
    SAT boresight, minus the boresight encoder reading. Unlike [`SATV1PointingModel`][], the
    periodic terms correct the encoder readings. With $\mathrm{az}_1 = \mathrm{az} +
    \mathrm{enc\_offset\_az}$ and $\mathrm{el}_1 = \mathrm{el} + \mathrm{enc\_offset\_el}$, they
    are corrected to

    $$
    \begin{aligned}
    \mathrm{az}' &= \mathrm{az}_1 + \mathrm{acec} \cos\mathrm{az}_1 - \mathrm{aces} \sin\mathrm{az}_1
        - \mathrm{harmonic\_2az\_cos} \cos 2\mathrm{az}_1
        + \mathrm{harmonic\_2az\_sin} \sin 2\mathrm{az}_1
        + \mathrm{npae}\,(\tan\mathrm{el}_1 - \tan 60^\circ), \\
    \mathrm{el}' &= \mathrm{el}_1 - \mathrm{harmonic\_el\_cos} \cos\mathrm{az}_1
        + \mathrm{harmonic\_el\_sin} \sin\mathrm{az}_1
        + \mathrm{harmonic\_2el\_cos} \cos 2\mathrm{az}_1
        - \mathrm{harmonic\_2el\_sin} \sin 2\mathrm{az}_1 \\
        &\quad + \mathrm{el\_sag}\,(1 / \sin\mathrm{el}_1 - 1 / \sin 60^\circ), \\
    \mathrm{roll}' &= \mathrm{roll} - \mathrm{enc\_offset\_boresight},
    \end{aligned}
    $$

    and the rotation from the telescope frame to the horizon frame is

    $$
    q = B\,R_z(-\mathrm{az}')\,R_y(\pi/2 - \mathrm{el}')\,
        X_\mathrm{offset}\,X_\mathrm{rot}\,R_z(\mathrm{roll}')\,X_\mathrm{rot}^{-1},
    $$

    with $B$ the base tilt, and $X_\mathrm{offset}$ and $X_\mathrm{rot}$ as in
    [`SATV1PointingModel`][].

    Attributes:
        enc_offset_az: Azimuth encoder offset, TPOINT $-$IA.
        enc_offset_el: Elevation encoder offset.
        enc_offset_boresight: Boresight rotation encoder offset.
        fp_offset_xi0: Collimation: $\xi$ of the centre of the boresight rotation, relative to the
            optical axis. It is fixed to the mount.
        fp_offset_eta0: Collimation: $\eta$ of that offset.
        fp_rot_xi0: $\xi$ of the centre of the boresight rotation in the focal plane.
        fp_rot_eta0: $\eta$ of the centre of the boresight rotation in the focal plane.
        base_tilt_cos: Base tilt, TPOINT AN.
        base_tilt_sin: Base tilt, TPOINT AW.
        harmonic_el_sin: First-order elevation harmonic, TPOINT HESA.
        harmonic_el_cos: First-order elevation harmonic, TPOINT HECA.
        harmonic_2el_sin: Second-order elevation harmonic, TPOINT HESA2.
        harmonic_2el_cos: Second-order elevation harmonic, TPOINT HECA2.
        harmonic_2az_sin: Second-order azimuth harmonic, TPOINT HASA2.
        harmonic_2az_cos: Second-order azimuth harmonic, TPOINT HACA2.
        acec: Azimuth centering error, cosine term, TPOINT HACA.
        aces: Azimuth centering error, sine term, TPOINT HASA.
        npae: Non-perpendicularity of the azimuth and elevation axes.
        el_sag: Elevation sag, in $1 / \sin\mathrm{el}$.
    """

    enc_offset_az: Float[Array, '...'] = _eqx_array_field()
    enc_offset_el: Float[Array, '...'] = _eqx_array_field()
    enc_offset_boresight: Float[Array, '...'] = _eqx_array_field()
    fp_offset_xi0: Float[Array, '...'] = _eqx_array_field()
    fp_offset_eta0: Float[Array, '...'] = _eqx_array_field()
    fp_rot_xi0: Float[Array, '...'] = _eqx_array_field()
    fp_rot_eta0: Float[Array, '...'] = _eqx_array_field()
    base_tilt_cos: Float[Array, '...'] = _eqx_array_field()
    base_tilt_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_el_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_el_cos: Float[Array, '...'] = _eqx_array_field()
    harmonic_2el_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_2el_cos: Float[Array, '...'] = _eqx_array_field()
    harmonic_2az_sin: Float[Array, '...'] = _eqx_array_field()
    harmonic_2az_cos: Float[Array, '...'] = _eqx_array_field()
    acec: Float[Array, '...'] = _eqx_array_field()
    aces: Float[Array, '...'] = _eqx_array_field()
    npae: Float[Array, '...'] = _eqx_array_field()
    el_sag: Float[Array, '...'] = _eqx_array_field()

    def quaternion(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Quaternion:
        az = jnp.asarray(az) + self.enc_offset_az
        el = jnp.asarray(el) + self.enc_offset_el
        roll = jnp.asarray(roll) - self.enc_offset_boresight

        # the sag, the non-perpendicularity and the harmonics all read the offset encoders
        pivot = jnp.pi / 3
        cos_az, sin_az = jnp.cos(az), jnp.sin(az)
        cos_2az, sin_2az = jnp.cos(2 * az), jnp.sin(2 * az)
        delta_az = (
            self.acec * cos_az
            - self.aces * sin_az
            - self.harmonic_2az_cos * cos_2az
            + self.harmonic_2az_sin * sin_2az
            + self.npae * (jnp.tan(el) - jnp.tan(pivot))
        )
        delta_el = (
            -self.harmonic_el_cos * cos_az
            + self.harmonic_el_sin * sin_az
            + self.harmonic_2el_cos * cos_2az
            - self.harmonic_2el_sin * sin_2az
            + self.el_sag * (1 / jnp.sin(el) - 1 / jnp.sin(pivot))
        )
        az, el = az + delta_az, el + delta_el

        fp_rot = XiEtaAngles(self.fp_rot_xi0, self.fp_rot_eta0).to_quaternion()
        return (
            _base_tilt(self.base_tilt_cos, self.base_tilt_sin)
            * euler(2, -az)
            * euler(1, jnp.pi / 2 - el)
            * XiEtaAngles(self.fp_offset_xi0, self.fp_offset_eta0).to_quaternion()
            * fp_rot
            * euler(2, roll)
            * fp_rot.conj()
        )


def _detector_direction(
    model: AbstractPointingModel,
    az: ArrayLike,
    el: ArrayLike,
    roll: ArrayLike,
    q_det: Quaternion | None,
) -> Float[Array, '... 3']:
    """The unit vectors in the horizon frame the detectors look at, the boresight by default."""
    if q_det is None:
        return model.direction(az, el, roll)
    return (model.quaternion(az, el, roll) * q_det).rotate_vector(ZAXIS)


class Observations(NamedTuple):
    """Pointing observations of point sources.

    All the arrays must broadcast to a common shape, the shape of the observations.

    Attributes:
        az: Azimuth encoder readings.
        el: Elevation encoder readings.
        roll: Roll encoder readings.
        az_obs: Observed (apparent) azimuth of the source.
        el_obs: Observed (apparent) elevation of the source.
        q_det: Quaternions of the detectors that observed the sources (identity by default).
        sigma: Uncertainty of the observed positions, in radians on the sky (1 by default).
    """

    az: ArrayLike
    el: ArrayLike
    roll: ArrayLike
    az_obs: ArrayLike
    el_obs: ArrayLike
    q_det: Quaternion | None = None
    sigma: ArrayLike | None = None
