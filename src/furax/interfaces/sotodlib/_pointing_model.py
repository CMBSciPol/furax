"""Ports of the pointing models of `sotodlib.coords.pointing_model`."""

import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import ArrayLike, Float

from furax.obs.coords import euler, xieta_to_quat
from furax.obs.pointing import AbstractPointingModel

__all__ = [
    'SOTODLibSATv1PointingModel',
    'SOTODLibSATv2PointingModel',
]


class SOTODLibSATv1PointingModel(AbstractPointingModel):
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

    enc_offset_az: Float[ArrayLike, '...'] = 0.0
    enc_offset_el: Float[ArrayLike, '...'] = 0.0
    enc_offset_boresight: Float[ArrayLike, '...'] = 0.0
    fp_offset_xi0: Float[ArrayLike, '...'] = 0.0
    fp_offset_eta0: Float[ArrayLike, '...'] = 0.0
    fp_rot_xi0: Float[ArrayLike, '...'] = 0.0
    fp_rot_eta0: Float[ArrayLike, '...'] = 0.0
    az_rot: Float[ArrayLike, '...'] = 0.0
    base_tilt_cos: Float[ArrayLike, '...'] = 0.0
    base_tilt_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_2el_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_2el_cos: Float[ArrayLike, '...'] = 0.0
    harmonic_2az_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_2az_cos: Float[ArrayLike, '...'] = 0.0
    acec: Float[ArrayLike, '...'] = 0.0
    aces: Float[ArrayLike, '...'] = 0.0

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

        fp_rot = xieta_to_quat(self.fp_rot_xi0, self.fp_rot_eta0)
        return (
            harmonics
            * _base_tilt(self.base_tilt_cos, self.base_tilt_sin)
            * euler(2, -az)
            * euler(1, jnp.pi / 2 - el)
            * xieta_to_quat(self.fp_offset_xi0, self.fp_offset_eta0)
            * fp_rot
            * euler(2, roll)
            * fp_rot.conj()
        )


class SOTODLibSATv2PointingModel(AbstractPointingModel):
    r"""The pointing model of the Simons Observatory small aperture telescopes, `sat_v2`.

    A port of `sotodlib.coords.pointing_model.model_sat_v2`, with the same parameters and sign
    conventions, so that SO parameters can be used as they are. The roll is that of sotodlib's
    SAT boresight, minus the boresight encoder reading. Unlike [`SOTODLibSATv1PointingModel`][], the
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
    [`SOTODLibSATv1PointingModel`][].

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

    enc_offset_az: Float[ArrayLike, '...'] = 0.0
    enc_offset_el: Float[ArrayLike, '...'] = 0.0
    enc_offset_boresight: Float[ArrayLike, '...'] = 0.0
    fp_offset_xi0: Float[ArrayLike, '...'] = 0.0
    fp_offset_eta0: Float[ArrayLike, '...'] = 0.0
    fp_rot_xi0: Float[ArrayLike, '...'] = 0.0
    fp_rot_eta0: Float[ArrayLike, '...'] = 0.0
    base_tilt_cos: Float[ArrayLike, '...'] = 0.0
    base_tilt_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_el_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_el_cos: Float[ArrayLike, '...'] = 0.0
    harmonic_2el_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_2el_cos: Float[ArrayLike, '...'] = 0.0
    harmonic_2az_sin: Float[ArrayLike, '...'] = 0.0
    harmonic_2az_cos: Float[ArrayLike, '...'] = 0.0
    acec: Float[ArrayLike, '...'] = 0.0
    aces: Float[ArrayLike, '...'] = 0.0
    npae: Float[ArrayLike, '...'] = 0.0
    el_sag: Float[ArrayLike, '...'] = 0.0

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
            self.harmonic_el_sin * sin_az
            - self.harmonic_el_cos * cos_az
            + self.harmonic_2el_cos * cos_2az
            - self.harmonic_2el_sin * sin_2az
            + self.el_sag * (1 / jnp.sin(el) - 1 / jnp.sin(pivot))
        )
        az, el = az + delta_az, el + delta_el

        fp_rot = xieta_to_quat(self.fp_rot_xi0, self.fp_rot_eta0)
        return (
            _base_tilt(self.base_tilt_cos, self.base_tilt_sin)
            * euler(2, -az)
            * euler(1, jnp.pi / 2 - el)
            * xieta_to_quat(self.fp_offset_xi0, self.fp_offset_eta0)
            * fp_rot
            * euler(2, roll)
            * fp_rot.conj()
        )


def _base_tilt(c: ArrayLike, s: ArrayLike) -> Quaternion:
    """The base tilt of TPOINT parameters AN = `c` and AW = `s`: the rotation vector (-s, c, 0)."""
    c, s = jnp.asarray(c), jnp.asarray(s)
    return Quaternion.from_rotation_vector(jnp.stack([-s, c, jnp.zeros_like(c)], axis=-1))
