"""Pointing models for alt-azimuthal mounts"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple, Self, dataclass_transform

import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, ArrayLike, Float

from furax.core.utils import register_dataclass_with_keys

from ..coords import ZAXIS, euler, xieta_to_quat

__all__ = [
    'AbstractPointingModel',
    'BasicPointingModel',
    'Observations',
]


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class AbstractPointingModel(ABC):
    """A telescope pointing model: the boresight in the horizon frame, from encoder readings.

    The parameters are the fields of a subclass, angles in radians, converted to floating-point
    arrays on construction.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True)(cls)
        # unflattening bypasses `__init__`, so that JAX transformations can hold any leaves
        register_dataclass_with_keys(cls)

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            object.__setattr__(self, field.name, jnp.asarray(getattr(self, field.name), float))

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
        """The boresight in the horizon frame, `q_hor_bore`.

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


class BasicPointingModel(AbstractPointingModel):
    r"""A basic telescope pointing model.

    This model only takes into account encoder zero points and collimation error. For encoder
    readings (az, el, roll), the rotation from the telescope frame to the horizon frame is

    $$
    q = R_z(-(\mathrm{az} + \mathrm{az_0}))\,R_y(\pi/2 - (\mathrm{el} + \mathrm{el_0}))\,
        X(\mathrm{col}_\xi, \mathrm{col}_\eta)\,R_z(\mathrm{roll} + \mathrm{roll_0}),
    $$

    where $X(\xi, \eta)$ takes the centre of the focal plane to $(\xi, \eta)$, see
    [`xieta_to_quat`][]. The collimation error is fixed to the mount: it does not turn with the
    boresight rotation.

    Attributes:
        az0: Azimuth encoder zero point.
        el0: Elevation encoder zero point.
        roll0: Roll (boresight rotation) encoder zero point.
        col_xi: Collimation: $\xi$ of the boresight, relative to the optical axis.
        col_eta: Collimation: $\eta$ of the boresight, relative to the optical axis.
    """

    az0: Float[ArrayLike, '...'] = 0.0
    el0: Float[ArrayLike, '...'] = 0.0
    roll0: Float[ArrayLike, '...'] = 0.0
    col_xi: Float[ArrayLike, '...'] = 0.0
    col_eta: Float[ArrayLike, '...'] = 0.0

    def quaternion(self, az: ArrayLike, el: ArrayLike, roll: ArrayLike) -> Quaternion:
        az, el, roll = jnp.asarray(az), jnp.asarray(el), jnp.asarray(roll)
        return (
            euler(2, -(az + self.az0))
            * euler(1, jnp.pi / 2 - (el + self.el0))
            * xieta_to_quat(self.col_xi, self.col_eta)
            * euler(2, roll + self.roll0)
        )


def _detector_direction(
    model: AbstractPointingModel,
    az: ArrayLike,
    el: ArrayLike,
    roll: ArrayLike,
    q_bore_det: Quaternion | None,
) -> Float[Array, '... 3']:
    """The unit vectors in the horizon frame the detectors look at, the boresight by default."""
    if q_bore_det is None:
        return model.direction(az, el, roll)
    return (model.quaternion(az, el, roll) * q_bore_det).rotate_vector(ZAXIS)


class Observations(NamedTuple):
    """Pointing observations of point sources.

    All the arrays must broadcast to a common shape, the shape of the observations.

    Attributes:
        az: Azimuth encoder readings.
        el: Elevation encoder readings.
        roll: Roll encoder readings.
        az_obs: Observed (apparent) azimuth of the source.
        el_obs: Observed (apparent) elevation of the source.
        q_bore_det: Quaternions, in the boresight frame, of the detectors that observed the sources
            (identity by default).
        sigma: Uncertainty of the observed positions, in radians on the sky (1 by default).
    """

    az: ArrayLike
    el: ArrayLike
    roll: ArrayLike
    az_obs: ArrayLike
    el_obs: ArrayLike
    q_bore_det: Quaternion | None = None
    sigma: ArrayLike | None = None
