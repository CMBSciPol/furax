import math

import jax
import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Key

from ..coords import tangent_basis, vec_to_azel
from ._fitting import ARCSEC
from .modeling import AbstractPointingModel, Observations, _detector_direction

__all__ = ['simulate_observations']


def simulate_observations(
    key: Key[Array, ''],
    true_model: AbstractPointingModel,
    n: int,
    noise_arcsec: float = 0.0,
    q_det: Quaternion | None = None,
    el_range: tuple[float, float] = (math.radians(20), math.radians(85)),
    roll_range: tuple[float, float] = (math.radians(-60), math.radians(60)),
) -> Observations:
    """Simulate observations of point sources by a telescope with the given pointing errors.

    The encoder readings are drawn uniformly, and the observed positions are those predicted by
    the true model, displaced by white noise on the sky.

    Args:
        key: PRNG key.
        true_model: Pointing model used to simulate the observations.
        n: Number of observations.
        noise_arcsec: Standard deviation of the noise per sky axis, in arcseconds.
        q_det: Detector quaternions of shape (n_det,). Each observation is made by a random
            detector. By default, the observations are made by the boresight.
        el_range: Range of the elevation encoder readings, in radians.
        roll_range: Range of the roll encoder readings, in radians.

    Returns:
        The simulated observations, with `sigma` set to the noise level.
    """
    key_az, key_el, key_roll, key_det, key_noise = jax.random.split(key, 5)
    az = jax.random.uniform(key_az, (n,), minval=0, maxval=2 * jnp.pi)
    el = jax.random.uniform(key_el, (n,), minval=el_range[0], maxval=el_range[1])
    roll = jax.random.uniform(key_roll, (n,), minval=roll_range[0], maxval=roll_range[1])
    if q_det is not None:
        q_det = q_det[jax.random.randint(key_det, (n,), 0, q_det.shape[0])]

    direction = _detector_direction(true_model, az, el, roll, q_det)
    az_true, el_true = vec_to_azel(direction)
    e_xel, e_el = tangent_basis(az_true, el_true)
    noise = noise_arcsec * ARCSEC * jax.random.normal(key_noise, (n, 2))
    direction = direction + noise[:, :1] * e_xel + noise[:, 1:] * e_el
    az_obs, el_obs = vec_to_azel(direction)

    sigma = None if noise_arcsec == 0 else jnp.full(n, noise_arcsec * ARCSEC)
    return Observations(az, el, roll, az_obs, el_obs, q_det=q_det, sigma=sigma)
