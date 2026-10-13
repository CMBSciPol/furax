"""Ground pickup template of the single-observation mapmakers.

[`GroundTemplateOperator`][] models the ground signal as a map on a horizon (azimuth, elevation)
grid, observed through the detectors' pointing, polarizer and half-wave plate.
"""

import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import Array
from jaxtyping import DTypeLike, Float

from furax import AbstractLinearOperator
from furax.math import coords
from furax.obs import HWPOperator, LinearPolarizerOperator
from furax.obs.landscapes import HorizonLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import ValidStokesLiteral

__all__ = [
    'GroundTemplateOperator',
]


class GroundTemplateOperator(AbstractLinearOperator):
    """Operator for ground signal templates.

    The template amplitudes form a two-dimensional (elevation, azimuth) IQU map of the ground
    observed that is shared across detectors for the observation range. This class only contains a
    factory method. All argument angles should be provided in radians.
    """

    @classmethod
    def create(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        boresight_rotation: Float[Array, ' samps'],
        detector_quaternions: Quaternion,
        hwp_angles: Float[Array, ' samps'],
        stokes: ValidStokesLiteral,
        dtype: DTypeLike,
        landscape: HorizonLandscape | None = None,
        batch_samples: int | None = None,
    ) -> AbstractLinearOperator:
        # Compute landscape if not provided
        if landscape is None:
            horizon_landscape: HorizonLandscape = cls.get_landscape(
                azimuth_resolution=azimuth_resolution,
                elevation_resolution=elevation_resolution,
                boresight_azimuth=boresight_azimuth,
                boresight_elevation=boresight_elevation,
                detector_quaternions=detector_quaternions,
                stokes=stokes,
                dtype=dtype,
            )
        else:
            horizon_landscape = landscape

        boresight_quaternions = coords.AzElAngles(
            boresight_azimuth, boresight_elevation, boresight_rotation
        ).to_quaternion()
        det_gamma = coords.gamma_angle(detector_quaternions)

        n_dets = detector_quaternions.shape[0]
        n_samps = boresight_azimuth.size

        pointing = PointingOperator.create(
            horizon_landscape,
            boresight_quaternions,
            detector_quaternions,
            batch_samples=batch_samples,
        )

        polarizer = LinearPolarizerOperator.create(
            shape=(n_dets, n_samps),
            dtype=dtype,
            stokes=stokes,
            angles=det_gamma[:, None].astype(dtype),
        )

        if stokes == 'I':
            return polarizer @ pointing

        hwp = HWPOperator.create(
            shape=(n_dets, n_samps), dtype=dtype, stokes=stokes, angles=hwp_angles.astype(dtype)
        )

        return polarizer @ hwp @ pointing

    @classmethod
    def get_landscape(
        cls,
        azimuth_resolution: float,
        elevation_resolution: float,
        boresight_azimuth: Float[Array, ' samps'],
        boresight_elevation: Float[Array, ' samps'],
        detector_quaternions: Quaternion,
        stokes: ValidStokesLiteral,
        dtype: DTypeLike,
    ) -> HorizonLandscape:
        # First, set up a grid of (az, el) pairs
        n_grid = 10
        az_grid = jnp.linspace(jnp.min(boresight_azimuth), jnp.max(boresight_azimuth), n_grid)
        el_grid = jnp.linspace(jnp.min(boresight_elevation), jnp.max(boresight_elevation), n_grid)
        az_mesh, el_mesh = jnp.meshgrid(az_grid, el_grid, indexing='ij')
        qbore_mesh = coords.AzElAngles(
            az_mesh, el_mesh, jnp.zeros_like(az_mesh)
        ).to_quaternion()  # (N_GRID, N_GRID)
        qfull_mesh = (
            qbore_mesh[None] * detector_quaternions[:, None, None]
        )  # (ndet, N_GRID, N_GRID)
        det_az_mesh, det_el_mesh, _ = coords.AzElAngles.from_quaternion(qfull_mesh)

        # Azimuth angle is first restricted to to [0,2pi),
        # and unwrapped along the elevation grid, azimuth grid, and detector axes in order
        det_az_mesh = jnp.unwrap(
            jnp.unwrap(jnp.unwrap(det_az_mesh % (2 * jnp.pi), axis=2), axis=1), axis=0
        )

        # Allow small margins
        az_min = jnp.min(det_az_mesh) - 1e-4
        az_max = jnp.max(det_az_mesh) + 1e-4
        el_min = jnp.min(det_el_mesh) - 1e-4
        el_max = jnp.max(det_el_mesh) + 1e-4

        n_alt = int(np.ceil((el_max - el_min) / elevation_resolution))
        n_az = int(np.ceil((az_max - az_min) / azimuth_resolution))

        landscape = HorizonLandscape(
            shape=(n_az, n_alt),
            altitude_limits=(el_min, el_max),
            azimuth_limits=(az_min, az_max),
            stokes=stokes,
            dtype=dtype,
        )

        return landscape
