from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import litebird_sim as lbs
import numpy as np
from astropy import wcs
from jaxtyping import Array, Bool, Float

from furax.mapmaking import AbstractLazyObservation, AbstractSatelliteObservation
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.obs.landscapes import HealpixLandscape, StokesLandscape


class LBSObservation(AbstractSatelliteObservation[lbs.Observation]):
    @classmethod
    def from_file(
        cls, filename: str | Path, requested_fields: list[str] | None = None
    ) -> LBSObservation:
        # check that file exists
        file = Path(filename)
        if not file.exists():
            raise FileNotFoundError(f'File {filename} does not exist')

        # TODO: support requested_fields
        obs = lbs.io.read_one_observation(file)
        if obs is None:
            raise RuntimeError(f'could not read {file} with litebird_sim')
        return cls(obs)

    @property
    def name(self) -> str:
        jdtime_min = round(self.data.start_time.jd * 60)
        return f'obs_{jdtime_min}_lbs'

    @property
    def telescope(self) -> str:
        return 'LB'

    @property
    def n_samples(self) -> int:
        return self.data.n_samples  # type: ignore[no-any-return]

    @property
    def detectors(self) -> list[str]:
        # assumes we have all the detectors
        return [det['name'] for det in self.data.detectors_global]

    @property
    def n_detectors(self) -> int:
        return self.data.n_detectors  # type: ignore[no-any-return]

    @property
    def sample_rate(self) -> float:
        return self.data.sample_rate_hz  # type: ignore[no-any-return]

    def get_tods(self) -> Array:
        tods = jnp.array(self.data.tod, dtype=jnp.float64)
        return jnp.atleast_2d(tods)

    def get_detector_offset_angles(self) -> Array:
        return jnp.array(self.data.pol_angle_rad, dtype=jnp.float64)

    def get_hwp_angles(self) -> Array:
        return jnp.array(self.data.get_hwp_angle(), dtype=jnp.float64)

    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        # TODO: take into account global and local flags
        return jnp.ones((self.n_detectors, self.n_samples), dtype=bool)

    def get_timestamps(self) -> Float[Array, ' a']:
        return jnp.array(
            self.data.get_times(normalize=False, astropy_times=False), dtype=jnp.float64
        )

    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], wcs.WCS]:
        raise NotImplementedError

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, ' ...'], Float[Array, ' ...']]:
        if not isinstance(landscape, HealpixLandscape):
            raise RuntimeError('only healpix is supported')
        pointings, _hwp_angles = self.data.get_pointings()
        # pointings have shape (N_det, N_samples, 3)
        theta, phi, psi = np.moveaxis(pointings, -1, 0)
        pixel_indices = landscape.world2index(theta, phi)
        spin_angles = jnp.array(psi, dtype=jnp.float64)
        return pixel_indices, spin_angles

    def get_noise_model(self) -> None | NoiseModel:
        return AtmosphericNoiseModel(
            sigma=jnp.array(self.data.net_ukrts * 1e-6),
            alpha=jnp.array(-self.data.alpha),
            fk=jnp.array(self.data.fknee_mhz * 1e-3),
            f0=jnp.array(self.data.fmin_hz),
        )

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        # TODO: coordinate system
        qbore = self.data.pointing_provider.bore2ecliptic_quats.quats
        return jnp.array(qbore, dtype=jnp.float64)

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        return jnp.concatenate([q.quats for q in self.data.quat], dtype=jnp.float64)


class LazyLBSObservation(AbstractLazyObservation[lbs.Observation]):
    interface_class = LBSObservation
