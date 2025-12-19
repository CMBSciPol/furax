from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import jax.numpy as jnp
import litebird_sim as lbs
from astropy import wcs
from jaxtyping import Array, Bool, Float

from furax.mapmaking import AbstractLazyObservation, AbstractSatelliteObservation
from furax.mapmaking.noise import NoiseModel
from furax.obs.landscapes import StokesLandscape

T = TypeVar('T')


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
    def _local_detectors(self) -> list[dict[str, Any]]:
        """The list of detectors available to this Observation"""
        return [self.data.detectors_global[i] for i in self.data.det_idx]

    @cached_property
    def detectors(self) -> list[str]:
        return [det['name'] for det in self._local_detectors]

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
        raise NotImplementedError

    def get_hwp_angles(self) -> Array:
        return jnp.array(self.data.get_hwp_angle(), dtype=jnp.float64)

    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        raise NotImplementedError

    def get_timestamps(self) -> Float[Array, ' a']:
        return self.data.get_times(normalize=False, astropy_times=False)  # type: ignore[no-any-return]

    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], wcs.WCS]:
        raise NotImplementedError

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, ' ...'], Float[Array, ' ...']]:
        raise NotImplementedError

    def get_noise_model(self) -> None | NoiseModel:
        raise NotImplementedError

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        raise NotImplementedError

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        raise NotImplementedError


class LazyLBSObservation(AbstractLazyObservation[lbs.Observation]):
    interface_class = LBSObservation
