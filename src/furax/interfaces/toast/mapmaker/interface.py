from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import toast
from astropy import units as u
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray
from toast.observation import default_values as defaults

from .utils import get_local_meridian_angle


@jax.tree_util.register_dataclass
@dataclass
class ObservationData:
    observation: toast.Observation
    det_selection: list[str] | None = None
    det_mask: int = defaults.det_mask_nonscience

    # the names of the fields we need
    det_data: str = defaults.det_data
    pixels: str = defaults.pixels
    quats: str = defaults.quats
    hwp_angle: str | None = defaults.hwp_angle
    noise_model: str | None = defaults.noise_model

    _cross_psd: tuple[Float[Array, ' freq'], Float[Array, 'det det freq']] | None = None

    @property
    def samples(self) -> int:
        return self.observation.n_local_samples  # type: ignore[no-any-return]

    @cached_property
    def dets(self) -> list[str]:
        """Returns a list of the detector names."""
        local_selection: list[str] = self.observation.select_local_detectors(
            selection=self.det_selection, flagmask=self.det_mask
        )
        return local_selection

    @property
    def focal_plane(self) -> toast.Focalplane:
        return self.observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        return self.focal_plane.sample_rate.to_value(u.Hz)  # type: ignore[no-any-return]

    def get_tods(self) -> Array:
        """Returns the timestream data."""
        # furax's LinearPolarizerOperator assumes power, TOAST assumes temperature
        return 0.5 * jnp.array(self.observation.detdata[self.det_data][self.dets, :])

    def get_pixels(self) -> Array:
        """Returns the pixel indices."""
        return jnp.array(self.observation.detdata[self.pixels][self.dets, :])

    def get_expanded_quats(self):  # type: ignore[no-untyped-def]
        """Returns expanded pointing quaternions."""
        return self.observation.detdata[self.quats][self.dets, :]

    def get_det_angles(self) -> Array:
        """Returns the detector angles on the sky."""
        func = np.vectorize(get_local_meridian_angle, signature='(n,k)->(n)')
        return jnp.array(func(self.get_expanded_quats()))

    def get_offsets(self) -> Array:
        """Returns the detector offset angles."""
        fp = self.focal_plane
        return jnp.array([fp[det]['gamma'].to_value(u.rad) for det in self.dets])

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        if self.hwp_angle is None:
            raise ValueError('HWP angle field not provided.')
        return jnp.array(self.observation.shared[self.hwp_angle].data)

    def get_psd_model(self) -> tuple[Array, Array]:
        """Returns frequencies and PSD values of the noise model."""
        if self.noise_model is None:
            raise ValueError('Noise model not provided.')
        model = self.observation[self.noise_model]
        freq = jnp.array([model.freq(det) for det in self.dets])
        psd = jnp.array([model.psd(det) for det in self.dets])
        return freq, psd

    def get_scanning_intervals(self) -> Int[Array, 'a 2'] | NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """
        if (
            not hasattr(self.observation, 'intervals')
            or 'scanning' not in self.observation.intervals
        ):
            # Scanning information missing, first compute the intervals
            toast.ops.AzimuthIntervals().apply(self.observation)
        intervals = self.observation.intervals['scanning']
        return np.array(intervals[['first', 'last']].tolist())

    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        if 'azimuth' not in self.observation.shared:
            raise ValueError('Azimuth field not provided.')
        return jnp.array(self.observation.shared['azimuth'].data)

    def get_elapsed_time(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self.observation.shared['times'].data
        return jnp.array(timestamps - timestamps[0])
