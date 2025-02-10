from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray
from sotodlib.core import AxisManager

from furax.mapmaking import GroundObservationData


@jax.tree_util.register_dataclass
class SotodlibObservationData(GroundObservationData):
    observation: AxisManager

    @property
    def n_samples(self) -> int:
        return self.observation.signal.shape[-1]  # type: ignore[no-any-return]

    @cached_property
    def dets(self) -> list[str]:
        """Returns a list of the detector names."""
        return self.observation.dets.vals  # type: ignore[no-any-return]

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        duration: float = self.observation.timestamps[-1] - self.observation.timestamps[0]
        return self.n_samples / duration

    def get_tods(self) -> Array:
        """Returns the timestream data."""
        return jnp.array(self.observation.signal)

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        return jnp.array(self.observation.hwp_angle)

    def get_scanning_intervals(self, det_ind: int = 0) -> Int[Array, 'a 2'] | NDArray[Any]:
        """Returns scanning intervals of the chosen detector.
        The output is a list of the starting and ending sample indices
        """
        return np.array(
            self.observation.preprocess.turnaround_flags.turnarounds.ranges[det_ind]
            .complement()
            .ranges()
        )

    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        return jnp.array(self.observation)

    def get_elapsed_time(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self.get_timestamps()
        return jnp.array(timestamps - timestamps[0])

    def get_timestamps(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        return jnp.array(self.observation.timestamps)

    def get_scanning_mask(self, det_ind: int = 0) -> NDArray[np.bool_]:
        """Returns scanning intervals of the chosen detector.
        The output is a boolean mask
        """
        # Assumes that the detectors have identical scanning intervals,
        return (  # type: ignore[no-any-return]
            self.observation.preprocess.turnaround_flags.turnarounds.ranges[det_ind]
            .complement()
            .mask()
        )

    def get_noise_fits(self, fmin: float) -> NDArray[np.float64]:
        """Returns fitted values of the noise psd with 1/f and white noise,
        either using the fitted parameters from the preprocessing,
        or fitting the model directly.
        """
        preproc = self.observation.preprocess

        if 'psdT' in preproc.keys():
            f = preproc.psdT.freqs
            fit = preproc.noiseT_fit.fit  # columns: (fknee, w, alpha)
        elif 'Pxx_raw' in preproc.keys():
            f = preproc.Pxx_raw.freqs
            fit = preproc.noise_signal_fit.fit  # columns: (fknee, w, alpha)
        else:
            # Estimate psd
            raise NotImplementedError('Self-psd evaluation not implemented')
        imin = np.argmin(np.abs(f - fmin))
        noiseT_fit_eval = np.zeros((fit.shape[0], f.size), dtype=float)  # (dets, freqs)
        noiseT_fit_eval[:, imin:] = fit[:, [1]] * (
            1 + (fit[:, [0]] / f[None, imin:]) ** fit[:, [2]]
        )
        noiseT_fit_eval[:, :imin] = noiseT_fit_eval[:, [imin]]

        return np.array(noiseT_fit_eval)

    def get_white_noise_fit(self) -> NDArray[np.float64]:
        """Returns fitted values of the white noise,
        obtained as a reult of a 1/f + white noise model fitting.
        Uses either the fitted parameters from the preprocessing,
        or fitting the model directly.
        """

        preproc = self.observation.preprocess
        if 'psdT' in preproc.keys():
            fit = preproc.noiseT_fit.fit  # columns: (fknee, w, alpha)
        elif 'Pxx_raw' in preproc.keys():
            fit = preproc.noise_signal_fit.fit  # columns: (fknee, w, alpha)
        else:
            # Estimate psd
            raise NotImplementedError('Self-psd evaluation not implemented')
        return fit[:, 1]  # type: ignore[no-any-return]
