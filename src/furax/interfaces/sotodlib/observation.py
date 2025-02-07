from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray
from sotodlib.core import AxisManager

from ..observation import ObservationData


@jax.tree_util.register_dataclass
@dataclass
class SotodlibObservationData(ObservationData):
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

    @abstractmethod
    def get_scanning_intervals(self, det_ind: int = 0) -> Int[Array, 'a 2'] | NDArray[Any]:
        """Returns scanning intervals of the first detector.
        The output is a list of the starting and ending sample indices
        """
        return np.array(
            self.observation.preprocess.turnaround_flags.turnarounds.ranges[det_ind]
            .complement()
            .ranges()
        )

    @abstractmethod
    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        return jnp.array(self.observation)

    @abstractmethod
    def get_elapsed_time(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self.observation.timestamps
        return jnp.array(timestamps - timestamps[0])
