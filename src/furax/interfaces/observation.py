from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from jaxtyping import Array, Float, Int
from numpy.typing import NDArray


@jax.tree_util.register_dataclass
@dataclass
class ObservationData:
    @property
    @abstractmethod
    def n_samples(self) -> int: ...

    @cached_property
    @abstractmethod
    def dets(self) -> list[str]:
        """Returns a list of the detector names."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        ...

    @abstractmethod
    def get_tods(self) -> Array:
        """Returns the timestream data."""
        ...

    @abstractmethod
    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        ...

    @abstractmethod
    def get_scanning_intervals(self) -> Int[Array, 'a 2'] | NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """
        ...

    @abstractmethod
    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        ...

    @abstractmethod
    def get_elapsed_time(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        ...
