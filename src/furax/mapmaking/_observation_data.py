from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import numpy as np
from astropy.wcs import WCS
from jaxtyping import Array, Float
from numpy.typing import NDArray

from furax.obs.landscapes import StokesLandscape

from .noise import NoiseModel


@jax.tree_util.register_dataclass
@dataclass
class GroundObservationData:
    """Dataclass for ground-based observation data.

    This class defines what data is needed for making maps with ground-based data.
    It is meant to be used as a base class for interfacing with different containers
    (e.g. toast's ``Observation``, sotodlib's ``AxisManager``, ...)
    """

    @property
    @abstractmethod
    def n_samples(self) -> int: ...

    @cached_property
    @abstractmethod
    def dets(self) -> list[str]:
        """Returns a list of the detector names."""
        ...

    @property
    def n_dets(self) -> int:
        return len(self.dets)

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
    def get_scanning_intervals(self) -> NDArray[Any]:
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

    @abstractmethod
    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], WCS]:
        """Returns the shape and object corresponding to a WCS projection"""
        ...

    @abstractmethod
    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, '...'], Float[Array, '...']]:
        """Obtain pointing information and spin angles from the observation"""
        ...

    @abstractmethod
    def get_noise_model(self) -> None | NoiseModel:
        """Load precomputed noise model from the data, if present. Otherwise, return None"""
        ...

    def get_scanning_mask(self) -> NDArray[Any]:
        """Returns a boolean mask constructed with scanning intervals"""
        intervals = self.get_scanning_intervals()
        mask = np.zeros(self.n_samples, dtype=bool)
        for l, u in intervals:
            mask[l:u] = True

        return mask
