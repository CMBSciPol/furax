from abc import abstractmethod
from functools import cached_property
from typing import Any, Generic, TypeVar

import numpy as np
from astropy.wcs import WCS
from jaxtyping import Array, Float
from numpy.typing import NDArray

from furax.obs.landscapes import StokesLandscape

from .noise import NoiseModel

T = TypeVar('T')


class AbstractGroundObservation(Generic[T]):
    """Dataclass for ground-based observation data.

    This class defines what data is needed for making maps with ground-based data.
    It is meant to be used as a base class for interfacing with different containers
    (e.g. toast's ``Observation``, sotodlib's ``AxisManager``, ...)
    """

    def __init__(self, data: T) -> None:
        self.data = data

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Returns the number of samples in the observation."""

    @cached_property
    @abstractmethod
    def detectors(self) -> list[str]:
        """Returns a list of the detector names."""

    @property
    def n_detectors(self) -> int:
        return len(self.detectors)

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""

    @abstractmethod
    def get_tods(self) -> Array:
        """Returns the timestream data."""

    @abstractmethod
    def get_detector_offset_angles(self) -> Array:
        """Returns the detector offset angles ('gamma')."""

    @abstractmethod
    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""

    @abstractmethod
    def get_scanning_intervals(self) -> NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """

    @abstractmethod
    def get_sample_mask(self) -> Float[Array, 'dets samps']:
        """Returns sample mask of the TOD,
        which is 1 at valid samples and 0 at invalid ones.
        """
        ...

    @abstractmethod
    def get_left_scan_mask(self) -> Float[Array, ' samps']:
        """Returns sample mask of the TOD for left-going scans,
        which is 1 at valid samples and 0 at invalid ones.
        """

    @abstractmethod
    def get_right_scan_mask(self) -> Float[Array, ' samps']:
        """Returns sample mask of the TOD for right-going scans,
        which is 1 at valid samples and 0 at invalid ones.
        """

    @abstractmethod
    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""

    @abstractmethod
    def get_elevation(self) -> Float[Array, ' a']:
        """Returns the elevation of the boresight for each sample"""

    @abstractmethod
    def get_elapsed_times(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""

    @abstractmethod
    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], WCS]:
        """Returns the shape and object corresponding to a WCS projection"""

    @abstractmethod
    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, '...'], Float[Array, '...']]:
        """Obtain pointing information and spin angles from the observation"""

    @abstractmethod
    def get_noise_model(self) -> None | NoiseModel:
        """Load a pre-computed noise model from the data, if present. Otherwise, return None"""

    def get_scanning_mask(self) -> NDArray[Any]:
        """Returns a boolean mask constructed with scanning intervals"""
        intervals = self.get_scanning_intervals()
        mask = np.zeros(self.n_samples, dtype=bool)
        for l, u in intervals:
            mask[l:u] = True

        return mask

    @abstractmethod
    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        """Returns the boresight quaternions at each time sample"""

    @abstractmethod
    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        """Returns the quaternion offsets of the detectors"""
