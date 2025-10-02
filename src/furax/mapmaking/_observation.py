from abc import abstractmethod
from functools import cached_property
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from astropy.wcs import WCS
from jaxtyping import Array, Bool, Float
from numpy.typing import NDArray

from furax.math.quaternion import qmul, to_lonlat_angles
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
    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        """Returns sample mask of the TOD,
        which is 1 at valid samples and 0 at invalid ones.
        """

    @abstractmethod
    def get_left_scan_mask(self) -> Bool[Array, ' samps']:
        """Returns sample mask of the TOD for left-going scans,
        which is 1 at valid samples and 0 at invalid ones.
        """

    @abstractmethod
    def get_right_scan_mask(self) -> Bool[Array, ' samps']:
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

    def get_scanning_mask(self) -> Bool[Array, '...']:
        """Returns a boolean mask constructed with scanning intervals"""
        intervals = self.get_scanning_intervals()
        mask = jnp.zeros(self.n_samples, dtype=bool)
        for l, u in intervals:
            mask[l:u] = True

        return mask

    @abstractmethod
    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        """Returns the boresight quaternions at each time sample"""

    @abstractmethod
    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        """Returns the quaternion offsets of the detectors"""

    def get_detector_pointing_lonlat(
        self,
        thin_samples: int = 1,
        use_scanning_mask: bool = True,
        use_degrees: bool = True,
    ) -> tuple[Float[Array, 'det samp'], Float[Array, 'det samp']]:
        """
        Compute the pointing trajectory in longitude/latitude coordinates for all detectors.

        This method calculates the sky coordinates (longitude, latitude) for each detector
        at each time sample by combining boresight quaternions with detector offset quaternions.

        Parameters
        ----------
        thin_samples : int, default=1
            Factor by which to subsample the time axis. If > 1, only every nth sample
            is included in the output. Applied after scanning mask if both are used.
        use_scanning_mask : bool, default=True
            Whether to apply the scanning mask to exclude non-scanning periods.
            When True, only samples during active scanning are included.
        use_degrees : bool, default=True
            If True, return angles in degrees. If False, return in radians.

        Returns
        -------
        alpha : Float[Array, 'det samp']
            Longitude coordinates (RA or azimuth) for each detector and sample.
            Shape is (n_detectors, n_samples_final) where n_samples_final depends
            on scanning mask and subsampling.
        delta : Float[Array, 'det samp']
            Latitude coordinates (Dec or elevation) for each detector and sample.
            Shape is (n_detectors, n_samples_final).

        Notes
        -----
        - The coordinate system depends on the boresight quaternion convention:
          typically equatorial (RA/Dec) or horizontal (Az/El)
        - Processing order: scanning mask is applied first, then subsampling
        - The quaternion multiplication combines the boresight pointing with
          detector offsets to get the absolute pointing of each detector

        Examples
        --------
        >>> # Get pointing in degrees for all detectors
        >>> lon, lat = obs.get_detector_pointing_lonlat()
        >>> print(f"Shape: {lon.shape}")  # (n_dets, n_samples_scanning)

        >>> # Get pointing in radians without scanning mask, subsampled by 10
        >>> lon_rad, lat_rad = obs.get_detector_pointing_lonlat(
        ...     thin_samples=10, use_scanning_mask=False, use_degrees=False
        ... )

        """
        # Get quaternions for boresight and detector offsets
        boresight_quaternions = self.get_boresight_quaternions()  # (n_samples, 4)
        detector_quaternions = self.get_detector_quaternions()  # (n_dets, 4)

        # Apply scanning mask first if requested
        if use_scanning_mask:
            mask = self.get_scanning_mask()  # (n_samples,) boolean array
            boresight_quaternions = boresight_quaternions[mask, :]

        # Apply subsampling after masking if requested
        if thin_samples > 1:
            boresight_quaternions = boresight_quaternions[::thin_samples, :]

        # Combine boresight pointing with detector offsets via quaternion multiplication
        # Broadcasting: (1, n_final_samples, 4) * (n_dets, 1, 4) -> (n_dets, n_final_samples, 4)
        qdet_full = qmul(
            boresight_quaternions[None, :, :],  # (1, n_final_samples, 4)
            detector_quaternions[:, None, :],  # (n_dets, 1, 4)
        )  # Result: (n_dets, n_final_samples, 4)

        # Convert quaternions to longitude/latitude angles in radians
        alpha, delta, _ = to_lonlat_angles(qdet_full)

        # Convert to degrees if requested
        if use_degrees:
            alpha = jnp.degrees(alpha)
            delta = jnp.degrees(delta)

        return alpha, delta
