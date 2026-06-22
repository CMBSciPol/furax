from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha1
from pathlib import Path
from typing import Any, ClassVar, Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from astropy.wcs import WCS
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, UInt32
from numpy.typing import NDArray

from furax.math.quaternion import qmul, to_lonlat_angles
from furax.obs.landscapes import ProjectionType, StokesLandscape
from furax.obs.stokes import ValidStokesType

from .noise import NoiseModel

T = TypeVar('T')


class ReaderField(StrEnum):
    """Canonical names of the data fields an observation reader can load.

    Members are plain strings (``StrEnum``), so they interoperate with string keys in
    field dictionaries and set membership tests. Using members instead of bare literals
    makes typos a name-resolution error caught by linting/type-checking.
    """

    METADATA = 'metadata'
    SAMPLE_DATA = 'sample_data'
    VALID_SAMPLE_MASKS = 'valid_sample_masks'
    VALID_SCANNING_MASKS = 'valid_scanning_masks'
    TIMESTAMPS = 'timestamps'
    HWP_ANGLES = 'hwp_angles'
    DETECTOR_QUATERNIONS = 'detector_quaternions'
    BORESIGHT_QUATERNIONS = 'boresight_quaternions'
    NOISE_MODEL_FITS = 'noise_model_fits'


@register_dataclass
@dataclass
class HashedObservationMetadata:
    """Hashed version of some metadata fields for JAX compatibility."""

    uid: UInt32[np.ndarray | Array, '']
    telescope_uid: UInt32[np.ndarray | Array, '']
    detector_uids: UInt32[np.ndarray | Array, '*#dets']

    @classmethod
    def from_observation(cls, obs: AbstractObservation[T]) -> Self:
        return cls(
            uid=_names_to_uids(obs.name),
            telescope_uid=_names_to_uids(obs.telescope),
            detector_uids=_names_to_uids(obs.detectors),
        )

    @classmethod
    def structure_for(cls, n_dets: int) -> Self:
        return cls(
            uid=jax.ShapeDtypeStruct((), dtype=np.uint32),  # type: ignore[arg-type]
            telescope_uid=jax.ShapeDtypeStruct((), dtype=np.uint32),  # type: ignore[arg-type]
            detector_uids=jax.ShapeDtypeStruct((n_dets,), dtype=np.uint32),  # type: ignore[arg-type]
        )


def _names_to_uids(names: str | list[str] | np.ndarray) -> UInt32[np.ndarray, ...]:
    """Converts names to unsigned 32-bit integers using hashing."""
    # SHA-1 hash truncated to a non-negative 32-bit integer
    to_int = lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0x7FFFFFFF
    return np.vectorize(to_int, otypes=[np.uint32])(names)  # type: ignore[no-any-return]


class AbstractObservation(ABC, Generic[T]):
    """Abstract class for interfacing with any observation data.

    This class defines what data is needed for making maps. It is meant to be
    used as a base class for interfacing with different containers (e.g. toast's
    ``Observation``, sotodlib's ``AxisManager``, litebird_sim's ``Observation``, etc.)
    """

    AVAILABLE_READER_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            ReaderField.METADATA,
            ReaderField.SAMPLE_DATA,
            ReaderField.VALID_SAMPLE_MASKS,
            ReaderField.TIMESTAMPS,
            ReaderField.HWP_ANGLES,
            ReaderField.DETECTOR_QUATERNIONS,
            ReaderField.BORESIGHT_QUATERNIONS,
            ReaderField.NOISE_MODEL_FITS,
        }
    )
    """Supported data field names for all observations"""

    OPTIONAL_READER_FIELDS: ClassVar[frozenset[str]] = frozenset({ReaderField.NOISE_MODEL_FITS})
    """Optional data field names"""

    def __init__(self, data: T) -> None:
        self.data = data

    @classmethod
    @abstractmethod
    def from_file(
        cls, filename: str | Path, requested_fields: Collection[str] | None = None
    ) -> AbstractObservation[T]:
        """Loads the observation from a binary file.

        Args:
            filename: The binary file.
            requested_fields: List of data fields needed.
                If None, the entire file is loaded into memory.
                If `[]` (empty list), loads only what's needed to determine buffer shapes.
                Otherwise, loads whatever is needed to satisfy the request.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Observation name"""

    @property
    @abstractmethod
    def telescope(self) -> str:
        """Telescope name"""

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Returns the number of samples in the observation."""

    @property
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
    def get_tods(self) -> Float[np.ndarray, 'dets samps']:
        """Returns the timestream data.

        Returns a host (numpy) array: getters feed the reader's io_callback, which
        performs a single host->device transfer. Returning a device (jax) array would
        force a wasteful device->host->device round trip at the callback boundary.
        """

    def get_demodulated_tods(self, stokes: ValidStokesType = 'IQU') -> Any:
        """Returns demodulated timestream data as a Stokes pytree.

        Subclasses that support demodulated data should override this method.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support demodulated TODs')

    def get_demodulated_noise_models(self, stokes: ValidStokesType = 'IQU') -> Any:
        """Returns per-Stokes pre-computed noise model fit arrays as a Stokes pytree.

        Subclasses that support demodulated data should override this method.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support demodulated noise models'
        )

    @abstractmethod
    def get_detector_offset_angles(self) -> Float[np.ndarray, ' dets']:
        """Returns the detector offset angles ('gamma')."""

    @abstractmethod
    def get_hwp_angles(self) -> Float[np.ndarray, ' a']:
        """Returns the HWP angles."""

    def get_hwp_frequency(self) -> Float[np.ndarray, '']:
        """Returns the average HWP rotation frequency in Hz."""
        hwp_angles = self.get_hwp_angles()
        timestamps = self.get_timestamps()
        return np.asarray(
            (np.unwrap(hwp_angles)[-1] - hwp_angles[0]) / np.ptp(timestamps) / (2 * np.pi)
        )

    @abstractmethod
    def get_sample_mask(self) -> Bool[np.ndarray, 'dets samps']:
        """Returns boolean sample mask (True=valid) of the TOD."""

    @abstractmethod
    def get_timestamps(self) -> Float[np.ndarray, ' a']:
        """Returns timestamps (sec) of the samples"""

    def get_elapsed_times(self) -> Float[np.ndarray, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self.get_timestamps()
        return timestamps - timestamps[0]  # type: ignore[no-any-return]

    @abstractmethod
    def get_wcs_shape_and_kernel(
        self,
        resolution_arcmin: float,
        projection: ProjectionType = ProjectionType.CAR,
    ) -> tuple[tuple[int, int], WCS]:
        """Returns the shape and object corresponding to a WCS projection"""

    @abstractmethod
    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, ' ...'], Float[Array, ' ...']]:
        """Obtain pointing information and spin angles from the observation"""

    @abstractmethod
    def get_noise_model(self) -> None | NoiseModel:
        """Load a pre-computed noise model from the data, if present. Otherwise, return None"""

    @abstractmethod
    def get_boresight_quaternions(self) -> Float[np.ndarray, 'samp 4']:
        """Returns the boresight quaternions at each time sample"""

    @abstractmethod
    def get_detector_quaternions(self) -> Float[np.ndarray, 'det 4']:
        """Returns the quaternion offsets of the detectors"""


class AbstractSatelliteObservation(AbstractObservation[T]):
    """Class for interfacing with satellite observation data."""

    pass


class AbstractGroundObservation(AbstractObservation[T]):
    """Class for interfacing with ground-based observation data."""

    AVAILABLE_READER_FIELDS: ClassVar[frozenset[str]] = (
        AbstractObservation.AVAILABLE_READER_FIELDS
        | {
            ReaderField.VALID_SCANNING_MASKS,
        }
    )

    @abstractmethod
    def get_scanning_intervals(self) -> NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """

    @abstractmethod
    def get_left_scan_mask(self) -> Bool[np.ndarray, ' samps']:
        """Returns boolean mask (True=valid) for selection of left-going scans"""

    @abstractmethod
    def get_right_scan_mask(self) -> Bool[np.ndarray, ' samps']:
        """Returns boolean mask (True=valid) for selection of right-going scans"""

    @abstractmethod
    def get_azimuth(self) -> Float[np.ndarray, ' a']:
        """Returns the azimuth of the boresight for each sample"""

    @abstractmethod
    def get_elevation(self) -> Float[np.ndarray, ' a']:
        """Returns the elevation of the boresight for each sample"""

    def get_scanning_mask(self) -> Bool[np.ndarray, ' samp']:
        """Returns a boolean sample mask from scanning intervals (True=scanning)."""
        intervals = self.get_scanning_intervals()
        mask = np.zeros(self.n_samples, dtype=bool)
        for l, u in intervals:
            mask[l:u] = True

        return mask

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


class AbstractLazyObservation(ABC, Generic[T]):
    """Deferred handle to an observation: opens its backing store only when read.

    The default implementation is file-backed, but subclasses are free to back the
    observation by any source (e.g. a preprocessing database) by overriding ``get_data``.
    """

    interface_class: type[AbstractObservation[T]]

    @abstractmethod
    def get_data(self, requested_fields: Collection[str] | None = None) -> AbstractObservation[T]:
        """Loads observation data from the underlying source.

        Args:
            requested_fields: List of data fields needed.
                If None, the entire observation is loaded into memory.
                If `[]` (empty list), loads only what's needed to determine buffer shapes.
                Otherwise, loads whatever is needed to satisfy the request.
        """

    def probe_shape(self) -> tuple[int, int]:
        """Returns ``(n_detectors, n_samples)`` to size the padded buffers.

        The default opens the observation with the minimal field set; subclasses may
        override with a cheaper query.
        """
        data = self.get_data([])
        return data.n_detectors, data.n_samples


class FileBackedLazyObservation(AbstractLazyObservation[T]):
    """Lazy observation whose backing store is a single binary file."""

    def __init__(self, filename: str | Path):
        self.file = Path(filename)
        if not self.file.exists():
            raise FileNotFoundError(f'Observation file {self.file} does not exist')

    def get_data(self, requested_fields: Collection[str] | None = None) -> AbstractObservation[T]:
        return self.interface_class.from_file(self.file, requested_fields)
