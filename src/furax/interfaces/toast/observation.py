import typing
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as np
import toast
from astropy import units as u
from astropy.wcs import WCS
from jaxtyping import Array, Bool, Float
from numpy.typing import NDArray
from toast.observation import default_values as defaults

from furax.mapmaking import AbstractGroundObservation
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.mapmaking.utils import get_local_meridian_angle
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape


class ToastObservation(AbstractGroundObservation[toast.Data]):
    def __init__(
        self,
        data: toast.Data,  # Currently intended to have only 1 observation
        *,
        det_selection: list[str] | None = None,
        det_mask: int = defaults.det_mask_nonscience,
        det_data: str = defaults.det_data,
        pixels: str = defaults.pixels,
        quats: str = defaults.quats,
        hwp_angle: str | None = defaults.hwp_angle,
        noise_model: str | None = defaults.noise_model,
        azimuth: str = defaults.azimuth,
        elevation: str = defaults.elevation,
        boresight: str = defaults.boresight_radec,
        cross_psd: tuple[Float[Array, ' freq'], Float[Array, 'det det freq']] | None = None,
    ) -> None:
        super().__init__(data)
        self._det_selection = det_selection
        self._det_mask = det_mask
        self._det_data = det_data
        self._pixels = pixels
        self._quats = quats
        self._hwp_angle = hwp_angle
        self._noise_model = noise_model
        self._azimuth = azimuth
        self._elevation = elevation
        self._boresight = boresight
        self._cross_psd = cross_psd

    @property
    def _observation(self) -> toast.Observation:
        return self.data.obs[0]

    @property
    def n_samples(self) -> int:
        return self._observation.n_local_samples  # type: ignore[no-any-return]

    @cached_property
    def detectors(self) -> list[str]:
        """Returns a list of the detector names."""
        local_selection: list[str] = self._observation.select_local_detectors(
            selection=self._det_selection, flagmask=self._det_mask
        )
        return local_selection

    @property
    def _focal_plane(self) -> toast.Focalplane:
        return self._observation.telescope.focalplane

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        return self._focal_plane.sample_rate.to_value(u.Hz)  # type: ignore[no-any-return]

    def get_tods(self) -> Array:
        """Returns the timestream data."""
        # furax's LinearPolarizerOperator assumes power, TOAST assumes temperature
        tods = 0.5 * jnp.array(self._observation.detdata[self._det_data][self.detectors, :])
        return jnp.atleast_2d(tods)

    def _get_detector_angles(self) -> Array:
        """Returns the detector angles on the sky."""
        # stick to the TOAST storage convention because get_local_meridian_angle expects it
        quats = self._get_expanded_quats()
        angles = jnp.array(get_local_meridian_angle(quats))
        return jnp.atleast_2d(angles)

    def get_detector_offset_angles(self) -> Array:
        """Returns the detector offset angles."""
        fp = self._focal_plane
        return jnp.array([fp[det]['gamma'].to_value(u.rad) for det in self.detectors])

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        if self._hwp_angle is None or self._hwp_angle not in self._observation.shared:
            raise ValueError('HWP angle field not provided.')
        return jnp.array(self._observation.shared[self._hwp_angle].data)

    def _get_psd_model(self) -> tuple[Array, Array]:
        """Returns frequencies and PSD values of the noise model."""
        if self._noise_model is None or self._noise_model not in self._observation:
            raise ValueError('Noise model not provided.')
        model = self._observation[self._noise_model]
        freq = jnp.array([model.freq(det) for det in self.detectors])
        psd = jnp.array([model.psd(det) for det in self.detectors])
        return freq, psd

    def get_scanning_intervals(self) -> NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """
        if (
            not hasattr(self._observation, 'intervals')
            or 'scanning' not in self._observation.intervals
        ):
            # Scanning information missing, first compute the intervals
            toast.ops.AzimuthIntervals().apply(self.data)
        intervals = self._observation.intervals['scanning']
        return np.array(intervals[['first', 'last']].tolist())

    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        return jnp.array(self._observation.detdata['flags'].data == 0, dtype=bool)

    def get_left_scan_mask(self) -> Bool[Array, ' samps']:
        if not hasattr(self._observation, 'intervals'):
            # Scanning information missing, first compute the intervals
            toast.ops.AzimuthIntervals().apply(self.data)

        # Left scan means scanning FROM right TO left
        intervals_list = self._observation.intervals['scan_rightleft'][['first', 'last']].tolist()
        mask = jnp.zeros(self.n_samples, dtype=bool)
        for start, stop in intervals_list:
            mask[start:stop] = True
        return mask

    def get_right_scan_mask(self) -> Bool[Array, ' samps']:
        if not hasattr(self._observation, 'intervals'):
            # Scanning information missing, first compute the intervals
            toast.ops.AzimuthIntervals().apply(self.data)

        # Right scan means scanning FROM left TO right
        intervals_list = self._observation.intervals['scan_leftright'][['first', 'last']].tolist()
        mask = jnp.zeros(self.n_samples, dtype=bool)
        for start, stop in intervals_list:
            mask[start:stop] = True
        return mask

    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        if self._azimuth not in self._observation.shared:
            raise ValueError('Azimuth field not provided.')
        return jnp.array(self._observation.shared[self._azimuth].data)

    def get_elevation(self) -> Float[Array, ' a']:
        """Returns the elevation of the boresight for each sample"""
        if self._elevation not in self._observation.shared:
            raise ValueError('Elevation field not provided.')
        return jnp.array(self._observation.shared[self._elevation].data)

    def get_elapsed_times(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self._observation.shared['times'].data
        t0: float = timestamps[0]
        return jnp.array(timestamps) - t0

    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], WCS]:
        """Returns the shape and object corresponding to a WCS projection.
        Here, this is obtained while we compute the pointing and pixelisation."""

        det_pointing = toast.ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats=self._quats
        )
        det_pixels = toast.ops.PixelsWCS(
            detector_pointing=det_pointing,
            pixels=self._pixels,
            resolution=[resolution * u.arcmin, resolution * u.arcmin],
            dimensions=tuple(),
        )
        det_pixels.apply(self.data)

        # Un-flatten the pixel indices
        pix = self._get_pixel_indices() % det_pixels._n_pix
        pix_dec = pix // det_pixels.pix_lat
        pix_ra = pix % det_pixels.pix_lat
        tot_pix = np.stack([pix_dec, pix_ra], axis=-1)
        del self._observation.detdata[self._pixels]
        self._observation.detdata[self._pixels] = tot_pix

        return det_pixels.wcs_shape, det_pixels.wcs

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, '...'], Float[Array, '...']]:
        """Obtain pointing information and spin angles from the observation"""

        det_keys = self._observation.detdata.keys()
        if self._quats in det_keys and self._pixels in det_keys:
            indices = self._get_pixel_indices()
            spin_ang = self._get_detector_angles() - 2 * self.get_detector_offset_angles()[:, None]
            return indices, spin_ang

        elif isinstance(landscape, WCSLandscape):
            raise ValueError(
                'Pointing information is missing from the data. \
                        This is supposed to be obtained when computing \
                        the WCS kernel.'
            )

        det_pointing = toast.ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats=self._quats
        )

        if isinstance(landscape, HealpixLandscape):
            det_pixels = toast.ops.PixelsHealpix(
                detector_pointing=det_pointing,
                pixels=self._pixels,
                nside=landscape.nside,
                nest=False,
            )
            det_pixels.apply(self.data)
            indices = self._get_pixel_indices()
            spin_ang = self._get_detector_angles() - 2 * self.get_detector_offset_angles()[:, None]
            return indices, spin_ang
        else:
            raise ValueError('Invalid landscape type')

    def _get_pixel_indices(self) -> Array:
        """Returns the pixel indices."""
        pixels = jnp.array(self._observation.detdata[self._pixels][self.detectors, :])
        return jnp.atleast_2d(pixels)

    @typing.no_type_check
    def get_noise_model(self) -> None | NoiseModel:
        """Load noise model from the focalplane data, if present. Otherwise, return None"""

        noise_keys = ['psd_fmin', 'psd_fknee', 'psd_alpha', 'psd_net']
        fp_data = self._observation.telescope.focalplane.detector_data
        for key in noise_keys:
            if key not in fp_data.colnames:
                # Noise model cannot be loaded from the observation
                return None

        idets = np.argwhere(np.array(self.detectors)[:, None] == fp_data['name'][None, :])[:, 1]
        noise_model = AtmosphericNoiseModel(
            sigma=jnp.array(fp_data['psd_net'][idets].to(u.K * u.s**0.5).value),
            alpha=jnp.array(-fp_data['psd_alpha'][idets].value),  # Note toast's sign convention
            fk=jnp.array(fp_data['psd_fknee'][idets].to(u.Hz).value),
            f0=jnp.array(fp_data['psd_fmin'][idets].to(u.Hz).value),
        )
        return noise_model

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        if self._boresight not in self._observation.shared:
            raise ValueError('Boresight field not provided.')
        quats = jnp.array(self._observation.shared[self._boresight].data)
        return jnp.roll(quats, 1, axis=-1)

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        quats = jnp.array([self._focal_plane[d]['quat'] for d in self.detectors])
        quats = jnp.roll(quats, 1, axis=-1)
        return jnp.atleast_2d(quats)

    def _get_expanded_quats(self) -> Array:
        """Returns expanded pointing quaternions.

        These will be in the TOAST storage convention, i.e. vector-scalar!
        """
        quats = jnp.array(self._observation.detdata[self._quats][self.detectors, :])
        if quats.ndim >= 3:
            return quats
        # np.atleast_3d appends one new axis for 1d/2d inputs, we want to prepend it instead
        return jnp.moveaxis(jnp.atleast_3d(quats), -1, 0)
