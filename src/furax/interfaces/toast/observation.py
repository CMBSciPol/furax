import typing
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import toast
from astropy import units as u
from astropy.wcs import WCS
from jaxtyping import Array, Float
from numpy.typing import NDArray
from toast.observation import default_values as defaults

from furax.mapmaking import GroundObservationData
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.mapmaking.utils import get_local_meridian_angle
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape


@jax.tree_util.register_dataclass
@dataclass
class ToastObservationData(GroundObservationData):
    data: toast.Data  # Currently intended to have only 1 observation
    det_selection: list[str] | None = None
    det_mask: int = defaults.det_mask_nonscience

    # the names of the fields we need
    det_data: str = defaults.det_data
    pixels: str = defaults.pixels
    quats: str = defaults.quats
    hwp_angle: str | None = defaults.hwp_angle
    noise_model: str | None = defaults.noise_model
    azimuth: str = defaults.azimuth
    elevation: str = defaults.elevation
    boresight: str = defaults.boresight_radec

    _cross_psd: tuple[Float[Array, ' freq'], Float[Array, 'det det freq']] | None = None

    @property
    def observation(self) -> toast.Observation:
        return self.data.obs[0]

    @property
    def n_samples(self) -> int:
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
        tods = 0.5 * jnp.array(self.observation.detdata[self.det_data][self.dets, :])
        return jnp.atleast_2d(tods)

    def get_det_angles(self) -> Array:
        """Returns the detector angles on the sky."""
        # stick to the TOAST storage convention because get_local_meridian_angle expects it
        quats = self._get_expanded_quats()
        angles = jnp.array(get_local_meridian_angle(quats))
        return jnp.atleast_2d(angles)

    def get_det_offset_angles(self) -> Array:
        """Returns the detector offset angles."""
        fp = self.focal_plane
        return jnp.array([fp[det]['gamma'].to_value(u.rad) for det in self.dets])

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        if self.hwp_angle is None or self.hwp_angle not in self.observation.shared:
            raise ValueError('HWP angle field not provided.')
        return jnp.array(self.observation.shared[self.hwp_angle].data)

    def get_psd_model(self) -> tuple[Array, Array]:
        """Returns frequencies and PSD values of the noise model."""
        if self.noise_model is None or self.noise_model not in self.observation:
            raise ValueError('Noise model not provided.')
        model = self.observation[self.noise_model]
        freq = jnp.array([model.freq(det) for det in self.dets])
        psd = jnp.array([model.psd(det) for det in self.dets])
        return freq, psd

    def get_scanning_intervals(self) -> NDArray[Any]:
        """Returns scanning intervals.
        The output is a list of the starting and ending sample indices
        """
        if (
            not hasattr(self.observation, 'intervals')
            or 'scanning' not in self.observation.intervals
        ):
            # Scanning information missing, first compute the intervals
            toast.ops.AzimuthIntervals().apply(self.data)
        intervals = self.observation.intervals['scanning']
        return np.array(intervals[['first', 'last']].tolist())

    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        if self.azimuth not in self.observation.shared:
            raise ValueError('Azimuth field not provided.')
        return jnp.array(self.observation.shared[self.azimuth].data)

    def get_elevation(self) -> Float[Array, ' a']:
        """Returns the elevation of the boresight for each sample"""
        if self.elevation not in self.observation.shared:
            raise ValueError('Elevation field not provided.')
        return jnp.array(self.observation.shared[self.elevation].data)

    def get_elapsed_time(self) -> Float[Array, ' a']:
        """Returns time (sec) of the samples since the observation began"""
        timestamps = self.observation.shared['times'].data
        return jnp.array(timestamps - timestamps[0])

    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], WCS]:
        """Returns the shape and object corresponding to a WCS projection.
        Here, this is obtained while we compute the pointing and pixelisation."""

        det_pointing = toast.ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats=self.quats
        )
        det_pixels = toast.ops.PixelsWCS(
            detector_pointing=det_pointing,
            pixels=self.pixels,
            resolution=[resolution * u.arcmin, resolution * u.arcmin],
            dimensions=tuple(),
        )
        det_pixels.apply(self.data)

        # Un-flatten the pixel indices
        pix = self.get_pixels() % det_pixels._n_pix
        pix_dec = pix // det_pixels.pix_lat
        pix_ra = pix % det_pixels.pix_lat
        tot_pix = np.stack([pix_dec, pix_ra], axis=-1)
        del self.observation.detdata[self.pixels]
        self.observation.detdata[self.pixels] = tot_pix

        return det_pixels.wcs_shape, det_pixels.wcs

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Float[Array, '...'], Float[Array, '...']]:
        """Obtain pointing information and spin angles from the observation"""

        det_keys = self.observation.detdata.keys()
        if self.quats in det_keys and self.pixels in det_keys:
            indices = self.get_pixels()
            spin_ang = self.get_det_angles() - 2 * self.get_det_offset_angles()[:, None]
            return indices, spin_ang

        elif isinstance(landscape, WCSLandscape):
            raise ValueError(
                'Pointing information is missing from the data. \
                        This is supposed to be obtained when computing \
                        the WCS kernel.'
            )

        det_pointing = toast.ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats=self.quats
        )

        if isinstance(landscape, HealpixLandscape):
            det_pixels = toast.ops.PixelsHealpix(
                detector_pointing=det_pointing,
                pixels=self.pixels,
                nside=landscape.nside,
                nest=False,
            )
            det_pixels.apply(self.data)
            indices = self.get_pixels()
            spin_ang = self.get_det_angles() - 2 * self.get_det_offset_angles()[:, None]
            return indices, spin_ang
        else:
            raise ValueError('Invalid landscape type')

    def get_pixels(self) -> Array:
        """Returns the pixel indices."""
        pixels = jnp.array(self.observation.detdata[self.pixels][self.dets, :])
        return jnp.atleast_2d(pixels)

    @typing.no_type_check
    def get_noise_model(self) -> None | NoiseModel:
        """Load noise model from the focalplane data, if present. Otherwise, return None"""

        noise_keys = ['psd_fmin', 'psd_fknee', 'psd_alpha', 'psd_net']
        fp_data = self.observation.telescope.focalplane.detector_data
        for key in noise_keys:
            if key not in fp_data.colnames:
                # Noise model cannot be loaded from the observation
                return None

        idets = np.argwhere(np.array(self.dets)[:, None] == fp_data['name'][None, :])[:, 1]
        noise_model = AtmosphericNoiseModel(
            sigma=jnp.array(fp_data['psd_net'][idets].value),
            alpha=jnp.array(fp_data['psd_alpha'][idets].value),
            fk=jnp.array(fp_data['psd_fknee'][idets].value),
            f0=jnp.array(fp_data['psd_fmin'][idets].value),
        )
        return noise_model

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        if self.boresight not in self.observation.shared:
            raise ValueError('Boresight field not provided.')
        quats = jnp.array(self.observation.shared[self.boresight].data)
        return jnp.roll(quats, 1, axis=-1)

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        quats = jnp.array([self.focal_plane[d]['quat'] for d in self.dets])
        quats = jnp.roll(quats, 1, axis=-1)
        return jnp.atleast_2d(quats)

    def _get_expanded_quats(self) -> Array:
        """Returns expanded pointing quaternions.

        These will be in the TOAST storage convention, i.e. vector-scalar!
        """
        quats = jnp.array(self.observation.detdata[self.quats][self.dets, :])
        if quats.ndim >= 3:
            return quats
        # np.atleast_3d appends one new axis for 1d/2d inputs, we want to prepend it instead
        return jnp.moveaxis(jnp.atleast_3d(quats), -1, 0)
