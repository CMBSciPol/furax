from functools import cached_property
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
import pixell
import so3g.proj
import yaml
from astropy.wcs import WCS
from jaxtyping import Array, Bool, Float, Integer
from numpy.typing import NDArray
from sotodlib import coords
from sotodlib.core import AxisManager
from sotodlib.preprocess.preprocess_util import load_and_preprocess

from furax.mapmaking import AbstractGroundObservation, AbstractGroundObservationResource
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.math import quaternion
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape


class SOTODLibObservation(AbstractGroundObservation[AxisManager]):
    """Class for interfacing with sotodlib's AxisManager."""

    @classmethod
    def from_file(cls, filename: str | Path) -> 'SOTODLibObservation':
        """Loads the observation directly from a binary file."""
        if not Path(filename).exists():
            raise FileNotFoundError(f'File {filename} does not exist')
        if isinstance(filename, Path):
            filename = filename.as_posix()
        data = AxisManager.load(filename)
        return cls(data)

    @classmethod
    def from_preprocess(
        cls,
        preprocess_config: str | Path | dict[str, Any],
        observation_id: str,
        detector_selection: dict[str, str] | None = None,
    ) -> 'SOTODLibObservation':
        """Loads and preprocesses an observation.

        Args:
            preprocess_config: Preprocessing configuration as a path to a yaml file or dictionary.
            observation_id: Observation id.
                (e.g. 'obs_1714550584_satp3_1111111').
            detector_selection: Optional dictionary to select a subset of detectors
                (e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}).
        Returns:
            An instance of SOTODLibObservation.
        """
        if isinstance(preprocess_config, dict):
            config = preprocess_config
        else:
            # load the preprocessing config from a yaml file
            with open(preprocess_config) as file:
                config = yaml.safe_load(file)

        data = load_and_preprocess(observation_id, config, dets=detector_selection)
        return cls(data)

    @property
    def n_samples(self) -> int:
        return self.data.samps.count  # type: ignore[no-any-return]

    @cached_property
    def detectors(self) -> list[str]:
        return self.data.dets.vals  # type: ignore[no-any-return]

    @property
    def n_detectors(self) -> int:
        return self.data.dets.count  # type: ignore[no-any-return]

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        duration: float = self.data.timestamps[-1] - self.data.timestamps[0]
        return (self.n_samples - 1) / duration

    def get_tods(self) -> Array:
        """Returns the timestream data."""
        tods = jnp.array(self.data.signal, dtype=jnp.float64)
        return jnp.atleast_2d(tods)

    def get_detector_offset_angles(self) -> Array:
        """Returns the detector offset angles."""
        return jnp.array(self.data.focal_plane['gamma'])

    def get_hwp_angles(self) -> Array:
        """Returns the HWP angles."""
        return jnp.array(self.data.hwp_angle)

    def get_scanning_intervals(self, det_ind: int = 0) -> NDArray[Any]:
        """Returns scanning intervals of the chosen detector.
        The output is a list of the starting and ending sample indices
        """
        return np.array(
            self.data.preprocess.turnaround_flags.turnarounds.ranges[det_ind].complement().ranges()
        )

    def get_sample_mask(self) -> Bool[Array, 'dets samps']:
        try:
            return jnp.array((~self.data.flags.glitch_flags).mask(), dtype=bool)
        except KeyError:
            raise KeyError('Glitch flags unavailable in the observation')

    def get_left_scan_mask(self) -> Bool[Array, ' samps']:
        try:
            return jnp.array(self.data.flags.left_scan.mask(), dtype=bool)
        except KeyError:
            raise KeyError('Scan mask unavailable in the observation')

    def get_right_scan_mask(self) -> Bool[Array, ' samps']:
        try:
            return jnp.array(self.data.flags.right_scan.mask(), dtype=bool)
        except KeyError:
            raise KeyError('Scan mask unavailable in the observation')

    def get_azimuth(self) -> Float[Array, ' a']:
        """Returns the azimuth of the boresight for each sample"""
        return jnp.array(self.data.boresight.az)

    def get_elevation(self) -> Float[Array, ' a']:
        """Returns the elevation of the boresight for each sample"""
        return jnp.array(self.data.boresight.el)

    def get_wcs_shape_and_kernel(
        self,
        resolution: float = 8.0,  # units: arcmins
        projection: str = 'car',
    ) -> tuple[tuple[int, ...], WCS]:
        """Returns astropy WCS kernel object corresponding to the observed sky"""

        res = resolution * pixell.utils.arcmin
        wcs_kernel_init = coords.get_wcs_kernel('car', 0, 0, res=res)
        wcs_shape, wcs_kernel = coords.get_footprint(self.data, wcs_kernel_init)

        return wcs_shape, wcs_kernel

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Integer[Array, 'dets samps 2'], Float[Array, 'dets samps']]:
        """Obtain pointing information and spin angles from the observation"""

        # Projection Matrix class instance for the observation
        if isinstance(landscape, WCSLandscape):
            # TODO: pass 'cuts' keyword here for time slices (glitches etc)?
            P = coords.P.for_tod(self.data, wcs_kernel=landscape.wcs, comps='TQU', hwp=True)
        elif isinstance(landscape, HealpixLandscape):
            hp_geom = coords.healpix_utils.get_geometry(nside=landscape.nside, ordering='RING')
            P = coords.P.for_tod(self.data, geom=hp_geom)
        else:
            raise NotImplementedError(f'Landscape {landscape} not supported')

        # Projectionist object from P
        proj = P._get_proj()

        # Assembly containing the focal plane and boresight information
        assembly = P._get_asm()

        # Get the pixel indicies as before, but also obtain
        # the spin projection factors of size (n_samps,n_comps) for each detector,
        # which have 1, cos(2*p), sin(2*p) where p is the parallactic angle
        pixel_inds, spin_proj = proj.get_pointing_matrix(assembly)

        # TODO: check if this could be jnp array directly
        pixel_inds = np.array(pixel_inds)

        spin_proj = jnp.array(spin_proj, dtype=landscape.dtype)
        spin_ang = jnp.arctan2(spin_proj[..., 2], spin_proj[..., 1]) / 2.0

        return pixel_inds, spin_ang  # type: ignore[return-value]

    def get_timestamps(self) -> Float[Array, ' a']:
        """Returns timestamps (sec) of the samples"""
        return jnp.array(self.data.timestamps)

    def get_scanning_mask(self) -> Bool[Array, ' samp']:
        # Assume that all detectors have the same scanning intervals
        return jnp.array(
            self.data.preprocess.turnaround_flags.turnarounds.ranges[0].complement().mask(),
            dtype=bool,
        )

    def get_noise_model(self) -> None | NoiseModel:
        """Load precomputed noise model from the data, if present. Otherwise, return None"""

        preproc = self.data.get('preprocess')
        if preproc is None:
            return None

        if 'noiseT' in preproc:
            fit = preproc.noiseT
        else:
            return None

        # sotodlib fit's columns: (w, fknee, alpha), with
        # psd = wn**2 * (1 + (fknee / f) ** alpha)
        # Note the difference in the sign of alpha
        assert np.all(fit.noise_model_coeffs.vals == np.array(['white_noise', 'fknee', 'alpha']))

        return AtmosphericNoiseModel(
            sigma=jnp.array(fit.fit[:, 0]),
            alpha=jnp.array(-fit.fit[:, 2]),
            fk=jnp.array(fit.fit[:, 1]),
            f0=1e-5 * jnp.ones_like(fit.fit[:, 1]),
        )

    '''
    def get_noise_fits(self, fmin: float) -> NDArray[np.float64]:
        """Returns fitted values of the noise psd with 1/f and white noise,
        either using the fitted parameters from the preprocessing,
        or fitting the model directly.
        #TODO: reimplement below by calling get_noise_model() instead
        """
        preproc = self.observation.preprocess

        if 'psdT' in preproc:
            f = preproc.psdT.freqs
            fit = preproc.noiseT_fit.fit  # columns: (fknee, w, alpha)
        elif 'Pxx_raw' in preproc:
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
        if 'psdT' in preproc:
            fit = preproc.noiseT_fit.fit  # columns: (fknee, w, alpha)
        elif 'Pxx_raw' in preproc:
            fit = preproc.noise_signal_fit.fit  # columns: (fknee, w, alpha)
        else:
            # Estimate psd
            raise NotImplementedError('Self-psd evaluation not implemented')
        return fit[:, 1]  # type: ignore[no-any-return]
    '''

    def get_boresight_quaternions(self) -> Float[Array, 'samp 4']:
        """Returns the boresight quaternions at each time sample"""
        csl = so3g.proj.CelestialSightLine.az_el(
            self.data.timestamps,
            self.data.boresight.az,
            self.data.boresight.el,
            roll=self.data.boresight.roll,
            site='so',
            weather='typical',
        )
        return jnp.array(csl.Q, dtype=jnp.float64)

    def get_detector_quaternions(self) -> Float[Array, 'det 4']:
        """Returns the quaternion offsets of the detectors"""
        quats = quaternion.from_xieta_angles(
            jnp.array(self.data.focal_plane.xi, dtype=jnp.float64),
            jnp.array(self.data.focal_plane.eta, dtype=jnp.float64),
            jnp.array(self.data.focal_plane.gamma, dtype=jnp.float64),
        )

        return jnp.atleast_2d(quats)


class SOTODLibObservationResource(AbstractGroundObservationResource[SOTODLibObservation]):
    def request(self, field_names: list[str] | None = None) -> SOTODLibObservation:
        if field_names is None:
            # load just the info needed to determine the data shape
            fields = ['dets', 'samps']
        else:
            # translate request to sotodlib subfield names
            fields = []
            if 'sample_data' in field_names:
                fields.append('signal')
            if 'valid_sample_masks' in field_names:
                fields.append('flags.glitch_flags')
            if 'valid_scanning_masks' in field_names:
                fields.append('preprocess.turnaround_flags')
            if 'timestamps' in field_names:
                fields.append('timestamps')
            if 'hwp_angles' in field_names:
                fields.append('hwp_angle')
            if 'boresight_quaternions' in field_names:
                fields.append('boresight')
                if 'timestamps' not in fields:
                    fields.append('timestamps')
            if 'detector_quaternions' in field_names:
                fields.append('focal_plane')
            if 'noise_model_fits' in field_names:
                fields.append('preprocess.noiseT')

        with h5py.File(self.file.as_posix(), 'r') as h:
            aman = AxisManager.load(h, fields=fields)

        return SOTODLibObservation(aman)
