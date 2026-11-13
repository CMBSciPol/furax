from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal, Self, overload

import jax.numpy as jnp
import numpy as np
import pixell.utils
import so3g.proj
import sotodlib.preprocess.preprocess_util as pu
import yaml
from astropy.wcs import WCS
from jaxtyping import Array, Bool, Float, Integer
from numpy.typing import NDArray
from sotodlib import coords
from sotodlib.coords.helpers import get_deflected_sightline
from sotodlib.core import AxisManager
from sotodlib.core.flagman import has_any_cuts
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.mapmaking.utils import downsample_cut, downsample_obs
from sotodlib.preprocess.pcore import Pipeline

from furax.mapmaking import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    FileBackedLazyObservation,
    ObservationBufferShape,
    ObservationProbe,
    ReaderField,
    SampleRanges,
)
from furax.mapmaking.config import SotodlibConfig
from furax.mapmaking.noise import AtmosphericNoiseModel, NoiseModel
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    ProjectionType,
    StokesLandscape,
)
from furax.obs.stokes import (
    Stokes,
    StokesI,
    StokesIQU,
    StokesIQUV,
    StokesQU,
    StokesType,
    ValidStokesLiteral,
)

from ._cache import PreprocCache, cache_inputs

logger = logging.getLogger(__name__)


class NoDetectorsLeftError(RuntimeError):
    """Preprocessing cut every detector of the requested observation or group."""


# A detector selection for sotodlib, e.g. {'wafer_slot': ['ws0', 'ws1'], 'wafer.bandpass': 'f090'}
DetectorSelection = dict[str, str | list[str]]

# Part of the message of the LoaderError for an observation without a preprocessing record
_NO_RECORD = 'no metadata records found'

# The flags the loaders mask, under `flags`, which the probe rebuilds from the metadata
_MASKED_FLAGS = 'glitch_flags'

# Per-process cache of (configs, context) from sotodlib's get_preprocess_context, keyed by
# (config-file path, thread id). See _enable_preproc_context_cache.
_PREPROC_CONTEXT_CACHE: dict[tuple[str, int], tuple[Any, Any]] = {}


def _enable_preproc_context_cache() -> None:
    """Memoize sotodlib's ``get_preprocess_context`` per process and thread.

    ``multilayer_load_and_preprocess`` rebuilds the ``core.Context`` on every call and exposes
    no hook to inject a prebuilt one, so each observation load reopens the obsdb/obsfiledb sqlite
    indices. Wrapping ``get_preprocess_context`` with a cache builds the Context (and opens those
    indices) once per config layer; ``multilayer`` calls it by module-global name, so the patch
    takes effect without reimplementing the loader.

    The cache is keyed by thread id as well as config path: sotodlib's sqlite connections are
    thread-affine, and the reader runs the data load inside a ``jax.io_callback`` thread distinct
    from the shape-probe phase. A shared Context would raise "SQLite objects created in a thread
    can only be used in that same thread".
    """
    if getattr(pu.get_preprocess_context, '_furax_cached', False):
        return
    original = pu.get_preprocess_context

    @functools.wraps(original)
    def cached(configs: Any, context: Any = None) -> tuple[Any, Any]:
        # Only memoize the hot path: a config given by path with no caller-supplied context.
        # Keyed by config path (not context_file) so init/proc layers stay distinct even when
        # they share a context — each keeps its own appended preprocess archive.
        if context is not None or not isinstance(configs, str):
            return original(configs, context)
        key = (configs, threading.get_ident())
        if key not in _PREPROC_CONTEXT_CACHE:
            _PREPROC_CONTEXT_CACHE[key] = original(configs, context)
        return _PREPROC_CONTEXT_CACHE[key]

    cached._furax_cached = True  # ty: ignore[unresolved-attribute]
    # Relies on sotodlib calling get_preprocess_context by its module-global name (which
    # load_and_preprocess / multilayer_load_and_preprocess do): reassigning the module attribute
    # is what makes the cache take effect. A future sotodlib that imports it as a local alias, or
    # builds core.Context directly, would bypass this.
    pu.get_preprocess_context = cached  # ty: ignore[invalid-assignment]


class SOTODLibObservation(AbstractGroundObservation[AxisManager]):
    """Class for interfacing with sotodlib's AxisManager."""

    def __init__(
        self,
        data: AxisManager,
        sotodlib_config: SotodlibConfig | None = None,
        *,
        processed: bool = False,
    ) -> None:
        """Wraps an AxisManager, applying the steps that `sotodlib_config` asks for.

        Args:
            data: The observation data.
            sotodlib_config: The sotodlib-specific configuration.
            processed: Whether the data already went through the steps of `sotodlib_config`, such
                as the downsampling, e.g. when it is read back from a cache. If so, they are skipped.
        """
        sotodlib_config = sotodlib_config or SotodlibConfig()
        if sotodlib_config.downsample > 1 and not processed:
            # downsample_obs only Fourier-resamples 'signal' by default
            timestreams = ['signal', 'dsT', 'demodQ', 'demodU']
            data = downsample_obs(data, sotodlib_config.downsample, fft_resample=timestreams)
        super().__init__(data)
        self._sotodlib_config = sotodlib_config

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        requested_fields: Collection[str] | None = None,
        sotodlib_config: SotodlibConfig | None = None,
        *,
        processed: bool = False,
    ) -> Self:
        # check that file exists
        if not Path(filename).exists():
            raise FileNotFoundError(f'File {filename} does not exist')
        if isinstance(filename, Path):
            filename = filename.as_posix()

        sotodlib_config = sotodlib_config or SotodlibConfig()
        if requested_fields is None:
            # default is to load everything
            data = AxisManager.load(filename, fields=None)
            return cls(data, sotodlib_config, processed=processed)

        requested = set(requested_fields)
        # minimum information needed to determine buffer shapes and name the observation
        fields = {'dets', 'samp', 'obs_info'}
        # translate request to sotodlib subfield names (sets dedup overlapping requests)
        if ReaderField.SAMPLE_DATA in requested:
            if sotodlib_config.demodulated:
                fields |= {'dsT', 'demodQ', 'demodU'}
            else:
                fields.add('signal')
        if ReaderField.VALID_SAMPLE_MASKS in requested:
            fields.add(f'flags.{_MASKED_FLAGS}')
        if {ReaderField.VALID_SCANNING_MASKS, ReaderField.SCANNING_INTERVALS} & requested:
            fields.add('preprocess.turnaround_flags')
        if ReaderField.LEFT_SCAN_MASK in requested:
            fields.add('flags.left_scan')
        if ReaderField.RIGHT_SCAN_MASK in requested:
            fields.add('flags.right_scan')
        if {ReaderField.AZIMUTH, ReaderField.ELEVATION} & requested:
            fields.add('boresight')
        if ReaderField.TIMESTAMPS in requested:
            fields.add('timestamps')
        if ReaderField.HWP_ANGLES in requested:
            fields.add('hwp_angle')
        if ReaderField.BORESIGHT_QUATERNIONS in requested:
            fields |= {'boresight', 'timestamps'}
            if sotodlib_config.wobble_correction:
                fields |= {'wobble_params', 'det_info', 'hwp_angle'}
        if ReaderField.DETECTOR_QUATERNIONS in requested:
            fields.add('focal_plane')
        if ReaderField.NOISE_MODEL_FITS in requested:
            if sotodlib_config.noise_source == 'mapmaking':
                fields.add('preprocess.noiseQ_mapmaking')
            else:
                fields |= {'preprocess.noiseT', 'preprocess.noiseQ', 'preprocess.noiseU'}

        data = AxisManager.load(filename, fields=list(fields))
        return cls(data, sotodlib_config, processed=processed)

    @classmethod
    def from_preprocess(
        cls,
        preprocess_config: str | Path | dict[str, Any],
        observation_id: str,
        detector_selection: DetectorSelection | None = None,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> Self:
        """Loads and preprocesses an observation.

        Args:
            preprocess_config: Preprocessing configuration as a path to a yaml file or dictionary.
            observation_id: Observation id.
                (e.g. 'obs_1714550584_satp3_1111111').
            detector_selection: Optional dictionary to select a subset of detectors
                (e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}).
            sotodlib_config: Optional sotodlib-specific configuration.

        Returns:
            An instance of SOTODLibObservation.
        """
        if isinstance(preprocess_config, dict):
            preprocess_dict = preprocess_config
        else:
            # load the preprocessing config from a yaml file
            with open(preprocess_config) as file:
                preprocess_dict = yaml.safe_load(file)

        data = pu.load_and_preprocess(observation_id, preprocess_dict, dets=detector_selection)
        return cls(data, sotodlib_config)

    @classmethod
    def from_preproc_group(
        cls,
        observation_id: str,
        init_config: str | Path,
        proc_config: str | Path | None = None,
        detector_selection: DetectorSelection | None = None,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> Self:
        """Loads a (already preprocessed) observation directly from the preprocessing db.

        This reads the saved preprocessing products from the archive and re-applies the
        process pipeline, without writing any intermediate file. It mirrors the load path
        of ``preproc_or_load_group`` but skips the archive/proc-aman saving.

        Args:
            observation_id: Observation id (e.g. 'obs_1714550584_satp3_1111111').
            init_config: Base-layer preprocessing config (path or posix string).
            proc_config: Optional second-layer preprocessing config. If given, the
                two-layer (init+proc) load path is used.
            detector_selection: Optional detector restriction
                (e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}).
            sotodlib_config: Optional sotodlib-specific configuration.

        Returns:
            An instance of SOTODLibObservation.

        Raises:
            NoDetectorsLeftError: If no detectors remain after cuts (nothing to load).
        """
        _enable_preproc_context_cache()
        init_config = Path(init_config).as_posix()
        if proc_config is not None:
            aman = pu.multilayer_load_and_preprocess(
                observation_id,
                init_config,
                Path(proc_config).as_posix(),
                dets=detector_selection,
            )
        else:
            # load_and_preprocess returns (aman, full_aman); we only need the restricted one
            result = pu.load_and_preprocess(observation_id, init_config, dets=detector_selection)
            aman = result[0] if isinstance(result, tuple) else result
        if aman is None:
            raise NoDetectorsLeftError(f'no detectors left after cuts for {observation_id}')
        return cls(aman, sotodlib_config)

    @classmethod
    def from_preproc_groups(
        cls,
        observation_id: str,
        init_config: str | Path,
        proc_config: str | Path | None = None,
        detector_selection: DetectorSelection | None = None,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> Self:
        """Loads every preprocessing group of an observation as a single observation.

        The preprocessing runs per group of detectors (the config's `subobs.use`, e.g. wafer slot
        and bandpass). Each group matching `detector_selection` is loaded as by
        [`from_preproc_group`][furax.interfaces.sotodlib.observation.SOTODLibObservation.from_preproc_group], so it is preprocessed exactly as on its own,
        and the groups are joined along the detectors: with a bandpass selection, the observation
        spans the whole focal plane. Groups with no detectors left after cuts, or with no
        preprocessing record, are skipped.

        Args:
            observation_id: Observation id (e.g. 'obs_1714550584_satp3_1111111').
            init_config: Base-layer preprocessing config (path or posix string).
            proc_config: Optional second-layer preprocessing config.
            detector_selection: Optional detector restriction (e.g. {'wafer.bandpass': 'f150'}).
            sotodlib_config: Optional sotodlib-specific configuration.

        Returns:
            An instance of SOTODLibObservation.

        Raises:
            RuntimeError: If the groups cannot be listed, or no detectors remain in any of them.
            ValueError: If the groups do not share their samples.
        """
        sotodlib_config = sotodlib_config or SotodlibConfig()
        parts = []
        for selection in group_selections(observation_id, init_config, detector_selection):
            try:
                part = cls.from_preproc_group(
                    observation_id, init_config, proc_config, selection, sotodlib_config
                )
            except LoaderError as error:  # the group was not preprocessed
                if not _is_missing_record(error):
                    raise
                logger.warning('No preprocessing record for %s in %s', selection, observation_id)
                continue
            except NoDetectorsLeftError:
                continue
            parts.append(part)
        if not parts:
            raise NoDetectorsLeftError(f'no detectors left after cuts for {observation_id}')
        return cls(_join_groups(observation_id, parts), sotodlib_config, processed=True)

    @property
    def name(self) -> str:
        return self.data.obs_info.obs_id

    @property
    def telescope(self) -> str:
        return self.data.obs_info.get('telescope')

    @property
    def n_samples(self) -> int:
        return self.data.samps.count

    @property
    def detectors(self) -> list[str]:
        return self.data.dets.vals

    @property
    def n_detectors(self) -> int:
        return self.data.dets.count

    @property
    def sample_rate(self) -> float:
        """Returns the sampling rate (in Hz) of the data."""
        duration: float = self.data.timestamps[-1] - self.data.timestamps[0]
        return (self.n_samples - 1) / duration

    def get_tods(
        self, out: Float[np.ndarray, 'dets samps'] | None = None
    ) -> Float[np.ndarray, 'dets samps']:
        """Returns the timestream data, in the precision it is stored in."""
        # furax's LinearPolarizerOperator assumes power, sotodlib assumes temperature
        return self._scaled_tod(np.atleast_2d(np.asarray(self.data.signal)), 0.5, out)

    @overload
    def get_demodulated_tods(
        self, stokes: Literal['I'], out: np.ndarray | None = None
    ) -> StokesI: ...
    @overload
    def get_demodulated_tods(
        self, stokes: Literal['QU'], out: np.ndarray | None = None
    ) -> StokesQU: ...
    @overload
    def get_demodulated_tods(
        self, stokes: Literal['IQU'], out: np.ndarray | None = None
    ) -> StokesIQU: ...
    @overload
    def get_demodulated_tods(
        self, stokes: Literal['IQUV'], out: np.ndarray | None = None
    ) -> StokesIQUV: ...
    def get_demodulated_tods(
        self, stokes: ValidStokesLiteral = 'IQU', out: np.ndarray | None = None
    ) -> StokesType:
        """Returns the demodulated timestream data as a Stokes pytree, in the stored precision.

        'IQUV' is not supported.
        """
        if stokes == 'IQUV':
            raise NotImplementedError
        attrs = {'I': 'dsT', 'Q': 'demodQ', 'U': 'demodU'}
        streams = [np.atleast_2d(np.asarray(getattr(self.data, attrs[s]))) for s in stokes]
        # furax's LinearPolarizerOperator assumes power, sotodlib assumes temperature
        return Stokes.class_for(stokes).from_array(self._scaled_tods(streams, 0.5, out))

    def get_detector_offset_angles(self) -> Float[np.ndarray, ' dets']:
        """Returns the detector offset angles."""
        return np.asarray(self.data.focal_plane['gamma'])

    def get_hwp_angles(self) -> Float[np.ndarray, ' a']:
        """Returns the HWP angles."""
        return np.asarray(self.data.hwp_angle)

    def get_scanning_intervals(self, det_ind: int = 0) -> NDArray[Any]:
        """Returns scanning intervals of the chosen detector.

        The output is a list of the starting and ending sample indices
        """
        return np.array(
            self.data.preprocess.turnaround_flags.turnarounds.ranges[det_ind].complement().ranges()
        )

    def get_sample_mask(self) -> Bool[np.ndarray, 'dets samps']:
        try:
            return np.asarray((~getattr(self.data.flags, _MASKED_FLAGS)).mask(), dtype=bool)
        except KeyError as e:
            raise RuntimeError('Glitch flags unavailable in the observation') from e

    def get_left_scan_mask(self) -> Bool[np.ndarray, ' samps']:
        try:
            # Stored per-detector (RangesMatrix), but scan direction is shared across the
            # focal plane; collapse to the boresight 1D mask (cf. get_scanning_intervals).
            return np.asarray(self.data.flags.left_scan.mask()[0], dtype=bool)
        except KeyError as e:
            raise RuntimeError('Scan mask unavailable in the observation') from e

    def get_right_scan_mask(self) -> Bool[np.ndarray, ' samps']:
        try:
            return np.asarray(self.data.flags.right_scan.mask()[0], dtype=bool)
        except KeyError as e:
            raise RuntimeError('Scan mask unavailable in the observation') from e

    def get_azimuth(self) -> Float[np.ndarray, ' a']:
        """Returns the azimuth of the boresight for each sample."""
        return np.asarray(self.data.boresight.az)

    def get_elevation(self) -> Float[np.ndarray, ' a']:
        """Returns the elevation of the boresight for each sample."""
        return np.asarray(self.data.boresight.el)

    def get_wcs_shape_and_kernel(
        self,
        resolution_arcmin: float,
        projection: ProjectionType = ProjectionType.CAR,
    ) -> tuple[tuple[int, int], WCS]:
        """Returns astropy WCS kernel object corresponding to the observed sky."""
        res = resolution_arcmin * pixell.utils.arcmin
        wcs_kernel_init = coords.get_wcs_kernel(projection.name.lower(), 0, 0, res=res)
        wcs_shape, wcs_kernel = coords.get_footprint(self.data, wcs_kernel_init)

        return wcs_shape, wcs_kernel

    def get_pointing_and_spin_angles(
        self, landscape: StokesLandscape
    ) -> tuple[Integer[Array, 'dets samps 2'], Float[Array, 'dets samps']]:
        """Obtain pointing information and spin angles from the observation."""
        # Projection Matrix class instance for the observation
        if isinstance(landscape, AstropyWCSLandscape):
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

        # Get the pixel indices as before, but also obtain
        # the spin projection factors of size (n_samps,n_comps) for each detector,
        # which have 1, cos(2*p), sin(2*p) where p is the parallactic angle
        pixel_inds, spin_proj = proj.get_pointing_matrix(assembly)

        pixel_inds = jnp.array(pixel_inds)

        spin_proj = jnp.array(spin_proj, dtype=landscape.dtype)
        spin_ang = jnp.arctan2(spin_proj[..., 2], spin_proj[..., 1]) / 2.0

        return pixel_inds, spin_ang

    def get_timestamps(self) -> Float[np.ndarray, ' a']:
        """Returns timestamps (sec) of the samples."""
        return np.asarray(self.data.timestamps)

    def get_scanning_mask(self) -> Bool[np.ndarray, ' samp']:
        # Assume that all detectors have the same scanning intervals
        return np.asarray(
            self.data.preprocess.turnaround_flags.turnarounds.ranges[0].complement().mask(),
            dtype=bool,
        )

    def get_noise_model(self) -> None | NoiseModel:
        """Load precomputed noise model from the data, if present. Otherwise, return None."""
        try:
            fit = self._get_noise_fit_for_stoke('I')
        except ValueError:
            return None
        return AtmosphericNoiseModel(*fit)

    def get_demodulated_noise_model(self, stokes: ValidStokesLiteral = 'IQU') -> NoiseModel:
        """Returns a single noise model covering every requested stream.

        Each stream is fit independently (I/Q/U noise properties genuinely differ), so the
        per-detector parameters carry a leading Stokes axis rather than being separate models.
        """
        if stokes == 'IQUV':
            raise NotImplementedError
        fits = np.stack([self._get_noise_fit_for_stoke(s) for s in stokes], axis=1)
        return AtmosphericNoiseModel(*fits)

    def _get_noise_fit_for_stoke(self, stoke: Literal['I', 'Q', 'U']) -> NDArray[np.floating]:
        """Returns the (sigma, alpha, fk, f0) fit rows for one stream, shape ``(4, dets)``."""
        preproc = self.data.get('preprocess')
        if preproc is None:
            raise ValueError('No preprocess data available')
        if self._sotodlib_config.noise_source == 'mapmaking':
            sigma = np.asarray(preproc['noiseQ_mapmaking.white_noise'])
            ones = np.ones_like(sigma)  # fake alpha, fk, f0
            return np.stack([sigma, ones, ones, ones], axis=0)
        attr = {'I': 'noiseT', 'Q': 'noiseQ', 'U': 'noiseU'}[stoke]
        if attr not in preproc:
            raise ValueError(f'No {attr} noise model available')
        fit = getattr(preproc, attr)
        # sotodlib fit's columns: (w, fknee, alpha), with
        # psd = wn**2 * (1 + (fknee / f) ** alpha)
        # Note the difference in the sign of alpha
        expected = ['white_noise', 'fknee', 'alpha']
        actual = list(fit.noise_model_coeffs.vals)
        if actual != expected:
            raise ValueError(f'Unexpected noise model coefficients: {actual}, expected {expected}')
        sigma = np.asarray(fit.fit[:, 0])
        alpha = np.asarray(-fit.fit[:, 2])
        fk = np.asarray(fit.fit[:, 1])
        f0 = 1e-5 * np.ones_like(fk)
        return np.stack([sigma, alpha, fk, f0], axis=0)

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
        obtained as a result of a 1/f + white noise model fitting.
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
        return fit[:, 1]
    '''

    def get_boresight_quaternions(self) -> Float[np.ndarray, 'samp 4']:
        """Returns the boresight quaternions at each time sample."""
        site = self._sotodlib_config.site
        weather = self._sotodlib_config.weather
        if self._sotodlib_config.wobble_correction:
            wobble = self.data.get('wobble_params')
            if wobble is None:
                raise ValueError('wobble_params not found in observation data')
            csl = get_deflected_sightline(self.data, wobble, site=site, weather=weather)
        else:
            csl = so3g.proj.CelestialSightLine.az_el(
                self.data.timestamps,
                self.data.boresight.az,
                self.data.boresight.el,
                roll=self.data.boresight.roll,
                site=site,
                weather=weather,
            )
        return np.asarray(csl.Q, dtype=np.float64)

    def get_detector_quaternions(self) -> Float[np.ndarray, 'det 4']:
        """Returns the quaternion offsets of the detectors."""
        # Use so3g's CPU implementation so the result stays on host
        # quaternion convention is the same ordering (1,i,j,k)
        fp = self.data.focal_plane
        quats = so3g.proj.quat.rotation_xieta(fp.xi, fp.eta, fp.gamma)
        return np.atleast_2d(np.asarray(quats, dtype=np.float64))


class LazySOTODLibObservation(FileBackedLazyObservation[AxisManager]):
    interface_class = SOTODLibObservation

    def __init__(self, filename: str | Path, sotodlib_config: SotodlibConfig | None = None) -> None:
        super().__init__(filename)
        self._sotodlib_config = sotodlib_config

    def get_data(self, requested_fields: Collection[str] | None = None) -> SOTODLibObservation:
        return SOTODLibObservation.from_file(
            self.file, requested_fields, sotodlib_config=self._sotodlib_config
        )


class LazyPreprocSOTODLibObservation(AbstractLazyObservation[AxisManager]):
    """Lazy observation backed by the preprocessing database (no intermediate file).

    Loads on demand via ``SOTODLibObservation.from_preproc_group``, so observations are
    streamed straight from the preproc archive at mapmaking time and never copied to disk.

    With ``join_groups``, every preprocessing group matching ``detector_selection`` is loaded and
    the groups are joined into one observation (see
    [`SOTODLibObservation.from_preproc_groups`][furax.interfaces.sotodlib.observation.SOTODLibObservation.from_preproc_groups]),
    e.g. all the wafers of a bandpass. Otherwise the selection is loaded as one unit.

    With `cache_dir`, the first read saves each group loaded (preprocessed and downsampled) to
    `<cache_dir>/<key>/<observation_id>__<group>.h5`, and later reads load those files instead,
    only the requested fields. A group is named after its detector selection, so that runs
    share the files of the groups they have in common, whether they join the groups or not. A
    group skipped on its first read, because none of its detectors survived the cuts or it had no
    preprocessing record, is recorded in a `.skip` file next to the others and skipped again.
    `key` is a hash of every input that changes the loaded data: the content of the preprocessing
    configs and of their contexts, `SotodlibConfig.downsample`, and the sotodlib version. A
    `key.json` next to the files records these inputs. The other `SotodlibConfig` settings act
    after loading, so runs that differ only in them share a cache. The databases the configs point
    to, the preprocessing archive among them, are not part of the key: after preprocessing again
    under the same configs, delete the cache. A config staged by `furax-so-stage` is keyed as the
    config it was staged from, so that staged and unstaged runs share a cache.

    The files are written to a temporary name and renamed into place, so concurrent runs can
    share a cache and never read a partial file. A cached file that cannot be read is loaded
    from the preprocessing database again and replaced.
    """

    interface_class = SOTODLibObservation

    def __init__(
        self,
        observation_id: str,
        init_config: str | Path,
        proc_config: str | Path | None = None,
        detector_selection: DetectorSelection | None = None,
        sotodlib_config: SotodlibConfig | None = None,
        join_groups: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.observation_id = observation_id
        self.init_config = Path(init_config).resolve()
        self.proc_config = Path(proc_config).resolve() if proc_config else None
        self.detector_selection = detector_selection
        self.join_groups = join_groups
        self._sotodlib_config = sotodlib_config or SotodlibConfig()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None

    @property
    def name(self) -> str:
        """The observation id, followed by the wafer when each wafer is an observation of its own."""
        wafer = (self.detector_selection or {}).get('wafer_slot')
        if self.join_groups or not isinstance(wafer, str):
            return self.observation_id
        return f'{self.observation_id}__{wafer}'

    @functools.cached_property
    def cache_key_path(self) -> Path | None:
        """The directory of the cache key, `None` without `cache_dir`. It need not exist yet."""
        return self._cache.directory if self._cache else None

    def cache_paths(self) -> list[Path]:
        """The files the groups of the observation are cached in, none without `cache_dir`.

        The files need not exist yet.

        Raises:
            RuntimeError: If the groups cannot be listed.
        """
        if self._cache is None:
            return []
        return [self._cache.group_path(self.observation_id, s) for s in self._selections]

    def is_cached(self) -> bool:
        """Whether every group of the observation is cached, or recorded as skipped."""
        if self._cache is None:
            return False
        try:
            return self._cache.is_complete(self.cache_paths())
        except (RuntimeError, LoaderError):
            return False

    def get_data(self, requested_fields: Collection[str] | None = None) -> SOTODLibObservation:
        if requested_fields is not None and len(self._selections) > 1:
            # _join_groups checks that the groups share their samples through the timestamps
            requested_fields = {*requested_fields, ReaderField.TIMESTAMPS}
        parts = []
        for selection in self._selections:
            part = self._load_group(selection, requested_fields)
            if part is not None:
                parts.append(part)
        if not parts:
            raise NoDetectorsLeftError(f'no detectors left after cuts for {self.observation_id}')
        data = _join_groups(self.observation_id, parts)
        return SOTODLibObservation(data, self._sotodlib_config, processed=True)

    def probe(
        self, intervals: bool = False, sample_rate: bool = False, good_ranges: bool = False
    ) -> ObservationProbe:
        """What the mapmaker needs to know of the observation, from the preprocessing metadata.

        No TOD is read and no pipeline is run, and each group's metadata is loaded once.

        - The detector count is the number of detectors that the cuts of the process pipeline
          keep. It is selected from the stored cut flags the way the loaders do, so it is expected
          to equal the number loaded, but it follows their logic rather than their result. It is
          at least 1: an observation with every detector cut probes one detector and fails to load.
          With `join_groups`, it is summed over the groups that have a preprocessing record, as
          loaded by `from_preproc_groups`.
        - The sample count is `ceil(meta.samps.count / downsample)`. The process pipeline only
          trims samples (e.g. edge cuts) before downsampling, so this bounds the actual count
          without matching it. The bound does not depend on `cache_dir`, so that a cached
          observation is padded as an uncached one.
        - The sample rate is that of the observation's record in the obsdb, `n_samples` over
          `duration`, divided by `downsample`; `None` if the record lacks them. It is reported
          even when not requested.
        - The good ranges are those of the flags the loaders mask, `flags.glitch_flags`, which
          the `combine_flags` or `union_flags` steps of the process pipeline assemble from the
          flags the archive stores: the steps are run on the metadata, on the detectors left by
          the cuts, with the samples the pipeline trims (`trim_samps`) trimmed, and the flags are
          then downsampled as `downsample_obs` does. They cover the samples loaded, not those
          trimmed. They are not reported (`None`) when the flags cannot be followed this way:
          with a `proc_config`, whose pipeline may compute flags from the TOD, or when no step
          assembles them. The turnarounds `MapMakingConfig.scanning_mask` adds to the mask are
          not among them.
        """
        _enable_preproc_context_cache()
        # call get_preprocess_context through the module so the cached get_preprocess_context is used
        # (str, not Path: sotodlib and the cache key expect a posix string)
        configs, context = pu.get_preprocess_context(self.init_config.as_posix())
        downsample = self._sotodlib_config.downsample
        n_detectors, n_samps, n_intervals = 0, 0, 0
        rate = None
        # a proc_config's pipeline may compute flags from the TOD: they cannot be followed
        good_ranges = good_ranges and self.proc_config is None
        goods: list[SampleRanges | None] = []  # the good ranges of each group, if known
        found = False
        for selection in self._selections:
            try:
                meta = context.get_meta(self.observation_id, dets=selection)
                kept = self._detectors_after_cuts(meta, selection)
            except LoaderError:
                if not self.join_groups:
                    raise
                logger.warning(
                    'No preprocessing record for %s in %s', selection, self.observation_id
                )
                continue
            found = True
            n_detectors += len(kept)
            n_samps = max(n_samps, meta.samps.count)
            n_intervals = max(n_intervals, meta.subscans.count) if intervals else 0
            rate = _record_sample_rate(meta.obs_info, downsample)
            if good_ranges:
                meta.restrict('dets', kept)
                goods.append(self._good_ranges(meta, configs))
        if not found:
            raise RuntimeError(f'no preprocessing records for {self.observation_id}')
        shape = ObservationBufferShape(
            max(n_detectors, 1),
            -(-n_samps // downsample),  # ceil, matches downsample_obs
            n_intervals,
        )
        good = None
        known = [g for g in goods if g is not None]
        # the groups are joined along the detectors, sharing their samples (see `_join_groups`)
        if known and len(known) == len(goods) and len({g.n_samples for g in known}) == 1:
            good = SampleRanges(known[0].n_samples, [r for g in known for r in g.ranges])
        return ObservationProbe(shape, rate, good)

    def _good_ranges(self, meta: AxisManager, configs: dict[str, Any]) -> SampleRanges | None:
        """The good ranges of the detectors of `meta`, as loaded, from their metadata.

        `None` when no step of the process pipeline assembles the flags (see [`probe`][]).
        """
        flags = _glitch_flags(meta, configs)
        if flags is None:
            return None
        if (downsample := self._sotodlib_config.downsample) > 1:
            flags = downsample_cut(flags, downsample)
        good = ~flags
        return SampleRanges(good.shape[1], [r.ranges() for r in good.ranges])

    def _detectors_after_cuts(self, meta: AxisManager, selection: dict[str, Any]) -> NDArray[Any]:
        """The detectors of a group that the loaders keep, from its metadata `meta`.

        Follows `load_and_preprocess` and, with a `proc_config`, `multilayer_load_and_preprocess`,
        up to the point where they read the TOD.
        """
        configs, _ = pu.get_preprocess_context(self.init_config.as_posix())
        kept = _survivors(meta, configs)
        if self.proc_config is None:
            return kept
        configs_proc, context_proc = pu.get_preprocess_context(self.proc_config.as_posix())
        meta_proc = context_proc.get_meta(self.observation_id, dets=selection)
        meta_proc.restrict('dets', kept)
        return _survivors(meta_proc, configs_proc)

    @functools.cached_property
    def _selections(self) -> list[dict[str, Any]]:
        """The detector selections the observation is loaded and cached in, one file each."""
        if self.join_groups:
            return group_selections(self.observation_id, self.init_config, self.detector_selection)
        return [dict(self.detector_selection or {})]

    @functools.cached_property
    def _cache(self) -> PreprocCache | None:
        if self.cache_dir is None:
            return None
        inputs = cache_inputs(self.init_config, self.proc_config, self._sotodlib_config.downsample)
        return PreprocCache(self.cache_dir, inputs)

    def _load_group(
        self, selection: dict[str, Any], requested_fields: Collection[str] | None
    ) -> SOTODLibObservation | None:
        """The group of a detector selection, from the cache if it is there; `None` if skipped."""
        cache = self._cache
        if cache is None:
            return self._preprocess_group(selection)
        path = cache.group_path(self.observation_id, selection)
        if path.exists():
            try:
                return self._read_cached(path, requested_fields)
            except Exception:
                logger.warning('Cannot read %s, loading it again', path, exc_info=True)
        elif cache.is_skipped(path):
            return None
        obs = self._preprocess_group(selection)
        if obs is None:
            cache.skip(path)
            return None
        # Return what a later read of the file returns: the groups of a joined observation come
        # from the cache or from preprocessing, and must have the same fields to be joined.
        if cache.write(path, lambda tmp: obs.data.save(tmp.as_posix())):
            return self._read_cached(path, requested_fields)
        return obs

    def _preprocess_group(self, selection: dict[str, Any]) -> SOTODLibObservation | None:
        """The group of a detector selection, preprocessed; `None` if there is nothing to load."""
        # A preproc load cannot be field-subset: the full group is always returned, which
        # satisfies any requested fields. The cheap shape path lives in probe.
        try:
            return SOTODLibObservation.from_preproc_group(
                self.observation_id,
                self.init_config,
                self.proc_config,
                selection,
                sotodlib_config=self._sotodlib_config,
            )
        except LoaderError as error:
            # Only the missing record of a group of the observation is remembered: for a single
            # selection it is an error, and any other LoaderError may be transient
            if not (self.join_groups and _is_missing_record(error)):
                raise
            logger.warning('No preprocessing record for %s in %s', selection, self.observation_id)
            return None
        except NoDetectorsLeftError:
            return None

    def _read_cached(
        self, path: Path, requested_fields: Collection[str] | None
    ) -> SOTODLibObservation:
        return SOTODLibObservation.from_file(
            path, requested_fields, self._sotodlib_config, processed=True
        )


def _survivors(meta: AxisManager, configs: dict[str, Any]) -> NDArray[Any]:
    """The detectors of `meta` that the cuts of one preprocessing layer keep."""
    valid = meta.preprocess.get('valid_data') if 'preprocess' in meta else None
    if isinstance(valid, AxisManager):
        return meta.dets.vals[has_any_cuts(valid.valid_data)]
    keep = np.ones(meta.dets.count, dtype=bool)
    for process in Pipeline(configs['process_pipe'])[:]:
        selected = process.select(meta, in_place=False)
        if isinstance(selected, np.ndarray):
            keep &= selected
    return meta.dets.vals[keep]


def _record_sample_rate(obs_info: AxisManager, downsample: int) -> float | None:
    """The sample rate of the TOD as loaded, from the obsdb record of its observation."""
    n_samples = getattr(obs_info, 'n_samples', None)
    duration = getattr(obs_info, 'duration', None)
    if n_samples is None or not duration:
        return None
    # the obsdb duration spans the first to the last sample, as the timestamps do
    return (n_samples - 1) / duration / downsample


def _glitch_flags(meta: AxisManager, configs: dict[str, Any]) -> so3g.proj.RangesMatrix | None:
    """`flags.glitch_flags` as the process pipeline assembles it, from the metadata `meta`.

    Only the steps combining the stored flags into it, and those trimming samples, are followed;
    `None` if no step assembles it.
    """
    proc = meta.preprocess
    aman = AxisManager(meta.dets, meta.samps)
    built = False
    for process in Pipeline(configs['process_pipe'])[:]:
        cfgs = process.process_cfgs if isinstance(process.process_cfgs, dict) else {}
        if process.name in ('combine_flags', 'union_flags'):
            if cfgs.get('total_flags_label') == _MASKED_FLAGS:
                process.process(aman, proc)
                built = True
        elif trim := cfgs.get('trim_samps'):
            for am in (aman, proc):
                am.restrict(
                    'samps', (am.samps.offset + trim, am.samps.offset + am.samps.count - trim)
                )
    return getattr(aman.flags, _MASKED_FLAGS) if built else None


def _join_groups(observation_id: str, parts: list[SOTODLibObservation]) -> AxisManager:
    """The data of the groups, each already downsampled, joined along the detectors."""
    first = parts[0].data
    for part in parts[1:]:
        # also false when the groups have a different number of samples
        if not np.array_equal(part.data.timestamps, first.timestamps):
            raise ValueError(f'the preprocessing groups of {observation_id} differ in samples')
    # the samples are shared: the fields without a detector axis are the same in every group
    if len(parts) == 1:
        return first
    return AxisManager.concatenate([p.data for p in parts], 'dets', other_fields='first')


def _is_missing_record(error: LoaderError) -> bool:
    """Whether the error is the lack of a preprocessing record, not a failed read."""
    return _NO_RECORD in str(error)


def group_selections(
    observation_id: str, init_config: str | Path, detector_selection: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The detector selection of each preprocessing group of the observation that it matches."""
    _enable_preproc_context_cache()
    configs, context = pu.get_preprocess_context(Path(init_config).as_posix())
    group_by, groups, (error, message, _) = pu.get_groups(observation_id, configs, context)
    if error is not None:
        raise RuntimeError(f'cannot list the preprocessing groups of {observation_id}: {message}')
    selection = detector_selection or {}
    selections = []
    for values in groups:
        # sotodlib gives numpy scalars: as Python values, the selections print plainly in logs
        group = {
            _python_scalar(k): _python_scalar(v) for k, v in zip(group_by, values, strict=True)
        }
        if all(v in np.atleast_1d(selection.get(k, v)) for k, v in group.items()):
            selections.append(selection | group)
    return selections


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
