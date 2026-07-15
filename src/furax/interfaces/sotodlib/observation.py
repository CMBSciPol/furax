from __future__ import annotations

import functools
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any, Literal, overload

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
from sotodlib.mapmaking.utils import downsample_obs

from furax.mapmaking import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    FileBackedLazyObservation,
    ObservationBufferShapes,
    ReaderField,
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
            return original(configs, context)  # type: ignore[no-any-return]
        key = (configs, threading.get_ident())
        if key not in _PREPROC_CONTEXT_CACHE:
            _PREPROC_CONTEXT_CACHE[key] = original(configs, context)
        return _PREPROC_CONTEXT_CACHE[key]

    cached._furax_cached = True  # type: ignore[attr-defined]
    # Relies on sotodlib calling get_preprocess_context by its module-global name (which
    # load_and_preprocess / multilayer_load_and_preprocess do): reassigning the module attribute
    # is what makes the cache take effect. A future sotodlib that imports it as a local alias, or
    # builds core.Context directly, would bypass this.
    pu.get_preprocess_context = cached


class SOTODLibObservation(AbstractGroundObservation[AxisManager]):
    """Class for interfacing with sotodlib's AxisManager."""

    def __init__(self, data: AxisManager, sotodlib_config: SotodlibConfig | None = None) -> None:
        super().__init__(data)
        self._sotodlib_config = sotodlib_config or SotodlibConfig()

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        requested_fields: Collection[str] | None = None,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> SOTODLibObservation:
        # check that file exists
        if not Path(filename).exists():
            raise FileNotFoundError(f'File {filename} does not exist')
        if isinstance(filename, Path):
            filename = filename.as_posix()

        config = sotodlib_config or SotodlibConfig()
        if requested_fields is None:
            # default is to load everything
            data = AxisManager.load(filename, fields=None)
            return cls(data, config)

        requested = set(requested_fields)
        # minimum information needed to determine buffer shapes
        fields = {'dets', 'samp'}
        # translate request to sotodlib subfield names (sets dedup overlapping requests)
        if ReaderField.METADATA in requested:
            fields.add('obs_info')
        if ReaderField.SAMPLE_DATA in requested:
            if config.demodulated:
                fields |= {'dsT', 'demodQ', 'demodU'}
            else:
                fields.add('signal')
        if ReaderField.VALID_SAMPLE_MASKS in requested:
            fields.add('flags.glitch_flags')
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
            if config.wobble_correction:
                fields |= {'wobble_params', 'det_info', 'hwp_angle'}
        if ReaderField.DETECTOR_QUATERNIONS in requested:
            fields.add('focal_plane')
        if ReaderField.NOISE_MODEL_FITS in requested:
            if config.noise_source == 'mapmaking':
                fields.add('preprocess.noiseQ_mapmaking')
            else:
                fields |= {'preprocess.noiseT', 'preprocess.noiseQ', 'preprocess.noiseU'}

        data = AxisManager.load(filename, fields=list(fields))
        return cls(data, config)

    @classmethod
    def from_preprocess(
        cls,
        preprocess_config: str | Path | dict[str, Any],
        observation_id: str,
        detector_selection: dict[str, str] | None = None,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> SOTODLibObservation:
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
            config = preprocess_config
        else:
            # load the preprocessing config from a yaml file
            with open(preprocess_config) as file:
                config = yaml.safe_load(file)

        data = pu.load_and_preprocess(observation_id, config, dets=detector_selection)
        return cls(data, sotodlib_config)

    @classmethod
    def from_preproc_group(
        cls,
        observation_id: str,
        init_config: str | Path,
        proc_config: str | Path | None = None,
        detector_selection: dict[str, str] | None = None,
        downsample: int = 1,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> SOTODLibObservation:
        """Loads a (already preprocessed) observation directly from the preprocessing db.

        This reads the saved preprocessing products from the archive and re-applies the
        process pipeline, without writing any intermediate file. It mirrors the load path
        of ``preproc_or_load_group`` (as used by the ``furax-so-prepare`` tool) but skips
        the archive/proc-aman saving, so the result is identical to mapping a binary file
        produced by that tool.

        Args:
            observation_id: Observation id (e.g. 'obs_1714550584_satp3_1111111').
            init_config: Base-layer preprocessing config (path or posix string).
            proc_config: Optional second-layer preprocessing config. If given, the
                two-layer (init+proc) load path is used.
            detector_selection: Optional detector restriction
                (e.g. {'wafer_slot': 'ws0', 'wafer.bandpass': 'f150'}).
            downsample: Integer downsampling factor applied after preprocessing.
            sotodlib_config: Optional sotodlib-specific configuration.

        Returns:
            An instance of SOTODLibObservation.

        Raises:
            RuntimeError: If no detectors remain after cuts (nothing to load).
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
            raise RuntimeError(f'no detectors left after cuts for {observation_id}')
        if downsample > 1:
            aman = downsample_obs(aman, downsample)
        return cls(aman, sotodlib_config)

    @property
    def name(self) -> str:
        return self.data.obs_info.obs_id  # type: ignore[no-any-return]

    @property
    def telescope(self) -> str:
        return self.data.obs_info.get('telescope')  # type: ignore[no-any-return]

    @property
    def n_samples(self) -> int:
        return self.data.samps.count  # type: ignore[no-any-return]

    @property
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

    def get_tods(self) -> Float[np.ndarray, 'dets samps']:
        """Returns the timestream data."""
        # furax's LinearPolarizerOperator assumes power, sotodlib assumes temperature
        tods = np.asarray(self.data.signal, dtype=np.float64)
        return 0.5 * np.atleast_2d(tods)

    @overload
    def get_demodulated_tods(self, stokes: Literal['I']) -> StokesI: ...
    @overload
    def get_demodulated_tods(self, stokes: Literal['QU']) -> StokesQU: ...
    @overload
    def get_demodulated_tods(self, stokes: Literal['IQU']) -> StokesIQU: ...
    @overload
    def get_demodulated_tods(self, stokes: Literal['IQUV']) -> StokesIQUV: ...
    def get_demodulated_tods(self, stokes: ValidStokesLiteral = 'IQU') -> StokesType:
        """Returns the demodulated timestream data as a Stokes pytree.

        'IQUV' is not supported.
        """
        if stokes == 'IQUV':
            raise NotImplementedError
        kls = Stokes.class_for(stokes)
        tods = [self._get_demodulated_tod(s) for s in stokes]  # type: ignore[arg-type]
        return kls.from_array(np.stack(tods, axis=0))

    def _get_demodulated_tod(self, stoke: Literal['I', 'Q', 'U']) -> NDArray[np.float64]:
        attr = {'I': 'dsT', 'Q': 'demodQ', 'U': 'demodU'}[stoke]
        tod = np.asarray(getattr(self.data, attr), dtype=np.float64)
        return 0.5 * np.atleast_2d(tod)

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
            return np.asarray((~self.data.flags.glitch_flags).mask(), dtype=bool)
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

        # TODO: check if this could be jnp array directly
        pixel_inds = np.array(pixel_inds)

        spin_proj = jnp.array(spin_proj, dtype=landscape.dtype)
        spin_ang = jnp.arctan2(spin_proj[..., 2], spin_proj[..., 1]) / 2.0

        return pixel_inds, spin_ang  # type: ignore[return-value]

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
        """Returns a single noise model covering every requested Stokes leg.

        Each Stokes leg is fit independently (I/Q/U noise properties genuinely differ), so the
        per-detector parameters carry a leading Stokes axis rather than being separate models.
        """
        if stokes == 'IQUV':
            raise NotImplementedError
        fits = np.stack([self._get_noise_fit_for_stoke(s) for s in stokes], axis=1)  # type: ignore[arg-type]
        return AtmosphericNoiseModel(*fits)

    def _get_noise_fit_for_stoke(self, stoke: Literal['I', 'Q', 'U']) -> NDArray[np.floating]:
        """Returns the (sigma, alpha, fk, f0) fit rows for one Stokes leg, shape ``(4, dets)``."""
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
        return fit[:, 1]  # type: ignore[no-any-return]
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
    """

    interface_class = SOTODLibObservation

    def __init__(
        self,
        observation_id: str,
        init_config: str | Path,
        proc_config: str | Path | None = None,
        detector_selection: dict[str, str] | None = None,
        downsample: int = 1,
        sotodlib_config: SotodlibConfig | None = None,
    ) -> None:
        self.observation_id = observation_id
        self.init_config = Path(init_config).resolve()
        self.proc_config = Path(proc_config).resolve() if proc_config else None
        self.detector_selection = detector_selection
        self.downsample = downsample
        self._sotodlib_config = sotodlib_config

    @property
    def name(self) -> str:
        return self.observation_id

    def get_data(self, requested_fields: Collection[str] | None = None) -> SOTODLibObservation:
        # A preproc load cannot be field-subset: the full observation is always returned,
        # which satisfies any requested fields. The cheap shape path lives in probe_shape.
        return SOTODLibObservation.from_preproc_group(
            self.observation_id,
            self.init_config,
            self.proc_config,
            self.detector_selection,
            downsample=self.downsample,
            sotodlib_config=self._sotodlib_config,
        )

    def probe_shape(self, intervals: bool = False) -> ObservationBufferShapes:
        """Returns an upper bound on the variable padded-buffer dimensions for ``fields``.

        All values are read from the init preprocess metadata: no heavy I/O work is done.

        The value is only an *upper bound*, not the exact post-pipeline shape:

        - ``n_detectors`` is the count after applying ``detector_selection`` (wafer slot / bandpass)
          but before the process pipeline, which only ever restricts detectors further.
        - ``n_samples`` is ``ceil(meta.samps.count / downsample)``. The process pipeline only trims
          samples (e.g. edge cuts) before downsampling, so this bounds the actual samps count.
        """
        _enable_preproc_context_cache()
        # call get_preprocess_context through the module so the cached get_preprocess_context is used
        # (str, not Path: sotodlib and the cache key expect a posix string)
        _, context = pu.get_preprocess_context(self.init_config.as_posix())
        meta = context.get_meta(self.observation_id, dets=self.detector_selection)
        n_samps_ub = -(-meta.samps.count // self.downsample)  # ceil, matches downsample_obs
        return ObservationBufferShapes(
            meta.dets.count,
            n_samps_ub,
            meta.subscans.count if intervals else 0,
        )
