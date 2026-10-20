import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import healpy as hp
import jax.numpy as jnp
import numpy as np
import polars as pl
from astropy.io import fits
from jaxtyping import Array, Float, Integer

from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.stokes import Stokes, StokesType

from ._logger import logger as furax_logger

__all__ = [
    'MapMakingResults',
]

_AMPLITUDES_FILE = 'amplitudes.npz'
_DETECTORS_FILE = 'detectors.parquet'
_OBSERVATIONS_FILE = 'observations.csv'
_PROVENANCE_FILE = 'provenance.json'


@dataclass
class MapMakingResults:
    """The products of a mapmaking run.

    Here is a summary of the files that products are saved into:

        map / hit_map / icov     one file each, named after the field
        injected_map             injected_map, as the map
                                 FITS for HEALPix or WCS, .npy otherwise
        template_amplitudes      amplitudes.npz, one array per template
        observations             observations.csv
        detectors                detectors.parquet
        solver_stats             solver_stats.json
        provenance               provenance.json
    """

    map: StokesType
    """The estimated sky map"""

    landscape: StokesLandscape
    """The landscape corresponding to the map"""

    hit_map: Integer[Array, ' *dims']
    """The map of hit counts per pixel"""

    icov: Float[Array, 'stokes stokes *dims']
    """The per-pixel inverse noise covariance matrix (H^T N^{-1} H)"""

    solver_stats: dict[str, Any] | None = None
    """Statistics from the linear solver (e.g. num_steps, max_steps)"""

    template_amplitudes: dict[str, Any] | None = None
    """Estimated amplitudes of the explicit templates, keyed by template name: per observation, the
    amplitudes of [`Template`][furax.mapmaking.templates.Template], streams then detectors first"""

    injected_map: StokesType | None = None
    """The map of the simulated sky through the run (`injection.mode=response`)"""

    observations: pl.DataFrame | None = None
    """One row per observation, in the order of the observation axis of the per-observation
    products: `name`, whether it was `loaded`, `n_detectors`, `n_samples`, the
    `valid_sample_fraction` (null if the flags were not read), `load_seconds`, and the `error`
    that stopped an observation not loaded, its probe or its load (null if it was loaded)"""

    detectors: pl.DataFrame | None = None
    r"""One row per detector read. `observation` and `detector` are its indices along the
    observation and detector axes of the per-detector products, such as the amplitudes of a
    template with one set per detector. Then come its `name`, its focal-plane coordinates `xi`,
    `eta` and `gamma` in radians (null if the run did not read them), and the parameters of its
    noise model (`sigma`, and `alpha`, `fk`, `f0` for a $1/f$ model), suffixed by the stream on a
    demodulated TOD (`sigma_q`, `sigma_u`)"""

    provenance: dict[str, Any] | None = None
    """The code, environment and command that produced the run"""

    @property
    def failed_observations(self) -> list[str] | None:
        """The observations that failed to load and were excluded from the maps.

        Read from `observations`; `None` without it.
        """
        if self.observations is None:
            return None
        return self.observations.filter(~pl.col('loaded'))['name'].to_list()

    def save(self, out_dir: str | Path) -> None:
        """Write every product that is set, into the files listed above."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_array(np.array(self.map.data), 'map', out_dir)
        self._save_array(np.array(self.hit_map), 'hit_map', out_dir, column_names=['HITS'])
        self._save_icov(np.array(self.icov), out_dir)

        if self.injected_map is not None:
            self._save_array(np.array(self.injected_map.data), 'injected_map', out_dir)
        if self.template_amplitudes:
            entries = {name: np.array(a) for name, a in self.template_amplitudes.items()}
            np.savez(out_dir / _AMPLITUDES_FILE, **entries)  # ty: ignore[invalid-argument-type]
        if self.observations is not None:
            self.observations.write_csv(out_dir / _OBSERVATIONS_FILE)
        if self.detectors is not None:
            self.detectors.write_parquet(out_dir / _DETECTORS_FILE)
        if self.solver_stats is not None:
            (out_dir / 'solver_stats.json').write_text(json.dumps(self.solver_stats, indent=2))
        if self.provenance is not None:
            (out_dir / _PROVENANCE_FILE).write_text(json.dumps(self.provenance, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path, landscape: StokesLandscape) -> Self:
        """Load a previously saved MapMakingResults from disk.

        The maps are required; every other product comes back as ``None`` when absent.

        Args:
            out_dir: Directory containing the saved files.
            landscape: The landscape used when the results were saved.

        Raises:
            FileNotFoundError: If the directory or one of the maps is missing.
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f'Output directory not found: {out_dir}')

        return cls(
            map=cls._load_map(out_dir, landscape),
            landscape=landscape,
            hit_map=cls._load_hit_map(out_dir, landscape),
            icov=cls._load_icov(out_dir, landscape),
            injected_map=cls._load_injected_map(out_dir, landscape),
            solver_stats=cls._load_json(out_dir / 'solver_stats.json'),
            template_amplitudes=cls._load_amplitudes(out_dir),
            observations=_read_if_present(pl.read_csv, out_dir / _OBSERVATIONS_FILE),
            detectors=_read_if_present(pl.read_parquet, out_dir / _DETECTORS_FILE),
            provenance=cls._load_json(out_dir / _PROVENANCE_FILE),
        )

    @staticmethod
    def _load_array(
        name: str, out_dir: Path, landscape: StokesLandscape, n_fields: int
    ) -> np.ndarray:
        """Load a [n_fields, *pixel_dims] array from FITS or npy.

        For HEALPix landscapes with n_fields=1, a leading dimension is added
        so the returned shape is always [n_fields, npix].
        """
        if isinstance(landscape, (WCSLandscape, AstropyWCSLandscape)):
            path = out_dir / f'{name}.fits'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            with fits.open(path) as hdul:
                arr = np.asarray(hdul[0].data)
                return arr.astype(arr.dtype.newbyteorder('='), copy=False)
        elif isinstance(landscape, HealpixLandscape):
            path = out_dir / f'{name}.fits'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            if n_fields == 1:
                arr = np.array(hp.read_map(str(path), field=0))
            else:
                maps = hp.read_map(str(path), field=list(range(n_fields)))
                arr = np.stack(maps, axis=0)
            # hp.read_map with field=0 drops the leading dim; restore it
            if arr.ndim == len(landscape.shape):
                arr = arr[np.newaxis]
            return arr.astype(arr.dtype.newbyteorder('='), copy=False)
        else:
            path = out_dir / f'{name}.npy'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            return np.load(path)

    @staticmethod
    def _load_map(out_dir: Path, landscape: StokesLandscape, name: str = 'map') -> StokesType:
        ns = len(landscape.stokes)
        arr = MapMakingResults._load_array(name, out_dir, landscape, ns)
        stokes_cls = Stokes.class_for(landscape.stokes)
        return stokes_cls(*[jnp.array(arr[i]) for i in range(ns)])

    @staticmethod
    def _load_injected_map(out_dir: Path, landscape: StokesLandscape) -> StokesType | None:
        try:
            return MapMakingResults._load_map(out_dir, landscape, 'injected_map')
        except FileNotFoundError:
            return None

    @staticmethod
    def _load_hit_map(out_dir: Path, landscape: StokesLandscape) -> Array:
        if isinstance(landscape, (WCSLandscape, AstropyWCSLandscape)):
            path = out_dir / 'hit_map.fits'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            with fits.open(path) as hdul:
                arr = hdul[0].data
                return jnp.array(arr.astype(arr.dtype.newbyteorder('='), copy=False))
        elif isinstance(landscape, HealpixLandscape):
            path = out_dir / 'hit_map.fits'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            hits = hp.read_map(str(path), field=0)
            return jnp.array(hits.astype(hits.dtype.newbyteorder('='), copy=False))
        else:
            path = out_dir / 'hit_map.npy'
            if not path.exists():
                raise FileNotFoundError(f'Expected file not found: {path}')
            return jnp.array(np.load(path))

    @staticmethod
    def _load_icov(out_dir: Path, landscape: StokesLandscape) -> Array:
        stokes = landscape.stokes
        ns = len(stokes)
        n_upper = ns * (ns + 1) // 2
        arr_upper = MapMakingResults._load_array('icov', out_dir, landscape, n_upper)

        upper = [(i, j) for i in range(ns) for j in range(i, ns)]
        pixel_shape = arr_upper.shape[1:]
        icov = np.zeros((ns, ns, *pixel_shape), dtype=arr_upper.dtype)
        for k, (i, j) in enumerate(upper):
            icov[i, j] = arr_upper[k]
            if i != j:
                icov[j, i] = arr_upper[k]
        return jnp.array(icov)

    @staticmethod
    def _load_amplitudes(out_dir: Path) -> dict[str, Any] | None:
        """The amplitudes of each template, from the archive."""
        path = out_dir / _AMPLITUDES_FILE
        if not path.exists():
            return None
        with np.load(path) as archive:
            amplitudes = {name: jnp.array(archive[name]) for name in archive.files}
        return amplitudes or None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text()) if path.exists() else None

    def _save_icov(self, arr: np.ndarray, out_dir: Path) -> None:
        """Save the inverse covariance, storing only the upper triangle with stokes-aware names."""
        stokes = self.landscape.stokes
        ns = len(stokes)
        upper = [(i, j) for i in range(ns) for j in range(i, ns)]
        column_names: list[str] = [stokes[i] + stokes[j] for i, j in upper]
        arr_upper = np.stack([arr[i, j] for i, j in upper], axis=0)
        self._save_array(arr_upper, 'icov', out_dir, column_names=column_names)

    def _save_array(
        self, arr: np.ndarray, name: str, out_dir: Path, column_names: list[str] | None = None
    ) -> None:
        """Save a numpy array as FITS (WCS or HEALPix) or npy depending on the landscape."""
        if isinstance(self.landscape, WCSLandscape):
            hdu = fits.PrimaryHDU(arr, header=fits.Header(self.landscape.to_wcs().to_header()))
            hdu.writeto(out_dir / f'{name}.fits', overwrite=True)
        elif isinstance(self.landscape, AstropyWCSLandscape):
            hdu = fits.PrimaryHDU(arr, header=fits.Header(self.landscape.wcs.to_header()))
            hdu.writeto(out_dir / f'{name}.fits', overwrite=True)
        elif isinstance(self.landscape, HealpixLandscape):
            maps = [arr] if arr.ndim == 1 else list(arr.reshape(-1, arr.shape[-1]))
            hp.write_map(
                str(out_dir / f'{name}.fits'),
                maps,
                nest=self.landscape.nested,
                column_names=column_names,
                overwrite=True,
            )
        else:
            furax_logger.warning(
                f'saving {name} as npy: geometry information will not be embedded in the file'
            )
            np.save(out_dir / name, arr)


def _read_if_present(read: Any, path: Path) -> pl.DataFrame | None:
    return read(path) if path.exists() else None
