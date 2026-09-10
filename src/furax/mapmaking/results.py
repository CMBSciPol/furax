import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import jax.numpy as jnp
import numpy as np
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


@dataclass
class MapMakingResults:
    """The products of a mapmaking run.

    Here is a summary of the files that products are saved into:

        map / hit_map / icov     one file each, named after the field
                                 FITS for HEALPix or WCS, .npy otherwise
        noise_fits               noise_fits.npy
        template_amplitudes      amplitudes.npz  (<template> or <template>/<leg>)
        solver_stats             solver_stats.json
        failed_observations      failed_observations.txt, one name per line
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

    noise_fits: Float[Array, '...'] | None = None
    """The fitted noise PSD parameters"""

    failed_observations: list[str] | None = None
    """Names of observations that failed to load and were excluded from the maps"""

    template_amplitudes: dict[str, Any] | None = None
    """Estimated amplitudes of the explicit templates, keyed by template name"""

    def save(self, out_dir: str | Path) -> None:
        """Write every product that is set, into the files listed above."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._save_array(np.array(self.map.data), 'map', out_dir)
        self._save_array(np.array(self.hit_map), 'hit_map', out_dir, column_names=['HITS'])
        self._save_icov(np.array(self.icov), out_dir)

        if self.noise_fits is not None:
            np.save(out_dir / 'noise_fits.npy', np.array(self.noise_fits))
        if self.template_amplitudes:
            # One entry per array: `<template>`, or `<template>/<leg>` when Stokes-valued.
            entries: dict[str, np.ndarray] = {}
            for name, value in self.template_amplitudes.items():
                legs = value if isinstance(value, dict) else {'': value}
                for leg, amplitudes in legs.items():
                    key = f'{name}/{leg}' if leg else name
                    entries[key] = np.array(amplitudes)
            np.savez(out_dir / _AMPLITUDES_FILE, **entries)  # type: ignore[arg-type]
        if self.solver_stats is not None:
            (out_dir / 'solver_stats.json').write_text(json.dumps(self.solver_stats, indent=2))
        if self.failed_observations:
            (out_dir / 'failed_observations.txt').write_text(
                ''.join(f'{name}\n' for name in self.failed_observations)
            )

    @classmethod
    def load(cls, out_dir: str | Path, landscape: StokesLandscape) -> 'MapMakingResults':
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
            solver_stats=cls._load_solver_stats(out_dir),
            noise_fits=cls._load_noise_fits(out_dir),
            failed_observations=cls._load_failed_observations(out_dir),
            template_amplitudes=cls._load_amplitudes(out_dir),
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
            return np.load(path)  # type: ignore[no-any-return]

    @staticmethod
    def _load_map(out_dir: Path, landscape: StokesLandscape) -> StokesType:
        ns = len(landscape.stokes)
        arr = MapMakingResults._load_array('map', out_dir, landscape, ns)
        stokes_cls = Stokes.class_for(landscape.stokes)
        return stokes_cls(*[jnp.array(arr[i]) for i in range(ns)])

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
        """Rebuild the per-template mapping from the `<template>[/<leg>]` archive keys."""
        path = out_dir / _AMPLITUDES_FILE
        if not path.exists():
            return None
        amplitudes: dict[str, Any] = {}
        with np.load(path) as archive:
            for key in archive.files:
                name, is_stokes_valued, leg = key.partition('/')
                if is_stokes_valued:
                    amplitudes.setdefault(name, {})[leg] = jnp.array(archive[key])
                else:
                    amplitudes[name] = jnp.array(archive[key])
        return amplitudes or None

    @staticmethod
    def _load_noise_fits(out_dir: Path) -> Array | None:
        path = out_dir / 'noise_fits.npy'
        return jnp.array(np.load(path)) if path.exists() else None

    @staticmethod
    def _load_solver_stats(out_dir: Path) -> dict[str, Any] | None:
        path = out_dir / 'solver_stats.json'
        return json.loads(path.read_text()) if path.exists() else None

    @staticmethod
    def _load_failed_observations(out_dir: Path) -> list[str] | None:
        path = out_dir / 'failed_observations.txt'
        return path.read_text().split() if path.exists() else None

    def _save_icov(self, arr: np.ndarray, out_dir: Path) -> None:
        """Save the inverse covariance, storing only the upper triangle with stokes-aware names."""
        stokes = self.landscape.stokes
        ns = len(stokes)
        upper = [(i, j) for i in range(ns) for j in range(i, ns)]
        column_names = [stokes[i] + stokes[j] for i, j in upper]
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
