import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import jax
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
from furax.obs.stokes import Stokes, StokesPyTreeType

from ._logger import logger as furax_logger

__all__ = [
    'MapMakingResults',
]

_VALID_FIELDS = frozenset({'map', 'hit_map', 'icov', 'noise_fits', 'solver_stats'})
_REQUIRED_FIELDS = frozenset({'map', 'hit_map', 'icov'})


class _JsonEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'item'):  # numpy/jax scalars
            return obj.item()
        return super().default(obj)


@dataclass
class MapMakingResults:
    map: StokesPyTreeType
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

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        leaves = [np.array(leaf) for leaf in jax.tree.leaves(self.map)]
        self._save_array(np.array(leaves), 'map', out_dir)
        self._save_array(np.array(self.hit_map), 'hit_map', out_dir, column_names=['HITS'])
        self._save_icov(np.array(self.icov), out_dir)
        if self.noise_fits is not None:
            np.save(out_dir / 'noise_fits', np.array(self.noise_fits))
        if self.solver_stats is not None:
            with open(out_dir / 'solver_stats.json', 'w') as f:
                json.dump(self.solver_stats, f, indent=2, cls=_JsonEncoder)

    @classmethod
    def load(
        cls,
        out_dir: str | Path,
        landscape: StokesLandscape,
        fields: set[str] | list[str] | None = None,
    ) -> 'MapMakingResults':
        """Load a previously saved MapMakingResults from disk.

        Args:
            out_dir: Directory containing the saved files.
            landscape: The landscape used when the results were saved.
            fields: Fields to load. Defaults to all fields. Required fields
                (map, hit_map, icov) must always be included if specified.
        """
        out_dir = Path(out_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f'Output directory not found: {out_dir}')

        if fields is None:
            fields_to_load = _VALID_FIELDS
        else:
            fields_to_load = frozenset(fields)
            invalid = fields_to_load - _VALID_FIELDS
            if invalid:
                raise ValueError(
                    f'Unknown fields: {sorted(invalid)}. Valid fields: {sorted(_VALID_FIELDS)}'
                )
            missing_required = _REQUIRED_FIELDS - fields_to_load
            if missing_required:
                raise ValueError(f'Required fields cannot be excluded: {sorted(missing_required)}')

        sky_map = cls._load_map(out_dir, landscape)
        hit_map = cls._load_hit_map(out_dir, landscape)
        icov = cls._load_icov(out_dir, landscape)

        noise_fits = cls._load_noise_fits(out_dir) if 'noise_fits' in fields_to_load else None
        solver_stats = cls._load_solver_stats(out_dir) if 'solver_stats' in fields_to_load else None

        return cls(
            map=sky_map,
            landscape=landscape,
            hit_map=hit_map,
            icov=icov,
            solver_stats=solver_stats,
            noise_fits=noise_fits,
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
    def _load_map(out_dir: Path, landscape: StokesLandscape) -> StokesPyTreeType:
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
    def _load_noise_fits(out_dir: Path) -> Array | None:
        path = out_dir / 'noise_fits.npy'
        if not path.exists():
            return None
        return jnp.array(np.load(path))

    @staticmethod
    def _load_solver_stats(out_dir: Path) -> dict[str, Any] | None:
        path = out_dir / 'solver_stats.json'
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]

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
