from dataclasses import dataclass
from pathlib import Path
from typing import Any

import healpy as hp
import jax
import numpy as np
from astropy.io import fits
from jaxtyping import Array, Float, Integer

from furax.obs.landscapes import (
    AstropyWCSLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSLandscape,
)
from furax.obs.stokes import StokesPyTreeType

from ._logger import logger as furax_logger

__all__ = [
    'MapMakingResults',
]


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
