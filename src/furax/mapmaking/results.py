from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import healpy as hp
import jax
import numpy as np
import pixell.enmap
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
        # do not use asdict to avoid making copies
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if val is None:
                continue
            if isinstance(val, jax.Array) or isinstance(val, np.ndarray):
                self._save_array(np.array(val), field_.name, out_dir)
            elif isinstance(val, Stokes):
                leaves = [np.array(leaf) for leaf in jax.tree.leaves(val)]
                self._save_array(np.array(leaves), field_.name, out_dir)
            elif isinstance(val, pixell.enmap.ndmap):
                pixell.enmap.write_map(
                    (out_dir / f'{field_.name}.hdf').as_posix(), val, allow_modify=True
                )
            elif isinstance(val, StokesLandscape):
                pass  # landscape is not saved to file
            else:
                furax_logger.warning(f'not saving {field_.name}')

    def _save_array(self, arr: np.ndarray, name: str, out_dir: Path) -> None:
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
                str(out_dir / f'{name}.fits'), maps, nest=self.landscape.nested, overwrite=True
            )
        else:
            furax_logger.warning(
                f'saving {name} as npy: geometry information will not be embedded in the file'
            )
            np.save(out_dir / name, arr)
