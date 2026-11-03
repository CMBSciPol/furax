from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
import pixell.enmap
import pixell.utils
from astropy.wcs import WCS as AstropyWCS
from jaxtyping import DTypeLike

from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.stokes import ValidStokesLiteral

from ._geometry import minimum_enclosing_arc
from ._observation import AbstractLazyObservation, FileBackedLazyObservation, ReaderField
from .config import LandscapeConfig, WCSConfig


def build_landscape(
    observations: Sequence[AbstractLazyObservation[Any]], config: LandscapeConfig, dtype: DTypeLike
) -> StokesLandscape:
    """Build the sky grid a run solves on.

    The grid comes from the config alone for HEALPix and for a WCS with an explicit geometry. For
    a WCS without one, the observations are scanned for their combined footprint.
    """
    if config.healpix is not None:
        return HealpixLandscape(
            nside=config.healpix.nside,
            stokes=config.stokes,
            dtype=dtype,
            nested=config.healpix.ordering == 'nest',
        )
    assert config.wcs is not None  # LandscapeConfig sets exactly one of healpix and wcs
    if config.wcs.has_geometry:
        shape, wcs = _explicit_wcs_geometry(config.wcs)
        return WCSLandscape.from_wcs(shape, wcs, config.stokes, dtype)
    return scan_wcs_footprint(observations, config.wcs, config.stokes, dtype)


def scan_wcs_footprint(
    observations: Sequence[AbstractLazyObservation[Any]],
    wcs_config: WCSConfig,
    stokes: ValidStokesLiteral,
    dtype: DTypeLike,
) -> WCSLandscape:
    """Scan observations to determine the combined WCS footprint and build a WCSLandscape.

    Performs a preliminary pass over all observations, loading the pointing data needed
    to compute each observation's sky coverage, then combines them into a unified footprint.

    Restricted to single-process runs over file-backed observations: the scan is an
    unsharded loop calling ``get_data`` per observation. For file-backed observations that
    cheaply loads only the requested pointing fields, but for sources that cannot subset
    their fields (e.g. preproc-backed observations) it would run the full load/preprocess
    pipeline on every observation, on every process. Pre-compute the footprint and pass an
    explicit WCS landscape instead in those cases.
    """
    if jax.process_count() > 1:
        msg = (
            'Automatic WCS footprint scanning is only supported on a single process. '
            'Pre-compute the footprint and pass an explicit WCS landscape instead.'
        )
        raise RuntimeError(msg)
    if not all(isinstance(obs, FileBackedLazyObservation) for obs in observations):
        msg = (
            'Automatic WCS footprint scanning requires file-backed observations '
            '(get_data must cheaply subset pointing fields). Pre-compute the footprint'
            'and pass an explicit WCS landscape instead.'
        )
        raise RuntimeError(msg)
    res = wcs_config.resolution * pixell.utils.arcmin
    proj = wcs_config.projection.name.lower()
    pointing_fields = [ReaderField.BORESIGHT_QUATERNIONS, ReaderField.DETECTOR_QUATERNIONS]

    n = len(observations)
    corners_rad = np.empty((2, 2, n))  # [bottom-left, top-right] corners per observation
    for i, lazy_obs in enumerate(observations):
        obs = lazy_obs.get_data(pointing_fields)
        shape, wcs = obs.get_wcs_shape_and_kernel(
            resolution_arcmin=wcs_config.resolution, projection=wcs_config.projection
        )
        # RA _decreases_ left-to-right (increases toward the East)
        # so each corner pair is technically [[dec_lo, ra_hi], [dec_hi, ra_lo]]
        corners_rad[..., i] = pixell.enmap.corners(shape, wcs, corner=True)

    # find the corners of the smallest box covering all the individual patches
    dec_lo = corners_rad[0, 0].min()
    dec_hi = corners_rad[1, 0].max()
    # minimum_enclosing_arc expects [lo, hi] intervals with shape (2, n)
    ra_lo, ra_hi = minimum_enclosing_arc(corners_rad[::-1, 1].T)
    union_box = np.array([[dec_lo, ra_hi], [dec_hi, ra_lo]])

    # create the final shape and WCS objects for this covering box
    shape, wcs = pixell.enmap.geometry(pos=union_box, res=res, proj=proj)
    return WCSLandscape.from_wcs(shape, wcs, stokes, dtype)


def _explicit_wcs_geometry(wcs_config: WCSConfig) -> tuple[tuple[int, int], AstropyWCS]:
    """The shape and WCS of the explicit geometry of a WCS config: a geometry file or a patch."""
    if not wcs_config.has_geometry:
        raise ValueError('wcs_config must specify a geometry_file or patch.')
    if wcs_config.geometry_file is not None:
        return pixell.enmap.read_map_geometry(wcs_config.geometry_file)
    assert wcs_config.patch is not None  # ty assert: has_geometry guarantees patch is set here
    res = wcs_config.resolution * pixell.utils.arcmin
    half_w = np.radians(wcs_config.patch.width / 2)
    half_h = np.radians(wcs_config.patch.height / 2)
    ra, dec = np.radians(wcs_config.patch.center)
    corners = np.array([[dec - half_h, ra + half_w], [dec + half_h, ra - half_w]])
    return pixell.enmap.geometry(pos=corners, res=res, proj=wcs_config.projection.name.lower())
