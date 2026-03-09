from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from sotodlib import coords
from sotodlib.mapmaking.demod_mapmaker import project_rhs_demod

from furax.interfaces.sotodlib import LazySOTODLibObservation
from furax.mapmaking import MultiObservationMapMaker
from furax.mapmaking.acquisition import build_acquisition_operator
from furax.mapmaking.config import LandscapeConfig, Landscapes, MapMakingConfig
from furax.obs.landscapes import HealpixLandscape

FOLDER = Path(__file__).parents[2] / 'data' / 'sotodlib'
NSIDE = 16


def _make_config(demodulated: bool) -> MapMakingConfig:
    return MapMakingConfig(
        pointing_on_the_fly=True,
        landscape=LandscapeConfig(type=Landscapes.HPIX, nside=NSIDE),
        demodulated=demodulated,
    )


def _sotodlib_pointing(obs, hwp: bool):
    hp_geom = coords.healpix_utils.get_geometry(nside=NSIDE, ordering='RING')
    return coords.P.for_tod(obs.data, geom=hp_geom, comps='TQU', hwp=hwp)


def test_acquisition_no_hwp_vs_sotodlib():
    """Validate the acquisition transpose against sotodlib.

    The furax acquisition includes a LinearPolarizerOperator which applies a factor of 0.5:
        d = 0.5 * (I[pix] + Q[pix]*cos(2φ) + U[pix]*sin(2φ))

    When projecting back to the sky, H.T @ signal should satisfy:
        furax_map.{i,q,u} = 0.5 * sotodlib_P.to_map(signal).{T,Q,U}

    since sotodlib uses d = I + Q*cos(2φ) + U*sin(2φ) (no 0.5 factor).
    """

    lazy_obs = LazySOTODLibObservation(FOLDER / 'test_obs_2.h5')
    obs = lazy_obs.get_data()
    landscape = HealpixLandscape(nside=NSIDE, stokes='IQU', dtype=jnp.float64)
    h = build_acquisition_operator(
        landscape,
        obs.get_boresight_quaternions(),
        obs.get_detector_quaternions(),
        hwp_angles=None,
        pointing_on_the_fly=True,
    )

    tods = obs.get_tods()
    furax_map = h.T(tods)

    pmap = _sotodlib_pointing(obs, hwp=False)
    sotodlib_map = pmap.to_map(tod=obs.data, signal=np.array(tods, dtype=np.float32))

    # Furax TODs assume power, so they are 2x smaller
    for i, leaf in enumerate(jax.tree.leaves(furax_map)):
        assert_allclose(2 * leaf, sotodlib_map[i], rtol=1e-5)


def test_demod_acquisition_vs_sotodlib():
    """Validate the demodulated acquisition transpose against sotodlib's project_rhs_demod.

    The furax acquisition transpose H.T maps demodulated (I, Q, U) TODs to a sky map.
    This should match sotodlib's project_rhs_demod with unit detector weights.
    """

    lazy_obs = LazySOTODLibObservation(FOLDER / 'test_obs_2.h5')
    obs = lazy_obs.get_data()
    maker = MultiObservationMapMaker([lazy_obs], config=_make_config(demodulated=True))
    (h,) = maker.build_acquisitions()

    # Get demodulated TODs (I, Q, U)
    tod_iqu = obs.get_demodulated_tods(stokes='IQU')

    # Furax result: H.T @ tod
    furax_map = h.T(tod_iqu)

    # sotodlib result: project_rhs_demod with unit weights.
    pmap = _sotodlib_pointing(obs, hwp=True)
    sotodlib_map = project_rhs_demod(
        pmap,
        signalT=np.array(tod_iqu.i, dtype=np.float32),
        signalQ=np.array(tod_iqu.q, dtype=np.float32),
        signalU=np.array(tod_iqu.u, dtype=np.float32),
        det_weightsT=None,
        det_weightsQU=None,
    )

    # Furax TODs assume power, so they are 2x smaller
    for i, leaf in enumerate(jax.tree.leaves(furax_map)):
        assert_allclose(2 * leaf, sotodlib_map[i], rtol=1e-5)
