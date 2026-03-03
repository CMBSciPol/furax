import importlib.util
from pathlib import Path

import pytest

from furax.mapmaking import MultiObservationMapMaker
from furax.mapmaking.config import LandscapeConfig, Landscapes, MapMakingConfig

sotodlib_installed = importlib.util.find_spec('sotodlib') is not None


@pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed')
def test_demod_acquisition_vs_sotodlib():
    """Validate the demodulated acquisition transpose against sotodlib's project_rhs_demod.

    The furax acquisition transpose H.T maps demodulated (I, Q, U) TODs to a sky map.
    This should match sotodlib's project_rhs_demod with unit detector weights.
    """
    import numpy as np
    from sotodlib import coords
    from sotodlib.mapmaking.demod_mapmaker import project_rhs_demod

    from furax.interfaces.sotodlib import LazySOTODLibObservation

    nside = 16
    folder = Path(__file__).parents[1] / 'data' / 'sotodlib'
    obs = LazySOTODLibObservation(folder / 'test_obs.h5').get_data()

    config = MapMakingConfig(
        pointing_on_the_fly=True,
        landscape=LandscapeConfig(type=Landscapes.HPIX, nside=nside),
        demodulated=True,
    )

    # Build furax demodulated acquisition operator
    observations = [LazySOTODLibObservation(folder / 'test_obs.h5')]
    maker = MultiObservationMapMaker(observations, config=config)
    (h,) = maker.build_acquisitions()

    # Get demodulated TODs (I, Q, U)
    tod_iqu = obs.get_demodulated_tods(stokes='IQU')

    # Furax result: H.T @ tod
    furax_map = h.T(tod_iqu)

    # sotodlib result: project_rhs_demod with unit weights.
    # hwp=True reflects gamma angles (γ → -γ), matching furax's demod angle convention
    # where H.T rotates by α - 2γ (the negative of _get_angles's 2γ - α).
    hp_geom = coords.healpix_utils.get_geometry(nside=nside, ordering='RING')
    pmap = coords.P.for_tod(obs.data, geom=hp_geom, comps='TQU', hwp=True)
    sotodlib_map = project_rhs_demod(
        pmap,
        signalT=np.array(tod_iqu.i, dtype=np.float32),
        signalQ=np.array(tod_iqu.q, dtype=np.float32),
        signalU=np.array(tod_iqu.u, dtype=np.float32),
        det_weightsT=None,
        det_weightsQU=None,
    )

    # Compare: furax_map is a StokesIQU pytree, sotodlib_map has shape (3, npix).
    # Tolerance is loose because sotodlib's signal is cast to float32 before projection.
    np.testing.assert_allclose(np.array(furax_map.i), sotodlib_map[0], rtol=1e-5, atol=0)
    np.testing.assert_allclose(np.array(furax_map.q), sotodlib_map[1], rtol=1e-5, atol=0)
    np.testing.assert_allclose(np.array(furax_map.u), sotodlib_map[2], rtol=1e-5, atol=0)
