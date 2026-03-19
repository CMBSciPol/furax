import importlib.util
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from furax.core import CompositionOperator
from furax.mapmaking import (
    AbstractLazyObservation,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    HealpixLandscapeConfig,
    LandscapeConfig,
    SkyPatch,
    SotodlibConfig,
    WCSLandscapeConfig,
)
from furax.mapmaking.noise import WhiteNoiseModel
from furax.mapmaking.pointing import PointingOperator
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import Stokes, ValidStokesType

# Skip tests for interfaces that are not installed
sotodlib_installed = importlib.util.find_spec('sotodlib') is not None
toast_installed = importlib.util.find_spec('toast') is not None

# Parameters for all the tests below.
# Tests are parametrized over:
#   - PARAMS: observation interface (sotodlib, toast) and demodulation flag
#   - STOKES_TYPES: Stokes components ('I', 'QU', 'IQU')
#   - LANDSCAPE_TYPES: output map projection (healpix, CAR)
# Add more entries to any of these lists to extend coverage.

PARAMS = [
    pytest.param(
        'sotodlib',
        False,
        id='sotodlib',
        marks=pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed'),
    ),
    pytest.param(
        'sotodlib',
        True,
        id='sotodlib-demod',
        marks=pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed'),
    ),
    pytest.param(
        'toast',
        False,
        id='toast',
        marks=pytest.mark.skipif(not toast_installed, reason='toast is not installed'),
    ),
]
STOKES_TYPES = ['I', 'QU', 'IQU']
LANDSCAPE_TYPES = ['healpix', 'car']


@pytest.mark.parametrize('landscape_type', LANDSCAPE_TYPES)
@pytest.mark.parametrize('stokes', STOKES_TYPES)
@pytest.mark.parametrize('name,demodulated', PARAMS)
class TestMultiObsMapMaker:
    """Test the multi-observation mapmaker.

    Use a class in order to parametrize over multiple tests at once.
    """

    def test_blocks_vs_reader_structure(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        reader = ObservationReader(observations, demodulated=demodulated, stokes=stokes)
        blocks = maker.build_model()
        n_obs = jax.tree.leaves(blocks)[0].shape[0]
        assert n_obs == len(observations) == reader.count
        assert blocks.map_structure == maker.landscape.structure
        assert blocks.tod_structure == reader.out_structure['sample_data']

    def test_last_acquisition_operand_is_pointing(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        h = maker.build_model().H
        assert isinstance(h, CompositionOperator)
        assert isinstance(h.operands[-1], PointingOperator)

    @pytest.mark.parametrize('fit_models', [True, False])
    def test_white_noise_models_binned_or_demodulated(
        self, name, demodulated, stokes, landscape_type, fit_models
    ):
        observations = _observations(name, demodulated)
        config = _config(
            landscape_type, stokes, demodulated=demodulated, fit_noise_model=fit_models
        )
        maker = MultiObservationMapMaker(observations, config=config)
        noise_model = maker.build_model().noise_model
        if demodulated:
            # In demodulated case each block has a Stokes pytree of per-component WhiteNoiseModel's
            assert isinstance(noise_model, Stokes.class_for(stokes))
            assert all(
                isinstance(getattr(noise_model, stoke.lower()), WhiteNoiseModel) for stoke in stokes
            )
        else:
            assert isinstance(noise_model, WhiteNoiseModel)

    def test_rhs_shape(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        blocks = maker.build_model()
        rhs = maker.accumulate_rhs(blocks)
        assert rhs.shape == maker.landscape.shape

    def test_hits_are_nonnegative(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        blocks = maker.build_model()
        hits = maker.accumulate_hits(blocks)
        assert hits.shape == maker.landscape.shape
        assert jnp.all(hits >= 0)

    def test_full_mapmaker(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
        assert results.solver_stats is not None
        num_steps = results.solver_stats['num_steps']
        assert num_steps == 1, (
            f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
        )


def _observations(name: str, demodulated: bool = False) -> list[AbstractLazyObservation]:
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        from furax.interfaces.toast import LazyToastObservation

        files = [folder / 'test_obs.h5'] * 2
        return [LazyToastObservation(f) for f in files]
    elif name == 'sotodlib':
        from furax.interfaces.sotodlib import LazySOTODLibObservation

        sotodlib_config = SotodlibConfig(demodulated=True) if demodulated else None
        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
        return [LazySOTODLibObservation(f, sotodlib_config=sotodlib_config) for f in files]
    raise NotImplementedError


def _config(
    landscape_type: Literal['healpix', 'car'],
    stokes: ValidStokesType,
    demodulated: bool = False,
    fit_noise_model: bool = True,
) -> MapMakingConfig:
    if landscape_type == 'healpix':
        lc = LandscapeConfig(stokes=stokes, healpix=HealpixLandscapeConfig(nside=16))
    else:
        lc = LandscapeConfig(
            stokes=stokes,
            healpix=None,
            wcs=WCSLandscapeConfig(
                projection=ProjectionType.CAR,
                resolution=60.0,
                patch=SkyPatch(center=(0.0, 0.0), width=20.0, height=20.0),
            ),
        )
    return MapMakingConfig(
        pointing_on_the_fly=True,
        landscape=lc,
        fit_noise_model=fit_noise_model,
        nperseg=512,
        sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
    )
