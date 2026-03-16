import importlib.util
from math import prod
from pathlib import Path

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
from furax.mapmaking.config import LandscapeConfig, Landscapes, SotodlibConfig
from furax.mapmaking.noise import WhiteNoiseModel
from furax.mapmaking.pointing import PointingOperator
from furax.obs.stokes import Stokes, ValidStokesType

# Skip tests for interfaces that are not installed
sotodlib_installed = importlib.util.find_spec('sotodlib') is not None
toast_installed = importlib.util.find_spec('toast') is not None

# Parameters for all the tests below.
# We test sotodlib and toast, as well as sotodlib with demodulated data.
# Add more lines to test other interfaces/combinations.
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


def make_observations(name: str, demodulated: bool = False) -> list[AbstractLazyObservation]:
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


STOKES_PARAMS = ['I', 'QU', 'IQU']


def make_config(
    stokes: ValidStokesType = 'IQU',
    demodulated: bool = False,
    fit_noise_model: bool = True,
) -> MapMakingConfig:
    return MapMakingConfig(
        pointing_on_the_fly=True,
        landscape=LandscapeConfig(type=Landscapes.HPIX, nside=16),
        stokes=stokes,
        fit_noise_model=fit_noise_model,
        nperseg=512,
        sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
    )


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
class TestMultiObsMapMaker:
    """Test the multi-observation mapmaker.

    Use a class in order to parametrize over multiple tests at once.
    """

    def test_blocks_vs_reader_structure(self, name, demodulated, stokes):
        observations = make_observations(name, demodulated)
        config = make_config(stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        reader = ObservationReader(observations, demodulated=demodulated, stokes=stokes)
        blocks = maker.build_model()
        n_obs = jax.tree.leaves(blocks)[0].shape[0]
        assert n_obs == len(observations) == reader.count
        assert blocks.map_structure == maker.landscape.structure
        assert blocks.tod_structure == reader.out_structure['sample_data']

    def test_last_acquisition_operand_is_pointing(self, name, demodulated, stokes):
        observations = make_observations(name, demodulated)
        config = make_config(stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        h = maker.build_model().H
        assert isinstance(h, CompositionOperator)
        assert isinstance(h.operands[-1], PointingOperator)

    @pytest.mark.parametrize('fit_models', [True, False])
    def test_white_noise_models_binned_or_demodulated(self, name, demodulated, stokes, fit_models):
        observations = make_observations(name, demodulated)
        config = make_config(stokes=stokes, demodulated=demodulated, fit_noise_model=fit_models)
        maker = MultiObservationMapMaker(observations, config=config)
        noise_model = maker.build_model().noise_model
        if demodulated:
            # In demodulated case each block has a Stokes pytree of per-component WhiteNoiseModel's
            assert isinstance(noise_model, Stokes.class_for(config.stokes))
            assert all(
                isinstance(getattr(noise_model, stoke.lower()), WhiteNoiseModel)
                for stoke in config.stokes
            )
        else:
            assert isinstance(noise_model, WhiteNoiseModel)

    def test_rhs_shape(self, name, demodulated, stokes):
        observations = make_observations(name, demodulated)
        config = make_config(stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        blocks = maker.build_model()
        rhs = maker.accumulate_rhs(blocks)
        assert rhs.shape == maker.landscape.shape

    def test_hits_are_nonnegative(self, name, demodulated, stokes):
        observations = make_observations(name, demodulated)
        config = make_config(stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        blocks = maker.build_model()
        hits = maker.accumulate_hits(blocks)
        assert hits.shape == maker.landscape.shape
        assert jnp.all(hits >= 0)

    def test_full_mapmaker(self, name, demodulated, stokes):
        observations = make_observations(name, demodulated)
        config = make_config(stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        n_pixel = prod(results.map.shape)
        assert results.icov.shape == (n_stokes, n_stokes, n_pixel)
        assert results.solver_stats is not None
        num_steps = results.solver_stats['num_steps']
        assert num_steps == 1, (
            f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
        )
