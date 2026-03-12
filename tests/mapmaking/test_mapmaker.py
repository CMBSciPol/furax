import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from furax._config import Config
from furax.core import BlockColumnOperator
from furax.mapmaking import (
    AbstractLazyObservation,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.noise import WhiteNoiseModel
from furax.mapmaking.preconditioner import BJPreconditioner
from furax.obs.stokes import Stokes, StokesI, ValidStokesType

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


def make_observations(name: str) -> list[AbstractLazyObservation]:
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        from furax.interfaces.toast import LazyToastObservation

        files = [folder / 'test_obs.h5'] * 2
        return [LazyToastObservation(f) for f in files]
    elif name == 'sotodlib':
        from furax.interfaces.sotodlib import LazySOTODLibObservation

        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
        return [LazySOTODLibObservation(f) for f in files]
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
        demodulated=demodulated,
        fit_noise_model=fit_noise_model,
        nperseg=512,
    )


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_acquisitions(name, demodulated, stokes):
    observations = make_observations(name)
    config = make_config(stokes, demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    reader = ObservationReader(observations, demodulated=demodulated, stokes=stokes)
    operators = maker.build_acquisitions()
    assert len(operators) == len(observations)
    for op in operators:
        assert op.in_structure == maker.landscape.structure
        assert op.out_structure == reader.out_structure['sample_data']


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
@pytest.mark.parametrize('fit_models', [True, False])
def test_noise_models(name, demodulated, fit_models, stokes):
    observations = make_observations(name)
    config = make_config(stokes=stokes, demodulated=demodulated, fit_noise_model=fit_models)
    maker = MultiObservationMapMaker(observations, config=config)
    noise_models, _ = maker.noise_models_and_sample_rates()
    assert len(noise_models) == len(observations)
    # those must be white noise models in binned mapmaking
    # in the demodulated case each model is a Stokes pytree of per-component WhiteNoiseModels
    if demodulated:
        for model_tree in noise_models:
            assert isinstance(model_tree, Stokes.class_for(config.stokes))
            assert all(
                isinstance(getattr(model_tree, stoke.lower()), WhiteNoiseModel)
                for stoke in config.stokes
            )
    else:
        assert all(isinstance(model, WhiteNoiseModel) for model in noise_models)


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_accumulate_rhs(name, demodulated, stokes):
    observations = make_observations(name)
    config = make_config(stokes, demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    reader = ObservationReader(observations, demodulated=demodulated, stokes=stokes)
    tod_structure = reader.out_structure['sample_data']
    noise_models, sample_rates = maker.noise_models_and_sample_rates()
    w_blocks = MultiObservationMapMaker.noise_operator_blocks(
        noise_models,
        tod_structure,
        sample_rates,
        config.correlation_length,
        inverse=True,
    )
    h_blocks = maker.build_acquisitions()
    maskers = maker.build_sample_maskers(h_blocks[0].out_structure)
    rhs = maker.accumulate_rhs(h_blocks, w_blocks, maskers)
    assert rhs.shape == maker.landscape.shape


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_accumulate_hit_map(name, demodulated, stokes):
    observations = make_observations(name)
    config = make_config(stokes, demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    h_blocks = maker.build_acquisitions()
    maskers = maker.build_sample_maskers(h_blocks[0].out_structure)
    hit_map = maker.accumulate_hit_map(h_blocks, maskers)
    assert isinstance(hit_map, StokesI)
    assert hit_map.shape == maker.landscape.shape
    # hit maps should be nonnegative integers
    assert jnp.all(hit_map.i >= 0)
    assert jnp.all(hit_map.i == jnp.floor(hit_map.i))


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_binning(name, demodulated, stokes):
    observations = make_observations(name)
    config = make_config(stokes, demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    h_blocks = maker.build_acquisitions()
    h = BlockColumnOperator(h_blocks)
    system = BJPreconditioner.create((h.T @ h).reduce())
    assert system.in_structure == maker.landscape.structure
    zeros = system(maker.landscape.full(0))
    assert zeros.shape == maker.landscape.shape


@pytest.mark.parametrize('stokes', STOKES_PARAMS)
@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_full_mapmaker(name, demodulated, stokes):
    observations = make_observations(name)
    config = make_config(stokes, demodulated)
    maker = MultiObservationMapMaker(observations, config=config)

    num_steps = None

    def capture_iterations(solution):
        nonlocal num_steps
        num_steps = int(solution.stats['num_steps'])

    with Config(solver_callback=capture_iterations):
        results = maker.run()

    n_stokes, pixels = results.map.shape
    assert results.weights.shape == (n_stokes, n_stokes, pixels)
    assert num_steps == 1, f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
