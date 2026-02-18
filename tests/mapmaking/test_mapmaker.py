from pathlib import Path

import pytest

from furax.obs.stokes import Stokes

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.core import BlockColumnOperator
from furax.interfaces.sotodlib import LazySOTODLibObservation
from furax.interfaces.toast import LazyToastObservation
from furax.mapmaking import MapMakingConfig, MultiObservationMapMaker, ObservationReader
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.noise import WhiteNoiseModel
from furax.mapmaking.preconditioner import BJPreconditioner

# Parameters for all the tests below.
# We test sotodlib and toast, as well as sotodlib with demodulated data.
# Add more lines to test other interfaces/combinations.
PARAMS = [
    pytest.param('sotodlib', False, id='sotodlib'),
    pytest.param('sotodlib', True, id='sotodlib-demod'),
    pytest.param('toast', False, id='toast'),
]


def make_observations(name: str) -> list:
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        files = [folder / 'test_obs.h5'] * 2
        return [LazyToastObservation(f) for f in files]
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    return [LazySOTODLibObservation(f) for f in files]


def make_config(demodulated: bool = False) -> MapMakingConfig:
    return MapMakingConfig(
        pointing_on_the_fly=True,
        landscape=LandscapeConfig(type=Landscapes.HPIX, nside=16),
        demodulated=demodulated,
    )


@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_acquisitions(name, demodulated):
    observations = make_observations(name)
    config = make_config(demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    reader = ObservationReader(observations, demodulated=demodulated)
    operators = maker.build_acquisitions()
    assert len(operators) == len(observations)
    for op in operators:
        assert op.in_structure == maker.landscape.structure
        assert op.out_structure == reader.out_structure['sample_data']


@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_noise_models(name, demodulated):
    observations = make_observations(name)
    config = make_config(demodulated)
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


@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_accumulate_rhs(name, demodulated):
    observations = make_observations(name)
    config = make_config(demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    reader = ObservationReader(observations, demodulated=demodulated)
    tod_structure = reader.out_structure['sample_data']
    noise_models, sample_rates = maker.noise_models_and_sample_rates()
    w_blocks = [
        model.inverse_operator(
            tod_structure,
            sample_rate=fs,
            correlation_length=config.correlation_length,
        )
        for model, fs in zip(noise_models, sample_rates, strict=True)
    ]
    h_blocks = maker.build_acquisitions()
    maskers = maker.build_sample_maskers(h_blocks[0].out_structure)
    rhs = maker.accumulate_rhs(h_blocks, w_blocks, maskers)
    assert rhs.shape == maker.landscape.shape


@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_binning(name, demodulated):
    observations = make_observations(name)
    config = make_config(demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    h_blocks = maker.build_acquisitions()
    h = BlockColumnOperator(h_blocks)
    system = BJPreconditioner.create((h.T @ h).reduce())
    assert system.in_structure == maker.landscape.structure
    zeros = system(maker.landscape.full(0))
    assert zeros.shape == maker.landscape.shape


@pytest.mark.parametrize('name,demodulated', PARAMS)
def test_full_mapmaker(name, demodulated):
    observations = make_observations(name)
    config = make_config(demodulated)
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.run()
    stokes, pixels = results.map.shape
    assert results.map_weights.shape == (pixels, stokes, stokes)
