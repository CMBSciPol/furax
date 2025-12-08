from pathlib import Path

import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.core import BlockColumnOperator
from furax.interfaces.sotodlib import SOTODLibObservationResource
from furax.interfaces.toast import ToastObservationResource
from furax.mapmaking import GroundObservationReader, MapMakingConfig, MultiObservationMapMaker
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.noise import WhiteNoiseModel
from furax.mapmaking.preconditioner import BJPreconditioner


@pytest.fixture(
    params=[(SOTODLibObservationResource, 'sotodlib'), (ToastObservationResource, 'toast')]
)
def resources(request: pytest.FixtureRequest):
    cls, name = request.param
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        # only one test file exists
        files = [folder / 'test_obs.h5'] * 2
    else:
        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    return [cls(f) for f in files]


@pytest.fixture
def config():
    return MapMakingConfig(
        pointing_on_the_fly=True, landscape=LandscapeConfig(type=Landscapes.HPIX, nside=16)
    )


@pytest.fixture
def maker(resources, config):
    return MultiObservationMapMaker(resources, config=config)


@pytest.fixture
def reader(resources):
    return GroundObservationReader(resources)


def test_acquisitions(maker, reader):
    operators = maker.build_acquisitions()
    assert len(operators) == 2
    for op in operators:
        assert op.in_structure() == maker.landscape.structure
        assert op.out_structure() == reader.out_structure['sample_data']


def test_noise_models(maker):
    noise_models, _ = maker.noise_models_and_sample_rates()
    assert len(noise_models) == 2
    # those must be white noise models in binned mapmaking
    assert all(isinstance(model, WhiteNoiseModel) for model in noise_models)


@pytest.fixture
def w_blocks(config, maker, reader):
    tod_structure = reader.out_structure['sample_data']
    noise_models, sample_rates = maker.noise_models_and_sample_rates()
    return [
        model.inverse_operator(
            tod_structure,
            sample_rate=fs,
            correlation_length=config.correlation_length,
        )
        for model, fs in zip(noise_models, sample_rates, strict=True)
    ]


def test_accumulate_rhs(maker, w_blocks):
    h_blocks = maker.build_acquisitions()
    maskers = maker.build_sample_maskers()
    rhs = maker.accumulate_rhs(h_blocks, w_blocks, maskers)
    assert rhs.shape == maker.landscape.shape


def test_binning(maker):
    h_blocks = maker.build_acquisitions()
    h = BlockColumnOperator(h_blocks)
    system = BJPreconditioner.create((h.T @ h).reduce())
    assert system.in_structure() == maker.landscape.structure
    zeros = system(maker.landscape.full(0))
    assert zeros.shape == maker.landscape.shape


def test_full_mapmaker(maker):
    results = maker.run()
    assert 'map' in results
    assert 'weights' in results
