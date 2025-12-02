from pathlib import Path

import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.core import BlockColumnOperator
from furax.interfaces.sotodlib import SOTODLibObservationResource
from furax.interfaces.toast import ToastObservationResource
from furax.mapmaking import GroundObservationReader, MapMakingConfig, MultiObservationMapMaker
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.preconditioner import BJPreconditioner


@pytest.fixture(
    scope='module',
    params=[(SOTODLibObservationResource, 'sotodlib'), (ToastObservationResource, 'toast')],
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


@pytest.fixture(scope='module')
def config():
    return MapMakingConfig(
        pointing_on_the_fly=True, landscape=LandscapeConfig(type=Landscapes.HPIX, nside=16)
    )


def test_acquisitions(resources, config):
    maker = MultiObservationMapMaker(resources, config=config)
    reader = GroundObservationReader(resources)
    operators = maker.build_acquisitions()
    assert len(operators) == 2
    for op in operators:
        assert op.in_structure() == maker.landscape.structure
        assert op.out_structure() == reader.out_structure['sample_data']


def test_weightings(resources, config):
    maker = MultiObservationMapMaker(resources, config=config)
    reader = GroundObservationReader(resources)
    tod_structure = reader.out_structure['sample_data']
    operators = maker.build_weight_operators(tod_structure)
    assert len(operators) == 2
    for op in operators:
        assert op.in_structure() == tod_structure


def test_accumulate_rhs(resources, config):
    maker = MultiObservationMapMaker(resources, config=config)
    acquisitions = maker.build_acquisitions()
    reader = GroundObservationReader(resources)
    tod_structure = reader.out_structure['sample_data']
    weightings = maker.build_weight_operators(tod_structure)
    rhs = maker.accumulate_rhs(acquisitions, weightings)
    assert rhs.shape == maker.landscape.shape


def test_binning(resources, config):
    maker = MultiObservationMapMaker(resources, config=config)
    ops = maker.build_acquisitions()
    h = BlockColumnOperator(ops)
    system = BJPreconditioner.create((h.T @ h).reduce())
    assert system.in_structure() == maker.landscape.structure
    zeros = system(maker.landscape.full(0))
    assert zeros.shape == maker.landscape.shape


def test_full_mapmaker(resources, config):
    maker = MultiObservationMapMaker(resources, config=config)
    results = maker.run()
    assert 'map' in results
    assert 'weights' in results
