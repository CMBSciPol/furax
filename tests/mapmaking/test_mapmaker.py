from pathlib import Path

import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.core import BlockColumnOperator
from furax.interfaces.sotodlib import SOTODLibReader
from furax.interfaces.toast import ToastReader
from furax.mapmaking import (
    MapMakingConfig,
    MultiObservationBinnedMapMaker,
)
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.preconditioner import BJPreconditioner


@pytest.fixture(scope='module', params=[(SOTODLibReader, 'sotodlib'), (ToastReader, 'toast')])
def reader(request: pytest.FixtureRequest):
    cls, name = request.param
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        # only one test file exists
        files = [folder / 'test_obs.h5'] * 2
    else:
        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    return cls(files)


@pytest.fixture(scope='module')
def config():
    return MapMakingConfig(
        pointing_on_the_fly=True, landscape=LandscapeConfig(type=Landscapes.HPIX, nside=16)
    )


def test_acquisitions(reader, config):
    maker = MultiObservationBinnedMapMaker(reader, config=config)
    operators = maker.build_acquisitions()
    assert len(operators) == 2
    for op in operators:
        assert op.in_structure() == maker.landscape.structure
        assert op.out_structure() == reader.out_structure['sample_data']


def test_accumulate_rhs(reader, config):
    maker = MultiObservationBinnedMapMaker(reader, config=config)
    operators = maker.build_acquisitions()
    rhs = maker.accumulate_rhs(operators)
    assert rhs.shape == maker.landscape.shape


def test_full_acquisition(reader, config):
    maker = MultiObservationBinnedMapMaker(reader, config=config)
    ops = maker.build_acquisitions()
    h = BlockColumnOperator(ops)
    system = BJPreconditioner.create((h.T @ h).reduce())
    assert system.in_structure() == maker.landscape.structure
    zeros = system(maker.landscape.full(0))
    assert zeros.shape == maker.landscape.shape


def test_full_mapmaker(reader, config):
    maker = MultiObservationBinnedMapMaker(reader, config=config)
    results = maker.run()
    assert 'map' in results
    assert 'weights' in results
