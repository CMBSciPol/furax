from pathlib import Path

import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.core import BlockColumnOperator
from furax.interfaces.sotodlib import SOTODLibReader
from furax.io.readers import AbstractReader
from furax.mapmaking import MapMakingConfig
from furax.mapmaking.config import LandscapeConfig, Landscapes
from furax.mapmaking.mapmaker import MultiObservationBinnedMapMaker
from furax.mapmaking.preconditioner import BJPreconditioner
from furax.obs.landscapes import HealpixLandscape


@pytest.fixture(scope='module')
def maker():
    return MultiObservationBinnedMapMaker(
        config=MapMakingConfig(
            pointing_on_the_fly=True, landscape=LandscapeConfig(type=Landscapes.HPIX)
        )
    )


@pytest.fixture(scope='module')
def reader():
    folder = Path(__file__).parents[1] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    return SOTODLibReader(files)


def test_acquisitions(reader: AbstractReader, maker: MultiObservationBinnedMapMaker) -> None:
    landscape = HealpixLandscape(stokes='IQU', nside=16)
    operators = maker.build_acquisitions(reader, landscape)
    assert len(operators) == 2
    for op in operators:
        assert op.in_structure() == landscape.structure
        assert op.out_structure() == reader.out_structure['signal']


def test_accumulate_rhs(reader: AbstractReader, maker: MultiObservationBinnedMapMaker) -> None:
    landscape = HealpixLandscape(stokes='IQU', nside=16)
    operators = maker.build_acquisitions(reader, landscape)
    rhs = maker.accumulate_rhs(reader, landscape, operators)
    assert rhs.shape == landscape.shape


def test_full_acquisition(reader: AbstractReader, maker: MultiObservationBinnedMapMaker) -> None:
    landscape = HealpixLandscape(stokes='IQU', nside=16)
    ops = maker.build_acquisitions(reader, landscape)
    h = BlockColumnOperator(ops)
    sys = BJPreconditioner.create((h.T @ h).reduce())
    assert sys.in_structure() == landscape.structure
    zeros = sys(landscape.full(0))
    assert zeros.shape == landscape.shape


def test_full_mapmaker(reader: AbstractReader, maker: MultiObservationBinnedMapMaker) -> None:
    results = maker.run(reader)
    assert 'map' in results
    assert 'weights' in results
