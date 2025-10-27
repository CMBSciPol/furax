from pathlib import Path

import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.interfaces.sotodlib import SOTODLibReader
from furax.mapmaking.mapmaker import MultiObservationMapMaker


def test_accumulate():
    folder = Path(__file__).parents[1] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    # ndet1 = 2
    # nsample1 = 1000
    # ndet2 = 14
    # nsample2 = 10000

    reader = SOTODLibReader(files)
    maker = MultiObservationMapMaker(reader)
    operators = maker.build_acquisitions()
    assert len(operators) == 2

    rhs = maker.accumulate_rhs(operators)
    assert rhs.shape == maker.landscape.shape
