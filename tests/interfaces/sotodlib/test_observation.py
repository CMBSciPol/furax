from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from furax.interfaces.sotodlib import SOTODLibObservation

FOLDER = Path(__file__).parents[2] / 'data/sotodlib'
FILE = FOLDER / 'test_obs.h5'


def test_from_file() -> None:
    ndet = 2
    nsample = 1000

    obs = SOTODLibObservation.from_file(FILE)
    assert obs.n_detectors == ndet
    assert obs.n_samples == nsample


def test_from_file_missing() -> None:
    with pytest.raises(FileNotFoundError):
        SOTODLibObservation.from_file('non_existing_file.h5')


def test_get_hwp_frequency() -> None:
    obs = SOTODLibObservation.from_file(FILE)
    freq = obs.get_hwp_frequency()
    assert_allclose(float(freq), 2.0, rtol=1e-5)
