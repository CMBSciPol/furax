from pathlib import Path

import pytest

from furax.interfaces.sotodlib import SOTODLibObservation


def test_from_file() -> None:
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    file = folder / 'test_obs.h5'
    ndet = 2
    nsample = 1000

    obs = SOTODLibObservation.from_file(file)
    assert obs.n_detectors == ndet
    assert obs.n_samples == nsample


def test_from_file_missing() -> None:
    with pytest.raises(FileNotFoundError):
        SOTODLibObservation.from_file('non_existing_file.h5')
