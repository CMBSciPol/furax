from pathlib import Path

import pytest

from furax.interfaces.litebird_sim import LBSObservation


def test_from_file() -> None:
    folder = Path(__file__).parents[2] / 'data/litebird_sim'
    file = folder / 'test_obs.h5'
    ndet = 2
    nsample = 7200

    obs = LBSObservation.from_file(file)
    assert obs.n_detectors == ndet
    assert obs.n_samples == nsample


def test_from_file_missing() -> None:
    with pytest.raises(FileNotFoundError):
        LBSObservation.from_file('non_existing_file.h5')
