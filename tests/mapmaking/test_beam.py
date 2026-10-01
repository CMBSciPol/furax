import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax.mapmaking._beam import load_beam
from furax.mapmaking.config import PointingConfig
from furax.obs.stokes import Stokes

NODES = np.array([[1.0, 0.0, 0.0, 0.0], [np.cos(0.01), np.sin(0.01), 0.0, 0.0]])
WEIGHTS = np.array([0.7, 0.3])


def test_no_path_gives_no_beam() -> None:
    assert load_beam(PointingConfig()) is None


@pytest.mark.parametrize(
    'weights, stokes',
    [
        (WEIGHTS, None),
        (np.stack([WEIGHTS, WEIGHTS]), 'QU'),
        (np.stack([[0.5, 0.5], WEIGHTS, WEIGHTS]), 'IQU'),
    ],
    ids=['shared', 'QU-equal', 'IQU-equal-QU'],
)
def test_accepts_a_beam_the_mapmakers_can_model(tmp_path, weights, stokes) -> None:
    path = tmp_path / 'beam.npz'
    if stokes is None:
        np.savez(path, nodes=NODES, weights=weights)
    else:
        np.savez(path, nodes=NODES, weights=weights, stokes=stokes)

    beam = load_beam(PointingConfig(beam=str(path)))

    assert beam is not None
    assert_array_equal(beam.nodes.wxyz, NODES)
    if stokes is None:
        assert_array_equal(beam.weights, weights)
    else:
        assert isinstance(beam.weights, Stokes)
        assert beam.weights.stokes == stokes
        assert_array_equal(beam.weights.data, weights)


@pytest.mark.parametrize(
    'nodes, weights, stokes, match',
    [
        (np.stack([NODES] * 3), WEIGHTS, '', 'only nodes shared by every detector'),
        (NODES, np.stack([WEIGHTS, WEIGHTS[::-1]]), 'QU', 'differ between Q and U'),
        (NODES, np.stack([WEIGHTS, WEIGHTS, WEIGHTS[::-1]]), 'IQU', 'differ between Q and U'),
    ],
    ids=['per-detector-nodes', 'QU-different', 'IQU-different-QU'],
)
def test_rejects_a_beam_the_mapmakers_cannot_model(tmp_path, nodes, weights, stokes, match):
    path = tmp_path / 'beam.npz'
    np.savez(path, nodes=nodes, weights=weights, stokes=stokes)
    with pytest.raises(ValueError, match=match):
        load_beam(PointingConfig(beam=str(path)))
