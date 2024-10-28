import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from furax.landscapes import HealpixLandscape, StokesLandscape, StokesPyTree, ValidStokesType
from furax.samplings import Sampling


def test_healpix_landscape(stokes: ValidStokesType) -> None:
    nside = 64
    npixel = 12 * nside**2

    landscape = HealpixLandscape(nside, stokes)

    sky = landscape.ones()
    assert isinstance(sky, StokesPyTree.class_for(stokes))
    assert sky.shape == (npixel,)
    for stoke in stokes:
        leaf = getattr(sky, stoke.lower())
        assert isinstance(leaf, Array)
        assert leaf.size == npixel
        assert_array_equal(leaf, 1.0)


@pytest.mark.parametrize(
    'pixel, expected_index',
    [
        ((-0.5 - 1e-15, -0.5), -1),
        ((-0.5, -0.5 - 1e-15), -1),
        ((1.5 + 1e-15, -0.5), -1),
        ((1.5 - 1e-15, -0.5 - 1e-15), -1),
        ((-0.5 - 1e-15, 4.5 - 1e-15), -1),
        ((-0.5, 4.5 + 1e-15), -1),
        ((1.5 + 1e-15, 4.5 - 1e-15), -1),
        ((1.5 - 1e-15, 4.5 + 1e-15), -1),
        ((-0.5, -0.5), 0),
        ((-0.5, 0.5 - 1e-15), 0),
        ((0.5, -0.5), 0),
        ((0.5, 0.5 - 1e-15), 0),
        ((0, 0), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((1, 1), 3),
        ((0, 4), 8),
        ((1, 4), 9),
    ],
)
def test_pixel2index(pixel: tuple[float, float], expected_index: int) -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    landscape = CARStokesLandscape((5, 2), 'I')
    actual_index = landscape.pixel2index(*pixel)
    assert_array_equal(actual_index, expected_index)


def test_get_coverage() -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    samplings = Sampling(
        jnp.array([0.0, 1, 0, 1, 1, 1, 0]), jnp.array([0.0, 0, 0, 3, 0, 1, 0]), jnp.array(0.0)
    )
    landscape = CARStokesLandscape((5, 2), 'I')
    coverage = landscape.get_coverage(samplings)
    assert_array_equal(coverage, [[3, 2], [0, 1], [0, 0], [0, 1], [0, 0]])
