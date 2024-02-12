import pytest
from jax import Array
from numpy.testing import assert_array_equal

from astrosim.landscapes import HealpixLandscape


@pytest.mark.parametrize('stokes', ['I', 'IQU'])
def test_healpix_landscape(stokes) -> None:
    nside = 64
    npixel = 12 * nside**2
    landscape = HealpixLandscape(nside, stokes)
    sky = landscape.ones()
    assert sky.nside == nside
    assert sky.npixel == npixel
    for stoke in stokes:
        leaf = getattr(sky, stoke)
        assert isinstance(leaf, Array)
        assert leaf.size == npixel
        assert_array_equal(leaf, 1.0)
