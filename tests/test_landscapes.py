from typing import get_args

import pytest
from jax import Array
from numpy.testing import assert_array_equal

from astrosim.landscapes import HealpixLandscape, ValidStokesType, stokes_pytree_cls


@pytest.mark.parametrize('stokes', get_args(ValidStokesType))
def test_healpix_landscape(stokes) -> None:
    nside = 64
    npixel = 12 * nside**2

    landscape = HealpixLandscape(nside, stokes)

    sky = landscape.ones()
    assert isinstance(sky, stokes_pytree_cls(stokes))

    assert sky.shape == (npixel,)
    for stoke in stokes:
        leaf = getattr(sky, stoke)
        assert isinstance(leaf, Array)
        assert leaf.size == npixel
        assert_array_equal(leaf, 1.0)
