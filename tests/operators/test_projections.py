from typing import get_args

import equinox
import jax.numpy as jnp
import pytest

from astrosim.landscapes import HealpixLandscape, ValidStokesType, stokes_pytree_cls
from astrosim.operators.projections import SamplingOperator


@pytest.mark.parametrize('stokes', get_args(ValidStokesType))
def test_direct(stokes) -> None:
    nside = 1
    landscape = HealpixLandscape(nside, stokes)
    cls = stokes_pytree_cls(stokes)
    x_as_dict = {
        stoke: jnp.arange(12, dtype=landscape.dtype) * (i + 1) for i, stoke in enumerate(stokes)
    }
    x = cls(**x_as_dict)
    indices = jnp.array([[2, 3, 2]])
    proj = SamplingOperator(landscape, indices)

    y = proj(x)

    expected_y = cls(**{stoke: x_as_dict[stoke][indices] for stoke in stokes})
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('stokes', get_args(ValidStokesType))
def test_transpose(stokes) -> None:
    nside = 1
    landscape = HealpixLandscape(nside, stokes)
    cls = stokes_pytree_cls(stokes)
    x_as_dict = {stoke: jnp.array([[1, 2, 3]]) * (i + 1) for i, stoke in enumerate(stokes)}
    x = cls(**x_as_dict)
    indices = jnp.array([[2, 3, 2]])
    proj = SamplingOperator(landscape, indices)

    y = proj.T(x)

    array = jnp.array([0.0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_y = cls(*[array * i for i in range(1, len(stokes) + 1)])
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)
