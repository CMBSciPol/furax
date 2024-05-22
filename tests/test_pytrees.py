import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from astrosim.landscapes import stokes_pytree_cls


@pytest.mark.parametrize('shape', [(10,), (2, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize(
    'factory, value',
    [
        (lambda c, s, d: c.zeros(s, d), 0),
        (lambda c, s, d: c.ones(s, d), 1),
        (lambda c, s, d: c.full(s, 2, d), 2),
    ],
)
def test_zeros(stokes, shape, dtype, factory, value) -> None:
    cls = stokes_pytree_cls(stokes)
    pytree = factory(cls, shape, dtype)
    for stoke in stokes:
        array = getattr(pytree, stoke)
        assert array.shape == shape
        assert array.dtype == dtype
        assert_array_equal(array, value)


@pytest.mark.parametrize('shape', [(10,), (2, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_test_structure(stokes, shape, dtype) -> None:
    cls = stokes_pytree_cls(stokes)
    array = jnp.zeros(shape, dtype)
    pytree = cls(**{stoke: array for stoke in stokes})
    shape_pytree = jax.ShapeDtypeStruct(shape, dtype)
    expected_shape_pytree = cls(**{stoke: shape_pytree for stoke in stokes})

    assert pytree.shape == shape
    assert pytree.dtype == dtype
    assert pytree.shape_pytree(shape, dtype) == expected_shape_pytree
