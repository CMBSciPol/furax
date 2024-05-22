from itertools import chain, combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from astrosim.landscapes import StokesPyTree, stokes_pytree_cls


@pytest.mark.parametrize(
    'any_stokes',
    [''.join(_) for _ in chain.from_iterable(combinations('IQUV', n) for n in range(1, 5))],
)
def test_from_stokes(any_stokes: str) -> None:
    arrays = {stoke: jnp.ones(1) for stoke in any_stokes}
    if any_stokes not in ('I', 'QU', 'IQU', 'IQUV'):
        with pytest.raises(TypeError, match='Invalid Stokes vectors'):
            _ = StokesPyTree.from_stokes(**arrays)
    else:
        pytree = StokesPyTree.from_stokes(**arrays)
        assert type(pytree) is stokes_pytree_cls(any_stokes)


def test_from_iquv(stokes) -> None:
    arrays = {stoke: jnp.array(istoke) for istoke, stoke in enumerate('IQUV', 1)}
    cls = stokes_pytree_cls(stokes)
    pytree = cls.from_iquv(*arrays.values())
    assert type(pytree) is cls
    for stoke in stokes:
        assert getattr(pytree, stoke) == arrays[stoke]


def test_ravel(stokes) -> None:
    shape = (4, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = StokesPyTree.from_stokes(**arrays)
    raveled_pytree = pytree.ravel()
    for stoke in stokes:
        assert getattr(raveled_pytree, stoke).shape == (8,)


def test_reshape(stokes) -> None:
    shape = (4, 2)
    new_shape = (2, 2, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = StokesPyTree.from_stokes(**arrays)
    raveled_pytree = pytree.reshape(new_shape)
    for stoke in stokes:
        assert getattr(raveled_pytree, stoke).shape == new_shape


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
