from itertools import chain, combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax.landscapes import StokesPyTree, ValidStokesType


@pytest.mark.parametrize(
    'any_stokes',
    [''.join(_) for _ in chain.from_iterable(combinations('IQUV', n) for n in range(1, 5))],
)
def test_class_for(any_stokes: str) -> None:
    if any_stokes not in ('I', 'QU', 'IQU', 'IQUV'):
        with pytest.raises(ValueError, match='Invalid Stokes parameters'):
            _ = StokesPyTree.class_for(any_stokes)
    else:
        cls = StokesPyTree.class_for(any_stokes)
        assert cls.stokes == any_stokes


def test_from_stokes_args(stokes: ValidStokesType) -> None:
    arrays = [jnp.ones(1) for _ in stokes]
    pytree = StokesPyTree.from_stokes(*arrays)
    assert type(pytree) is StokesPyTree.class_for(stokes)


@pytest.mark.parametrize(
    'any_stokes',
    [''.join(_) for _ in chain.from_iterable(combinations('IQUV', n) for n in range(1, 5))],
)
def test_from_stokes_kwargs(any_stokes: str) -> None:
    kwargs = {stoke: jnp.ones(1) for stoke in any_stokes}
    if any_stokes not in ('I', 'QU', 'IQU', 'IQUV'):
        with pytest.raises(TypeError, match=f"Invalid Stokes vectors: '{any_stokes}'"):
            _ = StokesPyTree.from_stokes(**kwargs)
    else:
        pytree = StokesPyTree.from_stokes(**kwargs)
        assert type(pytree) is StokesPyTree.class_for(any_stokes)


def test_from_iquv(stokes: ValidStokesType) -> None:
    arrays = {stoke: jnp.array(istoke) for istoke, stoke in enumerate('IQUV', 1)}
    cls = StokesPyTree.class_for(stokes)
    pytree = cls.from_iquv(*arrays.values())
    assert type(pytree) is cls
    for stoke in stokes:
        assert getattr(pytree, stoke.lower()) == arrays[stoke]


def test_ravel(stokes: ValidStokesType) -> None:
    shape = (4, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = StokesPyTree.from_stokes(**arrays)
    raveled_pytree = pytree.ravel()
    for stoke in stokes:
        assert getattr(raveled_pytree, stoke.lower()).shape == (8,)


def test_reshape(stokes: ValidStokesType) -> None:
    shape = (4, 2)
    new_shape = (2, 2, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = StokesPyTree.from_stokes(**arrays)
    raveled_pytree = pytree.reshape(new_shape)
    for stoke in stokes:
        assert getattr(raveled_pytree, stoke.lower()).shape == new_shape


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
def test_zeros(stokes: ValidStokesType, shape: tuple[int, ...], dtype, factory, value) -> None:
    cls = StokesPyTree.class_for(stokes)
    pytree = factory(cls, shape, dtype)
    for stoke in stokes:
        array = getattr(pytree, stoke.lower())
        assert array.shape == shape
        assert array.dtype == dtype
        assert_array_equal(array, value)


@pytest.mark.parametrize('shape', [(10,), (2, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_structure(stokes: ValidStokesType, shape: tuple[int, ...], dtype) -> None:
    array = jnp.zeros(shape, dtype)
    pytree = StokesPyTree.from_stokes(*[array for _ in stokes])
    leaf_structure = jax.ShapeDtypeStruct(shape, dtype)
    expected_pytree_structure = StokesPyTree.from_stokes(*[leaf_structure for _ in stokes])

    assert pytree.shape == shape
    assert pytree.dtype == dtype
    assert pytree.structure == expected_pytree_structure
    assert pytree.structure_for(shape, dtype) == expected_pytree_structure


@pytest.mark.parametrize('shape', [(10,), (2, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_structure_for(stokes: ValidStokesType, shape: tuple[int, ...], dtype) -> None:
    structure = StokesPyTree.class_for(stokes).structure_for(shape, dtype)
    array = jnp.zeros(shape, dtype)
    pytree = StokesPyTree.from_stokes(*[array for _ in stokes])
    expected_structure = pytree.structure

    assert structure == expected_structure


def test_matmul(stokes: ValidStokesType) -> None:
    cls = StokesPyTree.class_for(stokes)
    x = cls.ones((2, 3))
    y = cls.full((2, 3), 2)
    assert x @ y == len(cls.stokes) * 12
