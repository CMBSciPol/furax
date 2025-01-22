import operator
from collections.abc import Callable
from itertools import chain, combinations
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import tree_equal
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_array_equal

from furax.obs.stokes import Stokes, ValidStokesType


@pytest.mark.parametrize(
    'any_stokes',
    [''.join(_) for _ in chain.from_iterable(combinations('IQUV', n) for n in range(1, 5))],
)
def test_class_for(any_stokes: str) -> None:
    if any_stokes not in ('I', 'QU', 'IQU', 'IQUV'):
        with pytest.raises(ValueError, match='Invalid Stokes parameters'):
            _ = Stokes.class_for(any_stokes)
    else:
        cls = Stokes.class_for(any_stokes)
        assert cls.stokes == any_stokes


def test_from_stokes_args(stokes: ValidStokesType) -> None:
    arrays = [jnp.ones(1) for _ in stokes]
    pytree = Stokes.from_stokes(*arrays)
    assert type(pytree) is Stokes.class_for(stokes)


@pytest.mark.parametrize(
    'any_stokes',
    [''.join(_) for _ in chain.from_iterable(combinations('IQUV', n) for n in range(1, 5))],
)
def test_from_stokes_kwargs(any_stokes: str) -> None:
    kwargs = {stoke: jnp.ones(1) for stoke in any_stokes}
    if any_stokes not in ('I', 'QU', 'IQU', 'IQUV'):
        with pytest.raises(TypeError, match=f"Invalid Stokes vectors: '{any_stokes}'"):
            _ = Stokes.from_stokes(**kwargs)
    else:
        pytree = Stokes.from_stokes(**kwargs)
        assert type(pytree) is Stokes.class_for(any_stokes)


def test_from_iquv(stokes: ValidStokesType) -> None:
    arrays = {stoke: jnp.array(istoke) for istoke, stoke in enumerate('IQUV', 1)}
    cls = Stokes.class_for(stokes)
    pytree = cls.from_iquv(*arrays.values())
    assert type(pytree) is cls
    for stoke in stokes:
        assert getattr(pytree, stoke.lower()) == arrays[stoke]


def test_ravel(stokes: ValidStokesType) -> None:
    shape = (4, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = Stokes.from_stokes(**arrays)
    raveled_pytree = pytree.ravel()
    for stoke in stokes:
        assert getattr(raveled_pytree, stoke.lower()).shape == (8,)


def test_reshape(stokes: ValidStokesType) -> None:
    shape = (4, 2)
    new_shape = (2, 2, 2)
    arrays = {k: jnp.ones(shape) for k in stokes}
    pytree = Stokes.from_stokes(**arrays)
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
    cls = Stokes.class_for(stokes)
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
    pytree = Stokes.from_stokes(*[array for _ in stokes])
    leaf_structure = jax.ShapeDtypeStruct(shape, dtype)
    expected_pytree_structure = Stokes.from_stokes(*[leaf_structure for _ in stokes])

    assert pytree.shape == shape
    assert pytree.dtype == dtype
    assert pytree.structure == expected_pytree_structure
    assert pytree.structure_for(shape, dtype) == expected_pytree_structure


@pytest.mark.parametrize('shape', [(10,), (2, 10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_structure_for(stokes: ValidStokesType, shape: tuple[int, ...], dtype) -> None:
    structure = Stokes.class_for(stokes).structure_for(shape, dtype)
    array = jnp.zeros(shape, dtype)
    pytree = Stokes.from_stokes(*[array for _ in stokes])
    expected_structure = pytree.structure

    assert structure == expected_structure


def test_matmul(stokes: ValidStokesType) -> None:
    cls = Stokes.class_for(stokes)
    x = cls.ones((2, 3))
    y = cls.full((2, 3), 2)
    assert x @ y == len(cls.stokes) * 12


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize(
    'operation, reverse, value, expected_value',
    [
        (operator.add, False, 2, 5.0),
        (operator.sub, False, 2, 1.0),
        (operator.mul, False, 2, 6.0),
        (operator.truediv, False, 2, 1.5),
        (operator.pow, False, 2, 9.0),
        (operator.add, True, 2, 5.0),
        (operator.sub, True, 2, -1.0),
        (operator.mul, True, 2, 6.0),
        (operator.truediv, True, 2, 2 / 3),
        (operator.pow, True, 2, 8.0),
        (operator.add, False, 2.0, 5.0),
        (operator.sub, False, 2.0, 1.0),
        (operator.mul, False, 2.0, 6.0),
        (operator.truediv, False, 2.0, 1.5),
        (operator.pow, False, 2.0, 9.0),
        (operator.add, True, 2.0, 5.0),
        (operator.sub, True, 2.0, -1.0),
        (operator.mul, True, 2.0, 6.0),
        (operator.truediv, True, 2.0, 2 / 3),
        (operator.pow, True, 2.0, 8.0),
        (operator.add, False, jnp.array(2), 5.0),
        (operator.sub, False, jnp.array(2), 1.0),
        (operator.mul, False, jnp.array(2), 6.0),
        (operator.truediv, False, jnp.array(2), 1.5),
        (operator.pow, False, jnp.array(2), 9.0),
        (operator.add, True, jnp.array(2), 5.0),
        (operator.sub, True, jnp.array(2), -1.0),
        (operator.mul, True, jnp.array(2), 6.0),
        (operator.truediv, True, jnp.array(2), 2 / 3),
        (operator.pow, True, jnp.array(2), 8.0),
        (operator.add, False, jnp.array([2.0, 1]), jnp.array([5.0, 4])),
        (operator.sub, False, jnp.array([2.0, 1]), jnp.array([1.0, 2])),
        (operator.mul, False, jnp.array([2.0, 1]), jnp.array([6.0, 3])),
        (operator.truediv, False, jnp.array([2.0, 1]), jnp.array([1.5, 3])),
        (operator.pow, False, jnp.array([2.0, 1]), jnp.array([9.0, 3])),
        (operator.add, True, jnp.array([2.0, 1]), jnp.array([5.0, 4])),
        (operator.sub, True, jnp.array([2.0, 1]), jnp.array([-1.0, -2])),
        (operator.mul, True, jnp.array([2.0, 1]), jnp.array([6.0, 3])),
        (operator.truediv, True, jnp.array([2.0, 1]), jnp.array([2 / 3, 1 / 3])),
        (operator.pow, True, jnp.array([2.0, 1]), jnp.array([8.0, 1])),
    ],
)
def test_operation_scalar_or_array(
    stokes: ValidStokesType,
    operation: Callable[[Any, Any], Any],
    reverse: bool,
    value: float | Float[Array, ''],
    expected_value: float,
    do_jit: bool,
) -> None:
    shape = 3, 2
    x = Stokes.class_for(stokes).full(shape, 3.0)

    def func(x):
        if reverse:
            return operation(value, x)
        else:
            return operation(x, value)

    if do_jit:
        func = jax.jit(func)
    actual_y = func(x)

    expected_leaf = jnp.broadcast_to(expected_value, shape)
    expected_y = Stokes.class_for(stokes)(*len(stokes) * (expected_leaf,))
    assert tree_equal(actual_y, expected_y, rtol=1e-15)


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize(
    'operation, expected_value',
    [
        (operator.add, 5.0),
        (operator.sub, 1.0),
        (operator.mul, 6.0),
        (operator.truediv, 1.5),
        (operator.pow, 9.0),
    ],
)
def test_operation_pytree(
    stokes: ValidStokesType,
    operation: Callable[[Any, Any], Any],
    expected_value: float,
    do_jit: bool,
) -> None:
    cls = Stokes.class_for(stokes)
    a = cls.full((3, 2), 3)
    b = cls.full((3, 2), 2)
    expected_result = cls.full((3, 2), expected_value)
    if do_jit:
        operation = jax.jit(operation)
    result = operation(a, b)
    assert tree_equal(result, expected_result)


@pytest.mark.parametrize(
    'operation',
    [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.pow,
    ],
)
def test_operation_incompatible_pytree(operation: Callable[[Any, Any], Any]) -> None:
    a = Stokes.class_for('I').ones(())
    b = Stokes.class_for('IQU').ones(())
    with pytest.raises(TypeError, match='unsupported operand type'):
        operation(a, b)
