import equinox
import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from furax import MoveAxisOperator
from furax._base.core import IdentityOperator


@pytest.mark.parametrize(
    'source, destination, expected_shape',
    [
        (1, 2, (1, 3, 2, 4)),
        ((1,), (2,), (1, 3, 2, 4)),
        ((1, 0), (2, 3), (3, 4, 2, 1)),
    ],
)
def test_move_axis(
    source: tuple[int, ...], destination: tuple[int, ...], expected_shape: tuple[int, ...]
) -> None:
    in_structure = jax.ShapeDtypeStruct((1, 2, 3, 4), jnp.float64)
    op = MoveAxisOperator(source, destination, in_structure=in_structure)
    assert op.out_structure().shape == expected_shape
    x = jnp.ones(in_structure.shape, in_structure.dtype)
    y = op(x)
    assert y.dtype == in_structure.dtype
    assert y.shape == expected_shape


def test_move_axis_pytree() -> None:
    source = 0
    destination = 1
    in_structure = {'x': jax.ShapeDtypeStruct((1, 2, 3, 4), jnp.float64)}
    op = MoveAxisOperator(source, destination, in_structure=in_structure)
    assert op.out_structure() == {'x': jax.ShapeDtypeStruct((2, 1, 3, 4), jnp.float64)}
    x = {'x': jnp.ones((1, 2, 3, 4), jnp.float64)}
    assert equinox.tree_equal(op(x), {'x': jnp.ones((2, 1, 3, 4), jnp.float64)})


@pytest.mark.parametrize(
    'source, destination, expected_shape',
    [
        ((1,), (2,), (1, 3, 2, 4)),
        ((1, 0), (2, 3), (3, 4, 2, 1)),
    ],
)
def test_move_axis_transpose(
    source: tuple[int, ...], destination: tuple[int, ...], expected_shape: tuple[int, ...]
) -> None:
    in_structure = jax.ShapeDtypeStruct((1, 2, 3, 4), jnp.float64)
    op = MoveAxisOperator(source, destination, in_structure=in_structure)
    assert_array_equal(op.T.as_matrix().T, op.as_matrix())


def test_move_axis_inverse() -> None:
    in_structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float64)
    op = MoveAxisOperator(0, 1, in_structure=in_structure)
    assert op.I == op.T


def test_move_axis_orthogonal() -> None:
    in_structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float64)
    op = MoveAxisOperator(0, 1, in_structure=in_structure)
    reduced_op = (op.T @ op).reduce()
    assert isinstance(reduced_op, IdentityOperator)
    assert reduced_op.in_structure() == op.in_structure()

    reduced_op = (op @ op.T).reduce()
    assert isinstance(reduced_op, IdentityOperator)
    assert reduced_op.in_structure() == op.out_structure()
