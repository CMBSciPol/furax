import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, IdentityOperator, RavelOperator
from furax.tree import as_structure
from tests.helpers import arange


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize(
    'first_axis, last_axis, x, expected_y',
    [
        (0, 0, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (0, 1, arange(1, 2, 3, 4), arange(2, 3, 4)),
        (0, 2, arange(1, 2, 3, 4), arange(6, 4)),
        (0, 3, arange(1, 2, 3, 4), arange(24)),
        (1, 1, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (1, 2, arange(1, 2, 3, 4), arange(1, 6, 4)),
        (1, 3, arange(1, 2, 3, 4), arange(1, 24)),
        (2, 2, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (2, 3, arange(1, 2, 3, 4), arange(1, 2, 12)),
        (3, 3, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (0, -4, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (0, -3, arange(1, 2, 3, 4), arange(2, 3, 4)),
        (0, -2, arange(1, 2, 3, 4), arange(6, 4)),
        (0, -1, arange(1, 2, 3, 4), arange(24)),
        (1, -3, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (1, -2, arange(1, 2, 3, 4), arange(1, 6, 4)),
        (1, -1, arange(1, 2, 3, 4), arange(1, 24)),
        (2, -2, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (2, -1, arange(1, 2, 3, 4), arange(1, 2, 12)),
        (3, -1, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-4, 0, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-4, 1, arange(1, 2, 3, 4), arange(2, 3, 4)),
        (-4, 2, arange(1, 2, 3, 4), arange(6, 4)),
        (-4, 3, arange(1, 2, 3, 4), arange(24)),
        (-3, 1, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-3, 2, arange(1, 2, 3, 4), arange(1, 6, 4)),
        (-3, 3, arange(1, 2, 3, 4), arange(1, 24)),
        (-2, 2, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-2, 3, arange(1, 2, 3, 4), arange(1, 2, 12)),
        (-1, 3, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-4, -4, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-4, -3, arange(1, 2, 3, 4), arange(2, 3, 4)),
        (-4, -2, arange(1, 2, 3, 4), arange(6, 4)),
        (-4, -1, arange(1, 2, 3, 4), arange(24)),
        (-3, -3, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-3, -2, arange(1, 2, 3, 4), arange(1, 6, 4)),
        (-3, -1, arange(1, 2, 3, 4), arange(1, 24)),
        (-2, -2, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (-2, -1, arange(1, 2, 3, 4), arange(1, 2, 12)),
        (-1, -1, arange(1, 2, 3, 4), arange(1, 2, 3, 4)),
        (0, 1, {'x': arange(1, 2), 'y': arange(3, 4)}, {'x': arange(2), 'y': arange(12)}),
    ],
)
def test_ravel(
    first_axis: int,
    last_axis: int,
    x: PyTree[jax.Array],
    expected_y: PyTree[jax.Array],
    do_jit: bool,
) -> None:
    op = RavelOperator(first_axis, last_axis, in_structure=as_structure(x))
    if do_jit:
        jitted_op_t = jax.jit(lambda x: op.T.mv(x))
        jitted_op = jax.jit(lambda x: op.mv(x))
        y = jitted_op(x)
        x_reconstructed = jitted_op_t(y)
    else:
        y = op.mv(x)
        x_reconstructed = op.T.mv(y)
    assert tree_equal(y, expected_y)
    assert tree_equal(x_reconstructed, x)


@pytest.mark.parametrize('first_axis, last_axis', [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
@pytest.mark.parametrize(
    'in_structure',
    [
        jax.ShapeDtypeStruct((2, 3, 4), jnp.float32),
        {
            'a': jax.ShapeDtypeStruct((2, 3, 4), jnp.float32),
            'b': jax.ShapeDtypeStruct((4, 3, 2), jnp.float32),
        },
    ],
)
def test_as_matrix(
    first_axis: int, last_axis: int, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> None:
    op = RavelOperator(first_axis, last_axis, in_structure=in_structure)
    matrix = op.as_matrix()
    expected_matrix = AbstractLinearOperator.as_matrix(op)
    assert_array_equal(matrix, expected_matrix)

    transposed_operator_matrix = op.T.as_matrix()
    expected_transposed_matrix = AbstractLinearOperator.as_matrix(op.T)
    assert_array_equal(transposed_operator_matrix, expected_transposed_matrix)

    assert_array_equal(transposed_operator_matrix.T, matrix)


@pytest.mark.parametrize(
    'first_axis, last_axis, in_structure',
    [
        (0, -1, jax.ShapeDtypeStruct((2,), jnp.float32)),
        (
            1,
            1,
            {
                'a': jax.ShapeDtypeStruct((2, 3, 4), jnp.float32),
                'b': jax.ShapeDtypeStruct((4, 3, 2), jnp.float32),
            },
        ),
    ],
)
def test_reduction(
    first_axis: int, last_axis: int, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> None:
    op = RavelOperator(first_axis, last_axis, in_structure=in_structure)
    assert isinstance(op.reduce(), IdentityOperator)


@pytest.mark.parametrize('first_axis, last_axis', [(1, 0), (-1, -2)])
def test_invalid_axes1(first_axis: int, last_axis: int) -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='the first axis'):
        _ = RavelOperator(first_axis, last_axis, in_structure=structure)


@pytest.mark.parametrize('first_axis, last_axis', [(1, -3), (-1, 1)])
def test_invalid_axes2(first_axis: int, last_axis: int) -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='there are no dimensions'):
        _ = RavelOperator(first_axis, last_axis, in_structure=structure)


def test_ravel_rules1() -> None:
    op = RavelOperator(in_structure=jax.ShapeDtypeStruct((10, 20), jnp.float64))
    op_reduced = (op.T @ op).reduce()
    assert isinstance(op_reduced, IdentityOperator)
    assert op_reduced.in_structure() == op.in_structure()


def test_ravel_rules2() -> None:
    op = RavelOperator(in_structure=jax.ShapeDtypeStruct((10, 20), jnp.float64))
    op_reduced = (op @ op.T).reduce()
    assert isinstance(op_reduced, IdentityOperator)
    assert op_reduced.in_structure() == op.out_structure()
