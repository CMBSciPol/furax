import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, IdentityOperator, ReshapeOperator
from furax.tree import as_structure
from tests.helpers import arange


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize(
    'shape, x, expected_y',
    [
        ((2, 12), arange(1, 2, 3, 4), arange(2, 12)),
        (
            (2, 12),
            {'a': arange(1, 2, 3, 4), 'b': arange(3, 2, 4)},
            {'a': arange(2, 12), 'b': arange(2, 12)},
        ),
        ((1, -1, 4), arange(1, 6, 2, 4), arange(1, 12, 4)),
        (
            (1, -1, 4),
            {'a': arange(1, 6, 2, 4), 'b': arange(3, 2, 4)},
            {'a': arange(1, 12, 4), 'b': arange(1, 6, 4)},
        ),
        ((-1,), arange(1, 6, 2, 4), arange(48)),
        (
            (-1,),
            {'a': arange(1, 6, 2, 4), 'b': arange(3, 2, 4)},
            {'a': arange(48), 'b': arange(24)},
        ),
    ],
)
def test_reshape(
    shape: tuple[int, ...], x: PyTree[jax.Array], expected_y: PyTree[jax.Array], do_jit: bool
) -> None:
    op = ReshapeOperator(shape, in_structure=as_structure(x))
    if do_jit:
        jitted_op_t = jax.jit(lambda x: op.T.mv(x))
        jitted_op = jax.jit(lambda x: op.mv(x))
        y = jitted_op(x)
        y_t = jitted_op_t(x)
    else:
        y = op.mv(x)
        y_t = op.T.mv(x)
    assert tree_equal(y, expected_y)
    assert tree_equal(y_t, x)


@pytest.mark.parametrize(
    'shape, in_structure',
    [
        ((-1,), jax.ShapeDtypeStruct((2,), jnp.float32)),
        ((2, 3), jax.ShapeDtypeStruct((2, 3), jnp.float32)),
        (
            (2, -1),
            {
                'a': jax.ShapeDtypeStruct((2, 3), jnp.float32),
                'b': jax.ShapeDtypeStruct((2, 5), jnp.float32),
            },
        ),
    ],
)
def test_reduction(shape: tuple[int, ...], in_structure: PyTree[jax.ShapeDtypeStruct]) -> None:
    op = ReshapeOperator(shape, in_structure=in_structure)
    assert isinstance(op.reduce(), IdentityOperator)


@pytest.mark.parametrize('shape', [(), (7,)])
def test_invalid_shape1(shape: tuple[int, ...]) -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='invalid new shape'):
        _ = ReshapeOperator(shape, in_structure=structure)


@pytest.mark.parametrize('shape', [(-2,), (1, 2, -3)])
def test_invalid_shape2(shape: tuple[int, ...]) -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='reshape new sizes should be all positive'):
        _ = ReshapeOperator(shape, in_structure=structure)


def test_invalid_shape3() -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='can only specify one unknown dimension'):
        _ = ReshapeOperator((-1, 2, -1), in_structure=structure)


@pytest.mark.parametrize('shape', [(7, -1, 1), (2, 4, -1)])
def test_invalid_shape4(shape: tuple[int, ...]) -> None:
    structure = jax.ShapeDtypeStruct((1, 2, 3), jnp.float32)
    with pytest.raises(ValueError, match='cannot reshape array of shape'):
        _ = ReshapeOperator(shape, in_structure=structure)


@pytest.mark.parametrize('shape', [(3, 8), (-1,), (2, -1, 4)])
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
def test_as_matrix(shape: tuple[int, ...], in_structure: PyTree[jax.ShapeDtypeStruct]) -> None:
    op = ReshapeOperator(shape, in_structure=in_structure)
    matrix = op.as_matrix()
    expected_matrix = AbstractLinearOperator.as_matrix(op)
    assert_array_equal(matrix, expected_matrix)

    transposed_operator_matrix = op.T.as_matrix()
    expected_transposed_matrix = AbstractLinearOperator.as_matrix(op.T)
    assert_array_equal(transposed_operator_matrix, expected_transposed_matrix)

    assert_array_equal(transposed_operator_matrix.T, matrix)


def test_reshape_rules1() -> None:
    op = ReshapeOperator((-1,), in_structure=jax.ShapeDtypeStruct((10, 20), jnp.float64))
    op_reduced = (op.T @ op).reduce()
    assert isinstance(op_reduced, IdentityOperator)
    assert op_reduced.in_structure() == op.in_structure()


def test_reshape_rules2() -> None:
    op = ReshapeOperator((-1,), in_structure=jax.ShapeDtypeStruct((10, 20), jnp.float64))
    op_reduced = (op @ op.T).reduce()
    assert isinstance(op_reduced, IdentityOperator)
    assert op_reduced.in_structure() == op.out_structure()
