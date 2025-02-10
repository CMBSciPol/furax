import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from numpy.testing import assert_allclose, assert_array_equal

from furax import TreeOperator
from furax.obs.stokes import StokesIQU, StokesIQUV
from furax.tree import _tree_to_dense

u = jnp.arange(3, dtype=jnp.float32)
sdu = jax.ShapeDtypeStruct((3,), jnp.float32)
sd0 = jax.ShapeDtypeStruct((), jnp.float32)


@pytest.mark.parametrize(
    'tree, in_structure, x, expected_y',
    [
        (2, sdu, u, 2 * u),
        ({'a': {'x': 1, 'y': 2}}, {'x': sd0, 'y': sd0}, {'x': 4, 'y': 5}, {'a': 14}),
        ({'a': 2, 'b': 3}, sdu, u, {'a': 2 * u, 'b': 3 * u}),
        (
            StokesIQU(StokesIQU(1, 4, 2), StokesIQU(3, -2, 1), StokesIQU(0, 0, 1)),
            StokesIQU(sd0, sd0, sd0),
            StokesIQU(-1, 1, 2),
            StokesIQU(7, -3, 2),
        ),
    ],
)
def test_treemul(tree, in_structure, x, expected_y) -> None:
    op = TreeOperator(tree, in_structure=in_structure)
    actual_y = op(x)
    assert tree_equal(actual_y, expected_y)


@pytest.mark.parametrize(
    'tree, in_structure, expected_matrix',
    [
        (2, sd0, jnp.array([[2]])),
        ([[1, 2], [3, 4]], [sd0, sd0], jnp.array([[1, 2], [3, 4]])),
        ({'a': {'x': 1, 'y': 2}}, {'x': sd0, 'y': sd0}, jnp.array([[1, 2]])),
        ({'a': 2, 'b': 3}, sd0, jnp.array([[2], [3]])),
        (
            StokesIQU(StokesIQU(1, 4, 2), StokesIQU(3, -2, 1), StokesIQU(0, 0, 1)),
            StokesIQU(sd0, sd0, sd0),
            jnp.array([[1, 4, 2], [3, -2, 1], [0, 0, 1]]),
        ),
    ],
)
def test_transpose(tree, in_structure, expected_matrix) -> None:
    op = TreeOperator(tree, in_structure=in_structure)
    assert op.tree_shape == expected_matrix.shape
    actual_matrix = op.as_matrix()
    assert_array_equal(actual_matrix, expected_matrix)
    assert_array_equal(op.T.as_matrix(), expected_matrix.T)


@pytest.mark.parametrize(
    'tree1, in_structure1, tree2, in_structure2',
    [
        (2, sd0, 3, sd0),
        ([[1, 2], [3, 4]], [sd0, sd0], [[-1, 0], [2, 4]], [sd0, sd0]),
        (
            {'a': {'x': 1, 'y': 2}},
            {'x': sd0, 'y': sd0},
            {'x': [1, 2, 3], 'y': [-1, 0, 4]},
            [sd0, sd0, sd0],
        ),
        (
            StokesIQU(StokesIQU(1, 4, 2), StokesIQU(3, -2, 1), StokesIQU(4, 0, 1)),
            StokesIQU(sd0, sd0, sd0),
            StokesIQU(StokesIQU(3, 2, 1), StokesIQU(0, -2, 1), StokesIQU(1, -1, 1)),
            StokesIQU(sd0, sd0, sd0),
        ),
    ],
)
def test_multiplication(tree1, in_structure1, tree2, in_structure2) -> None:
    op1 = TreeOperator(tree1, in_structure=in_structure1)
    op2 = TreeOperator(tree2, in_structure=in_structure2)
    op = (op1 @ op2).reduce()
    assert isinstance(op, TreeOperator)
    assert_array_equal(op.as_matrix(), op1.as_matrix() @ op2.as_matrix())


def test_multiplication_by_transpose() -> None:
    op = TreeOperator(
        StokesIQUV(1, -1, 0, 0), in_structure=StokesIQUV.structure_for((), jnp.float32)
    )
    expected_matrix = jnp.array([[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert_array_equal((op.T @ op).as_matrix(), expected_matrix)


def test_jit1() -> None:
    sd = jax.ShapeDtypeStruct((100,), jnp.float32)

    def func(array, x):
        op = TreeOperator([1, array, 0], in_structure=[sd, sd, sd])
        return op(x)

    array_ = jnp.arange(100, dtype=jnp.int32)
    x_ = jnp.arange(50, 150, dtype=jnp.int32)

    jitted_func = jax.jit(func)
    assert tree_equal(func(array_, x_), jitted_func(array_, x_))


def test_jit2() -> None:
    sd = jax.ShapeDtypeStruct((100,), jnp.int32)

    def func(x):
        op = TreeOperator([1, array, 0], in_structure=[sd, sd, sd])
        return op(x)

    array = jnp.arange(100, dtype=jnp.int32)
    x_ = jnp.arange(50, 150, dtype=jnp.int32)

    jitted_func = jax.jit(func)
    assert tree_equal(func(x_), jitted_func(x_))


def test_inverse1() -> None:
    sd = jax.ShapeDtypeStruct((), jnp.float32)
    op = TreeOperator({'r1': [1.0, 2, 3], 'r2': [1, 0, -1]}, in_structure=[sd, sd, sd])
    matrix = jnp.array([[1, 2, 3], [1, 0, -1]])
    assert_allclose(op.I.as_matrix(), jnp.linalg.pinv(matrix))


def test_inverse2() -> None:
    sd = jax.ShapeDtypeStruct((), jnp.float32)
    op = TreeOperator(
        {'r1': [1.0, 2, 3], 'r2': [1, 0, jnp.array([1, -1])]}, in_structure=[sd, sd, sd]
    )
    matrix = jnp.array([[[1, 2, 3], [1, 0, 1]], [[1, 2, 3], [1, 0, -1]]])
    assert_allclose(
        _tree_to_dense(op.inner_treedef, op.outer_treedef, op.I), jnp.linalg.pinv(matrix)
    )
