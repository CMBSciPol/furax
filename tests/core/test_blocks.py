from collections.abc import Callable

import equinox
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

import furax as fx
from furax import (
    AbstractLinearOperator,
    DenseBlockDiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
)
from furax.core import AdditionOperator
from furax.core._blocks import (
    AbstractBlockOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
)


@pytest.fixture(scope='module')
def op_23() -> AbstractLinearOperator:
    return DenseBlockDiagonalOperator(
        jnp.arange(2 * 3).reshape(2, 3) + 1, in_structure=jax.ShapeDtypeStruct((3,), jnp.float32)
    )


@pytest.fixture(scope='module')
def op2_23() -> AbstractLinearOperator:
    return DenseBlockDiagonalOperator(
        jnp.arange(2 * 3).reshape(2, 3), in_structure=jax.ShapeDtypeStruct((3,), jnp.float32)
    )


@pytest.fixture(scope='module')
def op_33() -> AbstractLinearOperator:
    return DenseBlockDiagonalOperator(
        jnp.arange(3 * 3).reshape(3, 3) + 1, in_structure=jax.ShapeDtypeStruct((3,), jnp.float32)
    )


@pytest.fixture(scope='module')
def op_32() -> AbstractLinearOperator:
    return DenseBlockDiagonalOperator(
        jnp.arange(3 * 2).reshape(3, 2) + 1, in_structure=jax.ShapeDtypeStruct((2,), jnp.float32)
    )


@pytest.mark.parametrize('cls', [BlockRowOperator, BlockDiagonalOperator, BlockColumnOperator])
def test_operators(
    cls: AbstractBlockOperator,
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_23: AbstractLinearOperator,
    op2_23: AbstractLinearOperator,
) -> None:
    ops = [op_23, op2_23, op_23]
    op = cls(pytree_builder(*ops))
    assert op.block_leaves == ops


def test_block_row(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_32: AbstractLinearOperator,
    op_33: AbstractLinearOperator,
) -> None:
    op = BlockRowOperator(pytree_builder(op_32, op_33, op_32))
    expected_matrix = jnp.array(
        [
            [1, 2, 1, 2, 3, 1, 2],
            [3, 4, 4, 5, 6, 3, 4],
            [5, 6, 7, 8, 9, 5, 6],
        ],
        jnp.float32,
    )
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected_matrix)
    assert_array_equal(op.as_matrix(), expected_matrix)
    assert_array_equal(op.T.as_matrix().T, expected_matrix)


def test_block_diag(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_32: AbstractLinearOperator,
    op_23: AbstractLinearOperator,
) -> None:
    op = BlockDiagonalOperator(pytree_builder(op_32, op_23, op_32))
    expected_matrix = jnp.array(
        [
            [1, 2, 0, 0, 0, 0, 0],
            [3, 4, 0, 0, 0, 0, 0],
            [5, 6, 0, 0, 0, 0, 0],
            [0, 0, 1, 2, 3, 0, 0],
            [0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 1, 2],
            [0, 0, 0, 0, 0, 3, 4],
            [0, 0, 0, 0, 0, 5, 6],
        ],
        jnp.float32,
    )
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected_matrix)
    assert_array_equal(op.as_matrix(), expected_matrix)
    assert_array_equal(op.T.as_matrix().T, expected_matrix)


def test_block_column(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_23: AbstractLinearOperator,
    op_33: AbstractLinearOperator,
) -> None:
    op = BlockColumnOperator(pytree_builder(op_23, op_33, op_23))
    expected_matrix = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3],
            [4, 5, 6],
        ],
        jnp.float32,
    )
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected_matrix)
    assert_array_equal(op.as_matrix(), expected_matrix)
    assert_array_equal(op.T.as_matrix().T, expected_matrix)


def test_rule_block_row_block_diagonal(
    op_32: AbstractLinearOperator, op_33: AbstractLinearOperator, op_23: AbstractLinearOperator
) -> None:
    op = BlockRowOperator([op_32, op_33]) @ BlockDiagonalOperator([op_23, op_32])
    reduced_op = op.reduce()
    assert isinstance(reduced_op, BlockRowOperator)
    assert_array_equal(
        reduced_op.as_matrix(),
        jnp.hstack([op_32.as_matrix() @ op_23.as_matrix(), op_33.as_matrix() @ op_32.as_matrix()]),
    )


def test_rule_block_diagonal_block_diagonal(
    op_32: AbstractLinearOperator, op_33: AbstractLinearOperator, op_23: AbstractLinearOperator
) -> None:
    op = BlockDiagonalOperator([op_32, op_33]) @ BlockDiagonalOperator([op_23, op_32])
    reduced_op = op.reduce()
    assert isinstance(reduced_op, BlockDiagonalOperator)
    assert_array_equal(
        reduced_op.as_matrix(),
        jsl.block_diag(
            op_32.as_matrix() @ op_23.as_matrix(), op_33.as_matrix() @ op_32.as_matrix()
        ),
    )


def test_rule_block_diagonal_block_column(
    op_32: AbstractLinearOperator, op_33: AbstractLinearOperator, op_23: AbstractLinearOperator
) -> None:
    op = BlockDiagonalOperator([op_32, op_23]) @ BlockColumnOperator([op_23, op_33])
    reduced_op = op.reduce()
    assert isinstance(reduced_op, BlockColumnOperator)
    assert_array_equal(
        reduced_op.as_matrix(),
        jnp.vstack([op_32.as_matrix() @ op_23.as_matrix(), op_23.as_matrix() @ op_33.as_matrix()]),
    )


def test_rule_block_row_block_column(
    op_32: AbstractLinearOperator, op_33: AbstractLinearOperator, op_23: AbstractLinearOperator
) -> None:
    op = BlockRowOperator([op_32, op_33]) @ BlockColumnOperator([op_23, op_33])
    reduced_op = op.reduce()
    assert isinstance(reduced_op, AdditionOperator)
    assert_array_equal(
        reduced_op.as_matrix(),
        op_32.as_matrix() @ op_23.as_matrix() + op_33.as_matrix() @ op_33.as_matrix(),
    )


def test_jit_block_row(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_32: AbstractLinearOperator,
    op_33: AbstractLinearOperator,
) -> None:
    op = BlockRowOperator(pytree_builder(op_32, op_33))
    x = pytree_builder(jnp.array([1, 2]), jnp.array([3, 4, 5]))
    expected_y = op(x)
    jit_op = jax.jit(lambda x: BlockRowOperator.mv(op, x))
    assert equinox.tree_equal(jit_op(x), expected_y)


def test_jit_block_diagonal(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_32: AbstractLinearOperator,
    op_23: AbstractLinearOperator,
) -> None:
    op = BlockDiagonalOperator(pytree_builder(op_32, op_23))
    x = pytree_builder(jnp.array([1, 2]), jnp.array([3, 4, 5]))
    expected_y = op(x)
    jit_op = jax.jit(lambda x: BlockDiagonalOperator.mv(op, x))
    assert equinox.tree_equal(jit_op(x), expected_y)


def test_jit_block_column(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
    op_23: AbstractLinearOperator,
    op_33: AbstractLinearOperator,
) -> None:
    op = BlockColumnOperator(pytree_builder(op_33, op_23))
    x = pytree_builder(jnp.array([1, 2, 3]), jnp.array([3, 4, 5]))
    expected_y = op(x)
    jit_op = jax.jit(lambda x: BlockColumnOperator.mv(op, x))
    assert equinox.tree_equal(jit_op(x), expected_y)


def test_block_row_nested() -> None:
    structure = {
        'x': jax.ShapeDtypeStruct((2,), jnp.float16),
        'y': jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    op = BlockRowOperator({'a': IdentityOperator(structure), 'b': IdentityOperator(structure)})
    x = {'a': fx.tree.ones_like(structure), 'b': fx.tree.full_like(structure, 2)}
    y = op(x)
    expected_y = fx.tree.full_like(structure, 3)
    assert equinox.tree_equal(y, expected_y)


def test_block_diagonal_nested() -> None:
    structure = {
        'x': jax.ShapeDtypeStruct((2,), jnp.float16),
        'y': jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    op = BlockDiagonalOperator(
        {'a': IdentityOperator(structure), 'b': HomothetyOperator(2, structure)}
    )
    x = {'a': fx.tree.ones_like(structure), 'b': fx.tree.full_like(structure, 2)}
    y = op(x)
    expected_y = {'a': fx.tree.ones_like(structure), 'b': fx.tree.full_like(structure, 4)}
    assert equinox.tree_equal(y, expected_y)


def test_block_column_nested() -> None:
    structure = {
        'x': jax.ShapeDtypeStruct((2,), jnp.float16),
        'y': jax.ShapeDtypeStruct((3,), jnp.float32),
    }
    op = BlockColumnOperator(
        {'a': IdentityOperator(structure), 'b': HomothetyOperator(2, structure)}
    )
    x = fx.tree.ones_like(structure)
    y = op(x)
    expected_y = {'a': x, 'b': fx.tree.full_like(structure, 2)}
    assert equinox.tree_equal(y, expected_y)


def test_block_row_single_leaf() -> None:
    id = IdentityOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
    op = BlockRowOperator([id])
    x = jnp.array([1.0, 2.0], jnp.float32)
    y = op([x])
    assert isinstance(y, jax.Array)
    assert_array_equal(y, x)


def test_block_diagonal_single_leaf() -> None:
    id = IdentityOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
    op = BlockDiagonalOperator([id])
    x = jnp.array([1.0, 2.0], jnp.float32)
    y = op([x])
    assert isinstance(y, list)
    assert len(y) == 1
    assert isinstance(y[0], jax.Array)
    assert_array_equal(y[0], x)


def test_block_column_single_leaf() -> None:
    id = IdentityOperator(jax.ShapeDtypeStruct((2,), jnp.float32))
    op = BlockColumnOperator([id])
    x = jnp.array([1.0, 2.0], jnp.float32)
    y = op(x)
    assert isinstance(y, list)
    assert len(y) == 1
    assert isinstance(y[0], jax.Array)
    assert_array_equal(y[0], x)


def test_reduce_block_diagonal() -> None:
    op = BlockDiagonalOperator(
        {
            'a': IdentityOperator(jax.ShapeDtypeStruct((3,), dtype=jnp.float64)),
            'b': IdentityOperator(jax.ShapeDtypeStruct((), jnp.float32)),
        }
    )
    reduced_op = op.reduce()
    assert isinstance(reduced_op, IdentityOperator)
    assert reduced_op.in_structure() == op.in_structure()
