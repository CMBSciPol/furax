import string
import sys
from collections.abc import Callable
from typing import Any

import equinox
from numpy.testing import assert_array_equal

if sys.version_info < (3, 12):
    from itertools import islice

    def batched(iterable, n):
        if n < 1:
            raise ValueError('n must be at least one')
        iterator = iter(iterable)
        while batch := tuple(islice(iterator, n)):
            yield batch

else:
    from itertools import batched

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import pytest
from jaxtyping import PyTree

import furax as fx
from furax._base.blocks import (
    AbstractBlockOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
)
from furax._base.core import (
    AbstractLinearOperator,
    AdditionOperator,
    HomothetyOperator,
    IdentityOperator,
)
from furax._base.dense import DenseBlockDiagonalOperator


def pytree_dict_builder(*args: Any) -> dict[str, Any]:
    return {k: v for k, v in zip(string.ascii_lowercase, args)}


def pytree_list_builder(*args: Any) -> list[Any]:
    return list(args)


def pytree_tuple_builder(*args: Any) -> tuple[Any, ...]:
    return args


def pytree_nested_builder(*args: Any) -> dict[str, tuple[Any]]:
    return {k: v for k, v in zip(string.ascii_lowercase, batched(args, 2))}


@pytest.fixture(
    params=[pytree_dict_builder, pytree_list_builder, pytree_tuple_builder, pytree_nested_builder]
)
def pytree_builder(
    request: pytest.FixtureRequest,
) -> Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]]:
    """Parametrized fixture for I, QU, IQU and IQUV."""
    return request.param


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
    assert op.operators == ops


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
