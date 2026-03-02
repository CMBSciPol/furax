from collections.abc import Callable

import equinox
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, HomothetyOperator, IdentityOperator, square
from furax.core import (
    AdditionOperator,
    CompositionOperator,
    TransposeOperator,
)


class Op(AbstractLinearOperator):
    value: float

    def mv(self, x):
        return jnp.array([[0, self.value], [1, 0]]) @ x


_op_structure = jax.ShapeDtypeStruct((2,), jnp.float32)


def test_add(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1, in_structure=_op_structure), Op(2, in_structure=_op_structure)
    extra_operand = Op(3, in_structure=_op_structure)
    operands = pytree_builder(*operand_leaves)
    op = AdditionOperator(operands)
    added_op = op + extra_operand
    assert isinstance(added_op, AdditionOperator)
    assert isinstance(added_op.operands, list)
    assert len(added_op.operand_leaves) == 3
    expected_matrix = op.as_matrix() + extra_operand.as_matrix()
    assert_array_equal(AbstractLinearOperator.as_matrix(added_op), expected_matrix)
    assert_array_equal(added_op.as_matrix(), expected_matrix)


def test_radd(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1, in_structure=_op_structure), Op(2, in_structure=_op_structure)
    extra_operand = Op(3, in_structure=_op_structure)
    operands = pytree_builder(*operand_leaves)
    op = AdditionOperator(operands)
    added_op = extra_operand + op
    assert isinstance(added_op, AdditionOperator)
    assert isinstance(added_op.operands, list)
    assert len(added_op.operand_leaves) == 3
    expected_matrix = op.as_matrix() + extra_operand.as_matrix()
    assert_array_equal(AbstractLinearOperator.as_matrix(added_op), expected_matrix)
    assert_array_equal(added_op.as_matrix(), expected_matrix)


def test_sub(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1, in_structure=_op_structure), Op(2, in_structure=_op_structure)
    extra_operand = Op(3, in_structure=_op_structure)
    operands = pytree_builder(*operand_leaves)
    op = AdditionOperator(operands)
    subtracted_op = op - extra_operand
    assert isinstance(subtracted_op, AdditionOperator)
    assert isinstance(subtracted_op.operands, list)
    assert len(subtracted_op.operand_leaves) == 3
    expected_matrix = op.as_matrix() - extra_operand.as_matrix()
    assert_array_equal(AbstractLinearOperator.as_matrix(subtracted_op), expected_matrix)
    assert_array_equal(subtracted_op.as_matrix(), expected_matrix)


def test_transpose(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1, in_structure=_op_structure), Op(2, in_structure=_op_structure)
    operands = pytree_builder(*operand_leaves)
    op = AdditionOperator(operands)
    transposed_op = op.T
    assert isinstance(transposed_op, AdditionOperator)
    assert isinstance(transposed_op.operands, type(operands))
    assert jax.tree.all(
        transposed_op._tree_map(lambda leaf: isinstance(leaf, TransposeOperator)),
        is_leaf=lambda leaf: isinstance(leaf, AbstractLinearOperator),
    )
    assert jax.tree.all(
        transposed_op._tree_map(lambda leaf, orig: leaf.operator is orig, operands),
        is_leaf=lambda leaf: isinstance(leaf, AbstractLinearOperator),
    )
    expected_matrix = op.as_matrix().T
    assert_array_equal(AbstractLinearOperator.as_matrix(transposed_op), expected_matrix)
    assert_array_equal(transposed_op.as_matrix(), expected_matrix)


def test_neg(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1, in_structure=_op_structure), Op(2, in_structure=_op_structure)
    operands = pytree_builder(*operand_leaves)
    op = AdditionOperator(operands)
    neg_op = -op
    assert isinstance(neg_op, AdditionOperator)
    assert isinstance(neg_op.operands, type(operands))
    assert jax.tree.all(
        neg_op._tree_map(lambda leaf: isinstance(leaf, CompositionOperator)),
        is_leaf=lambda leaf: isinstance(leaf, AbstractLinearOperator),
    )
    expected_matrix = -op.as_matrix()
    assert_array_equal(AbstractLinearOperator.as_matrix(neg_op), expected_matrix)
    assert_array_equal(neg_op.as_matrix(), expected_matrix)


def test_reduce_1(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand = Op(1, in_structure=_op_structure)
    op = AdditionOperator(pytree_builder(operand))
    assert op.reduce() == operand


def test_reduce_2(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    h = HomothetyOperator(3.0, in_structure=jax.ShapeDtypeStruct((3,), jnp.float32))
    op = AdditionOperator(
        pytree_builder(CompositionOperator([h, h]), Op(2, in_structure=_op_structure))
    )
    assert isinstance(op, AdditionOperator)
    assert isinstance(op.operand_leaves[0], CompositionOperator)
    assert isinstance(op.operand_leaves[1], Op)
    reduced_op = op.reduce()
    assert isinstance(reduced_op, AdditionOperator)
    assert isinstance(reduced_op.operand_leaves[0], HomothetyOperator)
    assert isinstance(reduced_op.operand_leaves[1], Op)


def test_input_pytree1() -> None:
    dtype = jnp.float64
    structure = {'a': jax.ShapeDtypeStruct((3,), dtype), 'b': jax.ShapeDtypeStruct((2,), dtype)}
    op1 = HomothetyOperator(2, in_structure=structure)
    op2 = IdentityOperator(in_structure=structure)
    op = op1 + op2
    x = {'a': jnp.ones(3, dtype), 'b': jnp.ones(2, dtype)}
    actual_y = op(x)
    expected_y = {'a': jnp.full(3, 3.0, dtype), 'b': jnp.full(2, 3.0, dtype)}
    assert equinox.tree_equal(actual_y, expected_y)


def test_input_pytree2() -> None:
    dtype = jnp.float64
    structure = {'a': jax.ShapeDtypeStruct((3,), dtype), 'b': jax.ShapeDtypeStruct((3,), dtype)}

    @square
    class Op1(AbstractLinearOperator):
        def mv(self, x):
            return {'a': x['a'] + x['b'], 'b': x['b']}

    op1 = Op1(in_structure=structure)
    op2 = IdentityOperator(in_structure=structure)
    op = op1 + op2
    x = {'a': jnp.full(3, 3.0, dtype), 'b': jnp.full(3, 2.0, dtype)}
    actual_y = op(x)
    expected_y = {'a': jnp.full(3, 8.0, dtype), 'b': jnp.full(3, 4.0, dtype)}
    assert equinox.tree_equal(actual_y, expected_y)


def test_add_invalid_in_structure() -> None:
    _op3_structure = jax.ShapeDtypeStruct((3,), jnp.float32)

    with pytest.raises(ValueError, match='Incompatible linear operator input structures'):
        _ = Op(1, in_structure=_op_structure) + Op(1, in_structure=_op3_structure)


def test_add_invalid_out_structure() -> None:
    class Op_(AbstractLinearOperator):
        value: float

        def mv(self, x):
            return jnp.array([[0, self.value], [1, 0], [0, 0]]) @ x

    with pytest.raises(ValueError, match='Incompatible linear operator output structures'):
        _ = Op(1, in_structure=_op_structure) + Op_(1, in_structure=_op_structure)
