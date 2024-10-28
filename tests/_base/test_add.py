from collections.abc import Callable

import equinox
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

from furax._base.core import (
    AbstractLinearOperator,
    AdditionOperator,
    CompositionOperator,
    HomothetyOperator,
    IdentityOperator,
    TransposeOperator,
    square,
)


class Op(AbstractLinearOperator):
    value: float

    def mv(self, x):
        return jnp.array([[0, self.value], [1, 0]]) @ x

    def in_structure(self):
        return jax.ShapeDtypeStruct((2,), jnp.float32)


def test_add(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    operand_leaves = Op(1), Op(2)
    extra_operand = Op(3)
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
    operand_leaves = Op(1), Op(2)
    extra_operand = Op(3)
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
    operand_leaves = Op(1), Op(2)
    extra_operand = Op(3)
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
    operand_leaves = Op(1), Op(2)
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
    operand_leaves = Op(1), Op(2)
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
    operand = Op(1)
    op = AdditionOperator(pytree_builder(operand))
    assert op.reduce() == operand


def test_reduce_2(
    pytree_builder: Callable[[AbstractLinearOperator, ...], PyTree[AbstractLinearOperator]],
) -> None:
    h = HomothetyOperator(3.0, jax.ShapeDtypeStruct((3,), jnp.float32))
    op = AdditionOperator(pytree_builder(CompositionOperator([h, h]), Op(2)))
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
    op1 = HomothetyOperator(2, structure)
    op2 = IdentityOperator(structure)
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

        def in_structure(self):
            return structure

    op1 = Op1()
    op2 = IdentityOperator(structure)
    op = op1 + op2
    x = {'a': jnp.full(3, 3.0, dtype), 'b': jnp.full(3, 2.0, dtype)}
    actual_y = op(x)
    expected_y = {'a': jnp.full(3, 8.0, dtype), 'b': jnp.full(3, 4.0, dtype)}
    assert equinox.tree_equal(actual_y, expected_y)


def test_add_invalid_in_structure() -> None:
    class Op_(Op):
        def in_structure(self):
            return jax.ShapeDtypeStruct((3,), jnp.float32)

    with pytest.raises(ValueError, match='Incompatible linear operator input structures'):
        _ = Op(1) + Op_(1)


def test_add_invalid_out_structure() -> None:
    class Op_(Op):
        def out_structure(self):
            return jax.ShapeDtypeStruct((3,), jnp.float32)

    with pytest.raises(ValueError, match='Incompatible linear operator output structures'):
        _ = Op(1) + Op_(1)
