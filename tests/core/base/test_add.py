from collections.abc import Callable

import equinox
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_allclose, assert_array_equal

import furax
from furax import (
    AbstractLinearOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
    square,
)
from furax.core import (
    AdditionOperator,
    BlockRowOperator,
    CompositionOperator,
    DenseBlockDiagonalOperator,
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


# A sum over a list of operands whose own amplitudes differ in size, which share the map
_SIZES = (2, 3)
_LIST_STRIDES = {'map': None, 'own': 1}


def _struct(n: int) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct((n,), jnp.float64)


def _list_joint() -> list[AbstractLinearOperator]:
    """The joint operators, from `{'map', 'own'}` to a TOD of the size of each one's own part."""
    keys = jax.random.split(jax.random.key(0), 2 * len(_SIZES))
    return [
        BlockRowOperator(
            {
                'map': DenseBlockDiagonalOperator(
                    jax.random.normal(keys[2 * g], (n, 4)),
                    in_structure=_struct(4),
                ),
                'own': DiagonalOperator(jax.random.normal(keys[2 * g + 1], (n,))),
            }
        )
        for g, n in enumerate(_SIZES)
    ]


def _list_unknowns() -> dict:
    keys = jax.random.split(jax.random.key(1), 1 + len(_SIZES))
    own = [jax.random.normal(k, (n,)) for k, n in zip(keys[1:], _SIZES)]
    return {'map': jax.random.normal(keys[0], (4,)), 'own': own}


@pytest.mark.parametrize('sequential', [False, True])
def test_list_strided_sum_applies_each_operand_to_its_parts(sequential: bool) -> None:
    joint = _list_joint()
    normal = AdditionOperator([J.T @ J for J in joint], sequential, in_strides=_LIST_STRIDES)
    x = _list_unknowns()
    parts = [(J.T @ J)({'map': x['map'], 'own': x['own'][g]}) for g, J in enumerate(joint)]
    expected = {'map': sum(p['map'] for p in parts), 'own': [p['own'] for p in parts]}
    assert normal.in_structure == normal.out_structure == furax.tree.as_structure(x)
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-12), normal(x), expected)


def test_list_strided_sum_transposes_with_its_strides() -> None:
    # non-square: the unknowns to the TOD of each operand, one entry each
    op = AdditionOperator(_list_joint(), in_strides=_LIST_STRIDES, out_strides=1)
    assert [s.shape for s in op.out_structure] == [(n,) for n in _SIZES]
    assert_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-12)
    assert_allclose((op.T @ op).as_matrix(), op.as_matrix().T @ op.as_matrix(), rtol=1e-12)


def test_list_strided_sum_tags_need_the_same_strides_on_both_sides() -> None:
    structures = [{'map': _struct(4), 'own': _struct(n)} for n in _SIZES]
    scalings = [HomothetyOperator(2.0, in_structure=structure) for structure in structures]
    scaled = AdditionOperator(scalings, in_strides=_LIST_STRIDES)
    assert scaled.is_symmetric
    # a closed operand of a wider sum, as a sum over an ensemble is
    assert (scaled + scaled).operand_leaves == [scaled, scaled]
    joint = AdditionOperator(_list_joint(), in_strides=_LIST_STRIDES, out_strides=1)
    assert not joint.is_square and not joint.is_symmetric


@pytest.mark.parametrize(
    ('strides', 'match'),
    [
        ('segment', 'None or 1'),
        (None, 'differ on a part of stride None'),
        ({'map': None, 'other': 1}, 'do not fit the operands'),
    ],
    ids=['segment', 'shared-part-differs', 'not-a-prefix'],
)
def test_list_strided_sum_rejects_invalid_strides(strides, match) -> None:
    operands = [J.T @ J for J in _list_joint()]
    with pytest.raises(ValueError, match=match):
        AdditionOperator(operands, in_strides=strides, out_strides=_LIST_STRIDES)
