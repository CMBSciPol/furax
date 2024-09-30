import equinox
import jax
import jax.numpy as jnp
import pytest
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
    value: int

    def __init__(self, value: int):
        self.value = value

    def mv(self, x):
        return self.value * x[0]

    def in_structure(self):
        return jax.ShapeDtypeStruct((3,), jnp.float32)

    def transpose(self):
        return OpTranspose(self)


class OpTranspose(TransposeOperator):
    def mv(self, x):
        return None


def test_add_1() -> None:
    op1 = Op(2)
    op2 = Op(3)
    op = op1 + op2
    assert isinstance(op, AdditionOperator)
    assert_array_equal(op.as_matrix(), jnp.array([[5, 0, 0]]))


def test_add_2() -> None:
    op1 = Op(1)
    op2 = Op(2)
    op3 = Op(3)
    op = op1 + op2 + op3
    assert isinstance(op, AdditionOperator)
    assert len(op.operands) == 3
    assert_array_equal(op.as_matrix(), jnp.array([[6, 0, 0]]))


def test_add_3() -> None:
    op1 = Op(1)
    op2 = Op(2)
    op3 = Op(3)
    op = op1 + (op2 + op3)
    assert isinstance(op, AdditionOperator)
    assert len(op.operands) == 3
    assert_array_equal(op.as_matrix(), jnp.array([[6, 0, 0]]))


def test_sub() -> None:
    op1 = Op(2)
    op2 = Op(3)
    op = op1 - op2
    assert isinstance(op, AdditionOperator)
    assert_array_equal(op.as_matrix(), jnp.array([[-1, 0, 0]]))


def test_pytree1() -> None:
    dtype = jnp.float64
    structure = {'a': jax.ShapeDtypeStruct((3,), dtype), 'b': jax.ShapeDtypeStruct((2,), dtype)}
    op1 = HomothetyOperator(2, structure)
    op2 = IdentityOperator(structure)
    op = op1 + op2
    x = {'a': jnp.ones(3, dtype), 'b': jnp.ones(2, dtype)}
    actual_y = op(x)
    expected_y = {'a': jnp.full(3, 3.0, dtype), 'b': jnp.full(2, 3.0, dtype)}
    assert equinox.tree_equal(actual_y, expected_y)


def test_pytree2() -> None:
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


def test_add_invalid_instructure() -> None:
    class Op_(Op):

        def in_structure(self):
            return jax.ShapeDtypeStruct((2,), jnp.float32)

    with pytest.raises(ValueError, match='Incompatible linear operator input structures'):
        _ = Op(1) + Op_(1)


def test_add_invalid_outstructure() -> None:
    class Op_(Op):

        def out_structure(self):
            return jax.ShapeDtypeStruct((2,), jnp.float32)

    with pytest.raises(ValueError, match='Incompatible linear operator output structures'):
        _ = Op(1) + Op_(1)


def test_transpose() -> None:
    op1 = Op(2)
    op2 = Op(3)
    op = op1 + op2
    opT = op.T
    assert isinstance(opT, AdditionOperator)
    assert len(opT.operands) == 2
    assert isinstance(opT.operands[0], OpTranspose)
    assert opT.operands[0].operator is op1
    assert isinstance(opT.operands[1], OpTranspose)
    assert opT.operands[1].operator is op2


def test_reduce_1() -> None:
    op = AdditionOperator([Op(1)])
    assert op.reduce() is op.operands[0]


def test_reduce_2() -> None:
    h = HomothetyOperator(3.0, jax.ShapeDtypeStruct((3,), jnp.float32))
    op = AdditionOperator([CompositionOperator([h, h]), Op(2)])
    assert isinstance(op, AdditionOperator)
    assert isinstance(op.operands[0], CompositionOperator)
    reduced_op = op.reduce()
    assert isinstance(reduced_op, AdditionOperator)
    assert isinstance(reduced_op.operands[0], HomothetyOperator)
