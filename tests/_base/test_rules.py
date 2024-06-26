import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PyTree

from astrosim._base.core import (
    AbstractLazyInverseOrthogonalOperator,
    AbstractLinearOperator,
    CompositionOperator,
    HomothetyOperator,
    IdentityOperator,
    orthogonal,
    square,
)


class Op(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct]
    _out_structure: PyTree[jax.ShapeDtypeStruct]

    def __init__(self, in_shape=None, out_shape=None):
        if out_shape is None:
            out_shape = in_shape
        self._in_structure = jax.ShapeDtypeStruct(in_shape, np.float32)
        self._out_structure = jax.ShapeDtypeStruct(out_shape, np.float32)

    def mv(self, x):
        return None

    def in_structure(self):
        return self._in_structure

    def out_structure(self):
        return self._out_structure


@square
class Op1(AbstractLinearOperator):
    def mv(self, x):
        return 2 * x

    def in_structure(self):
        return jax.ShapeDtypeStruct((), np.float32)


@orthogonal
class Op2(AbstractLinearOperator):
    def mv(self, x):
        return 3 * x

    def transpose(self):
        return Op2T(self)

    def in_structure(self):
        return jax.ShapeDtypeStruct((), np.float32)


class Op2T(AbstractLazyInverseOrthogonalOperator):
    def mv(self, x):
        return 3 * x

    def in_structure(self):
        return jax.ShapeDtypeStruct((), np.float32)


def test_composition() -> None:

    op = Op1() @ Op2() @ Op1()
    assert len(op.operands) == 3
    x = jnp.array(1)
    assert op(x) == 12


@pytest.mark.parametrize(
    'op, composed_op',
    [
        (op := Op1(), op.I @ op),
        (op := Op1(), op @ op.I),
        (op := Op2(), op @ op.I),
        (op := Op2(), op.I @ op),
        (op := Op2(), op @ op.T),
        (op := Op2(), op.T @ op),
    ],
)
def test_inverse(op: AbstractLinearOperator, composed_op: CompositionOperator) -> None:
    assert isinstance(composed_op, IdentityOperator)
    assert composed_op.in_structure() == op.in_structure()
    assert composed_op.out_structure() == op.out_structure()


@pytest.mark.parametrize(
    'op, composed_op',
    [
        (op := Op1(), CompositionOperator([op.I, op])),
        (op := Op1(), CompositionOperator([op, op.I])),
        (op := Op2(), CompositionOperator([op, op.I])),
        (op := Op2(), CompositionOperator([op.I, op])),
        (op := Op2(), CompositionOperator([op, op.T])),
        (op := Op2(), CompositionOperator([op.T, op])),
    ],
)
def test_inverse(op: AbstractLinearOperator, composed_op: CompositionOperator) -> None:
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, IdentityOperator)
    assert reduced_op.in_structure() == op.in_structure()
    assert reduced_op.out_structure() == op.out_structure()


def test_homothety1() -> None:
    op1 = HomothetyOperator(2.0, jax.ShapeDtypeStruct((2,), np.float32))
    op2 = HomothetyOperator(6.0, jax.ShapeDtypeStruct((2,), np.float32))
    composed_op = op1 @ op2
    assert isinstance(composed_op, HomothetyOperator)
    assert composed_op.value == 12


def test_homothety2() -> None:
    op1 = HomothetyOperator(2.0, jax.ShapeDtypeStruct((2,), np.float32))
    op2 = HomothetyOperator(6.0, jax.ShapeDtypeStruct((2,), np.float32))
    composed_op = CompositionOperator([op1, op2])
    assert isinstance(composed_op, CompositionOperator)
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, HomothetyOperator)
    assert reduced_op.value == 12.0
    assert reduced_op.out_structure() == composed_op.out_structure() == op1.out_structure()
    assert reduced_op.in_structure() == composed_op.in_structure() == op2.in_structure()


@pytest.mark.parametrize(
    'composed_op, index',
    [
        (HomothetyOperator(2.0, jax.ShapeDtypeStruct((2,), np.float32)) @ Op((3,), (2,)), 0),
        (HomothetyOperator(2.0, jax.ShapeDtypeStruct((3,), np.float32)) @ Op((2,), (3,)), 1),
        (Op((3,), (2,)) @ HomothetyOperator(2.0, jax.ShapeDtypeStruct((3,), np.float32)), 0),
        (Op((2,), (3,)) @ HomothetyOperator(2.0, jax.ShapeDtypeStruct((2,), np.float32)), 1),
    ],
)
def test_homothety3(composed_op: CompositionOperator, index: int) -> None:
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, CompositionOperator)
    assert len(reduced_op.operands) == 2
    assert isinstance(reduced_op.operands[index], HomothetyOperator)
    assert isinstance(reduced_op.operands[1 - index], Op)
