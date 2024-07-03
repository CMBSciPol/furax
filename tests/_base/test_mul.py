import jax
import numpy as np

from astrosim._base.core import CompositionOperator
from astrosim.operators import AbstractLinearOperator, HomothetyOperator


class Op(AbstractLinearOperator):
    def mv(self, x):
        return 2 * x

    def in_structure(self):
        return jax.ShapeDtypeStruct((3,), np.float32)

    def out_structure(self):
        return jax.ShapeDtypeStruct((), np.float32)


def test_rmul() -> None:
    mul_op = 2 * Op()
    assert isinstance(mul_op, CompositionOperator)
    assert isinstance(mul_op.operands[0], HomothetyOperator)
    assert mul_op.operands[0].value == 2
    assert isinstance(mul_op.operands[1], Op)


def test_mul() -> None:
    mul_op = Op() * 2
    assert isinstance(mul_op, CompositionOperator)
    assert isinstance(mul_op.operands[0], HomothetyOperator)
    assert mul_op.operands[0].value == 2
    assert isinstance(mul_op.operands[1], Op)


def test_truediv() -> None:
    mul_op = Op() / 2
    assert isinstance(mul_op, CompositionOperator)
    assert isinstance(mul_op.operands[0], HomothetyOperator)
    assert mul_op.operands[0].value == 0.5
    assert isinstance(mul_op.operands[1], Op)


def test_neg() -> None:
    mul_op = -Op()
    assert isinstance(mul_op, CompositionOperator)
    assert isinstance(mul_op.operands[0], HomothetyOperator)
    assert mul_op.operands[0].value == -1
    assert isinstance(mul_op.operands[1], Op)


def test_pos() -> None:
    mul_op = +Op()
    assert isinstance(mul_op, Op)
