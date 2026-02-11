from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PyTree

from furax import AbstractLinearOperator, HomothetyOperator, IdentityOperator, orthogonal, square
from furax.core._base import AbstractLazyInverseOrthogonalOperator, CompositionOperator


class Op(AbstractLinearOperator):
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self._out_structure is None:
            object.__setattr__(self, '_out_structure', self.in_structure)

    def mv(self, x):
        return None

    @property
    def out_structure(self):
        return self._out_structure


@square
class Op1(AbstractLinearOperator):
    def mv(self, x):
        return 2 * x


@orthogonal
class Op2(AbstractLinearOperator):
    def mv(self, x):
        return 3 * x

    def transpose(self):
        return Op2T(self)


class Op2T(AbstractLazyInverseOrthogonalOperator):
    def mv(self, x):
        return 3 * x


_scalar_struct = jax.ShapeDtypeStruct((), np.float32)


def test_composition() -> None:
    op = (
        Op1(in_structure=_scalar_struct)
        @ Op2(in_structure=_scalar_struct)
        @ Op1(in_structure=_scalar_struct)
    )
    assert len(op.operands) == 3
    x = jnp.array(1)
    assert op(x) == 12


@pytest.mark.parametrize(
    'op, composed_op',
    [
        (op := Op1(in_structure=_scalar_struct), op.I @ op),
        (op := Op1(in_structure=_scalar_struct), op @ op.I),
        (op := Op2(in_structure=_scalar_struct), op @ op.I),
        (op := Op2(in_structure=_scalar_struct), op.I @ op),
        (op := Op2(in_structure=_scalar_struct), op @ op.T),
        (op := Op2(in_structure=_scalar_struct), op.T @ op),
    ],
)
def test_inverse_1(op: AbstractLinearOperator, composed_op: CompositionOperator) -> None:
    assert isinstance(composed_op, IdentityOperator)
    assert composed_op.in_structure == op.in_structure
    assert composed_op.out_structure == op.out_structure


@pytest.mark.parametrize(
    'op, composed_op',
    [
        (op := Op1(in_structure=_scalar_struct), CompositionOperator([op.I, op])),
        (op := Op1(in_structure=_scalar_struct), CompositionOperator([op, op.I])),
        (op := Op2(in_structure=_scalar_struct), CompositionOperator([op, op.I])),
        (op := Op2(in_structure=_scalar_struct), CompositionOperator([op.I, op])),
        (op := Op2(in_structure=_scalar_struct), CompositionOperator([op, op.T])),
        (op := Op2(in_structure=_scalar_struct), CompositionOperator([op.T, op])),
    ],
)
def test_inverse_2(op: AbstractLinearOperator, composed_op: CompositionOperator) -> None:
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, IdentityOperator)
    assert reduced_op.in_structure == op.in_structure
    assert reduced_op.out_structure == op.out_structure


def test_homothety1() -> None:
    op1 = HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32))
    op2 = HomothetyOperator(6.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32))
    composed_op = op1 @ op2
    assert isinstance(composed_op, HomothetyOperator)
    assert composed_op.value == 12


def test_homothety2() -> None:
    op1 = HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32))
    op2 = HomothetyOperator(6.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32))
    composed_op = CompositionOperator([op1, op2])
    assert isinstance(composed_op, CompositionOperator)
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, HomothetyOperator)
    assert reduced_op.value == 12.0
    assert reduced_op.out_structure == composed_op.out_structure == op1.out_structure
    assert reduced_op.in_structure == composed_op.in_structure == op2.in_structure


@pytest.mark.parametrize(
    'composed_op, index',
    [
        (
            HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32))
            @ Op(
                in_structure=jax.ShapeDtypeStruct((3,), np.float32),
                _out_structure=jax.ShapeDtypeStruct((2,), np.float32),
            ),
            0,
        ),
        (
            HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((3,), np.float32))
            @ Op(
                in_structure=jax.ShapeDtypeStruct((2,), np.float32),
                _out_structure=jax.ShapeDtypeStruct((3,), np.float32),
            ),
            1,
        ),
        (
            Op(
                in_structure=jax.ShapeDtypeStruct((3,), np.float32),
                _out_structure=jax.ShapeDtypeStruct((2,), np.float32),
            )
            @ HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((3,), np.float32)),
            0,
        ),
        (
            Op(
                in_structure=jax.ShapeDtypeStruct((2,), np.float32),
                _out_structure=jax.ShapeDtypeStruct((3,), np.float32),
            )
            @ HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((2,), np.float32)),
            1,
        ),
    ],
)
def test_homothety3(composed_op: CompositionOperator, index: int) -> None:
    reduced_op = composed_op.reduce()
    assert isinstance(reduced_op, CompositionOperator)
    assert len(reduced_op.operands) == 2
    assert isinstance(reduced_op.operands[index], HomothetyOperator)
    assert isinstance(reduced_op.operands[1 - index], Op)
