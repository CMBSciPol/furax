from dataclasses import field
from uuid import uuid1

import jax
import pytest

from furax import (
    AbstractLinearOperator,
    diagonal,
    lower_triangular,
    negative_semidefinite,
    orthogonal,
    positive_semidefinite,
    square,
    symmetric,
    tridiagonal,
    upper_triangular,
)

_in_struct = jax.ShapeDtypeStruct((2, 3), int)
_out_struct = jax.ShapeDtypeStruct((3, 2), int)


@pytest.fixture
def Op() -> type[AbstractLinearOperator]:
    class Op(AbstractLinearOperator):
        _out_structure: jax.ShapeDtypeStruct = field(default=_out_struct, metadata={'static': True})

        def mv(self, x):
            return 2 * x

        @property
        def out_structure(self):
            return self._out_structure

    cls: type[AbstractLinearOperator] = type('C' + uuid1().hex, (Op,), {})
    return cls


def test_not_square(Op) -> None:
    op = Op(in_structure=_in_struct)
    assert not op.is_square
    assert not op.is_symmetric
    assert not op.is_orthogonal
    assert not op.is_diagonal
    assert not op.is_tridiagonal
    assert not op.is_lower_triangular
    assert not op.is_upper_triangular
    assert not op.is_negative_semidefinite
    assert not op.is_positive_semidefinite

    assert op.in_structure != op.out_structure
    assert op.T is not op
    if op.is_square:
        assert op.I is not op


@pytest.mark.parametrize(
    'decorator',
    [
        square,
        symmetric,
        orthogonal,
        diagonal,
        tridiagonal,
        lower_triangular,
        upper_triangular,
        negative_semidefinite,
        positive_semidefinite,
    ],
)
def test_square(decorator, Op) -> None:
    decorator(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_square
    assert op.out_structure == op.in_structure


@pytest.mark.parametrize('decorator', [diagonal, symmetric])
def test_symmetric(decorator, Op) -> None:
    decorator(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_symmetric
    assert op.T is op


def test_orthogonal(Op) -> None:
    orthogonal(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_orthogonal
    assert Op.inverse is Op.transpose


def test_diagonal(Op) -> None:
    diagonal(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_diagonal


def test_tridiagonal(Op) -> None:
    tridiagonal(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_tridiagonal


def test_lower_triangular(Op) -> None:
    lower_triangular(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_lower_triangular


def test_upper_triangular(Op) -> None:
    upper_triangular(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_upper_triangular


def test_negative_semidefinite(Op) -> None:
    negative_semidefinite(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_negative_semidefinite


def test_positive_semidefinite(Op) -> None:
    positive_semidefinite(Op)
    op = Op(in_structure=_in_struct)
    assert op.is_positive_semidefinite


def test_subclass() -> None:
    @diagonal
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

    class SubOp(Op):
        pass

    op = SubOp(in_structure=jax.ShapeDtypeStruct((2, 3), int))

    assert op.is_diagonal
