from uuid import uuid1

import jax
import lineax as lx
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


@pytest.fixture
def Op() -> type[AbstractLinearOperator]:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return jax.ShapeDtypeStruct((2, 3), int)

        def out_structure(self):
            return jax.ShapeDtypeStruct((3, 2), int)

    cls: type[AbstractLinearOperator] = type('C' + uuid1().hex, (Op,), {})
    return cls


def test_not_square(Op) -> None:
    op = Op()
    assert not op.is_square
    assert not op.is_symmetric
    assert not op.is_orthogonal
    assert not op.is_diagonal
    assert not op.is_tridiagonal
    assert not op.is_lower_triangular
    assert not op.is_upper_triangular
    assert not op.is_negative_semidefinite
    assert not op.is_positive_semidefinite

    assert op.in_structure() != op.out_structure()
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
    op = Op()
    assert op.is_square
    assert Op.out_structure is Op.in_structure


@pytest.mark.parametrize('decorator', [diagonal, symmetric])
def test_symmetric(decorator, Op) -> None:
    decorator(Op)
    op = Op()
    assert op.is_symmetric
    assert op.T is op


def test_orthogonal(Op) -> None:
    orthogonal(Op)
    op = Op()
    assert op.is_orthogonal
    assert Op.inverse is Op.transpose


def test_diagonal(Op) -> None:
    diagonal(Op)
    op = Op()
    assert op.is_diagonal


def test_tridiagonal(Op) -> None:
    tridiagonal(Op)
    op = Op()
    assert op.is_tridiagonal


def test_lower_triangular(Op) -> None:
    lower_triangular(Op)
    op = Op()
    assert op.is_lower_triangular


def test_upper_triangular(Op) -> None:
    upper_triangular(Op)
    op = Op()
    assert op.is_upper_triangular


def test_negative_semidefinite(Op) -> None:
    negative_semidefinite(Op)
    op = Op()
    assert op.is_negative_semidefinite


def test_positive_semidefinite(Op) -> None:
    positive_semidefinite(Op)
    op = Op()
    assert op.is_positive_semidefinite


@pytest.mark.parametrize(
    'decorator',
    [
        diagonal,
        symmetric,
        tridiagonal,
        lower_triangular,
        upper_triangular,
        negative_semidefinite,
        positive_semidefinite,
    ],
)
def test_lineax_register(Op, decorator) -> None:
    decorator(Op)
    assert getattr(lx, f'is_{decorator.__name__}')(Op())


def test_lineax_subclass() -> None:
    @diagonal
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return jax.ShapeDtypeStruct((2, 3), int)

    class SubOp(Op):
        pass

    assert lx.is_diagonal(SubOp())
