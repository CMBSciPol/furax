import jax
import lineax as lx
import pytest

from astrosim.operators import (
    AbstractLinearOperator,
    diagonal,
    lower_triangular,
    negative_semidefinite,
    positive_semidefinite,
    square,
    symmetric,
    upper_triangular,
)


@pytest.mark.parametrize(
    'decorator',
    [
        diagonal,
        lower_triangular,
        negative_semidefinite,
        positive_semidefinite,
        symmetric,
        upper_triangular,
    ],
)
def test_register(decorator) -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return

    decorator(Op)
    assert getattr(lx, f'is_{decorator.__name__}')(Op())


@pytest.mark.parametrize(
    'decorator',
    [
        diagonal,
        lower_triangular,
        negative_semidefinite,
        positive_semidefinite,
        square,
        symmetric,
        upper_triangular,
    ],
)
def test_square(decorator) -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return

    decorator(Op)
    assert Op.out_structure is Op.in_structure


@pytest.mark.parametrize('decorator', [diagonal, symmetric])
def test_symmetric(decorator) -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return jax.ShapeDtypeStruct((2, 3), int)

    decorator(Op)
    op = Op()
    assert op.out_structure() == op.in_structure()
    assert op.T is op


def test_subclass() -> None:
    @diagonal
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

        def in_structure(self):
            return jax.ShapeDtypeStruct((2, 3), int)

    class SubOp(Op):
        pass

    assert lx.is_diagonal(SubOp())
