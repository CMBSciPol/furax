import jax
import lineax as lx
import pytest

from furax import (
    AbstractLinearOperator,
    diagonal,
    lower_triangular,
    negative_semidefinite,
    positive_semidefinite,
    symmetric,
    tridiagonal,
    upper_triangular,
)
from furax.interfaces.lineax import as_lineax_operator


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
def test_lineax_register(decorator) -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return 2 * x

    furax_op = decorator(Op)(in_structure=jax.ShapeDtypeStruct((2, 3), int))
    lineax_op = as_lineax_operator(furax_op)
    assert getattr(lx, f'is_{decorator.__name__}')(lineax_op)
