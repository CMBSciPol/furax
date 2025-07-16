import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, HomothetyOperator, IdentityOperator
from furax.core import AbstractLazyInverseOperator


def test_inverse(base_op) -> None:
    inv_op = base_op.I
    if isinstance(inv_op, AbstractLazyInverseOperator):
        assert inv_op.operator is base_op
        assert inv_op.I is base_op
    else:
        assert inv_op.I == base_op


def test_inverse_matmul(base_op) -> None:
    if isinstance(base_op, HomothetyOperator):
        assert isinstance(base_op @ base_op.I, HomothetyOperator)
        assert isinstance(base_op.I @ base_op, HomothetyOperator)
    else:
        assert isinstance((base_op @ base_op.I).reduce(), IdentityOperator)
        assert isinstance((base_op.I @ base_op).reduce(), IdentityOperator)


def test_inverse_dense(base_op_and_dense) -> None:
    base_op, dense = base_op_and_dense
    assert_allclose(jnp.linalg.inv(dense), base_op.I.as_matrix())


def test_parametrized_inverse_empty() -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return x

        def in_structure(self):
            return jax.ShapeDtypeStruct((2,), jnp.float32)

    op = Op()
    inv_op = op.I
    assert inv_op() is inv_op

    x = jnp.array([1.0, 2], jnp.float32)
    with pytest.raises(ValueError, match=r'instead of A\.I'):
        inv_op(x, option=True)


def test_parametrized_inverse() -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return m @ x

        def in_structure(self):
            return jax.ShapeDtypeStruct((4,), jnp.float64)

    m = jnp.array(
        [
            [1, 2, 3, 4],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    inv_m = jnp.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1 / 4, -1 / 4, -1 / 2, -3 / 4],
        ]
    )

    op = Op()
    b = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

    solver = lx._solver.NormalCG(rtol=1e-6, atol=1e-6, max_steps=5)
    inv_op = op.I(solver=solver, throw=True)
    assert_allclose(inv_op(b), inv_m @ b)

    solver = lx._solver.NormalCG(rtol=1e-6, atol=1e-6, max_steps=4)
    inv_op = op.I(solver=solver, throw=True)
    with pytest.raises(RuntimeError, match='The maximum number of solver steps was reached'):
        inv_op(b)
