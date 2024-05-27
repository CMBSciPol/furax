import jax.numpy as jnp
from numpy.testing import assert_allclose

from astrosim.operators import DiagonalOperator, IdentityOperator
from astrosim.operators.base import AbstractLazyInverseOperator


def test_inverse(base_op) -> None:
    inv_op = base_op.I
    if isinstance(inv_op, AbstractLazyInverseOperator):
        assert inv_op.operator is base_op
        assert inv_op.I is base_op
    else:
        assert inv_op.I == base_op


def test_inverse_matmul(base_op) -> None:
    assert isinstance(base_op @ base_op.I, IdentityOperator)
    assert isinstance(base_op.I @ base_op, IdentityOperator)


def test_inverse_dense(base_op_and_dense) -> None:
    base_op, dense = base_op_and_dense
    assert_allclose(jnp.linalg.inv(dense), base_op.I.as_matrix())


def test_inverse_diagonal_with_zeros() -> None:
    op = DiagonalOperator((jnp.arange(4), jnp.array(0.0)))
    assert_allclose(op.I.as_matrix(), jnp.diag(jnp.array([0.0, 1.0, 1 / 2, 1 / 3, 0.0])))
