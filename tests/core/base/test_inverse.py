import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax import HomothetyOperator, IdentityOperator
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
