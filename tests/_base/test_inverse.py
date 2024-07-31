import equinox
import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax._base.core import (
    AbstractLazyInverseOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
)


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
        assert isinstance(base_op @ base_op.I, IdentityOperator)
        assert isinstance(base_op.I @ base_op, IdentityOperator)


def test_inverse_dense(base_op_and_dense) -> None:
    base_op, dense = base_op_and_dense
    assert_allclose(jnp.linalg.inv(dense), base_op.I.as_matrix())


def test_inverse_diagonal_with_zeros1() -> None:
    op = DiagonalOperator((jnp.arange(4), jnp.array(0.0)))
    x = (jnp.ones(4), jnp.array(1.0))
    expected_y = jnp.array([0, 1, 1, 1]), jnp.array(0)
    equinox.tree_equal(op.I(op(x)), expected_y)


def test_inverse_diagonal_with_zeros2() -> None:
    op = DiagonalOperator((jnp.arange(4), jnp.array(0.0)))
    assert_allclose(op.I.as_matrix(), jnp.diag(jnp.array([0.0, 1.0, 1 / 2, 1 / 3, 0.0])))
