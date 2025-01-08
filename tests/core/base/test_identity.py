import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, IdentityOperator


@jdc.pytree_dataclass
class PyTreeTest:
    a: jax.Array
    b: int
    c: float


def test_identity1() -> None:
    struct = jax.ShapeDtypeStruct((2, 1), np.float32)
    op = IdentityOperator(struct)
    assert op.in_size() == 2
    assert op.out_size() == 2

    expected = jnp.eye(2)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_identity2() -> None:
    struct = PyTreeTest(
        jax.ShapeDtypeStruct((2,), np.float64),
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct((), np.float32),
    )
    op = IdentityOperator(struct)
    assert op.in_size() == 4
    assert op.out_size() == 4

    expected = jnp.eye(4)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)
