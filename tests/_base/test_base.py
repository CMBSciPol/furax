import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from numpy.testing import assert_array_equal

from furax._base.core import (
    AbstractLinearOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
)


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


def test_homothety1() -> None:
    struct = jax.ShapeDtypeStruct((2, 1), np.float32)
    op = HomothetyOperator(2.0, struct)
    assert op.in_size() == 2
    assert op.out_size() == 2

    expected = 2 * jnp.eye(2)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_homothety2() -> None:
    struct = PyTreeTest(
        jax.ShapeDtypeStruct((2,), np.float64),
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct((), np.float32),
    )
    op = HomothetyOperator(2, struct)
    assert op.in_size() == 4
    assert op.out_size() == 4

    expected = 2 * jnp.eye(4)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_diagonal1() -> None:
    values = jnp.array([[2], [3]], np.float32)
    op = DiagonalOperator(values)
    assert op.in_structure() == jax.ShapeDtypeStruct((2, 1), np.float32)
    assert op.in_size() == 2
    assert op.out_size() == 2

    expected = jnp.diag(jnp.array([2, 3]))
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_diagonal2() -> None:
    values = PyTreeTest(
        jnp.array([1, 2], dtype=np.float64),
        jnp.array(3, dtype=np.int32),
        jnp.array(4, dtype=np.float32),
    )
    struct = PyTreeTest(
        jax.ShapeDtypeStruct((2,), np.float64),
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct((), np.float32),
    )
    op = DiagonalOperator(values)
    assert op.in_size() == 4
    assert op.out_size() == 4
    assert op.in_structure() == struct

    expected = jnp.diag(jnp.array([1, 2, 3, 4]))
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)
