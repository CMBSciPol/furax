import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from numpy.testing import assert_array_equal

from astrosim._base.core import (
    AbstractLinearOperator,
    DiagonalOperator,
    HomothetyOperator,
    IdentityOperator,
    square,
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
    op = HomothetyOperator(struct, 2.0)
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
    op = HomothetyOperator(struct, 2)
    assert op.in_size() == 4
    assert op.out_size() == 4

    expected = 2 * jnp.eye(4)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_homothety_matmul() -> None:
    op1 = HomothetyOperator(jax.ShapeDtypeStruct((2,), np.float32), 2.0)
    op2 = HomothetyOperator(jax.ShapeDtypeStruct((2,), np.float32), 6.0)
    op = op1 @ op2
    assert isinstance(op, HomothetyOperator)
    assert op.value == 12.0
    assert op.in_structure() == op2.in_structure()


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


def test_as_matrix() -> None:
    @square
    class MyOperator(AbstractLinearOperator):
        n1 = 100
        n2 = 2
        matrix1: jax.Array = eqx.field(static=True)
        matrix2: jax.Array = eqx.field(static=True)

        def __init__(self) -> None:
            key = jax.random.key(0)
            key, subkey1, subkey2 = jax.random.split(key, 3)
            self.matrix1 = jax.random.randint(subkey1, (self.n1, self.n1), 0, 100)
            self.matrix2 = jax.random.randint(subkey2, (self.n2, self.n2), 0, 100)

        def mv(self, x):
            return (self.matrix1 @ x[0], self.matrix2 @ x[1])

        def in_structure(self):
            return (
                jax.ShapeDtypeStruct((self.n1,), np.int32),
                jax.ShapeDtypeStruct((self.n2,), np.int32),
            )

    op = MyOperator()

    expected = jax.scipy.linalg.block_diag(op.matrix1, op.matrix2)
    import time

    t0 = time.perf_counter()
    actual = op.as_matrix().block_until_ready()
    assert_array_equal(actual, expected)
