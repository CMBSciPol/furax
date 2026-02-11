import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, square


def test_operators_as_pytree() -> None:
    class MyOp(AbstractLinearOperator):
        a: int

        def __init__(self):
            object.__setattr__(self, 'a', 3)
            super().__init__(in_structure=jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64))

        def mv(self, x):
            return self.a * x

    original_op = MyOp()
    leaves, treedef = jax.tree.flatten(original_op)
    actual_op = treedef.unflatten(leaves)
    assert actual_op == original_op


@pytest.mark.parametrize(
    'structure, expected_dtype',
    [
        (jax.ShapeDtypeStruct((), jnp.complex64), jnp.complex64),
        (
            {
                'a': jax.ShapeDtypeStruct((), jnp.float16),
                'b': jax.ShapeDtypeStruct((), jnp.float32),
                'c': jax.ShapeDtypeStruct((), jnp.float64),
            },
            jnp.float64,
        ),
        (
            [jax.ShapeDtypeStruct((), jnp.bfloat16), jax.ShapeDtypeStruct((), jnp.float32)],
            jnp.float32,
        ),
    ],
)
def test_in_promoted_dtype(structure, expected_dtype):
    @square
    class MyOperator(AbstractLinearOperator):
        def mv(self, x):
            return None

    op = MyOperator(in_structure=structure)
    assert op.in_promoted_dtype == expected_dtype
    assert op.out_promoted_dtype == expected_dtype


def test_as_matrix() -> None:
    n1 = 100
    n2 = 2

    @square
    class MyOperator(AbstractLinearOperator):
        matrix1: jax.Array
        matrix2: jax.Array

        def mv(self, x):
            return (self.matrix1 @ x[0], self.matrix2 @ x[1])

    key = jax.random.key(0)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    op = MyOperator(
        matrix1=jax.random.randint(subkey1, (n1, n1), 0, 100),
        matrix2=jax.random.randint(subkey2, (n2, n2), 0, 100),
        in_structure=(
            jax.ShapeDtypeStruct((n1,), jnp.int32),
            jax.ShapeDtypeStruct((n2,), jnp.int32),
        ),
    )

    expected = jax.scipy.linalg.block_diag(op.matrix1, op.matrix2)
    actual = op.as_matrix().block_until_ready()
    assert_array_equal(actual, expected)
