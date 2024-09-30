import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

from furax._base.core import AbstractLinearOperator, square


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
        _in_structure: PyTree[jax.ShapeDtypeStruct]

        def __init__(self, in_structure):
            self._in_structure = in_structure

        def mv(self, x):
            return None

        def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
            return self._in_structure

    op = MyOperator(structure)
    assert op.in_promoted_dtype == expected_dtype
    assert op.out_promoted_dtype == expected_dtype


def test_as_matrix() -> None:
    @square
    class MyOperator(AbstractLinearOperator):
        n1 = 100
        n2 = 2
        matrix1: jax.Array
        matrix2: jax.Array

        def __init__(self) -> None:
            key = jax.random.key(0)
            key, subkey1, subkey2 = jax.random.split(key, 3)
            self.matrix1 = jax.random.randint(subkey1, (self.n1, self.n1), 0, 100)
            self.matrix2 = jax.random.randint(subkey2, (self.n2, self.n2), 0, 100)

        def mv(self, x):
            return (self.matrix1 @ x[0], self.matrix2 @ x[1])

        def in_structure(self):
            return (
                jax.ShapeDtypeStruct((self.n1,), jnp.int32),
                jax.ShapeDtypeStruct((self.n2,), jnp.int32),
            )

    op = MyOperator()

    expected = jax.scipy.linalg.block_diag(op.matrix1, op.matrix2)
    import time

    t0 = time.perf_counter()
    actual = op.as_matrix().block_until_ready()
    assert_array_equal(actual, expected)
