import jax
import jax.numpy as jnp
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator


def test() -> None:
    class Op(AbstractLinearOperator):
        def mv(self, x):
            return {'c': x['a'], 'd': x['a'][0] + 2 * x['b']}

        def in_structure(self):
            return {
                'a': jax.ShapeDtypeStruct((3,), jnp.float32),
                'b': jax.ShapeDtypeStruct((4,), jnp.float64),
            }

    op = Op()
    assert op.out_structure() == {
        'c': jax.ShapeDtypeStruct((3,), jnp.float32),
        'd': jax.ShapeDtypeStruct((4,), jnp.float64),
    }

    matrix = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 2, 0, 0, 0],
            [1, 0, 0, 0, 2, 0, 0],
            [1, 0, 0, 0, 0, 2, 0],
            [1, 0, 0, 0, 0, 0, 2],
        ],
        dtype=jnp.float64,
    )
    assert_array_equal(op.as_matrix(), matrix)
    assert_array_equal(op.T.as_matrix().T, matrix)
