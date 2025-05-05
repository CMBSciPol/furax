import jax.numpy as jnp
import pytest
from equinox import tree_equal
from jax.scipy.linalg import block_diag
from numpy.testing import assert_array_equal

from furax import SumOperator
from furax.tree import as_structure
from tests.helpers import arange

ARRAY1 = arange(2, 3, 4)
ARRAY2 = arange(5, 4, 3, 2)
PYTREE1 = {'a': ARRAY1, 'b': ARRAY2}
PYTREE2 = (ARRAY1, ARRAY2)


@pytest.mark.parametrize(
    'input, axis, expected',
    [
        (ARRAY1, None, jnp.sum(ARRAY1)),
        (ARRAY1, (), ARRAY1),
        (ARRAY1, (0,), jnp.sum(ARRAY1, axis=0)),
        (ARRAY1, (1,), jnp.sum(ARRAY1, axis=1)),
        (ARRAY1, (2,), jnp.sum(ARRAY1, axis=2)),
        (ARRAY1, (-1,), jnp.sum(ARRAY1, axis=2)),
        (ARRAY1, (0, 1), jnp.sum(ARRAY1, axis=(0, 1))),
        (PYTREE1, None, {'a': jnp.sum(ARRAY1), 'b': jnp.sum(ARRAY2)}),
        (PYTREE1, (), PYTREE1),
        (PYTREE1, (0,), {'a': jnp.sum(ARRAY1, axis=0), 'b': jnp.sum(ARRAY2, axis=0)}),
        (PYTREE1, (1,), {'a': jnp.sum(ARRAY1, axis=1), 'b': jnp.sum(ARRAY2, axis=1)}),
        (PYTREE1, (2,), {'a': jnp.sum(ARRAY1, axis=2), 'b': jnp.sum(ARRAY2, axis=2)}),
        (PYTREE1, (-1,), {'a': jnp.sum(ARRAY1, axis=2), 'b': jnp.sum(ARRAY2, axis=3)}),
        (PYTREE1, (0, 1), {'a': jnp.sum(ARRAY1, axis=(0, 1)), 'b': jnp.sum(ARRAY2, axis=(0, 1))}),
        (PYTREE1, {'a': None, 'b': None}, {'a': jnp.sum(ARRAY1), 'b': jnp.sum(ARRAY2)}),
        (PYTREE1, {'a': None, 'b': ()}, {'a': jnp.sum(ARRAY1), 'b': ARRAY2}),
        (PYTREE1, {'a': (), 'b': ()}, PYTREE1),
        (PYTREE1, {'a': (0,), 'b': None}, {'a': jnp.sum(ARRAY1, axis=0), 'b': jnp.sum(ARRAY2)}),
        (PYTREE1, {'a': 0, 'b': 1}, {'a': jnp.sum(ARRAY1, axis=0), 'b': jnp.sum(ARRAY2, axis=1)}),
        (PYTREE2, None, (jnp.sum(ARRAY1), jnp.sum(ARRAY2))),
        (PYTREE2, (), PYTREE2),
        (PYTREE2, 0, (jnp.sum(ARRAY1, axis=0), jnp.sum(ARRAY2, axis=0))),
        (PYTREE2, (0, 1), (jnp.sum(ARRAY1, axis=(0, 1)), jnp.sum(ARRAY2, axis=(0, 1)))),
        (PYTREE2, (None, 1), (jnp.sum(ARRAY1), jnp.sum(ARRAY2, axis=1))),
        (PYTREE2, ((0,), (1,)), (jnp.sum(ARRAY1, axis=0), jnp.sum(ARRAY2, axis=1))),
    ],
)
def test_sum(input, axis, expected) -> None:
    op = SumOperator(axis=axis, in_structure=as_structure(input))
    actual = op(input)
    tree_equal(actual, expected)


def test_as_matrix() -> None:
    x = {'a': arange(2, 3), 'b': arange(3, 2)}
    op = SumOperator({'a': None, 'b': 0}, in_structure=as_structure(x))
    matrix = op.as_matrix()
    expected_matrix = block_diag(
        jnp.array([[1, 1, 1, 1, 1, 1]]), jnp.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]])
    )
    assert_array_equal(matrix, expected_matrix)

    matrix = op.T.as_matrix()
    assert_array_equal(matrix, expected_matrix.T)
