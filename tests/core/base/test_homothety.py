import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, HomothetyOperator


def test_homothety1() -> None:
    struct = jax.ShapeDtypeStruct((2, 1), np.float32)
    op = HomothetyOperator(2.0, struct)
    assert op.in_size() == 2
    assert op.out_size() == 2

    expected = 2 * jnp.eye(2)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)
