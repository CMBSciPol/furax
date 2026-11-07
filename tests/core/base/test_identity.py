from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, IdentityOperator


@register_dataclass
@dataclass
class PyTreeTest:
    a: jax.Array
    b: int
    c: float


def test_identity1() -> None:
    struct = jax.ShapeDtypeStruct((2, 1), np.float32)
    op = IdentityOperator(in_structure=struct)
    assert op.in_size == 2
    assert op.out_size == 2

    expected = jnp.eye(2)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


def test_identity2() -> None:
    struct = PyTreeTest(
        jax.ShapeDtypeStruct((2,), np.float64),
        jax.ShapeDtypeStruct((), np.int32),
        jax.ShapeDtypeStruct((), np.float32),
    )
    op = IdentityOperator(in_structure=struct)
    assert op.in_size == 4
    assert op.out_size == 4

    expected = jnp.eye(4)
    assert_array_equal(op.as_matrix(), expected)
    assert_array_equal(AbstractLinearOperator.as_matrix(op), expected)


class Op(AbstractLinearOperator):
    def mv(self, x):
        return 2 * x


_structure = jax.ShapeDtypeStruct((3,), np.float32)


def test_identity_is_absorbed_on_the_left() -> None:
    op = Op(in_structure=_structure)
    assert IdentityOperator(in_structure=_structure) @ op is op


def test_identity_is_absorbed_on_the_right() -> None:
    op = Op(in_structure=_structure)
    assert op @ IdentityOperator(in_structure=_structure) is op


def test_identity_is_absorbed_by_a_composition() -> None:
    op = Op(in_structure=_structure)
    composition = op @ op
    assert composition @ IdentityOperator(in_structure=_structure) is composition
