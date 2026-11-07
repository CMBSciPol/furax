import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax import AbstractLinearOperator, ZeroOperator
from furax.core import CompositionOperator
from furax.core._base import structure_equal


class Op(AbstractLinearOperator):
    def mv(self, x):
        return 2 * x


_structure = jax.ShapeDtypeStruct((3,), np.float32)
_pytree_structure = {
    'a': jax.ShapeDtypeStruct((2,), np.float32),
    'b': jax.ShapeDtypeStruct((2, 3), np.float64),
}


def test_mv_returns_zeros() -> None:
    z = ZeroOperator(in_structure=_structure)
    assert_array_equal(z(jnp.array([1.0, 2.0, 3.0], jnp.float32)), jnp.zeros(3, jnp.float32))


def test_out_structure_defaults_to_the_input_structure() -> None:
    z = ZeroOperator(in_structure=_structure)
    assert z.out_structure == _structure


def test_out_structure_is_honoured() -> None:
    z = ZeroOperator(in_structure=_structure, out_structure=_pytree_structure)
    assert jax.eval_shape(z.mv, z.in_structure) == _pytree_structure


@pytest.mark.parametrize('zero', [0, 0.0, np.float64(0), jnp.float32(0)])
def test_multiplication_by_a_concrete_zero_returns_a_zero_operator(zero) -> None:
    op = Op(in_structure=_structure)
    for z in (zero * op, op * zero):
        assert isinstance(z, ZeroOperator)
        assert z.in_structure == _structure
        assert structure_equal(z.out_structure, op.out_structure)


def test_multiplication_by_a_traced_zero_multiplies_the_values() -> None:
    op = Op(in_structure=_structure)
    x = jnp.arange(3, dtype=jnp.float32)

    @jax.jit
    def f(c):
        return (c * op)(x)

    assert_array_equal(f(0.0), jnp.zeros(3, jnp.float32))


def test_transpose_swaps_the_structures() -> None:
    z = ZeroOperator(in_structure=_structure, out_structure=_pytree_structure)
    assert isinstance(z.T, ZeroOperator)
    assert z.T.in_structure == _pytree_structure
    assert z.T.out_structure == _structure


def test_as_matrix() -> None:
    z = ZeroOperator(in_structure=_structure, out_structure=jax.ShapeDtypeStruct((2,), np.float32))
    assert_array_equal(z.as_matrix(), jnp.zeros((2, 3), jnp.float32))


def test_zero_operator_absorbs_the_right_operand() -> None:
    op = Op(in_structure=_structure)
    z = ZeroOperator(in_structure=_structure) @ op
    assert isinstance(z, ZeroOperator)
    assert z.in_structure == _structure


def test_zero_operator_absorbs_the_left_operand() -> None:
    op = Op(in_structure=_structure)
    z = op @ ZeroOperator(in_structure=_structure)
    assert isinstance(z, ZeroOperator)
    assert z.in_structure == _structure
    assert structure_equal(z.out_structure, op.out_structure)


def test_zero_operator_absorbs_a_composition() -> None:
    op = Op(in_structure=_structure)
    z = (op @ op) @ ZeroOperator(in_structure=_structure)
    assert isinstance(z, ZeroOperator)


def test_composition_with_a_zero_operator_reduces_to_zero() -> None:
    op = Op(in_structure=_structure)
    composition = CompositionOperator([op, ZeroOperator(in_structure=_structure)])
    reduced = composition.reduce()
    assert isinstance(reduced, ZeroOperator)
    assert reduced.in_structure == _structure
    assert structure_equal(reduced.out_structure, op.out_structure)


def test_addition_with_a_zero_operator_reduces_to_the_other_operand() -> None:
    op = Op(in_structure=_structure)
    assert (op + 0 * op).reduce() is op
    assert (0 * op + op).reduce() is op


def test_addition_of_zero_operators_reduces_to_one() -> None:
    z = ZeroOperator(in_structure=_structure)
    assert isinstance((z + z).reduce(), ZeroOperator)


def test_inverse_raises() -> None:
    z = ZeroOperator(in_structure=_structure)
    with pytest.raises(ValueError, match='not invertible'):
        _ = z.I


def test_square_zero_operator_tags() -> None:
    z = ZeroOperator(in_structure=_structure)
    assert z.is_square
    assert z.is_symmetric
    assert z.is_diagonal
    assert z.is_idempotent
    assert z.is_positive_semidefinite
    assert z.is_negative_semidefinite


def test_rectangular_zero_operator_has_no_tags() -> None:
    z = ZeroOperator(in_structure=_structure, out_structure=jax.ShapeDtypeStruct((2,), np.float32))
    assert not z.is_square
    assert not z.is_symmetric


def test_jit() -> None:
    z = ZeroOperator(in_structure=_structure)
    y = jax.jit(lambda op, x: op(x))(z, jnp.ones(3, jnp.float32))
    assert_array_equal(y, jnp.zeros(3, jnp.float32))
