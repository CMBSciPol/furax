import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from furax import MaskOperator
from furax.tree import as_structure


def test_mask_operator_simple():
    """Test basic masking functionality."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask = jnp.array([True, False, True, True])
    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))

    # Test application of mask
    expected = jnp.array([1.0, 0.0, 3.0, 4.0])
    result = op(x)
    assert_array_equal(result, expected)


def test_mask_operator_broadcasting():
    """Test that broadcastable masks work correctly."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Broadcast mask across rows
    mask = jnp.array([True, False])
    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))

    expected = jnp.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    result = op(x)
    assert_array_equal(result, expected)


def test_mask_operator_non_broadcastable():
    """Test that non-broadcastable masks raise an error."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Mask shape (3,) cannot be broadcast to (3, 2)
    mask = jnp.array([True, False, True])

    with pytest.raises(
        ValueError, match='Boolean mask shape must be broadcastable to operator shape'
    ):
        _ = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))


def test_mask_operator_composition():
    """Test composition of mask operators via the InverseBinaryRule."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask1 = jnp.array([True, True, False, True])
    mask2 = jnp.array([True, False, True, True])

    op1 = MaskOperator.from_boolean_mask(mask1, in_structure=as_structure(x))
    op2 = MaskOperator.from_boolean_mask(mask2, in_structure=as_structure(x))

    # Apply rule to compose operators
    composed_op = op1 @ op2

    # The composed mask should be the AND of both masks
    expected = jnp.array([1.0, 0.0, 0.0, 4.0])
    result = composed_op(x)
    assert_array_equal(result, expected)


def test_mask_operator_zero_preserving():
    """Test that masked values are consistently zero."""
    x = jnp.array([1.0, -2.0, 3.0, -4.0])
    mask = jnp.array([False, True, False, True])
    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))

    # Test that masked values are exactly zero
    result = op(x)
    assert_array_equal(result[~mask], jnp.zeros_like(result[~mask]))
    # Test that unmasked values are preserved
    assert_array_equal(result[mask], x[mask])
