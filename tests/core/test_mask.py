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

    with pytest.raises(ValueError, match='Boolean mask shape must be broadcastable to leaf shape'):
        _ = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))


def test_mask_operator_composition():
    """Test composition of mask operators."""
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


def test_mask_operator_pytree_single_mask():
    """Test single mask broadcast to all leaves of a pytree input."""
    x = {'a': jnp.array([1.0, 2.0, 3.0, 4.0]), 'b': jnp.array([5.0, 6.0, 7.0, 8.0])}
    mask = jnp.array([True, False, True, True])
    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))

    result = op(x)
    assert_array_equal(result['a'], jnp.array([1.0, 0.0, 3.0, 4.0]))
    assert_array_equal(result['b'], jnp.array([5.0, 0.0, 7.0, 8.0]))


def test_mask_operator_pytree_of_masks():
    """Test pytree of masks applied to matching pytree input."""
    x = {'a': jnp.array([1.0, 2.0, 3.0, 4.0]), 'b': jnp.array([5.0, 6.0, 7.0, 8.0])}
    masks = {
        'a': jnp.array([True, False, True, True]),
        'b': jnp.array([False, True, False, True]),
    }
    op = MaskOperator.from_boolean_mask(masks, in_structure=as_structure(x))

    result = op(x)
    assert_array_equal(result['a'], jnp.array([1.0, 0.0, 3.0, 4.0]))
    assert_array_equal(result['b'], jnp.array([0.0, 6.0, 0.0, 8.0]))


def test_mask_operator_pytree_composition_single_mask():
    """Test composition of single-mask pytree operators."""
    x = {'a': jnp.array([1.0, 2.0, 3.0, 4.0]), 'b': jnp.array([5.0, 6.0, 7.0, 8.0])}
    struct = as_structure(x)

    mask1 = jnp.array([True, True, False, True])
    mask2 = jnp.array([True, False, True, True])

    op1 = MaskOperator.from_boolean_mask(mask1, in_structure=struct)
    op2 = MaskOperator.from_boolean_mask(mask2, in_structure=struct)

    composed_op = op1 @ op2
    result = composed_op(x)
    assert_array_equal(result['a'], jnp.array([1.0, 0.0, 0.0, 4.0]))
    assert_array_equal(result['b'], jnp.array([5.0, 0.0, 0.0, 8.0]))


def test_mask_operator_pytree_composition_pytree_masks():
    """Test composition of pytree-of-masks operators."""
    x = {'a': jnp.array([1.0, 2.0, 3.0, 4.0]), 'b': jnp.array([5.0, 6.0, 7.0, 8.0])}
    struct = as_structure(x)

    masks1 = {
        'a': jnp.array([True, True, False, True]),
        'b': jnp.array([True, True, True, False]),
    }
    masks2 = {
        'a': jnp.array([True, False, True, True]),
        'b': jnp.array([False, True, True, True]),
    }

    op1 = MaskOperator.from_boolean_mask(masks1, in_structure=struct)
    op2 = MaskOperator.from_boolean_mask(masks2, in_structure=struct)

    composed_op = op1 @ op2
    result = composed_op(x)
    # AND of masks per leaf
    assert_array_equal(result['a'], jnp.array([1.0, 0.0, 0.0, 4.0]))
    assert_array_equal(result['b'], jnp.array([0.0, 6.0, 7.0, 0.0]))


def test_mask_operator_pytree_to_boolean_mask():
    """Test that to_boolean_mask returns a pytree of boolean masks."""
    x = {'a': jnp.array([1.0, 2.0, 3.0]), 'b': jnp.array([4.0, 5.0, 6.0])}
    masks = {
        'a': jnp.array([True, False, True]),
        'b': jnp.array([False, True, False]),
    }
    op = MaskOperator.from_boolean_mask(masks, in_structure=as_structure(x))

    recovered = op.to_boolean_mask()
    assert isinstance(recovered, dict)
    assert_array_equal(recovered['a'], jnp.array([True, False, True]))
    assert_array_equal(recovered['b'], jnp.array([False, True, False]))
