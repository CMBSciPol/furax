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


# New tests for chunked functionality


def test_mask_operator_large_array_chunking():
    """Test that chunking works correctly for large arrays."""
    # Use size larger than default chunk_size (8192)
    n = 20000
    x = jnp.arange(n, dtype=jnp.float32)
    # Create alternating pattern mask
    mask = jnp.arange(n) % 2 == 0

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x))
    result = op(x)

    # Check that even indices are preserved, odd are zeroed
    assert_array_equal(result[::2], x[::2])
    assert_array_equal(result[1::2], jnp.zeros(n // 2))


def test_mask_operator_custom_chunk_size():
    """Test that custom chunk sizes work correctly."""
    n = 1000
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.arange(n) % 3 == 0

    # Test with different chunk sizes
    for chunk_size in [64, 256, 1024]:
        op = MaskOperator.from_boolean_mask(
            mask, in_structure=as_structure(x), chunk_size=chunk_size
        )
        result = op(x)
        expected = jnp.where(mask, x, 0)
        assert_array_equal(result, expected)


def test_mask_operator_chunking_matches_non_chunked():
    """Test that chunked and non-chunked results are identical."""
    n = 10000
    x = jnp.arange(n, dtype=jnp.float32) * 0.1
    mask = (jnp.arange(n) % 7) < 4  # More complex pattern

    # Small chunk size to force chunking
    op_chunked = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=512)

    # Large chunk size to avoid chunking
    op_full = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=20000)

    result_chunked = op_chunked(x)
    result_full = op_full(x)

    assert_array_equal(result_chunked, result_full)


def test_mask_operator_batched_chunking():
    """Test chunking with batched inputs."""
    batch_size = 8
    n = 15000
    x = jnp.arange(batch_size * n, dtype=jnp.float32).reshape(batch_size, n)

    # Different mask for each batch element
    mask = jnp.arange(batch_size * n).reshape(batch_size, n) % 5 != 0

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=2048)
    result = op(x)

    expected = jnp.where(mask, x, 0)
    assert_array_equal(result, expected)


def test_mask_operator_broadcast_with_chunking():
    """Test broadcasting with chunked processing."""
    batch_size = 4
    n = 12000
    x = jnp.arange(batch_size * n, dtype=jnp.float32).reshape(batch_size, n)

    # Broadcast mask across batch dimension
    mask = jnp.arange(n) % 3 == 0

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=1024)
    result = op(x)

    # Check each batch element
    for i in range(batch_size):
        expected_i = jnp.where(mask, x[i], 0)
        assert_array_equal(result[i], expected_i)


def test_mask_operator_chunk_size_boundary_conditions():
    """Test edge cases around chunk boundaries."""
    # Test when n is exactly chunk_size
    n = 8192
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.ones(n, dtype=bool)

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=8192)
    result = op(x)
    assert_array_equal(result, x)

    # Test when n is chunk_size + 1
    n = 8193
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.ones(n, dtype=bool)

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=8192)
    result = op(x)
    assert_array_equal(result, x)

    # Test when n is chunk_size - 1
    n = 8191
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.ones(n, dtype=bool)

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=8192)
    result = op(x)
    assert_array_equal(result, x)


def test_mask_operator_to_boolean_mask():
    """Test that to_boolean_mask() returns the original mask."""
    n = 1000
    original_mask = jnp.array([i % 5 != 2 for i in range(n)])
    x = jnp.zeros(n)

    op = MaskOperator.from_boolean_mask(original_mask, in_structure=as_structure(x))

    recovered_mask = op.to_boolean_mask()
    assert_array_equal(recovered_mask, original_mask)


def test_mask_operator_all_false():
    """Test behavior when all mask values are False."""
    n = 5000
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.zeros(n, dtype=bool)

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=1024)
    result = op(x)

    assert_array_equal(result, jnp.zeros(n))


def test_mask_operator_all_true():
    """Test behavior when all mask values are True."""
    n = 5000
    x = jnp.arange(n, dtype=jnp.float32)
    mask = jnp.ones(n, dtype=bool)

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=1024)
    result = op(x)

    assert_array_equal(result, x)


def test_mask_operator_dtype_validation():
    """Test that passing non-uint8 packed mask raises error."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    # Try to create with wrong dtype
    packed_mask = jnp.array([255, 0], dtype=jnp.int32)  # Wrong dtype

    with pytest.raises(ValueError, match='Expected an input array of unsigned byte data type'):
        _ = MaskOperator(packed_mask, in_structure=as_structure(x))


def test_mask_operator_multidimensional_leading_dims():
    """Test with multiple leading batch dimensions and chunking."""
    shape = (2, 3, 10000)
    x = jnp.arange(jnp.prod(jnp.array(shape)), dtype=jnp.float32).reshape(shape)

    # Create mask with same shape
    mask = (jnp.arange(jnp.prod(jnp.array(shape))).reshape(shape) % 11) < 7

    op = MaskOperator.from_boolean_mask(mask, in_structure=as_structure(x), chunk_size=2048)
    result = op(x)

    expected = jnp.where(mask, x, 0)
    assert_array_equal(result, expected)
