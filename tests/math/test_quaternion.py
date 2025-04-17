"""Tests for quaternion operations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from furax.math.quaternion import Quat, Vec3, qmul, qrot, qrot_xaxis, qrot_zaxis, to_rotation_matrix


def test_qmul_identity() -> None:
    """Test multiplication by identity quaternion."""
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    q = jnp.array([0.5, 0.5, 0.5, 0.5])

    result = qmul(identity, q)
    np.testing.assert_allclose(result, q)

    result = qmul(q, identity)
    np.testing.assert_allclose(result, q)


@pytest.mark.parametrize(
    'quaternion, inverse',
    [
        # Case 1: Basic unit quaternion
        (
            jnp.array([0.5, 0.5, 0.5, 0.5]),
            jnp.array([0.5, -0.5, -0.5, -0.5]),
        ),
        # Case 2: Identity quaternion (its own inverse)
        (
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        ),
        # Case 3: 90-degree rotation around X axis
        (
            jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
            jnp.array([math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0]),
        ),
        # Case 4: 90-degree rotation around Z axis
        (
            jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),
            jnp.array([math.cos(math.pi / 4), 0.0, 0.0, -math.sin(math.pi / 4)]),
        ),
    ],
)
def test_qmul_inverse(quaternion: Quat, inverse: Quat) -> None:
    """Test multiplication by inverse yields identity."""
    expected = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # Test q * q_inv = identity
    result = qmul(quaternion, inverse)
    np.testing.assert_allclose(result, expected, atol=1e-6)

    # Test q_inv * q = identity (commutativity doesn't hold for quaternions in general,
    # but q_inv * q should still yield identity)
    result_reversed = qmul(inverse, quaternion)
    np.testing.assert_allclose(result_reversed, expected, atol=1e-6)


@pytest.mark.parametrize(
    'shape1, shape2',
    [
        # Single quaternion cases
        ((4,), (4,)),
        ((4,), (3, 4)),
        ((2, 4), (4,)),
        # Batch quaternion cases
        ((3, 4), (3, 4)),
        ((1, 4), (3, 4)),
        ((3, 4), (1, 4)),
        # More complex batch shapes
        ((2, 3, 4), (2, 1, 4)),
        ((1, 3, 4), (5, 1, 4)),
    ],
)
def test_qmul_batch(shape1: tuple[int, ...], shape2: tuple[int, ...]) -> None:
    """Test batch quaternion multiplication with different shapes."""
    # Create random quaternion arrays with the given shapes
    qs1 = jnp.ones(shape1)
    qs2 = jnp.ones(shape2)

    # Get the result of qmul
    result = qmul(qs1, qs2)

    # Calculate the expected output shape using JAX's broadcast rules
    # The output should maintain the last dimension (4) for quaternions
    expected_shape = jnp.broadcast_shapes(shape1[:-1], shape2[:-1]) + (4,)

    # Check that the result has the expected shape
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    'quaternion, expected_matrix',
    [
        # Identity quaternion should give identity matrix
        (
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.eye(3),
        ),
        # Test 90-degree rotation around X axis
        (
            jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
            jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        ),
    ],
)
def test_to_rotation_matrix(quaternion: Quat, expected_matrix: Float[Array, '... 3 3']) -> None:
    """Test conversion from quaternion to rotation matrix."""
    result_matrix = to_rotation_matrix(quaternion)
    np.testing.assert_allclose(result_matrix, expected_matrix, atol=1e-8)


def test_qrot_identity() -> None:
    """Test rotating a vector with identity quaternion."""
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    vec = jnp.array([1.0, 2.0, 3.0])

    result = qrot(identity, vec)
    np.testing.assert_allclose(result, vec, atol=1e-8)


@pytest.mark.parametrize(
    'quaternion, vector, expected',
    [
        # Rotate 90 degrees around Z axis
        (
            jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
        ),
        # Rotate 90 degrees around X axis
        (
            jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ),
    ],
)
def test_qrot_90_degrees(quaternion: Quat, vector: Vec3, expected: Vec3) -> None:
    """Test rotating a vector by 90 degrees."""
    result = qrot(quaternion, vector)
    np.testing.assert_allclose(result, expected, atol=1e-8)


@pytest.mark.parametrize(
    'q_shape, v_shape',
    [
        # Shape of quaternions, shape of vectors
        ((2, 4), (2, 3)),  # Matching batch dimension
        ((1, 4), (3, 3)),  # Broadcasting quaternion batch
        ((2, 4), (1, 3)),  # Broadcasting vector batch
        ((3, 2, 4), (3, 2, 3)),  # Multiple batch dimensions
        ((3, 1, 4), (1, 5, 3)),  # Complex broadcasting
    ],
)
def test_qrot_batch(q_shape: tuple[int, ...], v_shape: tuple[int, ...]) -> None:
    """Test batch vector rotation."""
    # Create dummy arrays with the specified shapes
    qs = jnp.ones(q_shape)
    vecs = jnp.ones(v_shape)

    # Compute result
    results = qrot(qs, vecs)

    # Calculate expected output shape using JAX's broadcast rules
    # The output should have the batch dimensions broadcasted and maintain the vector dimension (3)
    expected_shape = jnp.broadcast_shapes(q_shape[:-1], v_shape[:-1]) + (3,)

    # Check that the result has the expected shape
    assert results.shape == expected_shape


@pytest.mark.parametrize(
    'quaternion, expected',
    [
        # Identity quaternion
        (
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 0.0]),
        ),
        # 90 degree rotation around Y axis
        (
            jnp.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]),
            jnp.array([0.0, 0.0, -1.0]),
        ),
    ],
)
def test_qrot_xaxis(quaternion: Quat, expected: Vec3) -> None:
    """Test rotating the X axis by quaternion."""
    result = qrot_xaxis(quaternion)
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    'quaternion, expected',
    [
        # Identity quaternion
        (
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ),
        # 90 degree rotation around X axis
        (
            jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
            jnp.array([0.0, -1.0, 0.0]),
        ),
    ],
)
def test_qrot_zaxis(quaternion: Quat, expected: Vec3) -> None:
    """Test rotating the Z axis by quaternion."""
    result = qrot_zaxis(quaternion)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_zero_quaternion() -> None:
    """Test behavior with zero quaternion (edge case)."""
    zero_q = jnp.zeros(4)

    # Rotating with zero quaternion should return the original value
    result_xaxis = qrot_xaxis(zero_q)
    np.testing.assert_allclose(result_xaxis, jnp.array([1.0, 0.0, 0.0]))

    result_zaxis = qrot_zaxis(zero_q)
    np.testing.assert_allclose(result_zaxis, jnp.array([0.0, 0.0, 1.0]))

    vec = jnp.array([1.0, 2.0, 3.0])
    result_rot = qrot(zero_q, vec)
    np.testing.assert_allclose(result_rot, vec)


def test_jit_compatibility():
    """Test that functions work with JAX JIT compilation."""
    q1 = jnp.array([0.5, 0.5, 0.5, 0.5])
    q2 = jnp.array([0.0, 1.0, 0.0, 0.0])
    vec = jnp.array([1.0, 0.0, 0.0])

    # Test all functions with JIT
    jitted_qmul = jax.jit(qmul)
    jitted_qrot = jax.jit(lambda q, v: qrot(q, v))
    jitted_qrot_xaxis = jax.jit(qrot_xaxis)
    jitted_qrot_zaxis = jax.jit(qrot_zaxis)

    # Execute the JIT-compiled functions
    result1 = jitted_qmul(q1, q2)
    result2 = jitted_qrot(q1, vec)
    result3 = jitted_qrot_xaxis(q1)
    result4 = jitted_qrot_zaxis(q1)

    # Compare with non-JITted versions
    np.testing.assert_allclose(result1, qmul(q1, q2))
    np.testing.assert_allclose(result2, qrot(q1, vec))
    np.testing.assert_allclose(result3, qrot_xaxis(q1))
    np.testing.assert_allclose(result4, qrot_zaxis(q1))


def test_quaternion_composition():
    """Test that composing rotations works correctly."""
    # 90 degree rotations around Z and X
    q_z_90 = jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    q_x_90 = jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])

    # Compose the rotations
    q_composed = qmul(q_x_90, q_z_90)

    # Apply composed rotation to X-axis
    # Rotating X by 90° around Z gives Y, then rotating Y by 90° around X gives Z
    result = qrot_xaxis(q_composed)
    expected = jnp.array([0.0, 0.0, 1.0])

    np.testing.assert_allclose(result, expected, atol=1e-8)
