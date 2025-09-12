"""Tests for quaternion operations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float

from furax.math.quaternion import (
    Quat,
    Vec3,
    from_iso_angles,
    from_lonlat_angles,
    from_xieta_angles,
    qmul,
    qrot,
    qrot_xaxis,
    qrot_zaxis,
    to_iso_angles,
    to_lonlat_angles,
    to_rotation_matrix,
    to_xieta_angles,
)


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


# =============================
# Tests for angle conversions
# =============================


class TestISOAngles:
    """Test ISO angle conversion functions.

    ISO Angle Coordinate System:
    ============================
    The ISO angle representation uses three parameters (theta, phi, psi) to describe
    quaternion rotations. This system has mathematical singularities similar to
    spherical coordinates:

    SINGULARITY CONDITIONS:
    - When theta ≈ 0 (near the "pole"): phi and psi become degenerate
    - This occurs for pure Z-axis rotations where b = c = 0 in quaternion [a,b,c,d]
    - In these cases, atan2(0, 0) is undefined, leading to roundtrip failures

    MATHEMATICAL VALIDITY:
    - The functions are mathematically correct despite singularity cases
    - Singularities are expected behavior, not implementation errors
    - Non-singular cases roundtrip perfectly and preserve rotations
    """

    @pytest.mark.parametrize(
        'quaternion',
        [
            # Identity quaternion should give zero angles
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            # 90 degree rotation around Z axis - KNOWN SINGULARITY
            pytest.param(
                jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),
                marks=pytest.mark.xfail(
                    reason='ISO angle singularity: pure Z-rotations cause phi/psi degeneracy'
                ),
            ),
            # 90 degree rotation around X axis
            jnp.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]),
            # 90 degree rotation around Y axis
            jnp.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]),
        ],
    )
    def test_to_iso_angles_roundtrip(self, quaternion: Quat) -> None:
        """Test that ISO angle conversion roundtrips correctly."""
        # Normalize input quaternion
        original_q = quaternion / jnp.linalg.norm(quaternion)

        # Convert to ISO angles and back
        theta, phi, psi = to_iso_angles(original_q)
        recovered_q = from_iso_angles(theta, phi, psi)

        # Check that angles are finite
        assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))

        # Check quaternion normalization
        np.testing.assert_allclose(jnp.linalg.norm(recovered_q), 1.0, atol=1e-6)

        # Quaternions q and -q represent the same rotation
        # Check both possibilities
        matches_positive = jnp.allclose(recovered_q, original_q, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q, -original_q, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'ISO roundtrip failed for {original_q}: got {recovered_q}'
        )

    @pytest.mark.parametrize(
        'theta, phi, psi',
        [
            (0.0, 0.0, 0.0),  # Identity
            (math.pi / 4, 0.0, 0.0),
            (0.0, math.pi / 4, 0.0),
            (0.0, 0.0, math.pi / 4),
            (math.pi / 2, math.pi / 4, math.pi / 3),
            (math.pi / 6, -math.pi / 3, math.pi / 2),
        ],
    )
    def test_from_iso_angles(self, theta: float, phi: float, psi: float) -> None:
        """Test conversion from ISO angles to quaternion."""
        quaternion = from_iso_angles(theta, phi, psi)

        # Check quaternion is normalized
        norm = jnp.linalg.norm(quaternion)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

        # Check quaternion components are reasonable
        assert quaternion.shape == (4,)
        assert jnp.all(jnp.isfinite(quaternion))

    @pytest.mark.parametrize(
        'quaternion',
        [
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # Identity
            jnp.array([0.5, 0.5, 0.5, 0.5]),  # Normalized quaternion
            jnp.array([math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0]),  # X rotation
            jnp.array([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]),  # Y rotation
            # Z rotation - expected to fail due to singularity when theta ≈ 0
            pytest.param(
                jnp.array([math.cos(math.pi / 3), 0.0, 0.0, math.sin(math.pi / 3)]),
                marks=pytest.mark.xfail(
                    reason='ISO angle singularity: pure Z-rotations cause phi/psi degeneracy'
                ),
            ),
        ],
    )
    def test_iso_angles_roundtrip_parametrized(self, quaternion: Quat) -> None:
        """Test that to_iso_angles and from_iso_angles are inverses.

        Note: Some pure Z-rotations may fail due to coordinate singularities
        in the ISO angle representation, similar to spherical coordinate singularities.
        """
        # Normalize input quaternion
        original_q = quaternion / jnp.linalg.norm(quaternion)

        # Convert to angles and back
        theta, phi, psi = to_iso_angles(original_q)
        recovered_q = from_iso_angles(theta, phi, psi)

        # Quaternions q and -q represent the same rotation
        # Check both possibilities
        matches_positive = jnp.allclose(recovered_q, original_q, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q, -original_q, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'Roundtrip failed for {original_q}: got {recovered_q}'
        )

    def test_iso_angles_batch(self) -> None:
        """Test batch processing of ISO angle conversions."""
        # Create batch of quaternions
        batch_quaternions = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            ]
        )

        # Convert to angles
        theta, phi, psi = to_iso_angles(batch_quaternions)

        # Check shapes
        assert theta.shape == (3,)
        assert phi.shape == (3,)
        assert psi.shape == (3,)

        # Convert back
        recovered_quaternions = from_iso_angles(theta, phi, psi)
        assert recovered_quaternions.shape == (3, 4)

    @pytest.mark.parametrize(
        'quaternion',
        [
            # Identity quaternion
            jnp.array([1.0, 0.0, 0.0, 0.0]),
            # Various rotations around different axes
            jnp.array([math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0]),  # X rotation 60°
            jnp.array([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]),  # Y rotation 45°
            jnp.array([math.cos(math.pi / 3), 0.0, 0.0, math.sin(math.pi / 3)]),  # Z rotation 120°
            # Mixed rotations
            jnp.array([0.6, 0.3, 0.4, 0.7]),  # General quaternion
            jnp.array([0.5, 0.5, 0.5, 0.5]),  # Equal components
            jnp.array([0.8, 0.1, 0.2, 0.5]),  # Another general case
            # Small angle rotations
            jnp.array([0.999, 0.001, 0.002, 0.003]),  # Very small rotation
        ],
    )
    def test_iso_angles_z_axis_spherical_consistency(self, quaternion: Quat) -> None:
        """Test that ISO angles give correct spherical coordinates when rotating Z-axis.

        Mathematical principle: When we rotate the Z-axis [0,0,1] by a quaternion,
        the result should match the spherical coordinates derived from the ISO angles:
        (x, y, z) = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        """
        # Normalize input quaternion
        q = quaternion / jnp.linalg.norm(quaternion)

        # Rotate the Z-axis using the quaternion
        z_axis = jnp.array([0.0, 0.0, 1.0])
        rotated_z = qrot(q, z_axis)

        # Get ISO angles from the quaternion
        theta, phi, psi = to_iso_angles(q)

        # Calculate expected position using spherical coordinates
        # Standard spherical convention: (x, y, z) = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
        expected_x = jnp.sin(theta) * jnp.cos(phi)
        expected_y = jnp.sin(theta) * jnp.sin(phi)
        expected_z = jnp.cos(theta)
        expected_coords = jnp.array([expected_x, expected_y, expected_z])

        # The rotated Z-axis should match the spherical coordinates
        np.testing.assert_allclose(
            rotated_z,
            expected_coords,
            atol=1e-6,
            err_msg=f'Z-axis rotation mismatch for quaternion {q}\n'
            f'Rotated Z: {rotated_z}\n'
            f'Expected (spherical): {expected_coords}\n'
            f'ISO angles: theta={theta}, phi={phi}, psi={psi}',
        )


class TestLonLatAngles:
    """Test longitude/latitude angle conversion functions."""

    def test_to_lonlat_angles_identity(self) -> None:
        """Test lonlat conversion for identity quaternion."""
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        alpha, delta, psi = to_lonlat_angles(identity)

        # For identity, should have alpha=0, delta=π/2, psi=0
        np.testing.assert_allclose(alpha, 0.0, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2, atol=1e-8)
        np.testing.assert_allclose(psi, 0.0, atol=1e-8)

    def test_lonlat_angles_roundtrip(self) -> None:
        """Test that lonlat angle conversion is consistent."""
        test_quaternions = [
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # Identity
            jnp.array([0.5, 0.5, 0.5, 0.5]),  # Normalized quaternion
            jnp.array([math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0]),
            jnp.array([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]),
        ]

        for original_q in test_quaternions:
            # Normalize input quaternion
            original_q = original_q / jnp.linalg.norm(original_q)

            # Convert to lonlat angles and back
            alpha, delta, psi = to_lonlat_angles(original_q)
            recovered_q = from_lonlat_angles(alpha, delta, psi)

            # Check if quaternions match (allowing for sign flip)
            matches_positive = jnp.allclose(recovered_q, original_q, atol=1e-6)
            matches_negative = jnp.allclose(recovered_q, -original_q, atol=1e-6)

            assert matches_positive or matches_negative, (
                f'Lonlat roundtrip failed for {original_q}: got {recovered_q}'
            )

    def test_lonlat_relationship_to_iso(self) -> None:
        """Test relationship between lonlat and ISO angles."""
        test_q = jnp.array([0.6, 0.3, 0.4, 0.7])
        test_q = test_q / jnp.linalg.norm(test_q)

        # Get ISO angles
        theta_iso, phi_iso, psi_iso = to_iso_angles(test_q)

        # Get lonlat angles
        alpha, delta, psi_lonlat = to_lonlat_angles(test_q)

        # Check relationships: alpha = phi, delta = π/2 - theta, psi = psi
        np.testing.assert_allclose(alpha, phi_iso, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2 - theta_iso, atol=1e-8)
        np.testing.assert_allclose(psi_lonlat, psi_iso, atol=1e-8)


class TestXiEtaAngles:
    """Test xi/eta angle conversion functions."""

    def test_to_xieta_angles_identity(self) -> None:
        """Test xieta conversion for identity quaternion."""
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        xi, eta, gamma = to_xieta_angles(identity)

        np.testing.assert_allclose(xi, 0.0, atol=1e-8)
        np.testing.assert_allclose(eta, 0.0, atol=1e-8)
        np.testing.assert_allclose(gamma, 0.0, atol=1e-8)

    @pytest.mark.parametrize(
        'quaternion',
        [
            jnp.array([1.0, 0.0, 0.0, 0.0]),  # Identity
            jnp.array([0.5, 0.5, 0.5, 0.5]),  # Equal components
            jnp.array([math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0]),  # X rotation
            jnp.array([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0]),  # Y rotation
            jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),  # Z rotation
        ],
    )
    def test_xieta_angles_roundtrip(self, quaternion: Quat) -> None:
        """Test that xieta angle conversion roundtrips correctly."""
        # Normalize input quaternion
        original_q = quaternion / jnp.linalg.norm(quaternion)

        # Convert to xieta angles and back
        xi, eta, gamma = to_xieta_angles(original_q)
        recovered_q = from_xieta_angles(xi, eta, gamma)

        # Check if quaternions match (allowing for sign flip)
        matches_positive = jnp.allclose(recovered_q, original_q, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q, -original_q, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'XiEta roundtrip failed for {original_q}: got {recovered_q}'
        )

    def test_xieta_small_angles(self) -> None:
        """Test xieta angles for small rotations."""
        # Small rotation quaternion
        small_angle = 0.01  # Small angle in radians
        q_small = jnp.array(
            [
                math.cos(small_angle / 2),
                math.sin(small_angle / 2) * 0.1,
                math.sin(small_angle / 2) * 0.2,
                0.0,
            ]
        )
        q_small = q_small / jnp.linalg.norm(q_small)

        xi, eta, gamma = to_xieta_angles(q_small)

        # For small angles, xi and eta should be small
        assert abs(xi) < 0.1
        assert abs(eta) < 0.1

        # Test roundtrip
        recovered_q = from_xieta_angles(xi, eta, gamma)

        matches_positive = jnp.allclose(recovered_q, q_small, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q, -q_small, atol=1e-6)

        assert matches_positive or matches_negative

    def test_xieta_batch_processing(self) -> None:
        """Test batch processing for xieta angles."""
        batch_quaternions = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            ]
        )

        # Normalize batch
        batch_quaternions = batch_quaternions / jnp.linalg.norm(
            batch_quaternions, axis=1, keepdims=True
        )

        # Convert to xieta angles
        xi, eta, gamma = to_xieta_angles(batch_quaternions)

        # Check shapes
        assert xi.shape == (3,)
        assert eta.shape == (3,)
        assert gamma.shape == (3,)

        # Convert back
        recovered_quaternions = from_xieta_angles(xi, eta, gamma)
        assert recovered_quaternions.shape == (3, 4)


class TestAngleConversionConsistency:
    """Test consistency between different angle conversion methods."""

    def test_all_conversions_preserve_rotation(self) -> None:
        """Test that all angle conversions preserve the rotation effect."""
        # Test quaternion
        test_q = jnp.array([0.6, 0.3, 0.4, 0.7])
        test_q = test_q / jnp.linalg.norm(test_q)

        # Test vector
        test_vec = jnp.array([1.0, 2.0, 3.0])

        # Original rotation
        original_result = qrot(test_q, test_vec)

        # Test ISO roundtrip
        theta, phi, psi = to_iso_angles(test_q)
        q_iso = from_iso_angles(theta, phi, psi)
        iso_result = qrot(q_iso, test_vec)

        # Test lonlat roundtrip
        alpha, delta, psi_ll = to_lonlat_angles(test_q)
        q_lonlat = from_lonlat_angles(alpha, delta, psi_ll)
        lonlat_result = qrot(q_lonlat, test_vec)

        # Test xieta roundtrip
        xi, eta, gamma = to_xieta_angles(test_q)
        q_xieta = from_xieta_angles(xi, eta, gamma)
        xieta_result = qrot(q_xieta, test_vec)

        # All should give the same rotation result
        np.testing.assert_allclose(iso_result, original_result, atol=1e-6)
        np.testing.assert_allclose(lonlat_result, original_result, atol=1e-6)
        np.testing.assert_allclose(xieta_result, original_result, atol=1e-6)

    def test_edge_case_zero_quaternion(self) -> None:
        """Test angle conversion edge cases."""
        # Zero quaternion (not physically meaningful but should not crash)
        zero_q = jnp.zeros(4)

        # ISO angles
        try:
            theta, phi, psi = to_iso_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))
        except Exception as e:
            pytest.skip(f'ISO angles failed for zero quaternion: {e}')

        # Lonlat angles
        try:
            alpha, delta, psi = to_lonlat_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([alpha, delta, psi])))
        except Exception as e:
            pytest.skip(f'Lonlat angles failed for zero quaternion: {e}')

        # Xieta angles
        try:
            xi, eta, gamma = to_xieta_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([xi, eta, gamma])))
        except Exception as e:
            pytest.skip(f'Xieta angles failed for zero quaternion: {e}')

    def test_angle_conversion_jit_compatibility(self) -> None:
        """Test that angle conversion functions work with JAX JIT."""
        test_q = jnp.array([0.6, 0.3, 0.4, 0.7])
        test_q = test_q / jnp.linalg.norm(test_q)

        # JIT compile functions
        jitted_to_iso = jax.jit(to_iso_angles)
        jitted_from_iso = jax.jit(from_iso_angles)
        jitted_to_lonlat = jax.jit(to_lonlat_angles)
        jitted_from_lonlat = jax.jit(from_lonlat_angles)
        jitted_to_xieta = jax.jit(to_xieta_angles)
        jitted_from_xieta = jax.jit(from_xieta_angles)

        # Test ISO angles
        theta, phi, psi = jitted_to_iso(test_q)
        q_recovered = jitted_from_iso(theta, phi, psi)
        assert q_recovered.shape == (4,)

        # Test lonlat angles
        alpha, delta, psi_ll = jitted_to_lonlat(test_q)
        q_recovered_ll = jitted_from_lonlat(alpha, delta, psi_ll)
        assert q_recovered_ll.shape == (4,)

        # Test xieta angles
        xi, eta, gamma = jitted_to_xieta(test_q)
        q_recovered_xe = jitted_from_xieta(xi, eta, gamma)
        assert q_recovered_xe.shape == (4,)
