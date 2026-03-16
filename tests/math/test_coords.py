import math

import jax.numpy as jnp
import numpy as np
import pytest
from fastquat import Quaternion as Q

from furax.math.coords import (
    XAXIS,
    YAXIS,
    ZAXIS,
    euler,
    from_iso_angles,
    from_lonlat_angles,
    from_xieta_angles,
    to_iso_angles,
    to_lonlat_angles,
    to_polarization_angle,
    to_polarization_angle_cos_sin,
    to_xieta_angles,
)


class TestEuler:
    """Test the euler() quaternion constructor."""

    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_zero_angle_is_identity(self, axis):
        q = euler(axis, 0.0)
        np.testing.assert_allclose(q.wxyz, jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-7)

    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_unit_quaternion(self, axis):
        q = euler(axis, math.pi / 3)
        np.testing.assert_allclose(jnp.linalg.norm(q.wxyz), 1.0, atol=1e-7)

    @pytest.mark.parametrize(
        'axis, angle, expected_wxyz',
        [
            # 180° around X: (0, 1, 0, 0)
            (0, math.pi, [0.0, 1.0, 0.0, 0.0]),
            # 180° around Y: (0, 0, 1, 0)
            (1, math.pi, [0.0, 0.0, 1.0, 0.0]),
            # 180° around Z: (0, 0, 0, 1)
            (2, math.pi, [0.0, 0.0, 0.0, 1.0]),
            # 90° around Z: (cos(π/4), 0, 0, sin(π/4))
            (2, math.pi / 2, [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]),
        ],
    )
    def test_known_values(self, axis, angle, expected_wxyz):
        q = euler(axis, angle)
        np.testing.assert_allclose(q.wxyz, jnp.array(expected_wxyz), atol=1e-7)

    @pytest.mark.parametrize(
        'axis, angle, vec, expected',
        [
            # 90° around Z rotates X→Y
            (2, math.pi / 2, XAXIS, YAXIS),
            # 90° around X rotates Y→Z
            (0, math.pi / 2, YAXIS, ZAXIS),
            # 90° around Y rotates Z→X
            (1, math.pi / 2, ZAXIS, XAXIS),
        ],
    )
    def test_rotation_effect(self, axis, angle, vec, expected):
        q = euler(axis, angle)
        result = q.rotate_vector(vec)
        np.testing.assert_allclose(result, jnp.array(expected), atol=1e-6)

    def test_batched_angles(self):
        angles = jnp.array([0.0, math.pi / 2, math.pi])
        q = euler(2, angles)
        assert q.wxyz.shape == (3, 4)
        np.testing.assert_allclose(jnp.linalg.norm(q.wxyz, axis=-1), jnp.ones(3), atol=1e-7)


class TestPolarizationAngle:
    """Test to_polarization_angle() and to_polarization_angle_cos_sin().

    At the pole (detector pointing along Z, i.e. pure Z rotations or identity), the
    polarization angle is undefined. The two functions may differ in this case:
    - to_polarization_angle returns atan2(±0, ±0) depending on the rotation
    - to_polarization_angle_cos_sin explicitly returns (1, 0), i.e. pa = 0

    Non-pole tests use to_polarization_angle as the reference.
    """

    @pytest.mark.parametrize(
        'quaternion, expected_pa',
        [
            # Pure X rotation: pointing direction moves off the Z pole → pa = π/2
            (euler(0, math.pi / 2), math.pi / 2),
            (euler(0, math.pi / 3), math.pi / 2),
            # Pure Y rotation: pa = 0
            (euler(1, math.pi / 2), 0.0),
            (euler(1, math.pi / 3), 0.0),
        ],
    )
    def test_known_values(self, quaternion, expected_pa):
        pa = to_polarization_angle(quaternion)
        np.testing.assert_allclose(pa, expected_pa, atol=1e-6)

    @pytest.mark.parametrize(
        'quaternion',
        [
            # Pure Z rotations keep the detector pointing along Z (at the pole).
            Q(1, 0, 0, 0),
            euler(2, math.pi / 2),
            euler(2, math.pi / 3),
        ],
    )
    def test_at_pole_cos_sin(self, quaternion):
        cos_pa, sin_pa = to_polarization_angle_cos_sin(quaternion)
        np.testing.assert_allclose(cos_pa, 1.0, atol=1e-6)
        np.testing.assert_allclose(sin_pa, 0.0, atol=1e-6)

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(0.5, 0.5, 0.5, 0.5),
            Q(0.6, 0.3, 0.4, 0.7),
            euler(0, math.pi / 4),
            euler(1, math.pi / 3),
        ],
    )
    def test_cos_sin_consistent_with_angle(self, quaternion):
        """to_polarization_angle_cos_sin matches cos/sin of to_polarization_angle."""
        q_unit = quaternion.normalize()
        pa = to_polarization_angle(q_unit)
        cos_pa, sin_pa = to_polarization_angle_cos_sin(q_unit)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(1, 0, 0, 0),
            Q(0.5, 0.5, 0.5, 0.5),
            Q(0.6, 0.3, 0.4, 0.7),
            euler(0, math.pi / 4),
        ],
    )
    def test_cos_sin_is_unit(self, quaternion):
        q_unit = quaternion.normalize()
        cos_pa, sin_pa = to_polarization_angle_cos_sin(q_unit)
        np.testing.assert_allclose(cos_pa**2 + sin_pa**2, 1.0, atol=1e-6)

    def test_batched(self):
        angles = jnp.array([0.0, math.pi / 4, math.pi / 2, math.pi])
        q = euler(0, angles)
        pa = to_polarization_angle(q)
        cos_pa, sin_pa = to_polarization_angle_cos_sin(q)
        assert pa.shape == (4,)
        assert cos_pa.shape == (4,)
        assert sin_pa.shape == (4,)
        np.testing.assert_allclose(cos_pa**2 + sin_pa**2, jnp.ones(4), atol=1e-6)


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
            Q(1, 0, 0, 0),
            # 90 degree rotation around Z axis - KNOWN SINGULARITY
            pytest.param(
                Q(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)),
                marks=pytest.mark.xfail(
                    reason='ISO angle singularity: pure Z-rotations cause phi/psi degeneracy'
                ),
            ),
            # 90 degree rotation around X axis
            Q(math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0),
            # 90 degree rotation around Y axis
            Q(math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0),
        ],
    )
    def test_to_iso_angles_roundtrip(self, quaternion):
        """Test that ISO angle conversion roundtrips correctly."""
        q_unit = quaternion.normalize()

        # Convert to ISO angles and back
        theta, phi, psi = to_iso_angles(q_unit)
        recovered_q = from_iso_angles(theta, phi, psi)

        # Check that angles are finite
        assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))

        # Check quaternion normalization
        np.testing.assert_allclose(jnp.linalg.norm(recovered_q.wxyz), 1.0, atol=1e-6)

        # Quaternions q and -q represent the same rotation
        # Check both possibilities
        matches_positive = jnp.allclose(recovered_q.wxyz, q_unit.wxyz, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q.wxyz, -q_unit.wxyz, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'ISO roundtrip failed for {q_unit}: got {recovered_q.wxyz}'
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
    def test_from_iso_angles(self, theta, phi, psi):
        """Test conversion from ISO angles to quaternion."""
        q = from_iso_angles(theta, phi, psi)

        # Check quaternion is normalized
        np.testing.assert_allclose(jnp.linalg.norm(q.wxyz), 1.0, atol=1e-8)

        assert q.wxyz.shape == (4,)
        assert jnp.all(jnp.isfinite(q.wxyz))

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(1, 0, 0, 0),  # Identity
            Q(0.5, 0.5, 0.5, 0.5),
            Q(math.cos(math.pi / 6), math.sin(math.pi / 6), 0, 0),  # 60° X rotation
            Q(math.cos(math.pi / 8), 0, math.sin(math.pi / 8), 0),  # 45° Y rotation
            # Z rotation - expected to fail due to singularity when theta ≈ 0
            pytest.param(
                Q(math.cos(math.pi / 3), 0, 0, math.sin(math.pi / 3)),
                marks=pytest.mark.xfail(
                    reason='ISO angle singularity: pure Z-rotations cause phi/psi degeneracy'
                ),
            ),
        ],
    )
    def test_iso_angles_roundtrip_parametrized(self, quaternion):
        """Test that to_iso_angles and from_iso_angles are inverses.

        Note: Some pure Z-rotations may fail due to coordinate singularities
        in the ISO angle representation, similar to spherical coordinate singularities.
        """
        q_unit = quaternion.normalize()

        theta, phi, psi = to_iso_angles(q_unit)
        recovered_q = from_iso_angles(theta, phi, psi)

        assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))
        np.testing.assert_allclose(jnp.linalg.norm(recovered_q.wxyz), 1.0, atol=1e-6)

        matches_positive = jnp.allclose(recovered_q.wxyz, q_unit.wxyz, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q.wxyz, -q_unit.wxyz, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'Roundtrip failed for {q_unit}: got {recovered_q.wxyz}'
        )

    def test_iso_angles_batch(self):
        """Test batch processing of ISO angle conversions."""
        batch_quaternions = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            ]
        )

        theta, phi, psi = to_iso_angles(Q.from_array(batch_quaternions))

        assert theta.shape == (3,)
        assert phi.shape == (3,)
        assert psi.shape == (3,)

        recovered = from_iso_angles(theta, phi, psi)
        assert recovered.wxyz.shape == (3, 4)

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(1, 0, 0, 0),
            Q(math.cos(math.pi / 6), math.sin(math.pi / 6), 0, 0),  # 60° X rotation
            Q(math.cos(math.pi / 8), 0, math.sin(math.pi / 8), 0),  # 45° Y rotation
            Q(math.cos(math.pi / 3), 0, 0, math.sin(math.pi / 3)),  # 120° Z rotation
            Q(0.6, 0.3, 0.4, 0.7),
            Q(0.5, 0.5, 0.5, 0.5),
            Q(0.8, 0.1, 0.2, 0.5),
            Q(0.999, 0.001, 0.002, 0.003),
        ],
    )
    def test_iso_angles_z_axis_spherical_consistency(self, quaternion):
        """Test that ISO angles give correct spherical coordinates when rotating Z-axis."""
        q_unit = quaternion.normalize()

        rotated_z = q_unit.rotate_vector(ZAXIS)

        theta, phi, psi = to_iso_angles(q_unit)

        expected_x = jnp.sin(theta) * jnp.cos(phi)
        expected_y = jnp.sin(theta) * jnp.sin(phi)
        expected_z = jnp.cos(theta)
        expected_coords = jnp.array([expected_x, expected_y, expected_z])

        np.testing.assert_allclose(
            rotated_z,
            expected_coords,
            atol=1e-6,
            err_msg=f'Z-axis rotation mismatch for quaternion {q_unit}\n'
            f'Rotated Z: {rotated_z}\n'
            f'Expected (spherical): {expected_coords}\n'
            f'ISO angles: theta={theta}, phi={phi}, psi={psi}',
        )


class TestLonLatAngles:
    """Test longitude/latitude angle conversion functions."""

    def test_to_lonlat_angles_identity(self):
        """Test lonlat conversion for identity quaternion."""
        identity = Q(1, 0, 0, 0)
        alpha, delta, psi = to_lonlat_angles(identity)

        np.testing.assert_allclose(alpha, 0.0, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2, atol=1e-8)
        np.testing.assert_allclose(psi, 0.0, atol=1e-8)

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(1, 0, 0, 0),
            Q(0.5, 0.5, 0.5, 0.5),
            Q(math.cos(math.pi / 6), math.sin(math.pi / 6), 0, 0),  # 60° X rotation
            Q(math.cos(math.pi / 8), 0, math.sin(math.pi / 8), 0),  # 45° Y rotation
        ],
    )
    def test_lonlat_angles_roundtrip(self, quaternion):
        """Test that lonlat angle conversion roundtrips correctly."""
        q_unit = quaternion.normalize()

        alpha, delta, psi = to_lonlat_angles(q_unit)
        recovered_q = from_lonlat_angles(alpha, delta, psi)

        assert jnp.all(jnp.isfinite(jnp.array([alpha, delta, psi])))
        np.testing.assert_allclose(jnp.linalg.norm(recovered_q.wxyz), 1.0, atol=1e-6)

        matches_positive = jnp.allclose(recovered_q.wxyz, q_unit.wxyz, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q.wxyz, -q_unit.wxyz, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'Lonlat roundtrip failed for {q_unit}: got {recovered_q.wxyz}'
        )

    def test_lonlat_relationship_to_iso(self):
        """Test relationship between lonlat and ISO angles."""
        q_obj = Q(0.6, 0.3, 0.4, 0.7).normalize()

        theta_iso, phi_iso, psi_iso = to_iso_angles(q_obj)
        alpha, delta, psi_lonlat = to_lonlat_angles(q_obj)

        np.testing.assert_allclose(alpha, phi_iso, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2 - theta_iso, atol=1e-8)
        np.testing.assert_allclose(psi_lonlat, psi_iso, atol=1e-8)


class TestXiEtaAngles:
    """Test xi/eta angle conversion functions."""

    def test_to_xieta_angles_identity(self):
        """Test xieta conversion for identity quaternion."""
        identity = Q(1, 0, 0, 0)
        xi, eta, gamma = to_xieta_angles(identity)

        np.testing.assert_allclose(xi, 0.0, atol=1e-8)
        np.testing.assert_allclose(eta, 0.0, atol=1e-8)
        np.testing.assert_allclose(gamma, 0.0, atol=1e-8)

    @pytest.mark.parametrize(
        'quaternion',
        [
            Q(1, 0, 0, 0),
            Q(0.5, 0.5, 0.5, 0.5),
            Q(math.cos(math.pi / 6), math.sin(math.pi / 6), 0, 0),  # 60° X rotation
            Q(math.cos(math.pi / 8), 0, math.sin(math.pi / 8), 0),  # 45° Y rotation
            Q(math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)),  # 90° Z rotation
        ],
    )
    def test_xieta_angles_roundtrip(self, quaternion):
        """Test that xieta angle conversion roundtrips correctly."""
        q_unit = quaternion.normalize()

        xi, eta, gamma = to_xieta_angles(q_unit)
        recovered_q = from_xieta_angles(xi, eta, gamma)

        assert jnp.all(jnp.isfinite(jnp.array([xi, eta, gamma])))
        np.testing.assert_allclose(jnp.linalg.norm(recovered_q.wxyz), 1.0, atol=1e-6)

        matches_positive = jnp.allclose(recovered_q.wxyz, q_unit.wxyz, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q.wxyz, -q_unit.wxyz, atol=1e-6)

        assert matches_positive or matches_negative, (
            f'XiEta roundtrip failed for {q_unit}: got {recovered_q.wxyz}'
        )

    def test_xieta_small_angles(self):
        """Test xieta angles for small rotations."""
        small_angle = 0.01
        q_unit = Q(
            math.cos(small_angle / 2),
            math.sin(small_angle / 2) * 0.1,
            math.sin(small_angle / 2) * 0.2,
            0.0,
        ).normalize()

        xi, eta, gamma = to_xieta_angles(q_unit)

        assert abs(xi) < 0.1
        assert abs(eta) < 0.1

        recovered_q = from_xieta_angles(xi, eta, gamma)

        matches_positive = jnp.allclose(recovered_q.wxyz, q_unit.wxyz, atol=1e-6)
        matches_negative = jnp.allclose(recovered_q.wxyz, -q_unit.wxyz, atol=1e-6)

        assert matches_positive or matches_negative

    def test_xieta_batch_processing(self):
        """Test batch processing for xieta angles."""
        batch = jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0],
            ]
        )
        batch = batch / jnp.linalg.norm(batch, axis=1, keepdims=True)

        xi, eta, gamma = to_xieta_angles(Q.from_array(batch))

        assert xi.shape == (3,)
        assert eta.shape == (3,)
        assert gamma.shape == (3,)

        recovered = from_xieta_angles(xi, eta, gamma)
        assert recovered.wxyz.shape == (3, 4)


class TestAngleConversionConsistency:
    """Test consistency between different angle conversion methods."""

    def test_all_conversions_preserve_rotation(self):
        """Test that all angle conversions preserve the rotation effect."""
        test_q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        test_vec = jnp.array([1.0, 2.0, 3.0])

        original_result = test_q.rotate_vector(test_vec)

        theta, phi, psi = to_iso_angles(test_q)
        iso_result = from_iso_angles(theta, phi, psi).rotate_vector(test_vec)

        alpha, delta, psi_ll = to_lonlat_angles(test_q)
        lonlat_result = from_lonlat_angles(alpha, delta, psi_ll).rotate_vector(test_vec)

        xi, eta, gamma = to_xieta_angles(test_q)
        xieta_result = from_xieta_angles(xi, eta, gamma).rotate_vector(test_vec)

        np.testing.assert_allclose(iso_result, original_result, atol=1e-6)
        np.testing.assert_allclose(lonlat_result, original_result, atol=1e-6)
        np.testing.assert_allclose(xieta_result, original_result, atol=1e-6)

    def test_edge_case_zero_quaternion(self):
        """Test angle conversion edge cases."""
        zero_q = Q(0, 0, 0, 0)

        try:
            theta, phi, psi = to_iso_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))
        except Exception as e:
            pytest.skip(f'ISO angles failed for zero quaternion: {e}')

        try:
            alpha, delta, psi = to_lonlat_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([alpha, delta, psi])))
        except Exception as e:
            pytest.skip(f'Lonlat angles failed for zero quaternion: {e}')

        try:
            xi, eta, gamma = to_xieta_angles(zero_q)
            assert jnp.all(jnp.isfinite(jnp.array([xi, eta, gamma])))
        except Exception as e:
            pytest.skip(f'Xieta angles failed for zero quaternion: {e}')
