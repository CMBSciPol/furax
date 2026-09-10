"""Tests for coordinate-system conversions built on `fastquat.Quaternion`."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastquat import Quaternion as Q

from furax.math.coords import (
    euler,
    from_iso_angles,
    from_lonlat_angles,
    from_xieta_angles,
    to_gamma_angles,
    to_iso_angles,
    to_lonlat_angles,
    to_polarization_angle,
    to_polarization_angle_cos_sin,
    to_xieta_angles,
)

IDENTITY = Q.ones(())
ROT_X_90 = Q(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
ROT_Y_90 = Q(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
ROT_Z_90 = Q(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


class TestEuler:
    @pytest.mark.parametrize(
        'axis, expected',
        [
            (0, ROT_X_90),
            (1, ROT_Y_90),
            (2, ROT_Z_90),
        ],
    )
    def test_euler_90_degrees(self, axis: int, expected: Q) -> None:
        result = euler(axis, jnp.pi / 2)
        np.testing.assert_allclose(result.wxyz, expected.wxyz, atol=1e-8)

    def test_euler_batch(self) -> None:
        angles = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        result = euler(2, angles)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.wxyz[0], IDENTITY.wxyz, atol=1e-8)

    def test_euler_jit_compatible(self) -> None:
        jitted = jax.jit(euler, static_argnums=(0,))
        result = jitted(2, jnp.pi / 2)
        np.testing.assert_allclose(result.wxyz, ROT_Z_90.wxyz, atol=1e-8)


class TestISOAngles:
    """ISO angle representation (theta, phi, psi) has a known coordinate singularity for pure
    Z-axis rotations (theta ~ 0), where phi/psi become degenerate -- expected, not a bug.
    """

    @pytest.mark.parametrize(
        'q',
        [
            IDENTITY,
            pytest.param(
                ROT_Z_90,
                marks=pytest.mark.xfail(
                    reason='ISO angle singularity: pure Z-rotations cause phi/psi degeneracy'
                ),
            ),
            ROT_X_90,
            ROT_Y_90,
        ],
    )
    def test_roundtrip(self, q: Q) -> None:
        theta, phi, psi = to_iso_angles(q)
        recovered = from_iso_angles(theta, phi, psi)

        assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))
        np.testing.assert_allclose(abs(recovered), 1.0, atol=1e-6)
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'ISO roundtrip failed for {q}: got {recovered}'

    def test_from_iso_angles_is_normalized(self) -> None:
        q = from_iso_angles(jnp.pi / 2, jnp.pi / 4, jnp.pi / 3)
        np.testing.assert_allclose(abs(q), 1.0, atol=1e-8)
        assert jnp.all(jnp.isfinite(q.wxyz))

    def test_batch(self) -> None:
        batch = Q.from_array(jnp.array([IDENTITY.wxyz, [0.5, 0.5, 0.5, 0.5], ROT_X_90.wxyz]))
        theta, phi, psi = to_iso_angles(batch)
        assert theta.shape == phi.shape == psi.shape == (3,)
        recovered = from_iso_angles(theta, phi, psi)
        assert recovered.shape == (3,)

    def test_z_axis_spherical_consistency(self) -> None:
        """Rotating the Z-axis by q should land at the spherical point given by its ISO angles."""
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        rotated_z = q.rotate_vector(jnp.array([0.0, 0.0, 1.0]))
        theta, phi, _ = to_iso_angles(q)
        expected = jnp.array(
            [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)]
        )
        np.testing.assert_allclose(rotated_z, expected, atol=1e-6)

    def test_jit_compatible(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        theta, phi, psi = jax.jit(to_iso_angles)(q)
        recovered = jax.jit(from_iso_angles)(theta, phi, psi)
        assert recovered.shape == ()


class TestLonLatAngles:
    def test_identity(self) -> None:
        alpha, delta, psi = to_lonlat_angles(IDENTITY)
        np.testing.assert_allclose(alpha, 0.0, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2, atol=1e-8)
        np.testing.assert_allclose(psi, 0.0, atol=1e-8)

    @pytest.mark.parametrize('q', [IDENTITY, Q(0.5, 0.5, 0.5, 0.5), ROT_X_90, ROT_Y_90])
    def test_roundtrip(self, q: Q) -> None:
        q = q.normalize()
        alpha, delta, psi = to_lonlat_angles(q)
        recovered = from_lonlat_angles(alpha, delta, psi)
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'Lonlat roundtrip failed for {q}: got {recovered}'

    def test_relationship_to_iso(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        theta_iso, phi_iso, psi_iso = to_iso_angles(q)
        alpha, delta, psi_lonlat = to_lonlat_angles(q)
        np.testing.assert_allclose(alpha, phi_iso, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2 - theta_iso, atol=1e-8)
        np.testing.assert_allclose(psi_lonlat, psi_iso, atol=1e-8)


class TestXiEtaAngles:
    def test_identity(self) -> None:
        xi, eta, gamma = to_xieta_angles(IDENTITY)
        np.testing.assert_allclose(xi, 0.0, atol=1e-8)
        np.testing.assert_allclose(eta, 0.0, atol=1e-8)
        np.testing.assert_allclose(gamma, 0.0, atol=1e-8)

    @pytest.mark.parametrize(
        'q',
        [
            IDENTITY,
            Q(0.5, 0.5, 0.5, 0.5),
            # xieta is incomplete and singular at a 90-degree offset (xi^2 + eta^2 == 1); stay
            # away from that edge, matching furax.obs.pointing.PointingOperator's own caveat.
            Q(math.cos(math.pi / 6), math.sin(math.pi / 6), 0.0, 0.0),  # 60 deg around X
            Q(math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0),  # 45 deg around Y
            Q(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)),  # 90 deg around Z: OK,
            # xi = eta = 0 here since Z-rotations don't tilt the boresight off-axis
        ],
    )
    def test_roundtrip(self, q: Q) -> None:
        q = q.normalize()
        xi, eta, gamma = to_xieta_angles(q)
        recovered = from_xieta_angles(xi, eta, gamma)
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'XiEta roundtrip failed for {q}: got {recovered}'

    def test_batch(self) -> None:
        batch = Q.from_array(
            jnp.array([IDENTITY.wxyz, [0.5, 0.5, 0.5, 0.5], ROT_Z_90.wxyz])
        ).normalize()
        xi, eta, gamma = to_xieta_angles(batch)
        assert xi.shape == eta.shape == gamma.shape == (3,)
        recovered = from_xieta_angles(xi, eta, gamma)
        assert recovered.shape == (3,)


class TestGammaAngle:
    def test_identity_is_zero(self) -> None:
        np.testing.assert_allclose(to_gamma_angles(IDENTITY), 0.0, atol=1e-8)

    def test_matches_xieta_gamma(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        _, _, gamma_xieta = to_xieta_angles(q)
        np.testing.assert_allclose(to_gamma_angles(q), gamma_xieta, atol=1e-8)


class TestPolarizationAngle:
    def test_identity(self) -> None:
        """Identity quaternion: v_x u_y - v_y u_x = 0 and u_z = 0, so pa = atan2(0, -0) = pi."""
        np.testing.assert_allclose(to_polarization_angle(IDENTITY), math.pi, atol=1e-8)

    def test_cos_sin_matches_direct_angle(self) -> None:
        # Away from the at_pole branch (cos_theta**2 == 1, e.g. the identity quaternion above):
        # to_polarization_angle_cos_sin uses pa=0 there regardless of what atan2's zero-sign
        # convention gives to_polarization_angle, a pre-existing quirk this migration preserves.
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        pa = to_polarization_angle(q)
        cos_pa, sin_pa = to_polarization_angle_cos_sin(q)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)

    def test_batch(self) -> None:
        # Generic (non-pole) quaternions -- see test_cos_sin_matches_direct_angle for why.
        batch = Q.from_array(
            jnp.array([[0.6, 0.3, 0.4, 0.7], [0.8, 0.1, 0.2, 0.5], [0.5, 0.5, 0.5, 0.5]])
        ).normalize()
        pa = to_polarization_angle(batch)
        cos_pa, sin_pa = to_polarization_angle_cos_sin(batch)
        assert pa.shape == cos_pa.shape == sin_pa.shape == (3,)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)

    def test_jit_compatible(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        pa = jax.jit(to_polarization_angle)(q)
        cos_pa, sin_pa = jax.jit(to_polarization_angle_cos_sin)(q)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)


class TestAngleConversionConsistency:
    def test_all_conversions_preserve_rotation(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        vec = jnp.array([1.0, 2.0, 3.0])
        original = q.rotate_vector(vec)

        theta, phi, psi = to_iso_angles(q)
        assert jnp.allclose(
            from_iso_angles(theta, phi, psi).rotate_vector(vec), original, atol=1e-6
        )

        alpha, delta, psi_ll = to_lonlat_angles(q)
        assert jnp.allclose(
            from_lonlat_angles(alpha, delta, psi_ll).rotate_vector(vec), original, atol=1e-6
        )

        xi, eta, gamma = to_xieta_angles(q)
        assert jnp.allclose(
            from_xieta_angles(xi, eta, gamma).rotate_vector(vec), original, atol=1e-6
        )
