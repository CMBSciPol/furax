"""Tests for coordinate-system conversions built on `fastquat.Quaternion`."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastquat import Quaternion as Q

from furax.math.coords import (
    ZAXIS,
    AzElAngles,
    IsoAngles,
    LonLatAngles,
    XiEtaAngles,
    euler,
    gamma_angle,
    gamma_angle_cos_sin,
    polarization_angle,
    polarization_angle_cos_sin,
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
            ROT_Z_90,
            ROT_Z_90 * Q(0.0, 0.0, 1.0, 0.0),  # at the south pole
            ROT_X_90,
            ROT_Y_90,
        ],
    )
    def test_roundtrip(self, q: Q) -> None:
        theta, phi, psi = IsoAngles.from_quaternion(q)
        recovered = IsoAngles(theta, phi, psi).to_quaternion()

        assert jnp.all(jnp.isfinite(jnp.array([theta, phi, psi])))
        np.testing.assert_allclose(abs(recovered), 1.0, atol=1e-6)
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'ISO roundtrip failed for {q}: got {recovered}'

    def test_to_quaternion_is_normalized(self) -> None:
        q = IsoAngles(jnp.pi / 2, jnp.pi / 4, jnp.pi / 3).to_quaternion()
        np.testing.assert_allclose(abs(q), 1.0, atol=1e-8)
        assert jnp.all(jnp.isfinite(q.wxyz))

    def test_batch(self) -> None:
        batch = Q.from_array(jnp.array([IDENTITY.wxyz, [0.5, 0.5, 0.5, 0.5], ROT_X_90.wxyz]))
        theta, phi, psi = IsoAngles.from_quaternion(batch)
        assert theta.shape == phi.shape == psi.shape == (3,)
        recovered = IsoAngles(theta, phi, psi).to_quaternion()
        assert recovered.shape == (3,)

    def test_z_axis_spherical_consistency(self) -> None:
        """Rotating the Z-axis by q should land at the spherical point given by its ISO angles."""
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        rotated_z = q.rotate_vector(jnp.array([0.0, 0.0, 1.0]))
        theta, phi, _ = IsoAngles.from_quaternion(q)
        expected = jnp.array(
            [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)]
        )
        np.testing.assert_allclose(rotated_z, expected, atol=1e-6)

    def test_jit_compatible(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        angles = jax.jit(IsoAngles.from_quaternion)(q)
        recovered = jax.jit(IsoAngles.to_quaternion)(angles)
        assert recovered.shape == ()


class TestLonLatAngles:
    def test_identity(self) -> None:
        alpha, delta, psi = LonLatAngles.from_quaternion(IDENTITY)
        np.testing.assert_allclose(alpha, 0.0, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2, atol=1e-8)
        np.testing.assert_allclose(psi, 0.0, atol=1e-8)

    @pytest.mark.parametrize('q', [IDENTITY, Q(0.5, 0.5, 0.5, 0.5), ROT_X_90, ROT_Y_90])
    def test_roundtrip(self, q: Q) -> None:
        q = q.normalize()
        alpha, delta, psi = LonLatAngles.from_quaternion(q)
        recovered = LonLatAngles(alpha, delta, psi).to_quaternion()
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'Lonlat roundtrip failed for {q}: got {recovered}'

    def test_relationship_to_iso(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        theta_iso, phi_iso, psi_iso = IsoAngles.from_quaternion(q)
        alpha, delta, psi_lonlat = LonLatAngles.from_quaternion(q)
        np.testing.assert_allclose(alpha, phi_iso, atol=1e-8)
        np.testing.assert_allclose(delta, math.pi / 2 - theta_iso, atol=1e-8)
        np.testing.assert_allclose(psi_lonlat, psi_iso, atol=1e-8)


class TestAzElAngles:
    def test_relationship_to_lonlat(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        az, el, psi = AzElAngles.from_quaternion(q)
        lon, lat, psi_lonlat = LonLatAngles.from_quaternion(q)
        np.testing.assert_allclose(az, -lon, atol=1e-8)
        np.testing.assert_allclose(el, lat, atol=1e-8)
        np.testing.assert_allclose(psi, psi_lonlat, atol=1e-8)

    def test_roundtrip(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        recovered = AzElAngles(*AzElAngles.from_quaternion(q)).to_quaternion()
        assert jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )


class TestXiEtaAngles:
    def test_identity(self) -> None:
        xi, eta, gamma = XiEtaAngles.from_quaternion(IDENTITY)
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
        xi, eta, gamma = XiEtaAngles.from_quaternion(q)
        recovered = XiEtaAngles(xi, eta, gamma).to_quaternion()
        matches = jnp.allclose(recovered.wxyz, q.wxyz, atol=1e-6) or jnp.allclose(
            recovered.wxyz, -q.wxyz, atol=1e-6
        )
        assert matches, f'XiEta roundtrip failed for {q}: got {recovered}'

    def test_batch(self) -> None:
        batch = Q.from_array(
            jnp.array([IDENTITY.wxyz, [0.5, 0.5, 0.5, 0.5], ROT_Z_90.wxyz])
        ).normalize()
        xi, eta, gamma = XiEtaAngles.from_quaternion(batch)
        assert xi.shape == eta.shape == gamma.shape == (3,)
        recovered = XiEtaAngles(xi, eta, gamma).to_quaternion()
        assert recovered.shape == (3,)


class TestGammaAngle:
    def test_identity_is_zero(self) -> None:
        np.testing.assert_allclose(gamma_angle(IDENTITY), 0.0, atol=1e-8)

    def test_matches_xieta_gamma(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        _, _, gamma_xieta = XiEtaAngles.from_quaternion(q)
        np.testing.assert_allclose(gamma_angle(q), gamma_xieta, atol=1e-8)


class TestPolarizationAngle:
    def test_identity(self) -> None:
        np.testing.assert_allclose(polarization_angle(IDENTITY), 0.0, atol=1e-8)

    def test_cos_sin_matches_direct_angle(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        pa = polarization_angle(q)
        cos_pa, sin_pa = polarization_angle_cos_sin(q)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)

    def test_batch(self) -> None:
        batch = Q.from_array(
            jnp.array([[0.6, 0.3, 0.4, 0.7], [0.8, 0.1, 0.2, 0.5], [0.5, 0.5, 0.5, 0.5]])
        ).normalize()
        pa = polarization_angle(batch)
        cos_pa, sin_pa = polarization_angle_cos_sin(batch)
        assert pa.shape == cos_pa.shape == sin_pa.shape == (3,)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)

    def test_jit_compatible(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        pa = jax.jit(polarization_angle)(q)
        cos_pa, sin_pa = jax.jit(polarization_angle_cos_sin)(q)
        np.testing.assert_allclose(cos_pa, jnp.cos(pa), atol=1e-6)
        np.testing.assert_allclose(sin_pa, jnp.sin(pa), atol=1e-6)


@pytest.mark.parametrize('theta', [0.0, math.pi], ids=['north', 'south'])
class TestPoles:
    """Where only phi + psi or psi - phi is defined, every angle takes phi = 0."""

    PSI = np.array([-2.5, -0.7, 0.0, 0.3, 1.9, 3.0])

    def _pointing(self, theta: float) -> Q:
        psi = jnp.asarray(self.PSI)
        return IsoAngles(jnp.full_like(psi, theta), jnp.zeros_like(psi), psi).to_quaternion()

    def test_iso_angles_roundtrip(self, theta: float) -> None:
        angles = IsoAngles.from_quaternion(self._pointing(theta))
        np.testing.assert_allclose(angles.phi, 0.0, atol=1e-15)
        np.testing.assert_allclose(angles.psi, self.PSI, atol=1e-14)

    def test_polarization_angle(self, theta: float) -> None:
        q = self._pointing(theta)
        np.testing.assert_allclose(polarization_angle(q), self.PSI, atol=1e-14)
        cos_pa, sin_pa = polarization_angle_cos_sin(q)
        np.testing.assert_allclose(cos_pa, np.cos(self.PSI), atol=1e-14)
        np.testing.assert_allclose(sin_pa, np.sin(self.PSI), atol=1e-14)

    def test_limit_along_the_meridian(self, theta: float) -> None:
        """Approaching the pole along phi = 0 gives the angle at the pole."""
        psi = jnp.asarray(self.PSI)
        near = abs(theta - 1e-7)
        q = IsoAngles(jnp.full_like(psi, near), jnp.zeros_like(psi), psi).to_quaternion()
        np.testing.assert_allclose(polarization_angle(q), self.PSI, atol=1e-12)


class TestAngleConversionConsistency:
    def test_all_conversions_preserve_rotation(self) -> None:
        q = Q(0.6, 0.3, 0.4, 0.7).normalize()
        vec = jnp.array([1.0, 2.0, 3.0])
        original = q.rotate_vector(vec)

        theta, phi, psi = IsoAngles.from_quaternion(q)
        assert jnp.allclose(
            IsoAngles(theta, phi, psi).to_quaternion().rotate_vector(vec), original, atol=1e-6
        )

        alpha, delta, psi_ll = LonLatAngles.from_quaternion(q)
        assert jnp.allclose(
            LonLatAngles(alpha, delta, psi_ll).to_quaternion().rotate_vector(vec),
            original,
            atol=1e-6,
        )

        xi, eta, gamma = XiEtaAngles.from_quaternion(q)
        assert jnp.allclose(
            XiEtaAngles(xi, eta, gamma).to_quaternion().rotate_vector(vec), original, atol=1e-6
        )


class TestConventions:
    """How the angles of the three coordinate systems relate, on random rotations."""

    @staticmethod
    def _rotations() -> Q:
        return Q.random(jax.random.key(0), (200,))

    @staticmethod
    def _wrap(angle: jax.Array) -> jax.Array:
        return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi

    def test_psi_is_the_polarization_angle(self) -> None:
        q = self._rotations()
        psi = IsoAngles.from_quaternion(q).psi
        np.testing.assert_allclose(self._wrap(polarization_angle(q) - psi), 0.0, atol=1e-12)

    def test_gamma_is_phi_plus_psi(self) -> None:
        q = self._rotations()
        _, phi, psi = IsoAngles.from_quaternion(q)
        np.testing.assert_allclose(self._wrap(gamma_angle(q) - phi - psi), 0.0, atol=1e-12)

    def test_gamma_cos_sin_matches_the_angle(self) -> None:
        q = self._rotations()
        cos_gamma, sin_gamma = gamma_angle_cos_sin(q)
        np.testing.assert_allclose(cos_gamma, jnp.cos(gamma_angle(q)), atol=1e-12)
        np.testing.assert_allclose(sin_gamma, jnp.sin(gamma_angle(q)), atol=1e-12)

    def test_gamma_opposite_the_boresight(self) -> None:
        """A detector looking opposite the boresight is at the south pole: gamma = psi, phi = 0."""
        psi = TestPoles.PSI
        q = IsoAngles(jnp.full(psi.shape, jnp.pi), jnp.zeros(psi.shape), jnp.asarray(psi))
        q = q.to_quaternion()
        np.testing.assert_allclose(gamma_angle(q), psi, atol=1e-14)
        cos_gamma, sin_gamma = gamma_angle_cos_sin(q)
        np.testing.assert_allclose(cos_gamma, np.cos(psi), atol=1e-14)
        np.testing.assert_allclose(sin_gamma, np.sin(psi), atol=1e-14)

    def test_xi_eta_are_orthographic_coordinates(self) -> None:
        xi, eta = jax.random.normal(jax.random.key(1), (2, 200)) * 0.1
        gamma = jax.random.uniform(jax.random.key(2), (200,), maxval=2 * jnp.pi)
        v = XiEtaAngles(xi, eta, gamma).to_quaternion().rotate_vector(ZAXIS)
        np.testing.assert_allclose(-v[..., 1], xi, atol=1e-12)
        np.testing.assert_allclose(-v[..., 0], eta, atol=1e-12)
