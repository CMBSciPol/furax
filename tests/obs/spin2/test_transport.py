import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.obs.spin2 import spin2_cos_sin, spin2_cos_sin_zs


def _atan2_reference(
    theta_n: np.ndarray, phi_n: np.ndarray, theta_x: np.ndarray, phi_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook two-bearing form of the transport pair, in float64 numpy.

    Independent of the haversine implementation under test: it forms both bearings explicitly with
    `arctan2` and doubles the angle. Accurate at degree separations, and the reason the shipped
    implementation does not use this form is that it loses precision at arcminute separations.
    """
    z_n, s_n = np.cos(theta_n), np.sin(theta_n)
    z_x, s_x = np.cos(theta_x), np.sin(theta_x)
    dphi = phi_x - phi_n
    cd, sd = np.cos(dphi), np.sin(dphi)
    cos_beta = np.clip(s_n * s_x * cd + z_n * z_x, -1.0, 1.0)
    sin_beta = np.sqrt(np.maximum(0.0, 1.0 - cos_beta**2))
    inv = 1.0 / sin_beta
    alpha = np.arctan2(s_x * sd * inv, (s_x * z_n * cd - z_x * s_n) * inv)
    gamma = np.arctan2(s_n * sd * inv, -(s_n * z_x * cd - z_n * s_x) * inv)
    delta = alpha - gamma
    return np.cos(2 * delta), np.sin(2 * delta)


def _random_directions(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, np.pi - 0.05, n), rng.uniform(0.0, 2 * np.pi, n)


class TestSpin2CosSin:
    def test_unit_norm(self) -> None:
        theta_n, phi_n = _random_directions(500, 0)
        theta_x, phi_x = _random_directions(500, 1)
        c, s = spin2_cos_sin(*map(jnp.asarray, (theta_n, phi_n, theta_x, phi_x)))
        assert_allclose(np.asarray(c) ** 2 + np.asarray(s) ** 2, 1.0, atol=1e-14)

    def test_antisymmetry(self) -> None:
        """Swapping the two directions negates delta: cos is even, sin is odd."""
        theta_n, phi_n = _random_directions(500, 2)
        theta_x, phi_x = _random_directions(500, 3)
        fwd = spin2_cos_sin(*map(jnp.asarray, (theta_n, phi_n, theta_x, phi_x)))
        bwd = spin2_cos_sin(*map(jnp.asarray, (theta_x, phi_x, theta_n, phi_n)))
        assert_allclose(np.asarray(fwd[0]), np.asarray(bwd[0]), atol=1e-12)
        assert_allclose(np.asarray(fwd[1]), -np.asarray(bwd[1]), atol=1e-12)

    @pytest.mark.parametrize(
        'theta_n, theta_x', [(0.3, 1.0), (1.0, 2.5), (0.1, np.pi - 0.1), (1.2, 1.2)]
    )
    def test_same_meridian_is_identity(self, theta_n: float, theta_x: float) -> None:
        c, s = spin2_cos_sin(
            jnp.asarray(theta_n), jnp.asarray(0.7), jnp.asarray(theta_x), jnp.asarray(0.7)
        )
        assert_allclose(float(c), 1.0, atol=1e-12)
        assert_allclose(float(s), 0.0, atol=1e-12)

    @pytest.mark.parametrize('theta, phi', [(0.5, 1.2), (0.0, 0.0), (np.pi, 0.0)])
    def test_coincident_is_identity(self, theta: float, phi: float) -> None:
        """The denominator is exactly zero here, so this exercises the guard."""
        args = (jnp.asarray(theta), jnp.asarray(phi))
        c, s = spin2_cos_sin(*args, *args)
        assert float(c) == 1.0
        assert float(s) == 0.0

    def test_agrees_with_atan2_reference(self) -> None:
        theta_n, phi_n = _random_directions(500, 4)
        theta_x, phi_x = _random_directions(500, 5)
        c, s = spin2_cos_sin(*map(jnp.asarray, (theta_n, phi_n, theta_x, phi_x)))
        c_ref, s_ref = _atan2_reference(theta_n, phi_n, theta_x, phi_x)
        assert_allclose(np.asarray(c), c_ref, atol=1e-12)
        assert_allclose(np.asarray(s), s_ref, atol=1e-12)

    @pytest.mark.parametrize('separation_arcmin', [30.0, 10.0, 3.4, 1.0])
    def test_agrees_with_atan2_reference_at_stencil_separations(
        self, separation_arcmin: float
    ) -> None:
        """The separations that matter are sub-pixel, not the degree scale.

        A neighbour sits about 3.4 arcmin from the sample at nside 1024. A test run only at degree
        separations passes on a formulation that has already lost precision where the interpolation
        weight actually sits.
        """
        sep = np.deg2rad(separation_arcmin / 60.0)
        # a ring of neighbours around each target, so every bearing is exercised
        theta_x = np.repeat(np.deg2rad([90.0, 30.0, 5.0, 1.0]), 64)
        phi_x = np.tile(np.linspace(0.0, 2 * np.pi, 64, endpoint=False), 4)
        bearing = np.tile(np.linspace(0.0, 2 * np.pi, 64, endpoint=False), 4)
        theta_n = theta_x + sep * np.cos(bearing)
        phi_n = phi_x + sep * np.sin(bearing) / np.sin(theta_x)

        c, s = spin2_cos_sin(*map(jnp.asarray, (theta_n, phi_n, theta_x, phi_x)))
        c_ref, s_ref = _atan2_reference(theta_n, phi_n, theta_x, phi_x)
        assert_allclose(np.asarray(c), c_ref, atol=1e-10)
        assert_allclose(np.asarray(s), s_ref, atol=1e-10)

    def test_poles_are_finite(self) -> None:
        theta_n = jnp.asarray([0.0, np.pi, 1e-8, np.pi - 1e-8])
        phi_n = jnp.asarray([0.0, 1.0, 2.0, 3.0])
        theta_x = jnp.asarray([1e-6, np.pi - 1e-6, 0.5, 2.0])
        phi_x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
        c, s = spin2_cos_sin(theta_n, phi_n, theta_x, phi_x)
        assert np.isfinite(np.asarray(c)).all()
        assert np.isfinite(np.asarray(s)).all()

    def test_zs_form_matches_angle_form(self) -> None:
        theta_n, phi_n = _random_directions(200, 6)
        theta_x, phi_x = _random_directions(200, 7)
        from_angles = spin2_cos_sin(*map(jnp.asarray, (theta_n, phi_n, theta_x, phi_x)))
        from_zs = spin2_cos_sin_zs(
            jnp.cos(jnp.asarray(theta_n)),
            jnp.sin(jnp.asarray(theta_n)),
            jnp.asarray(phi_n),
            jnp.cos(jnp.asarray(theta_x)),
            jnp.sin(jnp.asarray(theta_x)),
            jnp.asarray(phi_x),
        )
        assert_allclose(np.asarray(from_angles[0]), np.asarray(from_zs[0]), atol=1e-15)
        assert_allclose(np.asarray(from_angles[1]), np.asarray(from_zs[1]), atol=1e-15)

    def test_broadcasts(self) -> None:
        theta_n = jnp.asarray(np.random.default_rng(8).uniform(0.1, 3.0, (5, 4)))
        phi_n = jnp.asarray(np.random.default_rng(9).uniform(0.0, 6.0, (5, 4)))
        theta_x = jnp.asarray(np.random.default_rng(10).uniform(0.1, 3.0, (5, 1)))
        phi_x = jnp.asarray(np.random.default_rng(11).uniform(0.0, 6.0, (5, 1)))
        c, s = spin2_cos_sin(theta_n, phi_n, theta_x, phi_x)
        assert c.shape == (5, 4)
        assert s.shape == (5, 4)

    def test_gradient_is_finite_at_coincidence(self) -> None:
        """The guard must not leak a NaN into the gradient through the dead branch."""

        def f(theta: jax.Array) -> jax.Array:
            c, s = spin2_cos_sin(theta, jnp.asarray(0.4), jnp.asarray(0.9), jnp.asarray(0.4))
            return c + s

        grad = jax.grad(f)(jnp.asarray(0.9))
        assert np.isfinite(float(grad))


class TestSpin2CosSinFloat32:
    """`double_precision=False` runs the whole pipeline in float32."""

    @pytest.mark.insubprocess
    def test_stencil_separation_precision(self) -> None:
        jax.config.update('jax_enable_x64', False)

        sep = np.deg2rad(3.4 / 60.0)  # one nside 1024 neighbour distance
        theta_x = np.repeat(np.deg2rad([90.0, 30.0, 5.0, 1.0]), 64)
        bearing = np.tile(np.linspace(0.0, 2 * np.pi, 64, endpoint=False), 4)
        phi_x = np.tile(np.linspace(0.0, 2 * np.pi, 64, endpoint=False), 4)
        theta_n = theta_x + sep * np.cos(bearing)
        phi_n = phi_x + sep * np.sin(bearing) / np.sin(theta_x)

        c, s = spin2_cos_sin(
            *(jnp.asarray(a, dtype=jnp.float32) for a in (theta_n, phi_n, theta_x, phi_x))
        )
        assert c.dtype == jnp.float32
        c_ref, s_ref = _atan2_reference(theta_n, phi_n, theta_x, phi_x)
        # Measured on this fixture: the haversine form holds 7.5e-7, the textbook two-bearing form
        # in float32 gives 1.0e-4. The tolerance sits between the two, so a regression to the
        # textbook form fails here by two orders of magnitude.
        assert_allclose(np.asarray(c), c_ref, atol=5e-6)
        assert_allclose(np.asarray(s), s_ref, atol=5e-6)
