import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import HomothetyOperator
from furax.obs.atmosphere import (
    AtmospherePointingOperator,
    profile_neg_log_likelihood,
    simulate_kolmogorov_screen,
)
from furax.obs.landscapes import TangentialLandscape

HEIGHT = 100.0
DX = DY = 10.0
NDET = 4
NSAMP = 30


def _make_landscape():
    return TangentialLandscape.from_extent(
        x_size=5000.0, y_size=5000.0, dx=DX, dy=DY, height=HEIGHT, dtype=jnp.float64
    )


def _random_unit_quats(key, shape):
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def _make_operator(landscape, wind_velocity, seed=0):
    key = jax.random.PRNGKey(seed)
    k_bore, k_det = jax.random.split(key, 2)
    qbore = _random_unit_quats(k_bore, (NSAMP,))
    qbore = qbore.at[:, 1:3].multiply(0.02)
    qbore = qbore / jnp.linalg.norm(qbore, axis=-1, keepdims=True)
    qdet = _random_unit_quats(k_det, (NDET,))
    qdet = qdet.at[:, 1:3].multiply(0.01)
    qdet = qdet / jnp.linalg.norm(qdet, axis=-1, keepdims=True)
    times = jnp.arange(NSAMP, dtype=jnp.float64)
    return AtmospherePointingOperator.from_wind(
        landscape, qbore, qdet, wind_velocity, times, interpolate=True
    )


class TestProfileNegLogLikelihood:
    def test_returns_scalar(self):
        landscape = _make_landscape()
        v = jnp.array([2.0, 1.0])
        P = _make_operator(landscape, v)
        atm = simulate_kolmogorov_screen(landscape, jax.random.PRNGKey(1))
        d = P(atm)
        N_inv = HomothetyOperator(1.0, in_structure=d.structure)
        val = profile_neg_log_likelihood(P, d, N_inv)
        assert val.shape == ()

    def test_true_params_minimize_loss(self):
        """Profile NLL must be lower at the true wind velocity than at a wrong one."""
        landscape = _make_landscape()
        v_true = jnp.array([2.0, 1.0])
        v_wrong = jnp.array([5.0, -2.0])

        key = jax.random.PRNGKey(3)
        atm = simulate_kolmogorov_screen(landscape, key)

        P_true = _make_operator(landscape, v_true)
        d_obs = P_true(atm)
        N_inv = HomothetyOperator(1.0, in_structure=d_obs.structure)

        nll_true = profile_neg_log_likelihood(P_true, d_obs, N_inv)
        P_wrong = _make_operator(landscape, v_wrong)
        nll_wrong = profile_neg_log_likelihood(P_wrong, d_obs, N_inv)

        assert float(nll_true) < float(nll_wrong)

    def test_differentiable_wrt_wind_velocity(self):
        """jax.grad must flow through from_wind back to wind_velocity."""
        landscape = _make_landscape()
        v_true = jnp.array([2.0, 1.0])

        key = jax.random.PRNGKey(5)
        k_atm, k_bore, k_det = jax.random.split(key, 3)

        atm = simulate_kolmogorov_screen(landscape, k_atm)
        qbore = _random_unit_quats(k_bore, (NSAMP,))
        qbore = qbore.at[:, 1:3].multiply(0.02)
        qbore = qbore / jnp.linalg.norm(qbore, axis=-1, keepdims=True)
        qdet = _random_unit_quats(k_det, (NDET,))
        qdet = qdet.at[:, 1:3].multiply(0.01)
        qdet = qdet / jnp.linalg.norm(qdet, axis=-1, keepdims=True)
        times = jnp.arange(NSAMP, dtype=jnp.float64)

        P_true = AtmospherePointingOperator.from_wind(
            landscape, qbore, qdet, v_true, times, interpolate=True
        )
        d_obs = P_true(atm)
        N_inv = HomothetyOperator(1.0, in_structure=d_obs.structure)

        def loss(wind_velocity):
            P = AtmospherePointingOperator.from_wind(
                landscape, qbore, qdet, wind_velocity, times, interpolate=True
            )
            return profile_neg_log_likelihood(P, d_obs, N_inv)

        grad = jax.grad(loss)(jnp.array([0.0, 0.0]))
        assert grad.shape == (2,)
        # Gradient at zero should point toward the true velocity
        assert not jnp.allclose(grad, jnp.zeros(2))

    @pytest.mark.parametrize('noise_level', [0.01, 1.0])
    def test_noise_scaling(self, noise_level):
        """With white noise N = σ² I, NLL scales as 1/σ²."""
        landscape = _make_landscape()
        v = jnp.array([1.5, 0.5])
        P = _make_operator(landscape, v)
        atm = simulate_kolmogorov_screen(landscape, jax.random.PRNGKey(9))
        d = P(atm)

        N_inv_1 = HomothetyOperator(1.0, in_structure=d.structure)
        N_inv_s = HomothetyOperator(1.0 / noise_level**2, in_structure=d.structure)

        nll_1 = profile_neg_log_likelihood(P, d, N_inv_1)
        nll_s = profile_neg_log_likelihood(P, d, N_inv_s)

        assert_allclose(float(nll_s), float(nll_1) / noise_level**2, rtol=1e-5)
