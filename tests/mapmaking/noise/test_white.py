import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.config import NoiseFitConfig
from furax.mapmaking.noise import WhiteNoiseModel


class TestWhiteNoiseModel:
    def test_psd_is_sigma_squared(self):
        sigma = jnp.array([1.5, 2.0])
        model = WhiteNoiseModel(sigma=sigma, sample_rate=10.0)
        f = jnp.array([0.1, 1.0, 5.0])
        psd = model.psd(f)
        assert psd.shape == (2, 3)
        assert_allclose(psd, sigma[:, None] ** 2 * jnp.ones((1, 3)), rtol=1e-6)
        assert model.white() is model

    def test_inverse_covariance_zero_sigma(self):
        """sigma=0 detectors should produce zero output, not NaN/Inf."""
        sigma = jnp.array([0.0, 1.0])
        model = WhiteNoiseModel(sigma=sigma, sample_rate=10.0)
        struct = jax.ShapeDtypeStruct((2, 50), jnp.float64)
        y = model.inverse_covariance(struct)(jnp.ones((2, 50)))
        assert jnp.all(jnp.isfinite(y))
        assert_array_equal(y[0], jnp.zeros(50))

    def test_covariance_is_the_variance_per_sample(self):
        model = WhiteNoiseModel(sigma=jnp.array([1.0, 2.0]), sample_rate=10.0)
        struct = jax.ShapeDtypeStruct((2, 3), jnp.float64)
        # a one-sided PSD sigma² spreads over the 5 Hz up to Nyquist
        assert_allclose(model.covariance(struct)(jnp.ones((2, 3))), [[5.0] * 3, [20.0] * 3])

    def test_operator_inverse_round_trip(self):
        sigma = jnp.array([1.0, 2.0])
        model = WhiteNoiseModel(sigma=sigma, sample_rate=10.0)
        struct = jax.ShapeDtypeStruct((2, 50), jnp.float64)
        x = jnp.ones((2, 50))
        assert_allclose(model.inverse_covariance(struct)(model.covariance(struct)(x)), x, rtol=1e-5)


@pytest.mark.parametrize('lead', [(3,), (2, 3)], ids=['modulated', 'demodulated'])
def test_fit_recovers_sigma_from_a_draw(lead):
    sigma = jnp.linspace(1.0, 2.0, 3) * jnp.ones(lead)
    structure = jax.ShapeDtypeStruct((*lead, 2**14), jnp.float64)
    samples = WhiteNoiseModel(sigma, sample_rate=10.0).draw(jax.random.key(0), structure)
    model = WhiteNoiseModel.fit(
        samples,
        sample_rate=jnp.array(10.0),
        hwp_frequency=jnp.array(0.0),
        config=NoiseFitConfig(mask_hwp_harmonics=False),
    )
    assert_allclose(model.sigma, sigma, rtol=0.02)
