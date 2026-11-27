from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.config import NoiseFitConfig
from furax.mapmaking.noise import OneOverFNoiseModel, WhiteNoiseModel
from furax.mapmaking.noise.one_over_f import _fit_one_over_f_model
from tests.mapmaking.noise.helpers import make_freq, make_one_over_f_psd


class TestOneOverFNoiseModel:
    @pytest.fixture
    def model(self):
        return OneOverFNoiseModel(
            sigma=jnp.array([0.5, 1.0]),
            alpha=jnp.array([-2.5, -2.0]),
            fk=jnp.array([0.3, 0.5]),
            f0=jnp.array([0.01, 0.01]),
            sample_rate=10.0,
        )

    def test_psd_is_positive(self, model):
        psd = model.psd(jnp.linspace(0.05, 5.0, 30))
        assert psd.shape == (2, 30)
        assert jnp.all(psd > 0)

    def test_psd_zero_sigma(self):
        model = OneOverFNoiseModel(
            sigma=jnp.array([0.0, 1.0]),
            alpha=jnp.array([-2.5, -2.5]),
            fk=jnp.array([0.3, 0.3]),
            f0=jnp.array([0.01, 0.01]),
            sample_rate=10.0,
        )
        psd = model.psd(jnp.array([0.1, 1.0]))
        assert jnp.all(jnp.isfinite(psd))
        assert_array_equal(psd[0], jnp.zeros(2))

    def test_to_array(self, model):
        arr = model.to_array()
        assert arr.shape == (2, 4)
        assert_array_equal(arr, jnp.stack([model.sigma, model.alpha, model.fk, model.f0], axis=-1))

    def test_white(self, model):
        white = model.white()
        assert isinstance(white, WhiteNoiseModel)
        assert_array_equal(white.sigma, model.sigma)

    @pytest.mark.parametrize('method', ['covariance', 'inverse_covariance'])
    def test_operators_shape_and_no_nan(self, model, method):
        shape = (2, 200)
        struct = jax.ShapeDtypeStruct(shape, jnp.float64)
        with pytest.raises(ValueError, match='correlation_length'):
            getattr(model, method)(struct)
        model = replace(model, correlation_length=20)
        y = getattr(model, method)(struct)(jnp.ones(shape))
        assert y.shape == shape
        assert jnp.all(jnp.isfinite(y))


def test_fit_one_over_f_model_recovers_the_parameters():
    f = make_freq(256, 10.0)
    result = _fit_one_over_f_model(
        f,
        make_one_over_f_psd(f, sigma=0.5, alpha=-2.5, fk=0.3),
        sample_rate=jnp.array(10.0),
        hwp_frequency=jnp.array(0.0),
        config=NoiseFitConfig(mask_hwp_harmonics=False, max_iter=50),
    )
    assert result['fit'].shape == (1, 4)
    assert result['inv_fisher'].shape == (1, 4, 4)
    assert jnp.all(jnp.isfinite(result['loss']))
    # sigma, alpha, f_knee; the offset f0 is barely constrained by a spectrum that starts above it
    assert_allclose(result['fit'][0, :3], [0.5, -2.5, 0.3], rtol=0.05)


def test_fit_recovers_the_parameters_of_a_draw():
    truth = OneOverFNoiseModel(
        sigma=jnp.array([0.5, 1.0]),
        alpha=jnp.full(2, -2.5),
        fk=jnp.array([0.3, 0.5]),
        f0=jnp.full(2, 0.01),
        sample_rate=10.0,
    )
    samples = truth.draw(jax.random.key(0), jax.ShapeDtypeStruct((2, 2**16), jnp.float64))
    model = OneOverFNoiseModel.fit(
        samples,
        sample_rate=jnp.array(10.0),
        hwp_frequency=jnp.array(0.0),
        config=NoiseFitConfig(mask_hwp_harmonics=False, max_iter=50),
    )
    for name in ('sigma', 'alpha', 'fk'):
        assert_allclose(getattr(model, name), getattr(truth, name), rtol=0.1, err_msg=name)
