import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.config import NoiseFitConfig
from furax.mapmaking.noise import (
    AtmosphericNoiseModel,
    WhiteNoiseModel,
    _create_frequency_mask,
    _create_frequency_mask_from_config,
    apodization_window,
    fit_atmospheric_psd_model,
    fit_white_noise_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_freq(n: int = 256, sample_rate: float = 10.0) -> jax.Array:
    f = jnp.fft.rfftfreq(n, 1 / sample_rate)
    return f[1:]  # drop DC


def make_white_psd(f: jax.Array, sigma: float, n_dets: int = 1) -> jax.Array:
    noise = jnp.exp(0.1 * jax.random.normal(jax.random.key(0), (n_dets, len(f))))
    return sigma**2 * noise


def make_atm_psd(
    f: jax.Array,
    sigma: float = 0.5,
    alpha: float = -2.5,
    fk: float = 0.3,
    f0: float = 0.01,
    n_dets: int = 1,
) -> jax.Array:
    psd = sigma**2 * (1 + ((f + f0) / fk) ** alpha)
    noise = jnp.exp(0.05 * jax.random.normal(jax.random.key(42), (n_dets, len(f))))
    return psd[None, :] * noise


# ---------------------------------------------------------------------------
# WhiteNoiseModel
# ---------------------------------------------------------------------------


class TestWhiteNoiseModel:
    def test_psd_is_sigma_squared(self):
        sigma = jnp.array([1.5, 2.0])
        model = WhiteNoiseModel(sigma=sigma)
        f = jnp.array([0.1, 1.0, 5.0])
        psd = model.psd(f)
        assert psd.shape == (2, 3)
        assert_allclose(psd, sigma[:, None] ** 2 * jnp.ones((1, 3)), rtol=1e-6)

    def test_inverse_operator_zero_sigma(self):
        """sigma=0 detectors should produce zero output, not NaN/Inf."""
        sigma = jnp.array([0.0, 1.0])
        model = WhiteNoiseModel(sigma=sigma)
        struct = jax.ShapeDtypeStruct((2, 50), jnp.float64)
        y = model.inverse_operator(struct)(jnp.ones((2, 50)))
        assert jnp.all(jnp.isfinite(y))
        assert_array_equal(y[0], jnp.zeros(50))

    def test_operator_inverse_round_trip(self):
        sigma = jnp.array([1.0, 2.0])
        model = WhiteNoiseModel(sigma=sigma)
        struct = jax.ShapeDtypeStruct((2, 50), jnp.float64)
        x = jnp.ones((2, 50))
        assert_allclose(model.inverse_operator(struct)(model.operator(struct)(x)), x, rtol=1e-5)


# ---------------------------------------------------------------------------
# AtmosphericNoiseModel
# ---------------------------------------------------------------------------


class TestAtmosphericNoiseModel:
    @pytest.fixture
    def model(self):
        return AtmosphericNoiseModel(
            sigma=jnp.array([0.5, 1.0]),
            alpha=jnp.array([-2.5, -2.0]),
            fk=jnp.array([0.3, 0.5]),
            f0=jnp.array([0.01, 0.01]),
        )

    def test_psd_and_log_psd_consistent(self, model):
        f = jnp.linspace(0.05, 5.0, 30)
        psd = model.psd(f)
        assert psd.shape == (2, 30)
        assert jnp.all(psd > 0)
        assert_allclose(model.log_psd(f), jnp.log10(psd), rtol=1e-5)

    def test_psd_zero_sigma(self):
        model = AtmosphericNoiseModel(
            sigma=jnp.array([0.0, 1.0]),
            alpha=jnp.array([-2.5, -2.5]),
            fk=jnp.array([0.3, 0.3]),
            f0=jnp.array([0.01, 0.01]),
        )
        psd = model.psd(jnp.array([0.1, 1.0]))
        assert jnp.all(jnp.isfinite(psd))
        assert_array_equal(psd[0], jnp.zeros(2))

    def test_to_array(self, model):
        arr = model.to_array()
        assert arr.shape == (2, 4)
        assert_array_equal(arr, jnp.stack([model.sigma, model.alpha, model.fk, model.f0], axis=-1))

    def test_to_white_noise_model(self, model):
        white = model.to_white_noise_model()
        assert isinstance(white, WhiteNoiseModel)
        assert_array_equal(white.sigma, model.sigma)

    def test_operators_shape_and_no_nan(self, model):
        shape = (2, 200)
        struct = jax.ShapeDtypeStruct(shape, jnp.float64)
        kwargs = dict(sample_rate=10.0, correlation_length=20)
        x = jnp.ones(shape)
        for method in ('operator', 'inverse_operator'):
            y = getattr(model, method)(struct, **kwargs)(x)
            assert y.shape == shape
            assert jnp.all(jnp.isfinite(y)), f'{method} produced non-finite values'


# ---------------------------------------------------------------------------
# apodization_window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', ['chebwin', 'gaussian'])
def test_apodization_window_shape(kind):
    w = apodization_window(50, kind=kind)
    assert w.shape == (50,)


def test_apodization_window_invalid_kind():
    with pytest.raises(RuntimeError, match='not supported'):
        apodization_window(10, kind='bogus')


# ---------------------------------------------------------------------------
# _create_frequency_mask
# ---------------------------------------------------------------------------


def test_frequency_mask_f_min():
    f = jnp.array([0.0, 1.0, 2.0, 3.0])
    mask = _create_frequency_mask(f, f_min=jnp.array(2.0), f_max=None, f_mask_intervals=None)
    assert_array_equal(mask, jnp.array([0.0, 0.0, 1.0, 1.0]))


def test_frequency_mask_f_max():
    f = jnp.array([0.0, 1.0, 2.0, 3.0])
    mask = _create_frequency_mask(f, f_min=None, f_max=jnp.array(2.0), f_mask_intervals=None)
    assert_array_equal(mask, jnp.array([1.0, 1.0, 0.0, 0.0]))


def test_frequency_mask_intervals():
    f = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mask = _create_frequency_mask(
        f, f_min=None, f_max=None, f_mask_intervals=jnp.array([[1.0, 3.0]])
    )
    assert_array_equal(mask, jnp.array([1.0, 0.0, 0.0, 1.0, 1.0]))


def test_frequency_mask_hwp_harmonics_masked():
    """HWP 1f harmonic should be masked when mask_hwp_harmonics=True."""
    sample_rate = jnp.array(10.0)
    hwp_frequency = jnp.array(2.0)
    f = jnp.fft.rfftfreq(512, 1 / float(sample_rate))
    config = NoiseFitConfig(mask_hwp_harmonics=True, freq_mask_width=0.1)
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
    )
    idx = jnp.argmin(jnp.abs(f - 2.0))
    assert mask[idx] == 0.0


# ---------------------------------------------------------------------------
# Fitting (minimiser paths)
# ---------------------------------------------------------------------------


def test_fit_white_noise_model():
    f = make_freq(256, 10.0)
    Pxx = make_white_psd(f, sigma=1.0)
    result = fit_white_noise_model(
        f,
        Pxx,
        sample_rate=jnp.array(10.0),
        hwp_frequency=jnp.array(0.0),
        config=NoiseFitConfig(mask_hwp_harmonics=False),
    )
    assert result['fit'].shape == (1,)
    assert jnp.all(jnp.isfinite(result['fit']))


def test_fit_atmospheric_psd_model():
    f = make_freq(256, 10.0)
    Pxx = make_atm_psd(f)
    result = fit_atmospheric_psd_model(
        f,
        Pxx,
        sample_rate=jnp.array(10.0),
        hwp_frequency=jnp.array(0.0),
        config=NoiseFitConfig(mask_hwp_harmonics=False, max_iter=50),
    )
    assert result['fit'].shape == (1, 4)
    assert result['loss'].shape == (1,)
    assert result['num_iter'].shape == (1,)
    assert result['inv_fisher'].shape == (1, 4, 4)
    assert jnp.all(jnp.isfinite(result['fit']))
