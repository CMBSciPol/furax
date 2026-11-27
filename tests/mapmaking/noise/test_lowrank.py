from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax.mapmaking.config import LowRankNoiseFitConfig
from furax.mapmaking.noise import LowRankNoiseModel, bin_edges
from tests.mapmaking.noise.helpers import toy_low_rank_model

SAMPLE_RATE = 8.0
N_CHANNELS = 24
N_SAMPLES = 2**15
CONFIG = LowRankNoiseFitConfig(bin_edges=(0.02, 0.05, 0.1, 0.2), max_modes=4)


def simulate(key, amplitude, *, n_modes=2, sigma=None):
    """White noise per channel plus `n_modes` common modes, coloured below 0.2 Hz."""
    k_w, k_v, k_t = jax.random.split(key, 3)
    sigma = jnp.linspace(1.0, 2.0, N_CHANNELS) if sigma is None else sigma
    white = jax.random.normal(k_w, (N_CHANNELS, N_SAMPLES)) * sigma[:, None]
    vectors, _ = jnp.linalg.qr(jax.random.normal(k_v, (N_CHANNELS, n_modes)))
    series = jax.random.normal(k_t, (n_modes, N_SAMPLES))
    frequencies = jnp.fft.rfftfreq(N_SAMPLES, 1 / SAMPLE_RATE)
    low = (frequencies > 0.02) & (frequencies < 0.2)
    series = jnp.fft.irfft(jnp.fft.rfft(series) * low * 3.0, n=N_SAMPLES) * amplitude
    return white + (vectors * sigma[:, None]) @ series, vectors, sigma


def mode_count(op):
    return int(jnp.any(op.modes != 0, axis=0).sum())


def test_white_noise_has_no_modes():
    data, _, _ = simulate(jax.random.key(0), 0.0)
    op = LowRankNoiseModel.fit(data, sample_rate=SAMPLE_RATE, config=CONFIG)
    assert mode_count(op) == 0


def test_recovers_injected_common_modes():
    data, vectors, sigma = simulate(jax.random.key(1), 1.0)
    # a bin below the injected band, too narrow to find modes in by itself
    config = replace(CONFIG, bin_edges=(0.0005, 0.001, *CONFIG.bin_edges))
    op = LowRankNoiseModel.fit(data, sample_rate=SAMPLE_RATE, config=config)
    # the white level above the last edge is an uncoupled last bin
    assert op.diagonal.shape[0] == op.core.shape[0] + 1
    # the modes, from the whole mode band, are given once for every bin
    assert op.modes.ndim == 2
    assert mode_count(op) == 2
    # the fitted subspace contains the injected one
    fitted = np.asarray(op.modes / sigma[:, None])
    basis, _ = np.linalg.qr(fitted[:, np.any(fitted != 0, axis=0)])
    residual = np.asarray(vectors) - basis @ (basis.T @ np.asarray(vectors))
    assert np.linalg.norm(residual) < 0.2 * np.linalg.norm(vectors)
    # their power follows the injected spectrum, in 0.02-0.2 Hz only
    power = np.asarray(jnp.diagonal(op.core, axis1=1, axis2=2))[:, :2]
    assert (power[3:] > 5 * power[1]).all()


def test_diagonal_reproduces_channel_power_in_white_band():
    data, _, sigma = simulate(jax.random.key(2), 0.0)
    op = LowRankNoiseModel.fit(data, sample_rate=SAMPLE_RATE, config=CONFIG)
    np.testing.assert_allclose(op.diagonal[-1], sigma**2, rtol=0.15)
    np.testing.assert_allclose(op.diagonal[2], sigma**2, rtol=0.4)


def test_weight_is_the_inverse_of_the_fitted_covariance():
    data, _, _ = simulate(jax.random.key(6), 1.0)
    op = LowRankNoiseModel.fit(data, sample_rate=SAMPLE_RATE, config=CONFIG)
    # the operator applies to inputs of any length at the same sample rate
    x = jax.random.normal(jax.random.key(7), (N_CHANNELS, 256))
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    weight = op.inverse_covariance(structure)
    np.testing.assert_allclose(weight(op.covariance(structure)(x)), x, atol=1e-8)


def test_white_model_is_the_zero_lag_of_the_weight():
    data, _, _ = simulate(jax.random.key(8), 1.0)
    model = LowRankNoiseModel.fit(
        data, sample_rate=SAMPLE_RATE, lead=(2, N_CHANNELS // 2), config=CONFIG
    )
    # the time-domain diagonal of the weight, on a grid fine enough to resolve the bins
    structure = jax.ShapeDtypeStruct((N_CHANNELS, 2**14), jnp.float64)
    weight = model.inverse_covariance(structure)
    impulses = (
        jnp.zeros((N_CHANNELS, *structure.shape))
        .at[jnp.arange(N_CHANNELS), :, 0]
        .set(jnp.eye(N_CHANNELS))
    )
    zero_lag = jnp.diag(jax.vmap(weight)(impulses)[:, :, 0])
    sigma = model.white().sigma
    assert sigma.shape == (2, N_CHANNELS // 2)
    # sigma is the square root of the one-sided PSD, which is twice the variance over the rate
    np.testing.assert_allclose(2 / (SAMPLE_RATE * sigma.ravel() ** 2), zero_lag, rtol=2e-3)


@pytest.mark.parametrize('minutes', [10, 60])
def test_bins_hold_enough_frequencies_of_the_shortest_observation(minutes):
    sample_rate, n_samples = 8.0, minutes * 60 * 8
    config = LowRankNoiseFitConfig()
    edges = np.array(bin_edges(config, sample_rate, n_samples))
    assert edges[-1] == config.mode_band[1]
    widths = np.diff(edges)
    # the first frequency of each bin, as the model places them
    frequencies = np.diff(np.ceil(edges * n_samples / sample_rate))
    assert (frequencies >= config.min_bin_frequencies).all()
    # above the narrowest width, the edges grow by the ratio, the last bin merged if narrow
    width = config.min_bin_frequencies * sample_rate / n_samples
    geometric = widths[:-1] > width * (1 + 1e-9)
    assert geometric.any()
    np.testing.assert_allclose(edges[1:-1][geometric] / edges[:-2][geometric], config.bin_ratio)


@pytest.mark.parametrize(
    'options',
    [
        {'bin_edges': (0.1,)},
        {'bin_edges': (0.2, 0.1)},
        {'threshold_quantile': 0.5},
        {'bin_ratio': 1.0},
        {'mode_band': (0.2, 0.1)},
        {'bin_edges': (0.5, 2.0), 'mode_band': (1e-4, 1.8)},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        LowRankNoiseFitConfig(**options)


def test_mode_band_narrower_than_a_bin():
    config = LowRankNoiseFitConfig(mode_band=(1e-4, 0.002))
    with pytest.raises(ValueError):
        LowRankNoiseModel.fit(jnp.zeros((4, 4096)), sample_rate=SAMPLE_RATE, config=config)


@pytest.mark.parametrize('coupled', [2, 3], ids=['white-last-bin', 'all-coupled'])
@pytest.mark.parametrize('n_samples', [32, 33], ids=['even', 'odd'])
def test_modes_weighted_by_the_diagonal_inverse_are_its_product(coupled, n_samples):
    model = toy_low_rank_model(coupled=coupled)
    structure = jax.ShapeDtypeStruct((3, n_samples), jnp.float64)
    T = model.modes_operator(structure)
    inverse_diagonal = model.decoupled().inverse_covariance(structure)
    x = jax.random.normal(jax.random.key(0), T.in_structure.shape)
    expected = inverse_diagonal(T(x))
    np.testing.assert_allclose(T.weighted_by(inverse_diagonal.diagonal)(x), expected, atol=1e-12)


def _dense(model: LowRankNoiseModel, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """T and D⁻¹ as dense matrices, on TODs of `n_samples`."""
    structure = jax.ShapeDtypeStruct((model.diagonal.shape[1], n_samples), jnp.float64)
    T = np.asarray(model.modes_operator(structure).as_matrix())
    D_inv = np.asarray(model.decoupled().inverse_covariance(structure).as_matrix())
    return T, D_inv


@pytest.mark.parametrize('coupled', [2, 3])
@pytest.mark.parametrize('n_samples', [32, 33])
def test_modes_core_is_the_inverse_of_the_gram_of_the_modes(coupled, n_samples):
    """K = (1 + Tᵀ D⁻¹ T)⁻¹, zero and Nyquist frequencies included."""
    model = toy_low_rank_model(coupled=coupled)
    T, D_inv = _dense(model, n_samples)
    expected = np.linalg.inv(np.eye(T.shape[1]) + T.T @ D_inv @ T)
    actual = np.asarray(model.modes_core(n_samples).as_matrix())
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('coupled', [2, 3])
@pytest.mark.parametrize('n_samples', [32, 33])
def test_marginalized_modes_give_the_low_rank_weight(coupled, n_samples):
    """D⁻¹ - D⁻¹ T K Tᵀ D⁻¹ = N⁻¹: the modes with their prior are the Woodbury term."""
    model = toy_low_rank_model(coupled=coupled)
    T, D_inv = _dense(model, n_samples)
    structure = jax.ShapeDtypeStruct((3, n_samples), jnp.float64)
    expected = np.asarray(model.inverse_covariance(structure).as_matrix())
    weight = D_inv - D_inv @ T @ np.linalg.solve(np.eye(T.shape[1]) + T.T @ D_inv @ T, T.T @ D_inv)
    # the complex modes of the Woodbury inverse are exact off the zero and Nyquist frequencies only
    covariance = np.asarray(model.covariance(structure).as_matrix())
    np.testing.assert_allclose(np.linalg.inv(weight), covariance, atol=1e-10)
    if np.isrealobj(np.asarray(model.modes)):
        np.testing.assert_allclose(weight, expected, atol=1e-12)
