import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax.mapmaking.config import LowRankNoiseFitConfig
from furax.mapmaking.lowrank import fit_low_rank_noise

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
    return np.asarray(jnp.any(op.modes != 0, axis=1).sum(axis=-1))


def test_white_noise_has_no_modes():
    data, _, _ = simulate(jax.random.key(0), 0.0)
    op = fit_low_rank_noise(data, SAMPLE_RATE, config=CONFIG)
    assert mode_count(op).sum() == 0


def test_recovers_injected_common_modes():
    data, vectors, sigma = simulate(jax.random.key(1), 1.0)
    op = fit_low_rank_noise(data, SAMPLE_RATE, config=CONFIG)
    # the modes are injected in 0.02-0.2 Hz: found in the fitted bins, not above
    assert (mode_count(op)[1:] >= 2).all()
    # the white level above the last edge is an uncoupled last bin
    assert op.diagonal.shape[0] == op.modes.shape[0] + 1
    # the fitted subspace contains the injected one
    for b in range(1, len(CONFIG.bin_edges)):
        fitted = np.asarray(op.modes[b] / jnp.sqrt(sigma**2)[:, None])
        fitted = fitted[:, np.any(fitted != 0, axis=0)]
        basis, _ = np.linalg.qr(fitted)
        residual = np.asarray(vectors) - basis @ (basis.T @ np.asarray(vectors))
        assert np.linalg.norm(residual) < 0.2 * np.linalg.norm(vectors)


def test_diagonal_reproduces_channel_power_in_white_band():
    data, _, sigma = simulate(jax.random.key(2), 0.0)
    op = fit_low_rank_noise(data, SAMPLE_RATE, config=CONFIG)
    np.testing.assert_allclose(op.diagonal[-1], sigma**2, rtol=0.15)
    np.testing.assert_allclose(op.diagonal[2], sigma**2, rtol=0.4)


def test_weight_is_the_inverse_of_the_fitted_covariance():
    data, _, _ = simulate(jax.random.key(6), 1.0)
    op = fit_low_rank_noise(data, SAMPLE_RATE, config=CONFIG)
    # the operator applies to inputs of any length at the same sample rate
    x = jax.random.normal(jax.random.key(7), (N_CHANNELS, 256))
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    weight = op.inverse_operator(structure)
    np.testing.assert_allclose(weight(op.operator(structure)(x)), x, atol=1e-8)


def test_white_model_is_the_zero_lag_of_the_weight():
    data, _, _ = simulate(jax.random.key(8), 1.0)
    model = fit_low_rank_noise(data, SAMPLE_RATE, lead=(2, N_CHANNELS // 2), config=CONFIG)
    # the time-domain diagonal of the weight, on a grid fine enough to resolve the bins
    structure = jax.ShapeDtypeStruct((N_CHANNELS, 2**14), jnp.float64)
    weight = model.inverse_operator(structure)
    impulses = (
        jnp.zeros((N_CHANNELS, *structure.shape))
        .at[jnp.arange(N_CHANNELS), :, 0]
        .set(jnp.eye(N_CHANNELS))
    )
    zero_lag = jnp.diag(jax.vmap(weight)(impulses)[:, :, 0])
    sigma = model.to_white_noise_model().sigma
    assert sigma.shape == (2, N_CHANNELS // 2)
    np.testing.assert_allclose(1 / sigma.ravel() ** 2, zero_lag, rtol=2e-3)


@pytest.mark.parametrize(
    'config',
    [
        LowRankNoiseFitConfig(bin_edges=(0.1,)),
        LowRankNoiseFitConfig(bin_edges=(0.2, 0.1)),
        LowRankNoiseFitConfig(threshold_quantile=0.5),
    ],
)
def test_invalid_config(config):
    with pytest.raises(ValueError):
        fit_low_rank_noise(jnp.zeros((4, 4096)), SAMPLE_RATE, config=config)
