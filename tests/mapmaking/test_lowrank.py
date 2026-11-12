import jax
import jax.numpy as jnp
import numpy as np
import pytest

from furax.mapmaking.config import LowRankNoiseFitConfig
from furax.mapmaking.lowrank import LowRankNoiseModel, fit_low_rank_noise

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


def _toy_model(n_channels: int = 3, coupled: int = 2) -> LowRankNoiseModel:
    """Three bins, the first `coupled` ones coupled by complex modes, one of them padded."""
    rng = np.random.default_rng(0)
    modes = rng.standard_normal((coupled, n_channels, 2)) + 1j * rng.standard_normal(
        (coupled, n_channels, 2)
    )
    modes[1, :, 1] = 0
    core = np.array([[[2.0, 0.5], [0.5, 1.0]], [[1.5, 0.0], [0.0, 0.0]], [[1.0, 0.2], [0.2, 0.5]]])
    diagonal = 1 + rng.random((3, n_channels))
    return LowRankNoiseModel(
        jnp.asarray(diagonal),
        jnp.asarray(modes),
        jnp.asarray(core[:coupled]),
        edges=(1.0, 2.0),
        sample_rate=8.0,
        lead=(n_channels,),
    )


def _dense(model: LowRankNoiseModel, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """T and D⁻¹ as dense matrices, on TODs of `n_samples`."""
    structure = jax.ShapeDtypeStruct((model.diagonal.shape[1], n_samples), jnp.float64)
    T = np.asarray(model.modes_operator(structure).as_matrix())
    D_inv = np.asarray(model.diagonal_inverse_operator(structure).as_matrix())
    return T, D_inv


@pytest.mark.parametrize('coupled', [2, 3])
@pytest.mark.parametrize('n_samples', [32, 33])
def test_modes_core_is_the_inverse_of_the_gram_of_the_modes(coupled, n_samples):
    """K = (1 + Tᵀ D⁻¹ T)⁻¹, zero and Nyquist frequencies included."""
    model = _toy_model(coupled=coupled)
    T, D_inv = _dense(model, n_samples)
    expected = np.linalg.inv(np.eye(T.shape[1]) + T.T @ D_inv @ T)
    core = model.modes_core(n_samples)
    structure = jax.ShapeDtypeStruct(
        model.modes_operator(jax.ShapeDtypeStruct((3, n_samples), jnp.float64)).in_structure.shape,
        jnp.float64,
    )
    basis = np.eye(T.shape[1]).reshape(T.shape[1], *structure.shape)
    actual = np.stack(
        [
            np.asarray(
                core.apply(jnp.asarray(e), model.edges, model.sample_rate, n_samples)
            ).ravel()
            for e in basis
        ],
        axis=1,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('coupled', [2, 3])
@pytest.mark.parametrize('n_samples', [32, 33])
def test_marginalized_modes_give_the_low_rank_weight(coupled, n_samples):
    """D⁻¹ - D⁻¹ T K Tᵀ D⁻¹ = N⁻¹: the modes with their prior are the Woodbury term."""
    model = _toy_model(coupled=coupled)
    T, D_inv = _dense(model, n_samples)
    structure = jax.ShapeDtypeStruct((3, n_samples), jnp.float64)
    expected = np.asarray(model.inverse_operator(structure).as_matrix())
    weight = D_inv - D_inv @ T @ np.linalg.solve(np.eye(T.shape[1]) + T.T @ D_inv @ T, T.T @ D_inv)
    # the complex modes of the Woodbury inverse are exact off the zero and Nyquist frequencies only
    covariance = np.asarray(model.operator(structure).as_matrix())
    np.testing.assert_allclose(np.linalg.inv(weight), covariance, atol=1e-10)
    if np.isrealobj(np.asarray(model.modes)):
        np.testing.assert_allclose(weight, expected, atol=1e-12)
