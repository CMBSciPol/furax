from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import block_diag

from furax.mapmaking.noise import NoiseModel, OneOverFNoiseModel, WhiteNoiseModel
from tests.mapmaking.noise.helpers import toy_low_rank_model

SAMPLE_RATE = 8.0
N_DRAWS = 20_000
SIGMA = jnp.array([0.5, 1.0, 2.0])


def white(n_samples: int) -> tuple[WhiteNoiseModel, np.ndarray]:
    variance = np.asarray(SIGMA) ** 2 * SAMPLE_RATE / 2
    return WhiteNoiseModel(SIGMA, SAMPLE_RATE), np.diag(np.repeat(variance, n_samples))


def one_over_f(n_samples: int) -> tuple[OneOverFNoiseModel, np.ndarray]:
    model = OneOverFNoiseModel(
        sigma=SIGMA,
        alpha=jnp.full(3, -2.0),
        fk=jnp.array([0.5, 1.0, 2.0]),
        f0=jnp.full(3, 0.1),
        sample_rate=SAMPLE_RATE,
    )
    psd = np.asarray(model.psd(jnp.fft.rfftfreq(n_samples, 1 / SAMPLE_RATE)))
    variance = psd * SAMPLE_RATE / 2  # per sample, in each frequency
    # the covariance of a periodic stationary noise: the circulant matrix of its autocorrelation
    column = np.fft.irfft(variance, n=n_samples)
    lags = (np.arange(n_samples)[None, :] - np.arange(n_samples)[:, None]) % n_samples
    return model, block_diag(*column[:, lags])


def low_rank(
    n_samples: int, coupled: int, decoupled: bool = False
) -> tuple[NoiseModel, np.ndarray]:
    model = toy_low_rank_model(coupled=coupled)
    if decoupled:
        model = model.decoupled()
    structure = jax.ShapeDtypeStruct((3, n_samples), jnp.float64)
    return model, np.asarray(model.covariance(structure).as_matrix())


COLOURED = {
    'one-over-f': one_over_f,
    'low-rank': partial(low_rank, coupled=2),
    'low-rank-all-coupled': partial(low_rank, coupled=3),
    'binned': partial(low_rank, coupled=2, decoupled=True),
}


# a white draw is the same at any length; the coloured ones differ at the Nyquist frequency
@pytest.mark.parametrize(
    ('make', 'n_samples'),
    [pytest.param(white, 32, id='white')]
    + [
        pytest.param(make, n, id=f'{name}-{parity}')
        for name, make in COLOURED.items()
        for n, parity in ((32, 'even'), (33, 'odd'))
    ],
)
def test_draws_have_the_covariance_of_the_model(make, n_samples):
    model, expected = make(n_samples)
    structure = jax.ShapeDtypeStruct((3, n_samples), jnp.float64)
    draw = jax.jit(jax.vmap(lambda key: model.draw(key, structure)))
    draws = draw(jax.random.split(jax.random.key(0), N_DRAWS)).reshape(N_DRAWS, -1)
    empirical = np.asarray(draws.T @ draws) / N_DRAWS
    # the standard error of each entry of the empirical covariance of Gaussian draws
    error = np.sqrt((np.outer(np.diag(expected), np.diag(expected)) + expected**2) / N_DRAWS)
    assert np.all(np.abs(empirical - expected) < 6 * error)
