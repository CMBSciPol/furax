"""Spectra and models shared by the tests of the noise models."""

import jax
import jax.numpy as jnp
import numpy as np

from furax.mapmaking.noise import LowRankNoiseModel


def make_freq(n: int = 256, sample_rate: float = 10.0) -> jax.Array:
    f = jnp.fft.rfftfreq(n, 1 / sample_rate)
    return f[1:]  # drop DC


def make_white_psd(f: jax.Array, sigma: float, n_dets: int = 1) -> jax.Array:
    noise = jnp.exp(0.1 * jax.random.normal(jax.random.key(0), (n_dets, len(f))))
    return sigma**2 * noise


def make_one_over_f_psd(
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


def toy_low_rank_model(n_channels: int = 3, coupled: int = 2) -> LowRankNoiseModel:
    """Three bins, the first `coupled` ones coupled by complex modes, one of them unused in one."""
    rng = np.random.default_rng(0)
    modes = rng.standard_normal((n_channels, 2)) + 1j * rng.standard_normal((n_channels, 2))
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
