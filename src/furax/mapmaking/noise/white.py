"""White noise, uncorrelated in time and across detectors."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key, PyTree

from furax import AbstractLinearOperator
from furax.core import DiagonalOperator
from furax.tree import normal_like

from ..config import NoiseFitConfig
from ._base import NoiseModel, _per_sample
from .spectra import _create_frequency_mask_from_config, _detector_psd


@jax.tree_util.register_dataclass
@dataclass
class WhiteNoiseModel(NoiseModel):
    r"""White noise of each detector, of the one-sided PSD $\sigma^2$.

    Its standard deviation per sample is $\sigma \sqrt{f_s / 2}$ at the sample rate $f_s$.
    """

    sigma: Float[Array, ' dets']
    r"""The square root of the one-sided PSD of each detector, in units per $\sqrt{\mathrm{Hz}}$."""
    sample_rate: Float[Array, ''] | float
    """The sample rate of the TOD in Hz."""

    @jax.jit
    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        return jnp.broadcast_to(self.sigma[..., None] ** 2, (*self.sigma.shape, f.shape[-1]))

    def covariance(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        r"""The variance $\sigma^2 f_s / 2$ on every sample: the model has no correlation."""
        # ``sigma`` covers all axes but the trailing sample axis (``(ndets,)`` modulated,
        # ``(n_stokes, ndets)`` for a demodulated Stokes TOD); broadcast it over the samples.
        return DiagonalOperator(self._variance()[..., None], in_structure=in_structure)

    def inverse_covariance(
        self, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        """The inverse variance on every sample, zero for a channel of zero `sigma`."""
        variance = self._variance()
        inv_var = jnp.where(variance > 0, 1 / jnp.where(variance > 0, variance, 1), 0.0)
        return DiagonalOperator(inv_var[..., None], in_structure=in_structure)

    def draw(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[Array]:
        scale = DiagonalOperator(jnp.sqrt(self._variance())[..., None], in_structure=in_structure)
        return scale(normal_like(in_structure, key))

    def white(self) -> Self:
        return self

    def decoupled(self) -> Self:
        return self

    def detector_parameters(self) -> dict[str, Array]:
        return {'sigma': self.sigma}

    @classmethod
    def fit(
        cls,
        samples: Float[Array, '*dets samples'],
        *,
        sample_rate: Array,
        hwp_frequency: Array,
        n_real: int | Array | None = None,
        valid: Bool[Array, '...'] | None = None,
        config: NoiseFitConfig | None = None,
    ) -> Self:
        """The white level of each detector, from its TOD.

        The level is the square root of the mean of the Welch PSD above `config.high_freq_nyquist`
        of the Nyquist frequency, over the frequencies the config keeps
        (see [`NoiseFitConfig`][furax.mapmaking.config.NoiseFitConfig]).

        Args:
            samples: The TOD of each detector, the samples last.
            sample_rate: The sample rate of the TOD, in Hz.
            hwp_frequency: The rotation frequency of the HWP, in Hz, whose harmonics are masked.
            n_real: The number of samples before the padded tail, which is left out; `None` for
                no padding.
            valid: Whether each sample is valid, broadcast to the shape of `samples`; the Welch
                segments that hold a flagged sample are left out. `None` keeps them all.
            config: Fit options.
        """
        config = config or NoiseFitConfig()
        f, psd = _detector_psd(samples, sample_rate, n_real, valid, config.nperseg)
        mask = _create_frequency_mask_from_config(
            f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
        )
        threshold = sample_rate / 2 * config.high_freq_nyquist
        sigma = jax.vmap(lambda p: _approximate_white_noise(f, p, mask, threshold))(psd)
        return cls(sigma.reshape(samples.shape[:-1]), sample_rate=sample_rate)

    def _variance(self) -> Float[Array, ' dets']:
        """The variance of each detector per sample."""
        return _per_sample(self.sigma**2, self.sample_rate)


def _approximate_white_noise(
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    high_f_threshold: Array,
) -> Float[Array, '1']:
    # Obtain an approximate level estimated from high-frequency > high_f_threshold

    high_f_mask = jnp.logical_and(mask, f > high_f_threshold)
    return jnp.sqrt(jnp.sum(jnp.where(high_f_mask, Pxx, 0)) / jnp.sum(high_f_mask))
