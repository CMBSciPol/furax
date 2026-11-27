"""The 1/f noise model of each detector, and its fit."""

import logging
from dataclasses import dataclass, field
from typing import Any, Self

import cadre
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Key, PyTree

from furax import AbstractLinearOperator
from furax.core import SymmetricBandToeplitzOperator

from ..config import NoiseFitConfig
from ._base import NoiseModel, _per_sample, _stationary_draw
from .spectra import _create_frequency_mask_from_config, _detector_psd, apodization_window
from .white import WhiteNoiseModel, _approximate_white_noise

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class OneOverFNoiseModel(NoiseModel):
    r"""The 1/f noise of each detector, of the one-sided PSD $\sigma^2 (1 + ((f + f_0) / f_k)^\alpha)$.

    Its operators are Toeplitz, banded to `correlation_length` samples, which they need.
    """

    sigma: Float[Array, ' dets']
    alpha: Float[Array, ' dets']
    fk: Float[Array, ' dets']
    f0: Float[Array, ' dets']
    sample_rate: Float[Array, ''] | float
    """The sample rate of the TOD in Hz."""
    correlation_length: int | None = field(default=None, metadata={'static': True})
    """The band of the Toeplitz operators, in samples."""

    @jax.jit
    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        ratio = (f + self.f0[..., None]) / self.fk[..., None]
        return jnp.where(
            self.sigma[..., None] > 0,
            self.sigma[..., None] ** 2 * (1 + ratio ** self.alpha[..., None]),
            0,
        )

    def covariance(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        """A banded Toeplitz approximation of $N$, to `correlation_length` samples.

        Its band is the autocorrelation of the inverse spectrum of the band of
        [`inverse_covariance`][], cut and apodized again, so that the two nearly invert each other.
        Neither is exact: the correlations past the band are dropped.
        """
        sample_rate, correlation_length = self.sample_rate, self._correlation_length()
        window = apodization_window(correlation_length)

        fft_size = SymmetricBandToeplitzOperator._get_next_power_of_two(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = _per_sample(self.psd(freq), sample_rate)
        invntt = jnp.fft.irfft(1.0 / eval_psd, n=fft_size)[..., :correlation_length]
        inv_band = invntt * window
        # pad only the last dimension
        pad_width = [(0, 0)] * (inv_band.ndim - 1) + [(0, fft_size - (2 * inv_band.shape[-1] - 1))]
        padded_band = jnp.pad(inv_band, pad_width)
        symmetrised_band = jnp.concatenate([padded_band, inv_band[..., -1:0:-1]], axis=-1)
        eff_inv_psd = jnp.fft.rfft(symmetrised_band, n=fft_size).real
        new_band = jnp.fft.irfft(1.0 / eff_inv_psd, n=fft_size)[..., :correlation_length] * window
        return SymmetricBandToeplitzOperator(new_band, in_structure=in_structure)

    def inverse_covariance(
        self, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        """A banded Toeplitz approximation of $N^{-1}$, to `correlation_length` samples.

        Its band is the autocorrelation of the inverse PSD, cut at `correlation_length` samples and
        apodized.
        """
        sample_rate, correlation_length = self.sample_rate, self._correlation_length()

        fft_size = SymmetricBandToeplitzOperator._get_next_power_of_two(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = _per_sample(self.psd(freq), sample_rate)
        inv_psd = jnp.where(eval_psd > 0, 1 / eval_psd, 0.0)
        invntt = jnp.fft.irfft(inv_psd, n=fft_size)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(invntt * window, in_structure=in_structure)

    def draw(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[Array]:
        """A stationary realization of the noise (see [`NoiseModel.draw`][]).

        Its covariance is the circulant matrix of the PSD, not the banded [`covariance`][]: the
        band cuts and apodizes the correlations, which the draw keeps whole.
        """
        sample_rate = self.sample_rate
        n = in_structure.shape[-1]
        psd = self.psd(jnp.fft.rfftfreq(n, 1 / sample_rate))
        return _stationary_draw(key, psd, n, sample_rate).astype(in_structure.dtype)

    def decoupled(self) -> Self:
        return self

    def white(self) -> WhiteNoiseModel:
        r"""The white noise of level $\sigma$, the high-frequency plateau of the PSD."""
        return WhiteNoiseModel(sigma=self.sigma, sample_rate=self.sample_rate)

    def detector_parameters(self) -> dict[str, Array]:
        return {'sigma': self.sigma, 'alpha': self.alpha, 'fk': self.fk, 'f0': self.f0}

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
        """The 1/f model of each detector, from its TOD, by maximum Whittle likelihood.

        The likelihood is that of the Welch PSD, over the frequencies the config keeps
        (see [`NoiseFitConfig`][furax.mapmaking.config.NoiseFitConfig]). The fit starts from a power law fitted to the low frequencies above the
        white level.

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
        result = _fit_one_over_f_model(f, psd, sample_rate, hwp_frequency, config)
        params = result['fit'].T.reshape(4, *samples.shape[:-1])
        return cls(*params, sample_rate=sample_rate)

    def _correlation_length(self) -> int:
        """The band of the Toeplitz operators, which must be set."""
        if self.correlation_length is None:
            raise ValueError('The operators of the 1/f noise model need its correlation_length')
        return self.correlation_length


def _fit_one_over_f_model(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, 'dets freqs'],
    sample_rate: Array,
    hwp_frequency: Array,
    config: NoiseFitConfig | None = None,
) -> dict[str, Any]:
    """The fit of [`OneOverFNoiseModel.fit`][], with its diagnostics.

    Returns:
        `fit`, the parameters `(sigma, alpha, fk, f0)` of each detector, `(det, 4)`; `loss`, the
        Whittle negative log-likelihood at the fit, `(det,)`; `num_iter`, the optimizer
        iterations, `(det,)`; `inv_fisher`, the inverse Fisher matrix of the parameters,
        `(det, 4, 4)`; and `num_freq`, the number of frequencies fitted.
    """
    config = config or NoiseFitConfig()
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
    )
    nyquist = sample_rate / 2

    # one detector at a time: under vmap, every detector would iterate as long as the slowest,
    # and the optimizer state of all of them would be held at once
    results = jax.lax.map(
        lambda psd: _fit_psd_model_masked(
            f,
            psd,
            mask=mask,
            low_f_threshold=nyquist * config.low_freq_nyquist,
            high_f_threshold=nyquist * config.high_freq_nyquist,
            max_iter=config.max_iter,
            tol=config.tol,
        ),
        Pxx,
    )
    results['num_freq'] = jnp.sum(mask)

    def fit_details(results: dict[str, Any]) -> None:
        logger.debug(
            'PSD fit, per detector: loss per bin %s, iterations %s',
            np.asarray(results['loss']) / results['num_freq'],
            np.asarray(results['num_iter']),
        )

    # Decided when the fit is traced: no host callback unless the record would be logged
    if logger.isEnabledFor(logging.DEBUG):
        jax.debug.callback(fit_details, results)

    return results


def _fit_psd_model_masked(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, ' freqs'],
    mask: Float[Array, ' freqs'],
    low_f_threshold: Array,
    high_f_threshold: Array,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """The fit of one detector (see [`_fit_one_over_f_model`][]), over the unmasked frequencies.

    Args:
        f: Frequency array (Hz). Shape: (n_freq,)
        Pxx: Power spectral density values. Shape: (n_freq,)
        mask: Frequency mask which is 1 at valid frequencies and 0 otherwise
        low_f_threshold: Frequency below which the PSD is assumed to be dominated by 1/f noise.
            Used in choosing the starting point for alpha and f_knee
        high_f_threshold: Frequency above which the PSD is assumed to be dominated by white noise
            Used in choosing the starting point for sigma
        max_iter: Maximum number of optimizer iterations.
        tol: Convergence tolerance for the optimizer.

    Returns:
        One detector's entries of [`_fit_one_over_f_model`][], but `num_freq`.
    """
    # 1. Initial parameter estimate (sigma, alpha, f_knee, f0)
    init_params = _approximate_fit(
        f, Pxx, mask, low_f_threshold=low_f_threshold, high_f_threshold=high_f_threshold
    )

    # 2. Loss function in scaled variables (scaled = params / init_params)
    def loss_fn(scaled_params: Array) -> Array:
        params = scaled_params * init_params
        return _compute_whittle_neglnlike(params, f, Pxx, mask)

    # 3. Run optimisation with bounds on scaled parameters
    lo = jnp.array([1e-3, 1e-3, 1e-3, 1e-10])
    up = jnp.array([1e3, 1e3, 1e3, 1e3])
    scaled_params, state = cadre.minimize(
        loss_fn,
        jnp.ones_like(init_params),
        solver_name='optax_lbfgs',
        max_iter=max_iter,
        atol=0,
        rtol=tol,
        lower_bound=lo,
        upper_bound=up,
    )

    # 4. Extract results
    params = scaled_params * init_params
    loss_final = loss_fn(scaled_params)
    num_iter = state.iter_num

    # 5. Fisher information matrix
    scaled_fisher = 0.5 * jax.hessian(loss_fn)(scaled_params)
    inv_fisher = (
        jnp.linalg.pinv(scaled_fisher, rtol=1e-12) * init_params[None, :] * init_params[:, None]
    )

    return {
        'fit': params,
        'loss': loss_final,
        'num_iter': jnp.array(num_iter, dtype=int),
        'inv_fisher': inv_fisher,
    }


def _approximate_fit(
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    low_f_threshold: Array,
    high_f_threshold: Array,
) -> Float[Array, '4']:
    # Obtain an approximate fit for all parameters:
    # sigma: estimated from high-frequency > high_f_threshold
    # f0: set to be the minimum non-zero frequency
    # alpha, fk: fit power-law to (P-sigma**2) as a function of (f+f0) from low_frequency < low_f_threshold

    sigma = _approximate_white_noise(f, Pxx, mask, high_f_threshold=high_f_threshold)
    f0 = jnp.min(jnp.where(jnp.logical_and(mask, f > 0), f, jnp.inf))

    return _approximate_fit_given_two_parameters(sigma, f0, f, Pxx, mask, low_f_threshold)


def _approximate_fit_given_two_parameters(
    sigma: Float[Array, '1'],
    f0: Float[Array, '1'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    low_f_threshold: Array,
) -> Float[Array, '4']:
    # Given sigma and f0, obtain an approximate fit for alpha and fk
    # by fitting power-law to (P-sigma**2) as a function of (f+f0).

    mask = jnp.logical_and(mask, f < low_f_threshold)
    mask = jnp.logical_and(mask, Pxx > sigma**2)

    # TODO: ensure sum(mask) >= 2 required for this method to work
    weight = (jnp.sum(mask) > 0) * (1 / jnp.sum(mask))
    mean_logf = jnp.sum(jnp.where(mask, jnp.log(f + f0), 0)) * weight
    mean_logP = jnp.sum(jnp.where(mask, jnp.log(Pxx - sigma**2), 0)) * weight

    nom = jnp.sum(
        jnp.where(mask, (jnp.log(f + f0) - mean_logf) * (jnp.log(Pxx - sigma**2) - mean_logP), 0)
    )
    denom = jnp.sum(jnp.where(mask, (jnp.log(f + f0) - mean_logf) ** 2, 0))

    bf_alpha = nom * (denom > 0) * (1 / denom)
    bf_log_gamma = mean_logP - mean_logf * bf_alpha
    bf_fk = jnp.exp((jnp.log(sigma**2) - bf_log_gamma) * (bf_alpha > 0) * (1 / bf_alpha))

    return jnp.array([sigma, bf_alpha, bf_fk, f0])


def _one_over_f_model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = params
    return jnp.where(sigma > 0, sigma**2 * (1 + ((x + f0) / fk) ** alpha), 0)


def _compute_whittle_neglnlike(
    params: Float[Array, '4'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
) -> Float[Array, '1']:
    # Computes -2logL up to a constant, using Whittle's approximation
    # Factor of 1/Pxx is included in the log term for better interpretability,
    # even though it results in a constant offset
    Pxx_pred = _one_over_f_model(params, f)
    return jnp.sum(jnp.where(mask, jnp.log(Pxx_pred / Pxx) + (Pxx / Pxx_pred), 0))
