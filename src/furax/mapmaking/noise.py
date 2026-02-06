from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax_grid_search import optimize
from jaxtyping import Array, Float, PyTree
from optax import lbfgs
from optax import tree_utils as otu
from scipy.signal import get_window

from furax.core import (
    AbstractLinearOperator,
    DiagonalOperator,
    FourierOperator,
    SymmetricBandToeplitzOperator,
)

from ._logger import logger
from .config import NoiseFitConfig


@jax.tree_util.register_dataclass
@dataclass
class NoiseModel:
    """Dataclass for noise models used for ground observation data"""

    @property
    @abstractmethod
    def n_detectors(self) -> int: ...

    @abstractmethod
    def psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

    @abstractmethod
    def log_psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

    @abstractmethod
    def to_array(self) -> Float[Array, 'dets n']: ...

    @abstractmethod
    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator: ...

    @abstractmethod
    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator: ...

    def to_operator_fourier(
        self,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        sample_rate: float,
        inverse: bool = True,
    ) -> FourierOperator:
        """Fourier operator representation of the noise model"""
        func = (lambda f: 1.0 / self.psd(f)) if inverse else self.psd
        # do not use apodization -- sufficient padding is done by the FourierOperator
        return FourierOperator(func, in_structure, sample_rate=sample_rate, apodize=False)

    def l2_loss(
        self, f: Float[Array, ' a'], Pxx: Float[Array, 'dets a'], mask: Float[Array, ' a']
    ) -> Float[Array, '']:
        """l2 loss in log-log spacea with given frequency mask"""
        pred = self.log_psd(f)
        loss = jnp.trapezoid(((pred - jnp.log10(Pxx)) * mask[None, :]) ** 2, jnp.log10(f))
        return jnp.mean(loss)


@jax.tree_util.register_dataclass
@dataclass
class WhiteNoiseModel(NoiseModel):
    """Dataclass for the white noise model used for ground observation data"""

    sigma: Float[Array, ' dets']

    @property
    def n_detectors(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return self.sigma**2

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return 2 * jnp.log10(self.sigma)

    def to_array(self) -> Float[Array, 'dets n']:
        return self.sigma[:, None]

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        return DiagonalOperator(self.sigma[:, None] ** 2, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        inv_var = jnp.where(self.sigma > 0, 1.0 / (self.sigma**2), 0.0)
        return DiagonalOperator(inv_var[:, None], in_structure=in_structure)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
        sample_rate: Array,
        hwp_frequency: Array,
        config: NoiseFitConfig = NoiseFitConfig(),
    ) -> 'WhiteNoiseModel':
        """Fit a white noise model to data"""
        sigma = fit_white_noise_model(
            f, Pxx, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
        )['fit']
        return cls(sigma)


@jax.tree_util.register_dataclass
@dataclass
class AtmosphericNoiseModel(NoiseModel):
    """Dataclass for the 1/f noise model used for ground observation data"""

    sigma: Float[Array, ' dets']
    alpha: Float[Array, ' dets']
    fk: Float[Array, ' dets']
    f0: Float[Array, ' dets']

    @property
    def n_detectors(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return jnp.where(
            self.sigma > 0, self.sigma**2 * (1 + ((f + self.f0) / self.fk) ** self.alpha), 0
        )

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return 2 * jnp.log10(self.sigma) + jnp.log10(1 + ((f + self.f0) / self.fk) ** self.alpha)

    def to_array(self) -> Float[Array, 'dets n']:
        return jnp.stack([self.sigma, self.alpha, self.fk, self.f0], axis=-1)

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))
        window = apodization_window(correlation_length)

        fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = self.psd(freq)
        invntt = jnp.fft.irfft(1.0 / eval_psd, n=fft_size)[..., :correlation_length]
        inv_band = invntt * window
        # pad only the last dimension
        pad_width = [(0, 0)] * (inv_band.ndim - 1) + [(0, fft_size - (2 * inv_band.shape[-1] - 1))]
        padded_band = jnp.pad(inv_band, pad_width)
        symmetrised_band = jnp.concatenate([padded_band, inv_band[..., -1:0:-1]], axis=-1)
        eff_inv_psd = jnp.fft.rfft(symmetrised_band, n=fft_size).real
        new_band = jnp.fft.irfft(1.0 / eff_inv_psd, n=fft_size)[:, :correlation_length] * window
        return SymmetricBandToeplitzOperator(new_band, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the inverse autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))

        fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = self.psd(freq)
        inv_psd = jnp.where(eval_psd > 0, 1 / eval_psd, 0.0)
        invntt = jnp.fft.irfft(inv_psd, n=fft_size)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(invntt * window, in_structure)

    def to_white_noise_model(self) -> WhiteNoiseModel:
        return WhiteNoiseModel(sigma=self.sigma)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
        sample_rate: Array,
        hwp_frequency: Array,
        config: NoiseFitConfig = NoiseFitConfig(),
    ) -> 'AtmosphericNoiseModel':
        """Fit a atmospheric (1/f) noise model to data"""
        result = fit_atmospheric_psd_model(
            f, Pxx, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
        )
        return cls(*result['fit'].T)


def apodization_window(size: int, kind: str = 'chebwin') -> Float[Array, ' {size}']:
    window_type: tuple[Any, ...]
    if kind == 'gaussian':
        q_apo = 3  # apodization factor: cut happens at q_apo * sigma in the Gaussian window
        window_type = ('general_gaussian', 1, 1 / q_apo * size)
    elif kind == 'chebwin':
        at = 150  # attenuation level (dB)
        window_type = ('chebwin', at)
    else:
        raise RuntimeError(f'Apodization window {kind!r} is not supported.')

    window = jnp.array(get_window(window_type, 2 * size))
    window = jnp.fft.ifftshift(window)[:size]
    return window


def fit_white_noise_model(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, 'dets freqs'],
    sample_rate: Array,
    hwp_frequency: Array,
    config: NoiseFitConfig = NoiseFitConfig(),
) -> dict[str, Any]:
    """Fit a white noise model to the periodogram in log space.

    This function fits a model of the form:
        PSD(f) = sigma^2

    where:
        - sigma: white noise level

    Args:
        f: Frequency array (Hz). Shape: (n_freq,)
        Pxx: Power spectral density values. Shape: (n_detectors, n_freq)
        hwp_frequency: HWP rotation frequency (Hz)
        config: NoiseFitConfig instance

    Returns:
        A dictionary containing following keys:
            fit: Array of fitted parameters [sigma].
                Shape: (n_detectors, 1)

    """
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
    )
    nyquist = sample_rate / 2

    results = jax.vmap(
        lambda f, Pxx: {
            'fit': _approximate_white_noise(
                f,
                Pxx,
                mask=mask,
                high_f_threshold=nyquist * config.high_freq_nyquist,
            )
        },
        in_axes=(None, 0),
        out_axes={'fit': 0},
    )(f, Pxx)

    return results


def fit_atmospheric_psd_model(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, 'dets freqs'],
    sample_rate: Array,
    hwp_frequency: Array,
    config: NoiseFitConfig = NoiseFitConfig(),
) -> dict[str, Any]:
    """Fit a 1/f PSD model to the periodogram in log space.

    This function fits a model of the form:
        PSD(f) = sigma^2 * (1 + ((f + f0) / f_knee)^alpha)

    where:
        - sigma: white noise level
        - alpha: power law index (typically negative)
        - f_knee: knee frequency
        - f0: minimum frequency offset

    Args:
        f: Frequency array (Hz). Shape: (n_freq,)
        Pxx: Power spectral density values. Shape: (n_detectors, n_freq)
        sample_rate: Sampling rate (Hz)
        hwp_frequency: HWP rotation frequency (Hz)
        config: NoiseFitConfig instance

    Returns:
        A dictionary containing following keys:
            fit: Array of fitted parameters [sigma, alpha, f_knee, f_min].
                Shape: (n_detectors, 4)
            loss: Array contaning loss function values (-2logL) evaluated at the fitted parameters.
                Shape: (n_detectors,)
            num_iter: Array containing number of iterations spent to obtain the fit.
                Shape: (n_detectors,)
            fisher: Array containing fisher matrix values for the fitted parameters.
                Shape: (n_detectors, 4, 4)

    """
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
    )
    nyquist = sample_rate / 2

    results = jax.vmap(
        lambda f, Pxx: _fit_psd_model_masked(
            f,
            Pxx,
            mask=mask,
            low_f_threshold=nyquist * config.low_freq_nyquist,
            high_f_threshold=nyquist * config.high_freq_nyquist,
            max_iter=config.max_iter,
            tol=config.tol,
        ),
        in_axes=(None, 0),
        out_axes={'fit': 0, 'loss': 0, 'num_iter': 0, 'inv_fisher': 0, 'num_freq': None},
    )(f, Pxx)

    def fit_details(results: dict[str, Any]) -> None:
        logger.info(
            f'---- PSD model fit details ----\n'
            f'Number of frequency bins: {results["num_freq"]}\n'
            f'Mean loss per bin: {results["loss"] / results["num_freq"]}\n'
            f'Number of iterations: {results["num_iter"]}\n'
            f'-------------------------------'
        )

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
) -> dict['str', Any]:
    """Fit a 1/f PSD model to the periodogram in log space.

    This function fits a model of the form:
        PSD(f) = sigma^2 * (1 + ((f + f0) / f_knee)^alpha)

    where:
        - sigma: white noise level
        - alpha: power law index (typically negative)
        - f_knee: knee frequency
        - f0: minimum frequency offset

    Args:
        f: Frequency array (Hz). Shape: (n_freq,)
        Pxx: Power spectral density values. Shape: (n_detectors, n_freq)
        mask: Frequency mask which is 1 at valid frequencies and 0 otherwise
        low_f_threshold: Frequency below which the PSD is assumed to be dominated by 1/f noise.
            Used in choosing the starting point for alpha and f_knee
        high_f_threshold: Frequency above which the PSD is assumed to be dominated by white noise
            Used in choosing the starting point for sigma

    Returns:
        params: Array of fitted parameters [sigma, alpha, f_knee, f_min].
            Shape: (4,)

    Examples:
        >>> # Basic usage - fit entire frequency range (excluding DC)
        >>> params = fit_psd_model(f, Pxx)

        >>> # Fit only low frequencies
        >>> params = fit_psd_model(f, Pxx, f_min=0.01, f_max=1.0)

        >>> # Mask out specific frequency bands (e.g., HWP modes)
        >>> mask_intervals = jnp.array([[3.8, 7.8], [4.2, 8.2]])  # Mask 3.8-4.2 Hz and 7.8-8.2 Hz
        >>> params = fit_psd_model(f, Pxx, f_mask_intervals=mask_intervals)

    """

    init_params = _approximate_fit(
        f, Pxx, mask, low_f_threshold=low_f_threshold, high_f_threshold=high_f_threshold
    )

    loss = lambda scaled_params: _compute_whittle_neglnlike(
        scaled_params * init_params, f, Pxx, mask
    )
    opt = lbfgs()

    # bounds
    up = jnp.array([1e3, 1e3, 1e3, 1e3])
    lo = jnp.array([1e-3, 1e-3, 1e-3, 1e-10])

    # perform minimization
    scaled_params, state = optimize(
        jnp.ones_like(init_params),
        loss,
        opt,
        max_iter=max_iter,
        tol=tol,
        upper_bound=up,
        lower_bound=lo,
    )
    params = scaled_params * init_params
    scaled_fisher = 0.5 * jax.hessian(loss)(scaled_params)
    inv_fisher = (
        jnp.linalg.pinv(scaled_fisher, rtol=1e-12) * init_params[None, :] * init_params[:, None]
    )

    return {
        'fit': params,
        'loss': state.best_val,
        'num_iter': jnp.array(otu.tree_get(state, 'count'), dtype=int),
        'inv_fisher': inv_fisher,
        'num_freq': jnp.sum(mask),
    }


def _create_frequency_mask_from_config(
    f: Float[Array, ' a'],
    sample_rate: Array,
    hwp_frequency: Array,
    config: NoiseFitConfig = NoiseFitConfig(),
) -> Float[Array, ' a']:
    """Create a float mask for frequency selection and interval masking from a NoiseFitConfig."""

    hwp_freq = jnp.abs(hwp_frequency)
    ptc_freq = config.ptc_freq
    d = config.freq_mask_width / 2

    n_intervals = 3 * config.mask_hwp_harmonics + 2 * config.mask_ptc_harmonics
    intervals = jnp.zeros((n_intervals, 2), dtype=f.dtype)

    if config.mask_hwp_harmonics:
        # Mask 1, 2, 4 harmonics of HWP
        intervals = intervals.at[:3, :].set(
            [
                [hwp_freq - d, hwp_freq + d],
                [2 * hwp_freq - d, 2 * hwp_freq + d],
                [4 * hwp_freq - d, 4 * hwp_freq + d],
            ]
        )
    if config.mask_ptc_harmonics:
        # Mask 1, 2 harmonics of PTC
        intervals = intervals.at[-2:, :].set(
            [
                [ptc_freq - d, ptc_freq + d],
                [2 * ptc_freq - d, 2 * ptc_freq + d],
            ]
        )

    nyquist = sample_rate / 2
    f_min = nyquist * config.min_freq_nyquist
    f_max = nyquist * config.max_freq_nyquist

    return _create_frequency_mask(f, f_min=f_min, f_max=f_max, f_mask_intervals=intervals)


def _create_frequency_mask(
    f: Float[Array, ' a'],
    f_min: Array | None,
    f_max: Array | None,
    f_mask_intervals: Float[Array, 'n_intervals 2'] | None,
) -> Float[Array, ' a']:
    """Create a float mask for frequency selection and interval masking.

    Args:
        f: Frequency array (Hz). Shape: (n_freq,)
        f_min: Minimum frequency (inclusive) for fitting region. If None, no lower bound.
        f_max: Maximum frequency (exclusive) for fitting region. If None, no upper bound.
        f_mask_intervals: Array of shape (n_mask_intervals, 2) specifying frequency
            intervals to mask during fitting. Each row is [f_start, f_end) where
            f_start is inclusive and f_end is exclusive.

    Returns:
        mask: Float array where 1.0 indicates frequencies to include in fit, 0.0 to exclude.
            Shape: (n_freq,) with same dtype as f.
    """
    # Start with all frequencies included
    mask = jnp.ones_like(f, dtype=f.dtype)

    # Apply f_min bound (inclusive)
    if f_min is not None:
        mask = jnp.where(f >= f_min, mask, 0.0)

    # Apply f_max bound (exclusive)
    if f_max is not None:
        mask = jnp.where(f < f_max, mask, 0.0)

    # Apply mask intervals
    if f_mask_intervals is not None:
        # Check all intervals at once
        # Shape: (n_freq, n_intervals) - broadcasts f[:,None] against intervals
        in_any_interval = jnp.any(
            (f[:, None] >= f_mask_intervals[:, 0]) & (f[:, None] < f_mask_intervals[:, 1]),
            axis=1,
        )
        # Set mask to 0 where frequency is in any masked interval
        mask = jnp.where(in_any_interval, 0.0, mask)

    return mask


def _approximate_white_noise(
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    high_f_threshold: Array,
) -> Float[Array, '1']:
    # Obtain an approximate level estimated from high-frequency > high_f_threshold

    high_f_mask = jnp.logical_and(mask, f > high_f_threshold)
    return jnp.sqrt(jnp.sum(jnp.where(high_f_mask, Pxx, 0)) / jnp.sum(high_f_mask))


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


def _log_atmospheric_model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack_four(params)
    return 2 * jnp.log10(sigma) + jnp.log10(1 + ((x + f0) / fk) ** alpha)


def _atmospheric_model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack_four(params)
    return jnp.where(sigma > 0, sigma**2 * (1 + ((x + f0) / fk) ** alpha), 0)


def _unpack_four(params: Float[Array, '4']) -> tuple[Array, Array, Array, Array]:
    return params[0], params[1], params[2], params[3]


def _compute_normal_neglnlike(
    params: Float[Array, '4'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    k: float,
) -> Float[Array, '1']:
    # Computes -2logL up to a constant, assuming k effective degrees of freedom
    # Assumes that the PSD estimates follow a normal distribution
    Pxx_pred = _atmospheric_model(params, f)
    return jnp.sum(jnp.where(mask, (k / 2) * (Pxx / Pxx_pred - 1) ** 2, 0))


def _compute_chisq_neglnlike(
    params: Float[Array, '4'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    k: float,
) -> Float[Array, '1']:
    # Computes -2logL up to a constant, assuming k effective degrees of freedom
    # Assumes that the PSD estimates follow a chi-sq distribution of order k
    Pxx_pred = _atmospheric_model(params, f)
    return jnp.sum(jnp.where(mask, (k - 2) * jnp.log(Pxx_pred) + k * (Pxx / Pxx_pred), 0))


def _compute_whittle_neglnlike(
    params: Float[Array, '4'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
) -> Float[Array, '1']:
    # Computes -2logL up to a constant, using Whittle's approximation
    # Factor of 1/Pxx is included in the log term for better interpretability,
    # even though it results in a constant offset
    Pxx_pred = _atmospheric_model(params, f)
    return jnp.sum(jnp.where(mask, jnp.log(Pxx_pred / Pxx) + (Pxx / Pxx_pred), 0))
