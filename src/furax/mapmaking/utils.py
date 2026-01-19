from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax_grid_search import optimize
from jaxtyping import Array, Float
from optax import tree_utils as otu
from scipy.signal import get_window
from toast import qarray as qa

from ._logger import logger
from .config import NoiseFitConfig


@partial(np.vectorize, signature='(4)->()')
def get_local_meridian_angle(quat):  # type: ignore[no-untyped-def]
    """
    Compute angle between local meridian and orientation vector from quaternions.

    Assumes that the quaternions encode the rotation between the celestial frame
    and some other frame (e.g. detector or boresight frame). The "orientation vector"
    is the unit vector of the latter frame obtained by rotating the X axis of the
    celestial frame. For a detector this will be the polarization sensitive direction.
    The local meridian vector is obtained by projecting the -Z axis of the celestial
    frame onto the plane orthogonal to the pointing direction.

    taken from
    https://github.com/hpc4cmb/toast/blob/toast3/src/toast/ops/stokes_weights/kernels_numpy.py#L12
    """
    zaxis = np.array([0, 0, 1], dtype=np.float64)
    xaxis = np.array([1, 0, 0], dtype=np.float64)

    vd = qa.rotate(quat, zaxis)
    vo = qa.rotate(quat, xaxis)

    # The vector orthogonal to the line of sight that is parallel
    # to the local meridian.
    dir_ang = np.arctan2(vd[1], vd[0])
    dir_r = np.sqrt(1.0 - vd[2] * vd[2])
    vm_z = -dir_r
    vm_x = vd[2] * np.cos(dir_ang)
    vm_y = vd[2] * np.sin(dir_ang)

    # Compute the rotation angle from the meridian vector to the
    # orientation vector.  The direction vector is normal to the plane
    # containing these two vectors, so the rotation angle is:
    #
    # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
    #
    alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1])
        - vd[1] * (vm_x * vo[2] - vm_z * vo[0])
        + vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    )
    alpha_x = vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]

    alpha = np.arctan2(alpha_y, alpha_x)
    return alpha


def next_fast_fft_size(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))


@partial(jax.jit, static_argnames=['fft_size'])
def interpolate_psd(
    freq: Float[Array, 'a b'],
    psd: Float[Array, 'a b'],
    fft_size: int,
    rate: float = 1.0,
) -> Float[Array, 'a {fft_size // 2 + 1}']:
    """Perform a logarithmic interpolation of PSD values."""
    interp_freq = jnp.fft.rfftfreq(fft_size, 1 / rate)
    # shift by fixed amounts in frequency and amplitude to avoid zeros
    freq_shift = rate / fft_size
    psd_shift = 0.01 * jnp.min(jnp.where(psd > 0, psd, 0))
    log_x = jnp.log10(interp_freq + freq_shift)
    log_xp = jnp.log10(freq + freq_shift)
    log_fp = jnp.log10(psd + psd_shift)
    # vectorize jax.numpy.interp, which only works on one-dimensional arrays
    f = partial(jnp.interp, left='extrapolate', right='extrapolate')
    vf = jnp.vectorize(f, signature='(m),(n),(n)->(m)')
    interp_psd: Array = vf(log_x, log_xp, log_fp)
    interp_psd = jnp.power(10.0, interp_psd) - psd_shift
    return interp_psd


@partial(jax.jit, static_argnames=['nperseg', 'rate'])
def estimate_psd(
    tod: Float[Array, 'a b'], nperseg: int, rate: float
) -> Float[Array, 'a {nperseg // 2 + 1}']:
    # average the periodogram estimate over blocks of size nperseg
    f, Pxx = jax.scipy.signal.welch(tod, fs=rate, nperseg=nperseg)
    # fit and compute full size PSD from fitted parameters
    params = _fit_psd_model_legacy(f, Pxx)
    freq = jnp.fft.rfftfreq(nperseg, 1 / rate)
    func = jnp.vectorize(_model, signature='(p),(n)->(n)')
    psd: Array = func(params, freq)
    return psd


@partial(jnp.vectorize, signature='(n),(n)->(p)')
def _fit_psd_model_legacy(f: Float[Array, ' a'], Pxx: Float[Array, ' a']) -> Float[Array, '4']:
    """Legacy implementation of fit_psd_model. Use fit_psd_model() instead."""
    # minimise loss for params: (sigma, alpha, fknee, fmin/fknee)
    loss = lambda params: _compute_loss(
        params.at[3].multiply(params[2]), x=f[1:], y=jnp.log10(Pxx[1:])
    )
    opt = optax.lbfgs()

    # estimate white noise level from the top 10% high-freq PSD
    # the formula is slightly more accurate than a simple sum
    hi_Pxx = Pxx[-(Pxx.size // 10 + 1) :]
    sigma_init = jnp.sqrt(jnp.sum(hi_Pxx**2) / jnp.sum(hi_Pxx))

    # initial guess
    maxf = f[-1]
    init_params = jnp.array([sigma_init, -1.0, 0.1 * maxf, 1e-5])

    # bounds
    up = jnp.array([jnp.inf, -0.1, maxf, 1e-3])
    lo = jnp.array([0, -10, e := jnp.finfo(f.dtype).eps, e])

    # perform minimization
    tol = jnp.median(Pxx**2) * 1e-12
    params, _ = optimize(
        init_params,
        loss,
        opt,
        max_iter=300,
        tol=tol,
        upper_bound=up,
        lower_bound=lo,
    )

    return params.at[3].multiply(params[2])  # type: ignore[no-any-return]


def fit_white_noise_model(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, 'dets freqs'],
    sample_rate: Array,
    hwp_freq: Array,
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
        hwp_freq: HWP rotation frequency (Hz)
        config: NoiseFitConfig instance

    Returns:
        A dictionary containing following keys:
            fit: Array of fitted parameters [sigma].
                Shape: (n_detectors, 1)

    """
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_freq=hwp_freq, config=config
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


def fit_psd_model(
    f: Float[Array, ' freqs'],
    Pxx: Float[Array, 'dets freqs'],
    sample_rate: Array,
    hwp_freq: Array,
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
        hwp_freq: HWP rotation frequency (Hz)
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
        f, sample_rate=sample_rate, hwp_freq=hwp_freq, config=config
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
    def fit_details(results):
        logger.info(
            f'---- PSD model fit details ----\n'
            f'Number of frequency bins: {results["num_freq"]}\n'
            f'Mean loss per bin: {results["loss"] / results["num_freq"]}\n'
            f'Number of iterations: {results["num_iter"]}\n'
            f'-------------------------------')
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
    opt = optax.lbfgs()

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
    hwp_freq: Array,
    config: NoiseFitConfig = NoiseFitConfig(),
) -> Float[Array, ' a']:
    """Create a float mask for frequency selection and interval masking from a NoiseFitConfig."""

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


def _compute_normal_neglnlike(
    params: Float[Array, '4'],
    f: Float[Array, ' a'],
    Pxx: Float[Array, ' a'],
    mask: Float[Array, ' a'],
    k: float,
) -> Float[Array, '1']:
    # Computes -2logL up to a constant, assuming k effective degrees of freedom
    # Assumes that the PSD estimates follow a normal distribution
    Pxx_pred = _model(params, f)
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
    Pxx_pred = _model(params, f)
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
    Pxx_pred = _model(params, f)
    return jnp.sum(jnp.where(mask, jnp.log(Pxx_pred / Pxx) + (Pxx / Pxx_pred), 0))


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


def _compute_loss(
    params: Float[Array, '4'], x: Float[Array, ' a'], y: Float[Array, ' a']
) -> Float[Array, '1']:
    y_pred = _log_model(params, x)
    w = jnp.where(x > 0, 1.0, 0.0)
    return jnp.mean(optax.l2_loss(w * y_pred, w * y))


def _log_model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack(params)
    return 2 * jnp.log10(sigma) + jnp.log10(1 + ((x + f0) / fk) ** alpha)


def _model(params: Float[Array, '4'], x: Float[Array, ' a']) -> Float[Array, ' a']:
    sigma, alpha, fk, f0 = _unpack(params)
    return jnp.where(sigma > 0, sigma**2 * (1 + ((x + f0) / fk) ** alpha), 0)


def _unpack(params: Float[Array, '4']) -> tuple[Array, Array, Array, Array]:
    net = params[0]
    alpha = params[1]
    f_knee = params[2]
    f_min = params[3]
    return net, alpha, f_knee, f_min


def run_lbfgs(init_params: Any, fun: Any, opt: Any, max_iter: int, tol: float) -> tuple[Array, Any]:
    """Minimizes a function using the L-BFGS solver.

    From https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver
    """
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):  # type: ignore[no-untyped-def]
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):  # type: ignore[no-untyped-def]
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(continuing_criterion, step, init_carry)
    return final_params, final_state


@partial(jax.jit, static_argnames=['correlation_length'])
def psd_to_invntt(
    psd: Float[Array, '...'], correlation_length: int
) -> Float[Array, '... {correlation_length}']:
    """Compute the inverse autocorrelation function from PSD values.
    The result is apodized and cut at the specified correlation length.
    """
    invntt = jnp.fft.irfft(1 / psd)[..., :correlation_length]
    window = apodization_window(correlation_length)
    return invntt * window


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


@partial(jax.jit, static_argnames=['nperseg', 'rate'])
def compute_cross_psd(
    tod: Float[Array, 'det samp'], nperseg: int, rate: float
) -> tuple[Float[Array, ' freq'], Float[Array, 'det det freq']]:
    """Compute cross power spectral density in the time-ordered data,
    and return the frequency bins and the detector covariance at each bin
    """

    n_dets = tod.shape[0]
    inds_x, inds_y = jnp.indices((n_dets, n_dets))
    f, Pxy = jax.scipy.signal.csd(tod[inds_x, :], tod[inds_y, :], fs=rate, nperseg=nperseg)
    Pxy = jnp.real(Pxy)

    return f, Pxy


def estimate_filtered_psd(
    tod: Float[Array, 'det samp'],
    nperseg: int,
    rate: float,
    freq: Float[Array, ' freq'],
    csd: Float[Array, 'det det freq'],
    freq_threshold: float,
    n_modes: int,
) -> Float[Array, 'det freq']:
    """Compute PCA common mode filtered power spectral density.
    Cross covariance below the frequency threshold is eigen-decomposed,
    and n_modes eigenvectors with the largest eigenvalues are selected.
    These 'common modes' are then fitted out from the time-ordered-data,
    after which the power spectral densities is computed.
    """

    # Take low frequencies excluding zero
    f_slice = jnp.logical_and(freq < freq_threshold, freq > 0)
    low_pass_csd = jnp.sum(jnp.where(f_slice, csd, 0.0), axis=-1)

    # Eigen decomposition
    evals, evecs = jnp.linalg.eigh(low_pass_csd)

    # Select eigenvectors with largest eigenvalues
    W = evecs[:, -n_modes:]

    # Fit away the corresponding common modes from the data
    C = tod @ tod.T
    alpha = C @ W @ jnp.linalg.inv(W.T @ C @ W)
    new_tod = tod - alpha @ W.T @ tod

    f, psd = jax.scipy.signal.welch(new_tod, fs=rate, nperseg=nperseg)

    return psd
