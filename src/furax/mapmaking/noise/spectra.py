"""Spectral estimates and frequency masks the noise models are fitted on."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from scipy.signal import get_window

from ..config import NoiseFitConfig


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


def padding_aware_welch(
    tod: Array,
    sample_padding: Array | int,
    *,
    fs: Array,
    nperseg: int,
    valid: Bool[Array, '*dets samples'] | None = None,
) -> tuple[Array, Array]:
    """Estimate a Welch PSD without averaging windows from a padded tail or over flagged samples.

    The segment length sets the number of frequency bins, so it is static and cannot depend on
    `sample_padding`. When fewer than `nperseg` real samples are available, there is therefore no
    all-real segment to average and the estimate falls back to the first one, padding included.
    Keep `nperseg` at or below the real sample count to stay in the regime where this is exact.

    With `valid`, a detector's estimate also leaves out every segment that contains a sample
    flagged invalid, which is the maximum-likelihood estimate for segments that are independent.
    A detector with no fully valid segment falls back to the segments of real samples, flagged
    ones included.

    Args:
        tod: Time-ordered data with detector and sample axes.
        sample_padding: Number of padded samples at the end of the sample axis.
        fs: Sampling frequency.
        nperseg: Number of samples per Welch segment, clipped to the buffer length.
        valid: Whether each sample is valid, with the shape of `tod`, or `None` to keep them all.

    Returns:
        The sample frequencies and power spectral density.
    """
    input_samples = tod.shape[-1]
    nperseg = min(nperseg, input_samples)

    noverlap = nperseg // 2
    frequencies, _, spectrum = jax.scipy.signal.stft(
        tod,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant',  # ty: ignore[invalid-argument-type]
        boundary=None,
        padded=False,
    )
    periodograms = jnp.real(jnp.conjugate(spectrum) * spectrum)
    window = (jnp.ones(1) if nperseg == 1 else jnp.hanning(nperseg + 1)[:-1]).astype(tod.dtype)
    density_scale = window.sum() ** 2 / (fs * jnp.sum(window**2))
    periodograms *= density_scale
    end = None if nperseg % 2 else -1
    periodograms = periodograms.at[..., 1:end, :].mul(2)

    step = nperseg - noverlap
    real_samples = input_samples - sample_padding
    n_valid = jnp.maximum(1, 1 + (real_samples - nperseg) // step)
    segments = jnp.arange(periodograms.shape[-1])
    real = segments < n_valid  # (segments,)
    if valid is not None:
        # a segment is clean when no sample in [start, start + nperseg) is flagged
        flagged = jnp.cumsum(~valid, axis=-1, dtype=jnp.int32)
        flagged = jnp.concatenate([jnp.zeros_like(flagged[..., :1]), flagged], axis=-1)
        starts = segments * step
        clean = flagged[..., starts + nperseg] == flagged[..., starts]  # (*dets, segments)
        keep = real & clean
        # a detector without a clean segment keeps all the real ones
        keep = jnp.where(keep.any(axis=-1, keepdims=True), keep, real)
    else:
        keep = jnp.broadcast_to(real, periodograms.shape[:-2] + real.shape)
    count = keep.sum(axis=-1, keepdims=True)
    psd = jnp.sum(jnp.where(keep[..., None, :], periodograms, 0), axis=-1) / count
    return frequencies, psd


def _detector_psd(
    samples: Float[Array, '*dets samples'],
    sample_rate: Array,
    n_real: int | Array | None,
    valid: Bool[Array, '...'] | None,
    nperseg: int,
) -> tuple[Float[Array, ' freq'], Float[Array, 'det freq']]:
    """The Welch PSD of each detector, the detector axes flattened (see [`padding_aware_welch`][])."""
    n_samples = samples.shape[-1]
    flat = samples.reshape(-1, n_samples)
    if valid is not None:
        valid = jnp.broadcast_to(valid, samples.shape).reshape(-1, n_samples)
    padding = 0 if n_real is None else n_samples - n_real
    return padding_aware_welch(flat, padding, fs=sample_rate, nperseg=nperseg, valid=valid)


def _create_frequency_mask_from_config(
    f: Float[Array, ' a'],
    sample_rate: Array,
    hwp_frequency: Array,
    config: NoiseFitConfig | None = None,
) -> Float[Array, ' a']:
    """Create a float mask for frequency selection and interval masking from a NoiseFitConfig."""
    config = config or NoiseFitConfig()

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
