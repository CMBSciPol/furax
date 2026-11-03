"""Detector-correlated noise model: a diagonal plus a few common modes per frequency bin."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from furax.core import LowRankFourierOperator

from .config import LowRankNoiseFitConfig

__all__ = [
    'fit_low_rank_noise',
]

# Tracy-Widom (beta = 1) quantiles, for the fluctuation of the largest eigenvalue of pure noise.
_TRACY_WIDOM_QUANTILES = (0.9, 0.95, 0.99, 0.999)
_TRACY_WIDOM_VALUES = (0.4501, 0.9793, 2.0234, 3.2724)

# Neighbouring Fourier coefficients of a Hann-windowed segment are correlated; every other one is
# close to independent, which the eigenvalue threshold assumes.
_FREQUENCY_STRIDE = 2

# Modes beyond the strongest ones are examined, to leave room for the single-channel rejection.
_CANDIDATES_PER_MODE = 3

# The diagonal of a channel never falls below this fraction of its measured power in a bin.
_DIAGONAL_FLOOR = 1e-2


def fit_low_rank_noise(
    samples: Float[Array, 'channel sample'],
    sample_rate: float,
    *,
    valid: Bool[Array, 'channel sample'] | None = None,
    n_real: int | Array | None = None,
    config: LowRankNoiseFitConfig | None = None,
) -> LowRankFourierOperator:
    r"""Fit the noise covariance $N(f) = D + V G V^\top$ in frequency bins from the data.

    Each channel is whitened by its power in the high-frequency band, where it is assumed white.
    In every bin of `config.bin_edges`, the eigendecomposition of the channel covariance gives the
    modes: those above the largest eigenvalue that pure noise would produce, with at most
    `config.max_modes`, and none carried mostly by a single channel. The diagonal $D$ is the
    remaining power of each channel, so the model reproduces every channel's power in the bin.

    The covariance is estimated from overlapping segments. A segment contributes for a channel
    only when none of its samples is flagged for that channel, so flagged samples are never
    replaced by zeros, which would create correlations. Segments in which too few channels are
    clean, or that reach into the padded tail, are left out entirely.

    Args:
        samples: Time-ordered data of all channels, with the samples last.
        sample_rate: Sample rate in Hz.
        valid: Whether each sample is valid, or `None` to keep them all.
        n_real: Number of samples before the padded tail, or `None` for no padding.
        config: Fit options.

    Returns:
        The covariance, as an operator on arrays of the shape of `samples`. Its
        [`inverse`][furax.core.LowRankFourierOperator.inverse] is the noise weight.
    """
    config = config or LowRankNoiseFitConfig()
    edges = tuple(float(e) for e in config.bin_edges)
    if len(edges) < 2 or list(edges) != sorted(edges):
        raise ValueError(f'Expected at least two increasing bin edges, got {edges}')
    if not _TRACY_WIDOM_QUANTILES[0] <= config.threshold_quantile <= _TRACY_WIDOM_QUANTILES[-1]:
        raise ValueError(
            f'threshold_quantile must be in [0.9, 0.999], got {config.threshold_quantile}'
        )

    n_channels, n_samples = samples.shape
    nperseg = min(config.nperseg, n_samples)
    starts = jnp.arange(0, n_samples - nperseg + 1, nperseg // 2)
    frequencies = np.fft.rfftfreq(nperseg, 1 / sample_rate)

    clean, used = _segment_weights(valid, n_real, starts, nperseg, n_channels, n_samples, config)
    weights = clean & used[:, None]  # (segment, channel)
    spectra, band_power = _segment_spectra(
        samples, starts, nperseg, frequencies, edges[-1], config.whitening_band
    )

    # per-channel noise level, from the clean segments
    count = weights.sum(axis=0)  # (channel,)
    alive = count > 0
    level = jnp.where(weights, band_power, 0).sum(axis=0) / jnp.maximum(count, 1)
    median = jnp.nanmedian(jnp.where(alive, level, jnp.nan))
    level = jnp.where(alive, level, jnp.where(jnp.isnan(median), 1.0, median))
    n_used = jnp.maximum(used.sum(), 1)

    diagonals, modes, powers = [], [], []
    for lo, hi in itertools.pairwise(edges):
        columns = np.flatnonzero((frequencies >= lo) & (frequencies < hi) & (frequencies > 0))
        columns = columns[::_FREQUENCY_STRIDE]
        if columns.size == 0:
            diagonal, vectors, eigenvalues = level, None, None
        else:
            # whitened coefficients, scaled so that a channel missing some segments keeps unit power
            z = spectra[:, :, columns] / jnp.sqrt(level)[None, :, None]
            z = jnp.where(weights[:, :, None], z, 0)
            z = z * jnp.sqrt(n_used / jnp.maximum(count, 1))[None, :, None]
            z = jnp.moveaxis(z, 1, 0).reshape(n_channels, -1)
            vectors, eigenvalues, power = _fit_bin(z, n_used * columns.size, alive, config)
            diagonal = level * jnp.maximum(
                power - (vectors**2 * eigenvalues).sum(axis=1), _DIAGONAL_FLOOR * power
            )
        diagonals.append(diagonal)
        modes.append(vectors)
        powers.append(eigenvalues)

    k = config.max_modes
    # bins: [0, edge0) repeats the first fitted bin, then the fitted ones, then white above
    fitted = list(zip(diagonals, modes, powers, strict=True))
    zero_modes = jnp.zeros((n_channels, k), level.dtype)
    zero_power = jnp.zeros(k, level.dtype)
    bins = [fitted[0], *fitted, (level, None, None)]
    diagonal = jnp.stack([d for d, _, _ in bins])
    mode_array = jnp.stack(
        [zero_modes if v is None else jnp.sqrt(level)[:, None] * v for _, v, _ in bins]
    )
    core = jnp.stack([jnp.diag(zero_power if g is None else g) for _, _, g in bins])
    return LowRankFourierOperator(
        diagonal,
        mode_array,
        core,
        edges=edges,
        sample_rate=float(sample_rate),
        in_structure=jax.ShapeDtypeStruct(samples.shape, samples.dtype),
    )


def _segment_weights(
    valid: Bool[Array, 'channel sample'] | None,
    n_real: int | Array | None,
    starts: Array,
    nperseg: int,
    n_channels: int,
    n_samples: int,
    config: LowRankNoiseFitConfig,
) -> tuple[Bool[Array, 'segment channel'], Bool[Array, ' segment']]:
    """Which segments are clean for each channel, and which segments are used at all."""
    stops = starts + nperseg
    in_real = stops <= (n_samples if n_real is None else n_real)
    if valid is None:
        return jnp.ones((starts.size, n_channels), bool), in_real
    flagged = jnp.cumsum(~valid, axis=-1, dtype=jnp.int32)
    flagged = jnp.concatenate([jnp.zeros_like(flagged[:, :1]), flagged], axis=-1)
    clean = (flagged[:, stops] == flagged[:, starts]).T
    used = in_real & (clean.mean(axis=1) >= config.min_valid_fraction)
    # without any such segment, fall back to every segment of real samples
    return clean, jnp.where(used.any(), used, in_real)


def _segment_spectra(
    samples: Float[Array, 'channel sample'],
    starts: Array,
    nperseg: int,
    frequencies: np.ndarray,
    top: float,
    band: tuple[float, float],
) -> tuple[Array, Array]:
    """Windowed spectra of the segments up to `top`, and the mean power in the white band.

    The spectra are normalised so that white noise of per-sample variance σ² has E|X|² = σ².
    """
    window = jnp.hanning(nperseg + 1)[:-1].astype(samples.dtype)
    norm = jnp.sqrt(jnp.sum(window**2))
    n_low = int(np.searchsorted(frequencies, top))
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    if not in_band.any():
        raise ValueError(f'No frequency in the whitening band {band} Hz')

    def spectrum(start: Array) -> tuple[Array, Array]:
        segment = jax.lax.dynamic_slice_in_dim(samples, start, nperseg, axis=1)
        segment = (segment - segment.mean(axis=1, keepdims=True)) * window
        coefficients = jnp.fft.rfft(segment) / norm
        power = jnp.mean(jnp.abs(coefficients[:, in_band]) ** 2, axis=1)
        return coefficients[:, :n_low], power

    return jax.lax.map(spectrum, starts)


def _fit_bin(
    z: Array, n_columns: Array, alive: Bool[Array, ' channel'], config: LowRankNoiseFitConfig
) -> tuple[Array, Array, Array]:
    """Modes of one bin from its whitened Fourier coefficients, shape `(channel, column)`.

    Returns the unit vectors `(channel, max_modes)`, their eigenvalues `(max_modes,)` (zero for
    modes that are not kept) and the power of each channel.
    """
    n_channels = z.shape[0]
    power = jnp.where(
        alive, jnp.mean(jnp.abs(z) ** 2, axis=1) * z.shape[1] / jnp.maximum(n_columns, 1), 1.0
    )
    r = jnp.concatenate([z.real, z.imag], axis=1)  # (channel, 2 column)
    n_candidates = min(_CANDIDATES_PER_MODE * config.max_modes, n_channels, r.shape[1])

    if r.shape[1] < n_channels:  # fewer samples than channels: diagonalize the small Gram matrix
        values, right = jnp.linalg.eigh(r.T @ r)
        values, right = values[::-1][:n_candidates], right[:, ::-1][:, :n_candidates]
        vectors = (r @ right) / jnp.sqrt(jnp.maximum(values, 1e-300))
    else:
        values, vectors = jnp.linalg.eigh(r @ r.T)
        values, vectors = values[::-1][:n_candidates], vectors[:, ::-1][:, :n_candidates]

    eigenvalues = values / jnp.maximum(n_columns, 1)  # covariance eigenvalues; pure noise is ~1
    threshold = _noise_edge(jnp.sum(alive), 2 * n_columns, config.threshold_quantile) / 2
    accepted = (values > threshold) & (jnp.abs(vectors).max(axis=0) < config.single_lim)
    # strongest accepted modes first, at most max_modes of them
    order = jnp.argsort(~accepted, stable=True)[: config.max_modes]
    keep = accepted[order]
    pad = config.max_modes - order.size
    vectors = jnp.where(keep, vectors[:, order], 0)
    eigenvalues = jnp.where(keep, eigenvalues[order], 0)
    return (
        jnp.pad(vectors, ((0, 0), (0, pad))),
        jnp.pad(eigenvalues, (0, pad)),
        power,
    )


def _noise_edge(n_channels: Array, n_samples: Array, quantile: float) -> Array:
    """Quantile of the largest eigenvalue of `X Xᵀ` for pure noise.

    `X` is an `n_channels × n_samples` matrix of unit-variance Gaussian entries. This is Johnstone's
    Marchenko-Pastur and Tracy-Widom approximation.
    """
    p = jnp.sqrt(n_channels - 0.5)
    m = jnp.sqrt(n_samples - 0.5)
    centre = (m + p) ** 2
    scale = (m + p) * (1 / m + 1 / p) ** (1 / 3)
    fluctuation = jnp.interp(
        quantile, jnp.array(_TRACY_WIDOM_QUANTILES), jnp.array(_TRACY_WIDOM_VALUES)
    )
    return centre + scale * fluctuation
