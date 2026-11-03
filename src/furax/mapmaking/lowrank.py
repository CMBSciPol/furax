"""Detector-correlated noise model: a diagonal plus a few common modes per frequency bin."""

import itertools
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Inexact, PyTree

from furax.core import LowRankFourierOperator

from .config import LowRankNoiseFitConfig
from .noise import NoiseModel, WhiteNoiseModel

__all__ = [
    'LowRankNoiseModel',
    'fit_low_rank_noise',
]

# Tracy-Widom (beta = 1) quantiles, for the fluctuation of the largest eigenvalue of pure noise.
_TRACY_WIDOM_QUANTILES = (0.9, 0.95, 0.99, 0.999)
_TRACY_WIDOM_VALUES = (0.4501, 0.9793, 2.0234, 3.2724)

# The eigenvalue threshold assumes independent samples. With far more channels than samples, as in
# the low bins, even the 1/6 correlation of Hann coefficients two apart shifts the noise edge enough
# to let noise eigenvalues through: coefficients three apart, in disjoint segments, are independent.
_FREQUENCY_STRIDE = 3

# Modes beyond the strongest ones are examined, to leave room for the single-channel rejection.
_CANDIDATES_PER_MODE = 3

# The diagonal of a channel never falls below this fraction of its measured power in a bin.
_DIAGONAL_FLOOR = 1e-2


@jax.tree_util.register_dataclass
@dataclass
class LowRankNoiseModel(NoiseModel):
    r"""Noise covariance $N(f) = D_b + V_b G_b V_b^\dagger$ in frequency bins, across channels.

    The channels are the streams and detectors of a TOD, flattened stream-major, as
    [`LowRankFourierOperator`][furax.core.LowRankFourierOperator] takes them.
    """

    diagonal: Float[Array, 'bin channel']
    """The diagonal $D_b$ of each bin."""

    modes: Inexact[Array, 'bin channel mode']
    """The modes $V_b$ of each bin, padded with zeros."""

    core: Inexact[Array, 'bin mode mode']
    """The mode covariance $G_b$ of each bin."""

    edges: tuple[float, ...] = field(metadata={'static': True})
    """Interior bin boundaries in Hz (see [`LowRankFourierOperator.edges`][])."""

    sample_rate: float = field(metadata={'static': True})
    """Sample rate in Hz at which the bins are placed."""

    lead: tuple[int, ...] = field(metadata={'static': True})
    """The shape of the channel axes: `(streams, detectors)` for a demodulated TOD."""

    @property
    def n_detectors(self) -> int:
        return self.lead[-1]

    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        """The diagonal of $N(f)$: the power of each channel, correlated modes included."""
        bins = jnp.searchsorted(jnp.array(self.edges), f, side='right')
        power = (
            self.diagonal
            + jnp.einsum('bcj,bjk,bck->bc', self.modes, self.core, self.modes.conj()).real
        )
        return power[bins].T.reshape(*self.lead, f.size)

    def log_psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        return jnp.log10(self.psd(f))

    def to_array(self) -> Float[Array, 'dets n']:
        return self.diagonal.T

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> LowRankFourierOperator:
        """The covariance $N$, as an operator on TODs of structure `in_structure`."""
        return LowRankFourierOperator(
            self.diagonal,
            self.modes,
            self.core,
            edges=self.edges,
            sample_rate=self.sample_rate,
            in_structure=in_structure,
        )

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> LowRankFourierOperator:
        """The weight $N^{-1}$, as an operator on TODs of structure `in_structure`."""
        return self.operator(in_structure).inverse()

    def detector_parameters(self) -> WhiteNoiseModel:
        """The white noise level of each detector: the correlated modes have no per-detector form."""
        return self.to_white_noise_model()

    def to_white_noise_model(self) -> WhiteNoiseModel:
        """The white noise whose weight is the zero-lag of $N^{-1}$, channel by channel.

        It is the diagonal of the weight in the time domain: the average of the diagonal of
        $N^{-1}(f)$ over the frequencies up to Nyquist.
        """
        weight = self.inverse_operator(
            jax.ShapeDtypeStruct((self.diagonal.shape[1], 1), self.diagonal.dtype)
        )
        diagonal = (
            weight.diagonal
            + jnp.einsum('bcj,bjk,bck->bc', weight.modes, weight.core, weight.modes.conj()).real
        )
        bounds = np.array([0.0, *self.edges, self.sample_rate / 2]).clip(0, self.sample_rate / 2)
        fractions = jnp.asarray(np.diff(bounds) / (self.sample_rate / 2), diagonal.dtype)
        inverse_variance = fractions @ diagonal
        return WhiteNoiseModel(sigma=(1 / jnp.sqrt(inverse_variance)).reshape(self.lead))


def fit_low_rank_noise(
    samples: Float[Array, 'channel sample'],
    sample_rate: float,
    *,
    valid: Bool[Array, 'channel sample'] | None = None,
    n_real: int | Array | None = None,
    lead: tuple[int, ...] | None = None,
    config: LowRankNoiseFitConfig | None = None,
) -> LowRankNoiseModel:
    r"""Fit the noise covariance $N(f) = D + V G V^\top$ in frequency bins from the data.

    Each channel is whitened by its power in the high-frequency band, where it is assumed white.
    In every bin of `config.bin_edges`, the eigendecomposition of the channel covariance gives the
    modes: those above the largest eigenvalue that pure noise would produce, with at most
    `config.max_modes`, and none carried mostly by a single channel. The diagonal $D$ is the
    remaining power of each channel, so the model reproduces every channel's power in the bin.

    The covariance is estimated from disjoint segments, leaving out those that reach into the
    padded tail. The flagged samples are treated as missing, whatever they hold: in each segment,
    a channel's valid samples are taken relative to their mean and the flagged ones are set to
    zero, and the spectrum is normalized by the windowed power of the valid samples. Removing the
    mean first keeps a gap shared by several channels from writing the same offset into all of
    them, which would show as a correlation at every frequency. A channel contributes a segment
    only when enough of its samples are valid.

    Args:
        samples: Time-ordered data of all channels, with the samples last.
        sample_rate: Sample rate in Hz.
        valid: Whether each sample is valid, or `None` to keep them all.
        n_real: Number of samples before the padded tail, or `None` for no padding.
        lead: The shape of the channel axes, `(streams, detectors)` for a demodulated TOD; by
            default, a single axis.
        config: Fit options.

    Returns:
        The fitted noise model.
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
    starts = jnp.arange(0, n_samples - nperseg + 1, nperseg)
    frequencies = np.fft.rfftfreq(nperseg, 1 / sample_rate)

    used = starts + nperseg <= (n_samples if n_real is None else n_real)  # (segment,)
    spectra, band_power, valid_fraction, efficiency = _segment_spectra(
        samples, valid, starts, nperseg, frequencies, edges[-1], config.whitening_band
    )
    weights = used[:, None] & (valid_fraction >= config.min_valid_fraction)  # (segment, channel)
    # a channel without such a segment keeps all those where it has valid samples
    weights = jnp.where(weights.any(axis=0), weights, used[:, None] & (valid_fraction > 0))
    # zero-filled gaps correlate neighbouring coefficients: fewer independent samples
    efficiency = jnp.sum(jnp.where(weights, efficiency, 0)) / jnp.maximum(weights.sum(), 1)

    # per-channel noise level, from the segments it contributes
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
            n_independent = n_used * columns.size * efficiency
            vectors, eigenvalues, power = _fit_bin(
                z, n_used * columns.size, n_independent, alive, config
            )
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
    return LowRankNoiseModel(
        diagonal,
        mode_array,
        core,
        edges=edges,
        sample_rate=float(sample_rate),
        lead=(n_channels,) if lead is None else tuple(lead),
    )


def _segment_spectra(
    samples: Float[Array, 'channel sample'],
    valid: Bool[Array, 'channel sample'] | None,
    starts: Array,
    nperseg: int,
    frequencies: np.ndarray,
    top: float,
    band: tuple[float, float],
) -> tuple[Array, Array, Array, Array]:
    """Spectra of the segments, with their white-band power, valid fraction and efficiency.

    The windowed spectra go up to `top`, and every output is per segment and channel.

    The spectra are normalised by the windowed power of the valid samples, so that white noise of
    per-sample variance σ² has E|X|² = σ² under any mask. The efficiency is the effective number of
    samples of the masked window relative to that of the full window, `(Σu²)²/Σu⁴` for weights
    `u`: the fraction of independent coefficients that remain once gaps correlate them.
    """
    window = jnp.hanning(nperseg + 1)[:-1].astype(samples.dtype)
    n_low = int(np.searchsorted(frequencies, top))
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    if not in_band.any():
        raise ValueError(f'No frequency in the whitening band {band} Hz')
    if valid is None:
        valid = jnp.ones(samples.shape, bool)

    full = jnp.sum(window**2) ** 2 / jnp.sum(window**4)

    def spectrum(start: Array) -> tuple[Array, Array, Array, Array]:
        segment = jax.lax.dynamic_slice_in_dim(samples, start, nperseg, axis=1)
        mask = jax.lax.dynamic_slice_in_dim(valid, start, nperseg, axis=1)
        n_valid = mask.sum(axis=1, keepdims=True)
        mean = jnp.where(mask, segment, 0).sum(axis=1, keepdims=True) / jnp.maximum(n_valid, 1)
        weighted = jnp.where(mask, window, 0)
        norm = jnp.sqrt(jnp.sum(weighted**2, axis=1, keepdims=True))
        coefficients = jnp.fft.rfft(jnp.where(mask, segment - mean, 0) * window)
        coefficients = coefficients / jnp.where(norm > 0, norm, 1)
        power = jnp.mean(jnp.abs(coefficients[:, in_band]) ** 2, axis=1)
        effective = jnp.sum(weighted**2, axis=1) ** 2 / jnp.maximum(
            jnp.sum(weighted**4, axis=1), 1e-300
        )
        return coefficients[:, :n_low], power, n_valid[:, 0] / nperseg, effective / full

    return jax.lax.map(spectrum, starts)


def _fit_bin(
    z: Array,
    n_columns: Array,
    n_independent: Array,
    alive: Bool[Array, ' channel'],
    config: LowRankNoiseFitConfig,
) -> tuple[Array, Array, Array]:
    """Modes of one bin from its whitened Fourier coefficients, shape `(channel, column)`.

    The noise edge is that of `n_independent` samples, the columns' worth once their correlations
    are accounted for, scaled to the `n_columns` they are averaged over.

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
    n_independent = jnp.maximum(n_independent, 1)
    edge = _noise_edge(jnp.sum(alive), 2 * n_independent, config.threshold_quantile) / 2
    threshold = edge * n_columns / n_independent
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
