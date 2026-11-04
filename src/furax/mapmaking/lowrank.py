"""Detector-correlated noise model: a diagonal plus a few common modes per frequency bin."""

import itertools
import logging
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

logger = logging.getLogger(__name__)

# Tracy-Widom (beta = 1) quantiles, for the fluctuation of the largest eigenvalue of pure noise.
_TRACY_WIDOM_QUANTILES = (0.9, 0.95, 0.99, 0.999)
_TRACY_WIDOM_VALUES = (0.4501, 0.9793, 2.0234, 3.2724)

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
    n_real: int | Array | None = None,
    lead: tuple[int, ...] | None = None,
    config: LowRankNoiseFitConfig | None = None,
) -> LowRankNoiseModel:
    r"""Fit the noise covariance $N(f) = D + V G V^\top$ in frequency bins from the data.

    The covariance is estimated from the Fourier transform of the whole observation, its mean
    removed and its ends tapered over `config.taper` seconds. Every sample enters the fit,
    flagged or not. Each channel is whitened by its power in `config.whitening_band`, where it
    is assumed white. In every bin of `config.bin_edges`, the eigendecomposition of the channel
    covariance gives the modes: those above the largest eigenvalue that pure noise would produce,
    with at most `config.max_modes` and at most half the real samples of the bin, and none
    carried mostly by a single channel. The diagonal $D$ is the remaining power of each channel,
    corrected for the part of it that the modes absorb, so that the model reproduces every
    channel's power in the bin.

    Args:
        samples: Time-ordered data of all channels, with the samples last.
        sample_rate: Sample rate in Hz.
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
    frequencies = np.fft.rfftfreq(n_samples, 1 / sample_rate)
    spectra, level, efficiency = _observation_spectra(
        samples,
        n_samples if n_real is None else n_real,
        frequencies,
        edges[-1],
        config.whitening_band,
        max(round(config.taper * sample_rate), 1),
    )
    alive = level > 0
    median = jnp.nanmedian(jnp.where(alive, level, jnp.nan))
    level = jnp.where(alive, level, jnp.where(jnp.isnan(median), 1.0, median))

    diagonals, modes, powers = [], [], []
    for lo, hi in itertools.pairwise(edges):
        columns = np.flatnonzero((frequencies >= lo) & (frequencies < hi) & (frequencies > 0))
        if columns.size == 0:
            diagonal, vectors, eigenvalues = level, None, None
        else:
            z = spectra[:, columns] / jnp.sqrt(level)[:, None]
            n_independent = columns.size * efficiency
            vectors, eigenvalues, power = _fit_bin(z, columns.size, n_independent, alive, config)
            # The k sample modes span part of each channel's own noise too, a fraction k/n of
            # it in expectation over the n real samples, which the residual power lacks.
            n_kept = jnp.sum(eigenvalues > 0)
            residual = (power - (vectors**2 * eigenvalues).sum(axis=1)) / (
                1 - n_kept / (2 * n_independent)
            )
            diagonal = level * jnp.maximum(residual, _DIAGONAL_FLOOR * power)
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
    model = LowRankNoiseModel(
        diagonal,
        mode_array,
        core,
        edges=edges,
        sample_rate=float(sample_rate),
        lead=(n_channels,) if lead is None else tuple(lead),
    )
    # Decided when the fit is traced: no host callback unless the record would be logged
    if logger.isEnabledFor(logging.DEBUG):
        jax.debug.callback(_log_fit, *_fit_summary(model, level, alive))
    return model


def _fit_summary(
    model: LowRankNoiseModel, level: Float[Array, ' channel'], alive: Bool[Array, ' channel']
) -> tuple[Array, ...]:
    """Per-bin statistics of a fit, over the live channels, for the debug log.

    Returns the number of modes kept, the largest mode power over the white level, the quantiles
    (0, 1, 50 %) of the diagonal over the white level, and the quantiles (50, 99, 100 %) of the
    weight's diagonal over its zero-lag value, which the block-Jacobi preconditioner uses.
    """
    n_modes = jnp.sum(jnp.diagonal(model.core, axis1=1, axis2=2) > 0, axis=1)
    top_power = jnp.max(jnp.diagonal(model.core, axis1=1, axis2=2), axis=1)
    whitened = jnp.where(alive, model.diagonal / level, jnp.nan)
    diagonal_q = jnp.nanquantile(whitened, jnp.array([0.0, 0.01, 0.5]), axis=1).T
    weight = model.inverse_operator(
        jax.ShapeDtypeStruct((model.diagonal.shape[1], 1), model.diagonal.dtype)
    )
    weight_diagonal = (
        weight.diagonal
        + jnp.einsum('bcj,bjk,bck->bc', weight.modes, weight.core, weight.modes.conj()).real
    )
    zero_lag = 1 / model.to_white_noise_model().sigma.ravel() ** 2
    ratio = jnp.where(alive, weight_diagonal / zero_lag, jnp.nan)
    ratio_q = jnp.nanquantile(ratio, jnp.array([0.5, 0.99, 1.0]), axis=1).T
    return n_modes, top_power, diagonal_q, ratio_q


def _log_fit(n_modes: Array, top_power: Array, diagonal_q: Array, ratio_q: Array) -> None:
    lines = [
        f'  bin {b}: {int(n_modes[b])} modes, top power {float(top_power[b]):.3g},'
        f' D/white min/1%/median {np.array2string(np.asarray(diagonal_q[b]), precision=3)},'
        f' weight/zero-lag median/99%/max {np.array2string(np.asarray(ratio_q[b]), precision=3)}'
        for b in range(len(n_modes))
    ]
    logger.debug('Low-rank noise fit, per bin (powers relative to white):\n%s', '\n'.join(lines))


def _observation_spectra(
    samples: Float[Array, 'channel sample'],
    n_real: int | Array,
    frequencies: np.ndarray,
    top: float,
    band: tuple[float, float],
    n_taper: int,
) -> tuple[Array, Array, Array]:
    """Spectra of the whole observation up to `top`, with their white-band power and efficiency.

    The samples past `n_real` are left out. The spectra are normalised by the power of the
    window, so that white noise of per-sample variance σ² has E|X|² = σ². The efficiency is the
    effective number of samples of the window, `(Σw²)²/Σw⁴`, over the length of the transform:
    the fraction of independent coefficients.
    """
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    if not in_band.any():
        raise ValueError(f'No frequency in the whitening band {band} Hz')
    n_low = int(np.searchsorted(frequencies, top))
    n_samples = samples.shape[-1]
    t = jnp.arange(n_samples)
    real = t < n_real
    ramp = jnp.clip(jnp.minimum(t + 1, n_real - t) / n_taper, 0, 1)
    window = jnp.where(real, (1 - jnp.cos(jnp.pi * ramp)) / 2, 0).astype(samples.dtype)
    mean = jnp.where(real, samples, 0).sum(axis=1, keepdims=True) / n_real
    coefficients = jnp.fft.rfft((samples - mean) * window) / jnp.sqrt(jnp.sum(window**2))
    level = jnp.mean(jnp.abs(coefficients[:, in_band]) ** 2, axis=1)
    efficiency = jnp.sum(window**2) ** 2 / jnp.sum(window**4) / n_samples
    return coefficients[:, :n_low], level, efficiency


def _fit_bin(
    z: Array,
    n_columns: int,
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
    # strongest accepted modes first, at most max_modes of them, and at most half the real samples
    # so that the diagonal keeps enough of them
    order = jnp.argsort(~accepted, stable=True)[: config.max_modes]
    keep = accepted[order] & (jnp.arange(order.size) < jnp.floor(n_independent))
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
