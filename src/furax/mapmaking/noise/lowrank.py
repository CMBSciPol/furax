"""Detector-correlated noise model: a diagonal plus a few common modes per frequency bin."""

import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float, Inexact, Key, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.core import LowRankFourierOperator
from furax.core._lowrank import mode_power
from furax.tree import normal_like, unravel

from ..config import LowRankNoiseFitConfig
from ._base import NoiseModel, _amplitude_scale
from .binned import (
    _FFT_BATCH_SIZE,
    BinnedNoiseModel,
    _bin_bounds,
    _bin_fractions,
    _white_of_zero_lag,
)
from .white import WhiteNoiseModel

logger = logging.getLogger(__name__)


# Tracy-Widom (beta = 1) quantiles, for the fluctuation of the largest eigenvalue of pure noise.
_TRACY_WIDOM_QUANTILES = (0.9, 0.95, 0.99, 0.999)
_TRACY_WIDOM_VALUES = (0.4501, 0.9793, 2.0234, 3.2724)


# The diagonal of a channel never falls below this fraction of its measured power in a bin.
_DIAGONAL_FLOOR = 1e-2


@jax.tree_util.register_dataclass
@dataclass
class LowRankNoiseModel(NoiseModel):
    r"""Noise covariance $N(f) = D_b + V G_b V^\dagger$ in frequency bins, across channels.

    The channels are the streams and detectors of a TOD, flattened stream-major, as
    [`LowRankFourierOperator`][furax.core.LowRankFourierOperator] takes them.
    """

    diagonal: Float[Array, 'bin channel']
    """The diagonal $D_b$ of each bin, in variance per sample, as the operators take it: the
    one-sided PSD times half the sample rate."""

    modes: Inexact[Array, 'channel mode']
    """The modes $V$, the same in every bin but the last, which has none; padded with zeros."""

    core: Inexact[Array, 'coupled mode mode']
    """The mode covariance $G_b$ of the first bins."""

    edges: tuple[float, ...] = field(metadata={'static': True})
    """Interior bin boundaries in Hz (see [`LowRankFourierOperator.edges`][])."""

    sample_rate: float = field(metadata={'static': True})
    """Sample rate in Hz at which the bins are placed."""

    lead: tuple[int, ...] = field(metadata={'static': True})
    """The shape of the channel axes: `(streams, detectors)` for a demodulated TOD."""

    zero_lag: Float[Array, ' channel'] | None = None
    """The zero-lag of $N^{-1}$ of each channel, kept by [`select`][]: that of the model of the
    channels it was selected from, not of their marginal model."""

    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        """The diagonal of $N(f)$: the PSD of each channel, correlated modes included."""
        bins = jnp.searchsorted(jnp.array(self.edges), f, side='right')
        power = _diagonal_power(self.diagonal, self.modes, self.core)
        return (power[bins].T * 2 / self.sample_rate).reshape(*self.lead, f.size)

    def covariance(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> LowRankFourierOperator:
        r"""$N = D + V G V^\dagger$ in each frequency bin, exactly, periodic over the TOD."""
        return LowRankFourierOperator(
            self.diagonal,
            self.modes,
            self.core,
            edges=self.edges,
            sample_rate=self.sample_rate,
            in_structure=in_structure,
            batch_size=_FFT_BATCH_SIZE,
        )

    def inverse_covariance(
        self, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> LowRankFourierOperator:
        """$N^{-1}$, exactly, from the Woodbury identity in each frequency bin."""
        return self.covariance(in_structure).inverse()

    def select(self, channels: Array, lead: tuple[int, ...]) -> 'LowRankNoiseModel':
        """The marginal model of some of the channels, such as those of a detector tile.

        Args:
            channels: The indices of the channels kept, in the flat order of `lead`.
            lead: The shape of their channel axes.
        """
        return replace(
            self,
            diagonal=self.diagonal[:, channels],
            modes=self.modes[channels],
            lead=lead,
            zero_lag=self._zero_lag()[channels],
        )

    def draw(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[Array]:
        r"""A realization of the noise, of covariance `covariance(in_structure)` exactly.

        It is $D^{1/2} w + T a$, with $w$ and the amplitudes $a$ of the modes (see
        [`LowRankModesOperator`][]) standard normal (see [`NoiseModel.draw`][]).
        """
        key_decoupled, key_modes = jax.random.split(key)
        decoupled = self.decoupled()
        coefficients = decoupled._coefficients(key_decoupled, in_structure)
        T = self.modes_operator(in_structure)
        modes = T.coefficients(normal_like(T.in_structure, key_modes))
        coefficients = coefficients.at[:, : modes.shape[-1]].add(modes)
        return decoupled._to_tod(coefficients, in_structure)

    def decoupled(self) -> BinnedNoiseModel:
        r"""The term $D$ of $N = D + V G V^\dagger$, without the common modes.

        It is not the marginal noise of each channel, $D + \operatorname{diag}(V G V^\dagger)$: the
        modes are left out, not averaged in. A detector tile is weighted by its inverse, the
        modes being marginalized over the whole observation (see [`modes_core`][]).
        """
        return BinnedNoiseModel(self.diagonal, self.edges, self.sample_rate, self.lead)

    def modes_operator(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> 'LowRankModesOperator':
        """The correlated modes $T$, onto TODs of structure `in_structure`.

        See [`LowRankModesOperator`][].
        """
        return LowRankModesOperator.create(self, in_structure)

    def modes_core(self, n_samples: int) -> 'ModesCore':
        r"""$K = (1 + T^\top D^{-1} T)^{-1}$ on the amplitudes of the modes of a TOD of `n_samples`.

        With it, $N^{-1} = D^{-1} - D^{-1} T K T^\top D^{-1}$ (see [`LowRankModesOperator`][]). It is
        a $k \times k$ matrix per coupled bin, the same at each of its frequencies, but at the zero
        and Nyquist frequencies, where only the real parts of the coefficients enter.
        """
        V = self.modes
        L = _core_factors(self.core)
        n_coupled = L.shape[0]
        inverse_diagonal = 1 / self.diagonal[:n_coupled]
        modes_gram = jnp.einsum('ck,bc,cj->bkj', V.conj(), inverse_diagonal, V)
        gram = jnp.conj(L).mT @ modes_gram @ L
        generic = jnp.linalg.inv(jnp.eye(gram.shape[-1], dtype=gram.dtype) + gram)

        def real(b: int) -> Array:
            # the real parts of the coefficients, from the real and imaginary parts of the amplitudes
            factors = V @ L[b]
            basis = jnp.concatenate([factors.real, -factors.imag], axis=-1)
            gram = basis.T @ (inverse_diagonal[b][:, None] * basis)
            inverse = jnp.linalg.inv(jnp.eye(gram.shape[-1], dtype=gram.dtype) + gram)
            return (inverse + inverse.T) / 2

        bounds = _bin_bounds(self.edges, self.sample_rate, n_samples)
        nyquist = None
        if n_samples % 2 == 0 and bounds[n_coupled] == n_samples // 2 + 1:
            nyquist = real(n_coupled - 1)
        return ModesCore(
            (generic + jnp.conj(generic).mT) / 2,
            real(0),
            nyquist,
            self.edges,
            self.sample_rate,
            n_samples,
            in_structure=self._amplitudes(n_samples, self.diagonal.dtype),
        )

    def white(self) -> WhiteNoiseModel:
        """The white noise whose weight is the zero-lag of $N^{-1}$, channel by channel.

        It is the diagonal of the weight in the time domain: the average of the diagonal of
        $N^{-1}(f)$ over the frequencies up to Nyquist.
        """
        return _white_of_zero_lag(self._zero_lag(), self.sample_rate, self.lead)

    @classmethod
    def fit(
        cls,
        samples: Float[Array, 'channel sample'],
        *,
        sample_rate: float,
        n_real: int | Array | None = None,
        edges: tuple[float, ...] | None = None,
        lead: tuple[int, ...] | None = None,
        config: LowRankNoiseFitConfig | None = None,
    ) -> 'LowRankNoiseModel':
        r"""Fit the noise covariance $N(f) = D + V G V^\top$ in frequency bins from the data.

        The covariance is estimated from the Fourier transform of the whole observation, its mean
        removed and its ends tapered over `config.taper` seconds. Every sample enters the fit,
        flagged or not. The modes $V$ are the same at every frequency: they come from the channel
        covariance averaged over `config.mode_band`, each channel whitened by its level in
        `config.whitening_band`, as the eigenvectors above the largest eigenvalue that pure noise
        would produce, with at most `config.max_modes`.
        In every bin (see [`bin_edges`][furax.mapmaking.noise.bin_edges]), the power of the
        amplitudes $V^\top x$ of the modes gives the diagonal core $G$, and the power left in each
        channel once the modes are projected out gives the diagonal $D$. Above the last edge the
        noise is white, at the level of each channel in `config.whitening_band`.

        Args:
            samples: Time-ordered data of all channels, with the samples last.
            sample_rate: Sample rate in Hz.
            n_real: Number of samples before the padded tail, or `None` for no padding.
            edges: The edges of the bins in Hz; by default, those of
                [`bin_edges`][furax.mapmaking.noise.bin_edges] for all the samples.
            lead: The shape of the channel axes, `(streams, detectors)` for a demodulated TOD; by
                default, a single axis.
            config: Fit options.

        Returns:
            The fitted noise model.
        """
        config = config or LowRankNoiseFitConfig()
        n_samples = samples.shape[-1]
        n_real = n_samples if n_real is None else n_real
        edges = bin_edges(config, sample_rate, n_samples) if edges is None else edges
        spectra, level = _low_rank_spectra(samples, sample_rate, n_real, config)
        return _fit_low_rank_spectra(
            spectra, level, n_samples, n_real, sample_rate, edges, lead=lead, config=config
        )

    def _amplitudes(self, n_samples: int, dtype: Any) -> jax.ShapeDtypeStruct:
        """The amplitudes of the modes of a TOD of `n_samples`: `(frequency, mode, 2)`."""
        bounds = _bin_bounds(self.edges, self.sample_rate, n_samples)
        n_low = bounds[self.core.shape[0]]
        dtype = jnp.promote_types(dtype, jnp.float32)
        return jax.ShapeDtypeStruct((n_low, self.modes.shape[-1], 2), dtype)

    def _zero_lag(self) -> Float[Array, ' channel']:
        """The zero-lag of $N^{-1}$ of each channel (see `zero_lag`)."""
        if self.zero_lag is not None:
            return self.zero_lag
        weight = self.inverse_covariance(
            jax.ShapeDtypeStruct((self.diagonal.shape[1], 1), self.diagonal.dtype)
        )
        diagonal = _diagonal_power(weight.diagonal, weight.modes, weight.core)
        return _bin_fractions(self.edges, self.sample_rate, diagonal.dtype) @ diagonal


class LowRankModesOperator(AbstractLinearOperator):
    r"""The correlated modes of a low-rank noise model, from amplitudes of unit variance to a TOD.

    In each coupled bin $b$, with $G_b = L_b L_b^\dagger$ and $A_b = V L_b$, the amplitudes
    $\zeta_f$ of each frequency $f$ of the bin, complex, make the Fourier coefficients
    $\sqrt{n / w_f} \, A_b \zeta_f$ of the TOD of $n$ samples, $w_f$ being 1 at the zero and
    Nyquist frequencies and 2 elsewhere. With standard normal real and imaginary parts, the TOD has
    the covariance $V G_b V^\dagger$ of the modes, as
    [`LowRankFourierOperator`][furax.core.LowRankFourierOperator] defines it: weighting by the
    diagonal $D^{-1}$ and solving for the amplitudes under their prior marginalizes them into the
    weight $N^{-1} = (D + V G V^\dagger)^{-1}$.

    The amplitudes are `(frequency, mode, 2)`, the real and imaginary parts last, over the
    frequencies of the coupled bins.
    """

    modes: Inexact[Array, 'channel mode']
    """The modes $V$."""
    core_factors: Inexact[Array, 'coupled mode mode']
    """$L_b$ of each coupled bin."""
    scale: Float[Array, 'coupled channel'] | None
    """A diagonal $D_b$ by which the modes of each coupled bin are scaled, $D_b A_b$; `None` for
    none (see [`weighted_by`][furax.mapmaking.noise.LowRankModesOperator.weighted_by])."""
    edges: tuple[float, ...] = field(metadata={'static': True})
    sample_rate: float = field(metadata={'static': True})
    tod_treedef: Any = field(metadata={'static': True})
    """The pytree structure of the TOD."""
    tod_leaves: tuple[jax.ShapeDtypeStruct, ...] = field(metadata={'static': True})
    """The structure of each leaf of the TOD, its channels first and its samples last."""

    @classmethod
    def create(
        cls, model: LowRankNoiseModel, tod_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> 'LowRankModesOperator':
        """The modes of `model`, onto TODs of structure `tod_structure`."""
        leaves, treedef = jax.tree.flatten(tod_structure)
        structure = model._amplitudes(leaves[0].shape[-1], leaves[0].dtype)
        return cls(
            model.modes,
            _core_factors(model.core),
            None,
            model.edges,
            model.sample_rate,
            treedef,
            tuple(leaves),
            in_structure=structure,
        )

    def weighted_by(self, diagonal: Float[Array, 'bin channel']) -> 'LowRankModesOperator':
        r"""$D T$, for $D$ diagonal in each frequency bin, such as the weight $D^{-1}$ of a model.

        The modes give the frequencies of the coupled bins only, on which $D$ scales each channel
        by its value in the bin: $D T$ is the operator of the modes $D_b A_b$, which takes no more
        Fourier transforms than $T$.

        Args:
            diagonal: The value of $D$ in each bin, for each channel; the bins past the coupled
                ones are unused.
        """
        scale = diagonal[: len(self.core_factors)]
        if self.scale is not None:
            scale = scale * self.scale
        return type(self)(
            self.modes,
            self.core_factors,
            scale,
            self.edges,
            self.sample_rate,
            self.tod_treedef,
            self.tod_leaves,
            in_structure=self.in_structure,
        )

    def mv(self, x: Float[Array, 'frequency mode 2']) -> PyTree[Array]:
        n = self.tod_leaves[0].shape[-1]
        coefficients = self.coefficients(x)
        coefficients = jnp.pad(coefficients, [(0, 0), (0, n // 2 + 1 - coefficients.shape[-1])])
        channels = jnp.fft.irfft(coefficients, n=n).astype(self.tod_leaves[0].dtype)
        structure = jax.tree.unflatten(self.tod_treedef, list(self.tod_leaves))
        return unravel(structure, channels, trailing_ndim=1)

    def coefficients(
        self, x: Float[Array, 'frequency mode 2']
    ) -> Complex[Array, 'channel frequency']:
        """The rfft coefficients of the TOD of the amplitudes `x`, over the coupled frequencies.

        The TOD is their inverse rfft, which keeps only the real parts at the zero and Nyquist
        frequencies; the channels are flattened.
        """
        n = self.tod_leaves[0].shape[-1]
        bounds = _bin_bounds(self.edges, self.sample_rate, n)
        zeta = jax.lax.complex(x[..., 0], x[..., 1])
        n_coupled = len(self.core_factors)
        # L_b ζ_f bin by bin, then V once for all the frequencies
        mixed = jnp.concatenate(
            [
                self.core_factors[b].astype(zeta.dtype) @ zeta[start:stop].T
                for b, (start, stop) in enumerate(itertools.pairwise(bounds[: n_coupled + 1]))
            ],
            axis=-1,
        )
        coefficients = self.modes.astype(zeta.dtype) @ mixed
        if self.scale is not None:
            bins = np.repeat(np.arange(n_coupled), np.diff(bounds[: n_coupled + 1]))
            coefficients = coefficients * self.scale[bins].T
        return coefficients * _amplitude_scale(n, zeta.shape[0])


@symmetric
class ModesCore(AbstractLinearOperator):
    r"""$K$ of [`LowRankNoiseModel.modes_core`][], on the amplitudes of the modes of a TOD.

    The amplitudes are those of [`LowRankModesOperator`][], `(frequency, mode, 2)`.
    """

    generic: Inexact[Array, 'coupled mode mode']
    """The complex $k \times k$ matrix of each coupled bin."""
    zero: Float[Array, 'amplitude amplitude']
    """The real $2k \times 2k$ matrix of the zero frequency, on its real then imaginary parts."""
    nyquist: Float[Array, 'amplitude amplitude'] | None
    """That of the Nyquist frequency, when it is coupled."""
    edges: tuple[float, ...] = field(metadata={'static': True})
    sample_rate: float = field(metadata={'static': True})
    n_samples: int = field(metadata={'static': True})
    """The number of samples of the TOD, padding included."""

    def mv(self, x: Float[Array, 'frequency mode 2']) -> Float[Array, 'frequency mode 2']:
        bounds = _bin_bounds(self.edges, self.sample_rate, self.n_samples)
        zeta = jax.lax.complex(x[..., 0], x[..., 1])
        pieces = [
            zeta[start:stop] @ self.generic[b].astype(zeta.dtype).T
            for b, (start, stop) in enumerate(itertools.pairwise(bounds[: len(self.generic) + 1]))
        ]
        out = jnp.concatenate(pieces)
        out = jnp.stack([out.real, out.imag], axis=-1).astype(x.dtype)
        out = out.at[0].set(_apply_real(self.zero, x[0]))
        if self.nyquist is not None:
            out = out.at[-1].set(_apply_real(self.nyquist, x[-1]))
        return out


def bin_edges(
    config: LowRankNoiseFitConfig, sample_rate: float, n_samples: int
) -> tuple[float, ...]:
    """The edges in Hz of the bins of the model, as `config.bin_edges` gives them.

    Unless given, the edges run from the narrowest width, `config.min_bin_frequencies` of an
    observation of `n_samples`, to the top of `config.mode_band`, each `config.bin_ratio` above
    the previous one, by that width at least; a last bin narrower than it is merged into the
    previous one.
    """
    if config.bin_edges is not None:
        return tuple(float(e) for e in config.bin_edges)
    width = config.min_bin_frequencies * sample_rate / n_samples
    top = float(config.mode_band[1])
    if top < 2 * width:
        raise ValueError(f'The mode band {config.mode_band} Hz holds no bin of {width:.3g} Hz')
    edges = [width]
    while (step := max(edges[-1] * config.bin_ratio, edges[-1] + width)) < top:
        edges.append(step)
    if len(edges) > 1 and top - edges[-1] < width:
        edges.pop()
    return (*edges, top)


def _low_rank_spectra(
    samples: Float[Array, 'channel sample'],
    sample_rate: float,
    n_real: int | Array,
    config: LowRankNoiseFitConfig,
) -> tuple[Inexact[Array, 'channel frequency'], Float[Array, ' channel']]:
    """What the low-rank fit takes of each channel.

    That is its spectrum up to the top of the mode band, and its white
    level. Each channel is taken alone, so that a detector tile gives the rows of its channels
    (see [`_fit_low_rank_spectra`][]). The samples past `n_real` are padding.
    """
    frequencies = np.fft.rfftfreq(samples.shape[-1], 1 / sample_rate)
    return _observation_spectra(
        samples,
        n_real,
        frequencies,
        float(config.mode_band[1]),
        config.whitening_band,
        _taper_samples(config, sample_rate),
    )


def _fit_low_rank_spectra(
    spectra: Inexact[Array, 'channel frequency'],
    level: Float[Array, ' channel'],
    n_samples: int,
    n_real: int | Array,
    sample_rate: float,
    edges: tuple[float, ...],
    *,
    lead: tuple[int, ...] | None = None,
    config: LowRankNoiseFitConfig | None = None,
) -> LowRankNoiseModel:
    """The low-rank noise model of the channels of an observation, from [`_low_rank_spectra`][].

    The channels with no white power are dead: they take no part in the modes.

    Args:
        spectra: The spectrum of each channel up to the top of the mode band.
        level: The white level of each channel.
        n_samples: The number of samples of the TOD the spectra are of, its padding included.
        n_real: The number of samples before its padding.
        sample_rate: Sample rate in Hz.
        edges: The edges of the bins in Hz (see [`bin_edges`][furax.mapmaking.noise.bin_edges]).
        lead: The shape of the channel axes, as in [`LowRankNoiseModel.fit`][].
        config: Fit options.
    """
    config = config or LowRankNoiseFitConfig()
    n_channels = spectra.shape[0]
    efficiency = _window_efficiency(n_samples, n_real, _taper_samples(config, sample_rate))
    alive = level > 0
    median = jnp.nanmedian(jnp.where(alive, level, jnp.nan))
    level = jnp.where(alive, level, jnp.where(jnp.isnan(median), 1.0, median))

    vectors = _correlated_modes(spectra, level, sample_rate, n_samples, efficiency, alive, config)
    # the bins as the operator places them, each a range of frequencies
    bounds = np.maximum(_bin_bounds(edges, sample_rate, n_samples)[1:-1], 1)
    start, widths = bounds[0], np.diff(bounds)
    coefficients = spectra[:, start : bounds[-1]]
    amplitudes = vectors.T @ coefficients
    residual = coefficients - vectors @ amplitudes
    bins = jnp.asarray(np.repeat(np.arange(widths.size), widths))

    def mean(x: Array) -> Array:
        """The mean of `|x|²` over the frequencies of each bin, `(bin, row)`; 0 if empty."""
        sums = jax.ops.segment_sum(jnp.abs(x.T) ** 2, bins, widths.size)
        return sums / np.maximum(widths, 1)[:, None]

    filled = jnp.asarray(widths > 0)[:, None]
    diagonals = jnp.maximum(mean(residual), _DIAGONAL_FLOOR * mean(coefficients))
    diagonals = jnp.where(filled & alive & (diagonals > 0), diagonals, level)
    powers = jnp.where(filled, mean(amplitudes), 0)

    # bins: [0, edge0) repeats the first fitted bin, then the fitted ones; the white level above
    # is an uncoupled last bin, which the operator applies without an FFT
    diagonal = jnp.concatenate([diagonals[:1], diagonals, level[None]])
    core = jax.vmap(jnp.diag)(jnp.concatenate([powers[:1], powers]))
    model = LowRankNoiseModel(
        diagonal,
        vectors,
        core,
        edges=edges,
        sample_rate=float(sample_rate),
        lead=(n_channels,) if lead is None else tuple(lead),
    )
    # the zero-lag of the whole model, which the rows selected from it keep
    model = replace(model, zero_lag=model._zero_lag())
    # Decided when the fit is traced: no host callback unless the record would be logged
    if logger.isEnabledFor(logging.DEBUG):
        jax.debug.callback(_log_fit, *_fit_summary(model, level, alive))
    return model


def _apply_real(matrix: Float[Array, 'a a'], x: Float[Array, 'mode 2']) -> Float[Array, 'mode 2']:
    """A real matrix on the real then imaginary parts of the amplitudes of one frequency."""
    return (matrix @ x.T.reshape(-1)).reshape(2, -1).T


def _core_factors(core: Inexact[Array, 'coupled mode mode']) -> Inexact[Array, 'coupled mode mode']:
    r"""$L_b$ with $G_b = L_b L_b^\dagger$, from the eigendecomposition of $G_b$."""
    eigenvalues, vectors = jnp.linalg.eigh(core)
    return vectors * jnp.sqrt(jnp.clip(eigenvalues, 0))[..., None, :]


def _diagonal_power(
    diagonal: Float[Array, 'bin channel'],
    modes: Inexact[Array, 'coupled channel mode'],
    core: Inexact[Array, 'coupled mode mode'],
) -> Float[Array, 'bin channel']:
    """Per-channel power of each bin; the bins past the modes are uncoupled."""
    return diagonal.at[: core.shape[0]].add(mode_power(modes, core))


def _fit_summary(
    model: LowRankNoiseModel, level: Float[Array, ' channel'], alive: Bool[Array, ' channel']
) -> tuple[Array, ...]:
    """Per-bin statistics of a fit, over the live channels, for the debug log.

    Returns the number of modes kept, the largest mode power over the diagonal, the quantiles
    (0, 1, 50 %) of the diagonal over the white level, and the quantiles (50, 99, 100 %) of the
    weight's diagonal over its zero-lag value, which the block-Jacobi preconditioner uses.
    """
    n_uncoupled = model.diagonal.shape[0] - model.core.shape[0]
    core_diagonal = jnp.pad(jnp.diagonal(model.core, axis1=1, axis2=2), ((0, n_uncoupled), (0, 0)))
    n_modes = jnp.sum(core_diagonal > 0, axis=1)
    top_power = jnp.max(core_diagonal, axis=1)
    whitened = jnp.where(alive, model.diagonal / level, jnp.nan)
    diagonal_q = jnp.nanquantile(whitened, jnp.array([0.0, 0.01, 0.5]), axis=1).T
    weight = model.inverse_covariance(
        jax.ShapeDtypeStruct((model.diagonal.shape[1], 1), model.diagonal.dtype)
    )
    weight_diagonal = _diagonal_power(weight.diagonal, weight.modes, weight.core)
    zero_lag = model._zero_lag()
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
    logger.debug(
        'Low-rank noise fit, per bin (mode power relative to the diagonal, others to white):\n%s',
        '\n'.join(lines),
    )


def _observation_spectra(
    samples: Float[Array, 'channel sample'],
    n_real: int | Array,
    frequencies: np.ndarray,
    top: float,
    band: tuple[float, float],
    n_taper: int,
) -> tuple[Array, Array]:
    """Spectra of the whole observation up to `top`, with their white-band power.

    The samples past `n_real` are left out. The spectra are normalised by the power of the
    window, so that white noise of per-sample variance σ² has E|X|² = σ².
    """
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    if not in_band.any():
        raise ValueError(f'No frequency in the whitening band {band} Hz')
    n_low = int(np.searchsorted(frequencies, top))
    n_samples = samples.shape[-1]
    real = jnp.arange(n_samples) < n_real
    window = _window(n_samples, n_real, n_taper).astype(samples.dtype)
    mean = jnp.where(real, samples, 0).sum(axis=1, keepdims=True) / n_real
    coefficients = jnp.fft.rfft((samples - mean) * window) / jnp.sqrt(jnp.sum(window**2))
    level = jnp.mean(jnp.abs(coefficients[:, in_band]) ** 2, axis=1)
    return coefficients[:, :n_low], level


def _taper_samples(config: LowRankNoiseFitConfig, sample_rate: float) -> int:
    """The samples of the taper at each end of an observation, at least one."""
    return max(round(config.taper * sample_rate), 1)


def _window(n_samples: int, n_real: int | Array, n_taper: int) -> Array:
    """The cosine taper over `n_taper` samples at each end of the first `n_real`, zero after."""
    t = jnp.arange(n_samples)
    ramp = jnp.clip(jnp.minimum(t + 1, n_real - t) / n_taper, 0, 1)
    return jnp.where(t < n_real, (1 - jnp.cos(jnp.pi * ramp)) / 2, 0)


def _window_efficiency(n_samples: int, n_real: int | Array, n_taper: int) -> Array:
    """The fraction of independent Fourier coefficients of the window.

    That is its effective number of samples, `(Σw²)²/Σw⁴`, over the length of the transform.
    """
    window = _window(n_samples, n_real, n_taper)
    return jnp.sum(window**2) ** 2 / jnp.sum(window**4) / n_samples


def _correlated_modes(
    spectra: Inexact[Array, 'channel frequency'],
    level: Float[Array, ' channel'],
    sample_rate: float,
    n_samples: int,
    efficiency: Array,
    alive: Bool[Array, ' channel'],
    config: LowRankNoiseFitConfig,
) -> Float[Array, 'channel max_modes']:
    """The orthonormal modes of the channels, from their covariance over `config.mode_band`.

    Each channel is whitened by its white level, so that pure noise has unit eigenvalues. The
    kept eigenvalues less that unit floor, the channels' own noise along each mode, are
    un-whitened, and the modes re-orthonormalized in the channel space. The unused modes are zero.
    """
    k = config.max_modes
    _, start, stop, _ = _bin_bounds(config.mode_band, sample_rate, n_samples)
    start = max(start, 1)
    if stop <= start:
        return jnp.zeros((spectra.shape[0], k), level.dtype)
    coefficients = spectra[:, start:stop]
    sigma = jnp.where(alive, jnp.sqrt(level), jnp.inf)  # dead channels whitened to zero
    gram = (coefficients @ coefficients.conj().T).real / jnp.outer(sigma, sigma)
    unit, eigenvalues = _whitened_modes(
        gram, stop - start, (stop - start) * efficiency, alive, config
    )
    factors = jnp.sqrt(level)[:, None] * unit * jnp.sqrt(jnp.maximum(eigenvalues - 1, 0))
    vectors, singular, _ = jnp.linalg.svd(factors, full_matrices=False)
    tolerance = singular[0] * max(factors.shape) * jnp.finfo(factors.dtype).eps
    vectors = jnp.where(singular > tolerance, vectors, 0)
    return jnp.pad(vectors, ((0, 0), (0, k - vectors.shape[1])))


def _whitened_modes(
    gram: Float[Array, 'channel channel'],
    n_columns: int,
    n_independent: Array,
    alive: Bool[Array, ' channel'],
    config: LowRankNoiseFitConfig,
) -> tuple[Array, Array]:
    """The modes of the whitened Gram `Re(z z^H)` of `n_columns` Fourier coefficients.

    The noise edge is that of `n_independent` samples, the columns' worth once their correlations
    are accounted for, scaled to the columns they are averaged over.

    Returns the strongest unit vectors `(channel, mode)` and their eigenvalues `(mode,)`, in units
    of the white level; both are zero for the modes that are not kept.
    """
    n_candidates = min(config.max_modes, gram.shape[0])
    values, vectors = jnp.linalg.eigh(gram)
    values, vectors = values[::-1][:n_candidates], vectors[:, ::-1][:, :n_candidates]
    eigenvalues = values / n_columns  # covariance eigenvalues; pure noise is ~1
    n_independent = jnp.maximum(n_independent, 1)
    edge = _noise_edge(jnp.sum(alive), 2 * n_independent, config.threshold_quantile) / 2
    threshold = edge * n_columns / n_independent
    # at most half the real samples
    keep = (values > threshold) & (jnp.arange(n_candidates) < jnp.floor(n_independent))
    return jnp.where(keep, vectors, 0), jnp.where(keep, eigenvalues, 0)


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
