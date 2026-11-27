"""Noise of independent channels, its PSD constant in each frequency bin."""

from dataclasses import dataclass, field
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Key, PyTree

from furax.core import LowRankFourierOperator
from furax.tree import unravel

from ._base import NoiseModel, _stationary_coefficients
from .white import WhiteNoiseModel

# The channels the operators of a model transform at once: one at a time is several times slower,
# on a CPU as on a GPU, and a detector tile has 16 by default.
_FFT_BATCH_SIZE = 16


@jax.tree_util.register_dataclass
@dataclass
class BinnedNoiseModel(NoiseModel):
    """Independent channels, coloured in time: the PSD of each is constant in each frequency bin.

    It is the decoupled term $D$ of a [`LowRankNoiseModel`][furax.mapmaking.noise.LowRankNoiseModel].
    The channels are the streams and detectors of a TOD, flattened stream-major.
    """

    diagonal: Float[Array, 'bin channel']
    """The level $D_b$ of each channel in each bin, in variance per sample, as the operators take
    it: the one-sided PSD times half the sample rate."""

    edges: tuple[float, ...] = field(metadata={'static': True})
    """Interior bin boundaries in Hz (see [`LowRankFourierOperator.edges`][])."""

    sample_rate: float = field(metadata={'static': True})
    """Sample rate in Hz at which the bins are placed."""

    lead: tuple[int, ...] = field(metadata={'static': True})
    """The shape of the channel axes: `(streams, detectors)` for a demodulated TOD."""

    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        bins = jnp.searchsorted(jnp.array(self.edges), f, side='right')
        return (self.diagonal[bins].T * 2 / self.sample_rate).reshape(*self.lead, f.size)

    def covariance(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> LowRankFourierOperator:
        """$D$, exactly, periodic over the TOD."""
        return self._operator(self.diagonal, in_structure)

    def inverse_covariance(
        self, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> LowRankFourierOperator:
        """$D^{-1}$, exactly, periodic over the TOD."""
        return self._operator(1 / self.diagonal, in_structure)

    def draw(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[Array]:
        """A realization of the noise, of covariance `covariance(in_structure)` exactly."""
        return self._to_tod(self._coefficients(key, in_structure), in_structure)

    def decoupled(self) -> Self:
        return self

    def white(self) -> WhiteNoiseModel:
        """The white noise whose weight is the zero-lag of $D^{-1}$, channel by channel."""
        fractions = _bin_fractions(self.edges, self.sample_rate, self.diagonal.dtype)
        return _white_of_zero_lag(fractions @ (1 / self.diagonal), self.sample_rate, self.lead)

    def _coefficients(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> Complex[Array, 'channel frequency']:
        """The rfft coefficients of a draw on TODs of structure `in_structure`, its channels flat."""
        n = jax.tree.leaves(in_structure)[0].shape[-1]
        bins = jnp.searchsorted(
            jnp.array(self.edges), jnp.fft.rfftfreq(n, 1 / self.sample_rate), 'right'
        )
        return _stationary_coefficients(key, self.diagonal[bins].T, n)

    def _to_tod(
        self,
        coefficients: Complex[Array, 'channel frequency'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> PyTree[Array]:
        """The TOD of structure `in_structure` of the rfft `coefficients` of its flat channels."""
        leaves = jax.tree.leaves(in_structure)
        channels = jnp.fft.irfft(coefficients, n=leaves[0].shape[-1]).astype(leaves[0].dtype)
        return unravel(in_structure, channels, trailing_ndim=1)

    def _operator(
        self, diagonal: Float[Array, 'bin channel'], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> LowRankFourierOperator:
        """The operator of `diagonal` in each bin, which couples no channels."""
        n_bins, n_channels = diagonal.shape
        return LowRankFourierOperator(
            diagonal,
            jnp.zeros((n_channels, 1), diagonal.dtype),  # an inert mode
            jnp.zeros((n_bins, 1, 1), diagonal.dtype),
            edges=self.edges,
            sample_rate=self.sample_rate,
            in_structure=in_structure,
            batch_size=_FFT_BATCH_SIZE,
        )


def _bin_fractions(edges: tuple[float, ...], sample_rate: float, dtype: jnp.dtype) -> Array:
    """The fraction of the frequencies up to Nyquist that falls in each bin."""
    bounds = np.array([0.0, *edges, sample_rate / 2]).clip(0, sample_rate / 2)
    return jnp.asarray(np.diff(bounds) / (sample_rate / 2), dtype)


def _bin_bounds(edges: tuple[float, ...], sample_rate: float, n_samples: int) -> list[int]:
    """The first rfft frequency of each bin of `n_samples`, then the number of frequencies.

    The bins are placed as [`LowRankFourierOperator`][furax.core.LowRankFourierOperator] does.
    """
    n_frequencies = n_samples // 2 + 1
    inner = [int(np.ceil(edge * n_samples / sample_rate)) for edge in edges]
    return [0, *(min(max(i, 0), n_frequencies) for i in inner), n_frequencies]


def _white_of_zero_lag(
    zero_lag: Float[Array, ' channel'], sample_rate: float, lead: tuple[int, ...]
) -> WhiteNoiseModel:
    """The white noise whose weight per sample is `zero_lag`, channel by channel."""
    sigma = (zero_lag * sample_rate / 2) ** -0.5
    return WhiteNoiseModel(sigma=sigma.reshape(lead), sample_rate=sample_rate)
