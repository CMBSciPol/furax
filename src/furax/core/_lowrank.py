import itertools
from dataclasses import field
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Inexact, PyTree

from furax.tree import ravel, unravel

from ._base import AbstractLinearOperator, symmetric

__all__ = [
    'LowRankFourierOperator',
]


@symmetric
class LowRankFourierOperator(AbstractLinearOperator):
    r"""Diagonal plus low-rank operator across channels, applied frequency bin by frequency bin.

    The input is a set of channels sampled along the last axis. In each frequency bin $b$ the
    operator multiplies the Fourier coefficients $\hat x(f)$ of all channels by the same
    $n_\mathrm{ch} \times n_\mathrm{ch}$ matrix

    $$ M_b = D_b + V_b G_b V_b^\dagger, $$

    where $D_b$ is diagonal and $V_b$ holds $k$ modes across the channels. The inverse has the
    same form. [`inverse`][furax.core.LowRankFourierOperator.inverse] computes it with the Woodbury
    identity, so applying either operator costs one FFT pair plus products with
    $n_\mathrm{ch} \times k$ matrices, with no dense $n_\mathrm{ch}^2$ matrix and no band in the
    time domain.

    When correlations stop at some frequency, make `modes` and `core` one bin shorter than
    `diagonal`: the last bin is then uncoupled, $M_b = D_b$. The operator applies it in the time
    domain, so that only the frequencies below it are held in memory.

    The operator is symmetric when every $G_b$ is Hermitian. Channels are the leading axes of the
    leaves of the input pytree, flattened and concatenated in leaf order; every leaf has the same
    number of samples. Frequencies are in Hz and follow the FFT of the whole input, so the
    operator applies unchanged to inputs of any length at the same `sample_rate`. The transform is
    circulant: the end of the input wraps around to its start.

    Examples:
        Two channels sharing a common mode below 1 Hz, and uncoupled above it:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> diagonal = jnp.ones((2, 2))  # (bin, channel)
        >>> modes = jnp.array([[[1.0], [1.0]]])  # (bin - 1, channel, mode)
        >>> power = jnp.array([[[4.0]]])  # (bin - 1, mode, mode)
        >>> structure = jax.ShapeDtypeStruct((2, 100), jnp.float32)
        >>> op = LowRankFourierOperator(
        ...     diagonal, modes, power, edges=(1.0,), sample_rate=10.0, in_structure=structure
        ... )
        >>> inverse = op.inverse()
    """

    diagonal: Float[Array, 'bin channel']
    """The diagonal $D_b$ of each bin."""

    modes: Inexact[Array, 'coupled channel mode']
    """The modes $V_b$ of each bin, or of all bins but the last when it is uncoupled. Unused modes
    are padded with zeros."""

    core: Inexact[Array, 'coupled mode mode']
    """The Hermitian matrix $G_b$ of each bin holding modes."""

    edges: tuple[float, ...] = field(metadata={'static': True})
    """Interior bin boundaries in Hz, increasing. Bin $b$ spans `edges[b-1]` to `edges[b]`, the
    first one starts at zero and the last one ends at the Nyquist frequency."""

    sample_rate: float = field(metadata={'static': True})
    """Sample rate in Hz."""

    batch_size: int = field(default=1, metadata={'static': True})
    """Number of channels transformed together. Smaller batches take less memory; large ones can
    be slower on a GPU, while on a CPU tens of channels are usually faster than one."""

    def __post_init__(self) -> None:
        super().__post_init__()
        n_bins, n_channels = self.diagonal.shape
        if len(self.edges) != n_bins - 1 or list(self.edges) != sorted(self.edges):
            raise ValueError(f'Expected {n_bins - 1} increasing bin edges, got {self.edges}')
        n_coupled = self.modes.shape[0]
        if n_coupled not in (n_bins, n_bins - 1) or self.modes.shape[1] != n_channels:
            raise ValueError(
                f'Modes have shape {self.modes.shape}, expected {n_bins} or {n_bins - 1} bins '
                f'of {n_channels} channels'
            )
        if self.core.shape != (n_coupled, *(2 * (self.modes.shape[2],))):
            raise ValueError(
                f'Core has shape {self.core.shape}, expected a matrix per bin of the modes'
            )
        if self.batch_size < 1:
            raise ValueError(f'batch_size must be >= 1, got {self.batch_size}')
        n_input = sum(leaf.size // leaf.shape[-1] for leaf in jax.tree.leaves(self.in_structure))
        if n_input != n_channels:
            raise ValueError(f'The input has {n_input} channels, the operator {n_channels}')

    def mv(
        self, x: PyTree[Float[Array, '*channels sample']]
    ) -> PyTree[Float[Array, '*channels sample']]:
        filtered = self._filter(ravel(x, trailing_ndim=1))
        out = unravel(x, filtered, trailing_ndim=1)
        return jax.tree.map(lambda o, leaf: o.astype(leaf.dtype), out, x)

    def inverse(self) -> Self:
        r"""The inverse operator, from the Woodbury identity.

        With $M_b = D_b + V_b G_b V_b^\dagger$,

        $$ M_b^{-1} = D_b^{-1} - D_b^{-1} V_b \, (1 + G_b S_b)^{-1} G_b \, V_b^\dagger D_b^{-1}, $$

        where $S_b = V_b^\dagger D_b^{-1} V_b$. This form needs no inverse of $G_b$, so modes
        padded with zeros stay inert. The diagonal must be strictly positive.

        With complex modes the inverse is exact at every frequency but the zero one and, unless
        it is in an uncoupled last bin, the Nyquist one. The Fourier coefficients of a real input
        are real there, so only the real part of $M_b$ acts on them, and the inverse of a real
        part is not the real part of the inverse.
        """
        inverse_diagonal = 1 / self.diagonal
        modes = inverse_diagonal[: self.modes.shape[0], :, None] * self.modes
        gram = jnp.conj(self.modes).mT @ modes
        eye = jnp.eye(self.core.shape[-1], dtype=self.core.dtype)
        core = -jnp.linalg.solve(eye + self.core @ gram, self.core)
        core = (core + jnp.conj(core).mT) / 2
        return type(self)(
            inverse_diagonal,
            modes,
            core,
            edges=self.edges,
            sample_rate=self.sample_rate,
            in_structure=self.in_structure,
            batch_size=self.batch_size,
        )

    def as_matrix(self) -> Inexact[Array, 'a a']:
        r"""The operator as a dense matrix on the channels flattened in row-major order.

        Entry $((a, s), (b, t))$ is $\frac1n \sum_f w_f \, \mathrm{Re}\big(M_f[a, b] \,
        e^{2\pi i f (s - t) / n}\big)$, with $w_f = 1$ at the zero and Nyquist frequencies and
        $2$ elsewhere. The matrix has $(n_\mathrm{ch} n)^2$ entries: use it on small inputs.
        """
        n = jax.tree.leaves(self.in_structure)[0].shape[-1]
        n_frequencies = n // 2 + 1
        n_coupled = self.modes.shape[0]
        n_channels = self.diagonal.shape[1]

        diagonal = jax.vmap(jnp.diag)(self.diagonal)
        coupled = diagonal[:n_coupled] + self.modes @ self.core @ jnp.conj(self.modes).mT
        per_bin = jnp.concatenate([coupled, diagonal[n_coupled:]])
        bin_of_frequency = np.searchsorted(
            self._bin_bounds(n)[1:-1], np.arange(n_frequencies), 'right'
        )

        weight = np.full(n_frequencies, 2.0)
        weight[0] = 1.0
        if n % 2 == 0:
            weight[-1] = 1.0
        phase = np.exp(2j * np.pi * np.outer(np.arange(n_frequencies), np.arange(n)) / n)
        # circulant: entry (a, s), (b, t) depends on s - t only
        kernel = jnp.einsum('f,fab,fd->abd', weight, per_bin[bin_of_frequency], phase).real / n
        lag = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
        matrix = kernel[:, :, lag].transpose(0, 2, 1, 3)
        return matrix.reshape(n_channels * n, n_channels * n).astype(self.out_promoted_dtype)

    def _filter(self, channels: Float[Array, 'channel sample']) -> Float[Array, 'channel sample']:
        """The matrix of each bin applied to the channels, one batch of FFTs at a time.

        An uncoupled last bin holds most frequencies and needs no FFT, as a diagonal in the time
        domain. Only the difference to it, confined to the frequencies below, goes through the
        transforms.
        """
        n_channels, n_samples = channels.shape
        n_coupled = self.modes.shape[0]
        bounds = self._bin_bounds(n_samples)
        real_dtype = channels.dtype
        white = jnp.zeros((n_channels, 1), real_dtype)
        if n_coupled < self.diagonal.shape[0]:
            white = self.diagonal[-1].astype(real_dtype)[:, None]
        n_low = bounds[n_coupled]
        if n_low == 0:
            return white * channels

        low_spectrum = jax.lax.map(
            lambda channel: jnp.fft.rfft(channel)[:n_low], channels, batch_size=self.batch_size
        )

        complex_dtype = low_spectrum.dtype
        pieces = []
        for b, (start, stop) in enumerate(itertools.pairwise(bounds[: n_coupled + 1])):
            if start == stop:
                continue
            piece = low_spectrum[:, start:stop]
            modes = self.modes[b].astype(complex_dtype)
            core = self.core[b].astype(complex_dtype)
            diagonal = self.diagonal[b].astype(real_dtype)[:, None]
            pieces.append((diagonal - white) * piece + modes @ (core @ (modes.conj().T @ piece)))
        correction = jnp.concatenate(pieces, axis=-1)

        n_frequencies = bounds[-1]

        def apply(args):
            channel, correction_row, white_row = args
            full = jnp.pad(correction_row, (0, n_frequencies - n_low))
            return white_row * channel + jnp.fft.irfft(full, n=n_samples).astype(real_dtype)

        return jax.lax.map(apply, (channels, correction, white), batch_size=self.batch_size)

    def _bin_bounds(self, n_samples: int) -> list[int]:
        """Indices of the first rfft frequency of each bin, followed by the number of frequencies."""
        n_frequencies = n_samples // 2 + 1
        inner = [int(np.ceil(edge * n_samples / self.sample_rate)) for edge in self.edges]
        return [0, *(min(max(i, 0), n_frequencies) for i in inner), n_frequencies]
