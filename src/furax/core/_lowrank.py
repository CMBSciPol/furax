import itertools
from dataclasses import field
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Inexact, PyTree

from ._base import AbstractLinearOperator, symmetric

__all__ = [
    'LowRankFourierOperator',
]


@symmetric
class LowRankFourierOperator(AbstractLinearOperator):
    r"""Diagonal plus low-rank operator across channels, applied frequency bin by frequency bin.

    The input is a set of channels (detectors, Stokes streams) sampled along the last axis. In
    each frequency bin $b$ the operator multiplies the Fourier coefficients $\hat x(f)$ of all
    channels by the same $n_\mathrm{ch} \times n_\mathrm{ch}$ matrix

    $$ M_b = D_b + V_b G_b V_b^\dagger, $$

    where $D_b$ is diagonal and $V_b$ holds $k$ modes across the channels. With $G_b$ the mode
    powers this is a noise covariance $N(f)$; its inverse, which has the same form, is a noise
    weight. [`inverse`][furax.core.LowRankFourierOperator.inverse] computes it with the Woodbury
    identity, so applying either costs one FFT pair plus products with $n_\mathrm{ch} \times k$
    matrices, with no dense $n_\mathrm{ch}^2$ matrix and no band in the time domain.

    The operator is symmetric when every $G_b$ is Hermitian. Channels are the leading axes of the
    leaves of the input pytree, flattened and concatenated in leaf order; every leaf has the same
    number of samples. Frequencies are in Hz and follow the FFT of the whole input, so the
    operator applies unchanged to inputs of any length at the same `sample_rate`. The transform is
    circulant: the end of the input wraps around to its start.

    Examples:
        Two channels sharing a common mode below 1 Hz, and independent noise elsewhere:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> diagonal = jnp.ones((2, 2))  # (bin, channel)
        >>> modes = jnp.array([[[1.0], [1.0]], [[0.0], [0.0]]])  # (bin, channel, mode)
        >>> power = jnp.array([[[4.0]], [[0.0]]])  # (bin, mode, mode)
        >>> structure = jax.ShapeDtypeStruct((2, 100), jnp.float32)
        >>> noise = LowRankFourierOperator(
        ...     diagonal, modes, power, edges=(1.0,), sample_rate=10.0, in_structure=structure
        ... )
        >>> weight = noise.inverse()
    """

    diagonal: Float[Array, 'bin channel']
    """The diagonal $D_b$ of each bin."""

    modes: Inexact[Array, 'bin channel mode']
    """The modes $V_b$ of each bin. Unused modes are padded with zeros."""

    core: Inexact[Array, 'bin mode mode']
    """The Hermitian matrix $G_b$ of each bin."""

    edges: tuple[float, ...] = field(metadata={'static': True})
    """Interior bin boundaries in Hz, increasing. Bin $b$ spans `edges[b-1]` to `edges[b]`, the
    first one starts at zero and the last one ends at the Nyquist frequency."""

    sample_rate: float = field(metadata={'static': True})
    """Sample rate in Hz."""

    def __post_init__(self) -> None:
        super().__post_init__()
        n_bins, n_channels = self.diagonal.shape
        if len(self.edges) != n_bins - 1 or list(self.edges) != sorted(self.edges):
            raise ValueError(f'Expected {n_bins - 1} increasing bin edges, got {self.edges}')
        if self.modes.shape[:2] != (n_bins, n_channels):
            raise ValueError(
                f'Modes have shape {self.modes.shape}, expected {(n_bins, n_channels)}'
            )
        if self.core.shape != (n_bins, *(2 * (self.modes.shape[2],))):
            raise ValueError(
                f'Core has shape {self.core.shape}, expected a matrix per bin and mode'
            )
        n_input = sum(leaf.size // leaf.shape[-1] for leaf in jax.tree.leaves(self.in_structure))
        if n_input != n_channels:
            raise ValueError(f'The input has {n_input} channels, the operator {n_channels}')

    def mv(
        self, x: PyTree[Float[Array, '*channels sample']]
    ) -> PyTree[Float[Array, '*channels sample']]:
        leaves, treedef = jax.tree.flatten(x)
        n_samples = leaves[0].shape[-1]
        channels = jnp.concatenate([leaf.reshape(-1, n_samples) for leaf in leaves])
        spectrum = jnp.fft.rfft(channels)

        bounds = self._bin_bounds(n_samples)
        pieces = []
        for b, (start, stop) in enumerate(itertools.pairwise(bounds)):
            if start == stop:
                continue
            piece = spectrum[:, start:stop]
            modes = self.modes[b].astype(spectrum.dtype)
            core = self.core[b].astype(spectrum.dtype)
            diagonal = self.diagonal[b].astype(piece.real.dtype)
            amplitudes = core @ (modes.conj().T @ piece)
            pieces.append(diagonal[:, None] * piece + modes @ amplitudes)
        filtered = jnp.fft.irfft(jnp.concatenate(pieces, axis=-1), n=n_samples)

        sizes = [leaf.size // n_samples for leaf in leaves]
        splits = np.cumsum(sizes)[:-1]
        outputs = [
            part.reshape(leaf.shape).astype(leaf.dtype)
            for part, leaf in zip(jnp.split(filtered, splits), leaves, strict=True)
        ]
        return jax.tree.unflatten(treedef, outputs)

    def inverse(self) -> Self:
        r"""The inverse operator, from the Woodbury identity.

        With $M_b = D_b + V_b G_b V_b^\dagger$,

        $$ M_b^{-1} = D_b^{-1} - D_b^{-1} V_b \, (1 + G_b S_b)^{-1} G_b \, V_b^\dagger D_b^{-1}, $$

        where $S_b = V_b^\dagger D_b^{-1} V_b$. This form needs no inverse of $G_b$, so modes
        padded with zeros stay inert. The diagonal must be strictly positive.
        """
        inverse_diagonal = 1 / self.diagonal
        modes = inverse_diagonal[..., None] * self.modes
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
        )

    def _bin_bounds(self, n_samples: int) -> list[int]:
        """Indices of the first rfft frequency of each bin, followed by the number of frequencies."""
        n_frequencies = n_samples // 2 + 1
        inner = [int(np.ceil(edge * n_samples / self.sample_rate)) for edge in self.edges]
        return [0, *(min(max(i, 0), n_frequencies) for i in inner), n_frequencies]
