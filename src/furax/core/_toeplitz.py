from collections.abc import Callable
from dataclasses import field
from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import lax
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Inexact, PyTree

from ._base import AbstractLinearOperator, symmetric

__all__ = [
    'SymmetricBandToeplitzOperator',
    'dense_symmetric_band_toeplitz',
]


@symmetric
class SymmetricBandToeplitzOperator(AbstractLinearOperator):
    """Operator for symmetric band Toeplitz convolution.

    A Toeplitz matrix has constant diagonals. This operator is symmetric and
    exploits the band structure for efficient computation. For multidimensional
    band values, the operator is block diagonal.

    Available methods (N = matrix size, K = number of bands):
        - ``dense``: dense matrix multiplication
        - ``direct``: direct convolution
        - ``fft``: FFT on the whole input
        - ``overlap_save_parallel``: batched FFT on chunks (default)
        - ``overlap_save_sequential``: sequential FFT on chunks

    ============================  =========  ======
    Method                        Time       Memory
    ============================  =========  ======
    ``dense``                     O(N^2)     N^2
    ``direct``                    O(NK)      2N
    ``fft``                       O(N log N) 3N
    ``overlap_save_sequential``   O(N log K) 2N
    ``overlap_save_parallel``     O(N log K) 4N
    ============================  =========  ======

    Attributes:
        band_values: The band values (first element is the diagonal).
        method: The computation method.
        fft_size: FFT size for the fft and overlap methods.

    Example:
        >>> tod = jnp.ones((2, 5))
        >>> op = SymmetricBandToeplitzOperator(
        ...     jnp.array([[1., 0.5], [1, 0.25]]),
        ...     in_structure=jax.ShapeDtypeStruct(tod.shape, tod.dtype))
        >>> op(tod)
        Array([[1.5 , 2.  , 2.  , 2.  , 1.5 ],
               [1.25, 1.5 , 1.5 , 1.5 , 1.25]], dtype=float64)
        >>> op.as_matrix()
        Array([[1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.5 , 1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.5 , 1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.5 , 1.  , 0.5 , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.5 , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.25, 0.  , 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.  , 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25, 0.  ],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  , 0.25],
               [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 1.  ]],      dtype=float64)
    """

    METHODS: ClassVar[tuple[str, ...]] = (
        'dense',
        'direct',
        'fft',
        'overlap_save_parallel',
        'overlap_save_sequential',
    )
    band_values: Float[Array, '...']
    method: str = field(metadata={'static': True})
    fft_size: int | None = field(metadata={'static': True})

    def __init__(
        self,
        band_values: Float[Array, ' a'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        method: str = 'overlap_save_parallel',
        fft_size: int | None = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f'Invalid method {method}. Choose from: {", ".join(self.METHODS)}')

        band_number = 2 * band_values.shape[-1] - 1
        if fft_size is not None:
            if method not in ('fft', 'overlap_save_parallel', 'overlap_save_sequential'):
                raise ValueError('The FFT size is only used by the fft and overlap methods.')
            if fft_size < band_number:
                raise ValueError('The FFT size should not be less than the number of bands.')

        elif method == 'fft':
            signal_length = in_structure.shape[-1]
            padded_length = signal_length + band_number - 1
            fft_size = self._get_next_power_of_two(padded_length)

        elif method.startswith('overlap_save'):
            overlap = band_number - 1  # 2 * (K - 1) where K = band_values.shape[-1]
            fft_size = self._get_optimal_fft_size(overlap)

        object.__setattr__(self, 'band_values', band_values)
        object.__setattr__(self, 'method', method)
        object.__setattr__(self, 'fft_size', fft_size)
        object.__setattr__(self, 'in_structure', in_structure)

    @staticmethod
    def _get_next_power_of_two(n: int) -> int:
        """Return the smallest power of 2 >= n."""
        if n <= 1:
            return 1
        return int(2 ** np.ceil(np.log2(n)))

    @staticmethod
    def _get_optimal_fft_size(overlap: int) -> int:
        """Return the power-of-2 FFT size that minimizes the OLS computation cost.

        The cost per sample is proportional to F*log2(F) / (F - overlap), where F
        is the FFT size and (F - overlap) is the number of new samples per block.
        """
        if overlap <= 0:
            return 2
        min_power = int(np.ceil(np.log2(overlap + 1)))
        best_f: int = 2**min_power
        best_cost = best_f * min_power / (best_f - overlap)
        for p in range(min_power + 1, min_power + 30):
            f = 2**p
            cost = f * p / (f - overlap)
            if cost < best_cost:
                best_cost = cost
                best_f = f
            else:
                break
        return best_f

    def _get_func(self) -> Callable[[Array, Array], Array]:
        if self.method == 'dense':
            return self._apply_dense
        if self.method == 'direct':
            return self._apply_direct
        if self.method == 'fft':
            return self._apply_fft
        if self.method == 'overlap_save_parallel':
            return self._apply_overlap_save_parallel
        if self.method == 'overlap_save_sequential':
            return self._apply_overlap_save_sequential

        raise NotImplementedError

    def _apply_dense(self, x: Array, band_values: Array) -> Array:
        matrix = dense_symmetric_band_toeplitz(x.shape[-1], band_values)
        return matrix @ x

    def _apply_direct(self, x: Array, band_values: Array) -> Array:
        kernel = self._get_kernel(band_values)
        half_band_width = kernel.size // 2
        return jnp.convolve(jnp.pad(x, (half_band_width, half_band_width)), kernel, mode='valid')

    def _apply_fft(self, x: Array, band_values: Array) -> Array:
        assert self.fft_size is not None
        kernel = self._get_kernel(band_values)
        half_band_width = kernel.size // 2
        signal_length = x.shape[-1]
        H = jnp.fft.rfft(kernel, n=self.fft_size)
        x_padded = jnp.pad(x, (0, 2 * half_band_width), mode='constant')
        X_padded = jnp.fft.rfft(x_padded, n=self.fft_size)
        Y_padded = jnp.fft.irfft(X_padded * H, n=self.fft_size)
        if half_band_width == 0:
            return Y_padded[:signal_length]
        return Y_padded[half_band_width : half_band_width + signal_length]

    def _apply_overlap_save_parallel(self, x: Array, band_values: Array) -> Array:
        """Overlap-and-save with batched rfft via vmap. All blocks processed in parallel."""
        assert self.fft_size is not None
        kernel = self._get_kernel(band_values)
        half_band_width = kernel.size // 2
        H = jnp.fft.rfft(kernel, n=self.fft_size)
        l = x.shape[-1]
        overlap = 2 * half_band_width
        step_size = self.fft_size - overlap
        nblock = int(np.ceil((l + overlap) / step_size))
        total_length = (nblock - 1) * step_size + self.fft_size
        x_padding_start = overlap
        x_padding_end = total_length - overlap - l
        x_padded = jnp.pad(x, (x_padding_start, x_padding_end), mode='constant')
        y_length = l + x_padding_end

        block_starts = jnp.arange(nblock) * step_size

        def extract_block(start_idx: Array) -> Array:
            return lax.dynamic_slice(x_padded, (start_idx,), (self.fft_size,))

        blocks = jax.vmap(extract_block)(block_starts)
        blocks_fft = jnp.fft.rfft(blocks, n=self.fft_size, axis=-1)
        blocks_filtered = jnp.fft.irfft(blocks_fft * H[None, :], n=self.fft_size, axis=-1)
        y_full = blocks_filtered[:, overlap:].ravel()[:y_length]

        return y_full[half_band_width : half_band_width + l]

    def _apply_overlap_save_sequential(self, x: Array, band_values: Array) -> Array:
        """Overlap-and-save with sequential rfft via fori_loop. Low memory usage."""
        assert self.fft_size is not None
        kernel = self._get_kernel(band_values)
        half_band_width = kernel.size // 2
        H = jnp.fft.rfft(kernel, n=self.fft_size)
        l = x.shape[-1]
        overlap = 2 * half_band_width
        step_size = self.fft_size - overlap
        nblock = int(np.ceil((l + overlap) / step_size))
        total_length = (nblock - 1) * step_size + self.fft_size
        x_padding_start = overlap
        x_padding_end = total_length - overlap - l
        x_padded = jnp.pad(x, (x_padding_start, x_padding_end), mode='constant')
        y = jnp.zeros(l + x_padding_end, dtype=x.dtype)

        def func(iblock: int, y: Array) -> Array:
            position = iblock * step_size
            x_block = lax.dynamic_slice(x_padded, (position,), (self.fft_size,))
            X = jnp.fft.rfft(x_block, n=self.fft_size)
            y_block = jnp.fft.irfft(X * H, n=self.fft_size)
            y = lax.dynamic_update_slice(
                y, lax.dynamic_slice(y_block, (overlap,), (step_size,)), (position,)
            )
            return y

        y = lax.fori_loop(0, nblock, func, y)
        return y[half_band_width : half_band_width + l]  # type: ignore[no-any-return]

    def _get_kernel(self, band_values: Array) -> Array:
        """[4, 3, 2, 1] -> [1, 2, 3, 4, 3, 2, 1]"""
        return jnp.concatenate((band_values[-1:0:-1], band_values))

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        func = jnp.vectorize(self._get_func(), signature='(n),(k)->(n)')
        return func(x, self.band_values)  # type: ignore[no-any-return]

    def as_matrix(self) -> Inexact[Array, 'a a']:
        @partial(jnp.vectorize, signature='(n),(k)->(n,n)')
        def func(x: Array, band_values: Array) -> Array:
            return dense_symmetric_band_toeplitz(x.size, band_values)

        x = jnp.zeros(self.in_structure.shape, self.in_structure.dtype)
        blocks: Array = func(x, self.band_values)
        if blocks.ndim > 2:
            blocks = blocks.reshape(-1, blocks.shape[-1], blocks.shape[-1])
            matrix: Array = jsl.block_diag(*blocks)
            return matrix
        return blocks


def dense_symmetric_band_toeplitz(n: int, band_values: ArrayLike) -> Array:
    """Returns a dense Symmetric Band Toeplitz matrix."""
    band_values = jnp.asarray(band_values)
    output = jnp.zeros(n**2, dtype=band_values.dtype)
    band_width = band_values.size - 1
    for j in range(-band_width, band_width + 1):
        value = band_values[abs(j)]
        m = n - j
        if j >= 0:
            indices = j + jnp.arange(m) * (n + 1)
        else:
            indices = -n * j + jnp.arange(m) * (n + 1)
        output = output.at[indices].set(value)
    return output.reshape(n, n)
