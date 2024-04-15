from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax import lax
from jaxtyping import Array, Float, Inexact, PyTree

__all__ = [
    'SymmetricBandToeplitzOperator',
]


class SymmetricBandToeplitzOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    """Class to represent Symmetric Band Toeplitz matrices.

    Four methods are available, where N is the size of the Toepliz matrix and K the number
    of bands:
        - dense, using the dense matrix: O(N^2)
        - direct, using a direct convolution: O(NK)
        - fft, applying the DFT on the whole input: O(NlogN)
        - overlap_save, applying the DFT on chunked input: O(NlogK)
        - overlap_add, applying the DFT on chunked input: O(NlogK)
    """

    METHODS: ClassVar[tuple[str]] = ['dense', 'direct', 'fft', 'overlap_save']
    shape: tuple[int, int]
    band_values: Float[Array, '...'] = eqx.field(static=True)
    method: str
    fft_size: int

    def __init__(
        self,
        shape: tuple[int, int],
        band_values: Float[Array, ' a'],
        *,
        method: str = 'overlap_save',
        fft_size: int | None = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f'Invalid method {method}. Choose from: {", ".join(self.METHODS)}')

        band_number = 2 * band_values.size - 1
        if fft_size is not None:
            if not method.startswith('overlap_'):
                raise ValueError('The FFT size is only used by the overlap methods.')
            if fft_size < band_number:
                raise ValueError('The FFT size should not be less than the number of bands.')

        self.band_values = band_values
        self.shape = shape
        self.fft_size = fft_size
        self.method = method
        if fft_size is None and method.startswith('overlap_'):
            self.fft_size = self._get_default_fft_size(band_number)

    @staticmethod
    def _get_default_fft_size(band_number: int) -> int:
        additional_power = 1
        return int(2 ** (additional_power + np.ceil(np.log2(band_number))))

    def _get_func(self):
        if self.method == 'dense':
            return self._apply_dense
        if self.method == 'direct':
            return self._apply_direct
        if self.method == 'fft':
            return self._apply_fft
        if self.method == 'overlap_add':
            return self._apply_overlap_add
        if self.method == 'overlap_save':
            return self._apply_overlap_save

        raise NotImplementedError

    def _apply_dense(self, x):
        matrix = dense_symmetric_band_toeplitz(x.shape[-1], self.band_values)
        return matrix @ x

    def _apply_direct(self, x):
        kernel = self._get_kernel()
        half_band_width = kernel.size // 2
        return jnp.convolve(jnp.pad(x, (half_band_width, half_band_width)), kernel, mode='valid')

    def _apply_fft(self, x):
        kernel = self._get_kernel()
        half_band_width = kernel.size // 2
        H = jnp.fft.fft(kernel, x.shape[-1] + 2 * half_band_width)
        x_padded = jnp.pad(x, (0, 2 * half_band_width), mode='constant')
        X_padded = jnp.fft.fft(x_padded)
        Y_padded = jnp.fft.ifft(X_padded * H).real
        if half_band_width == 0:
            return Y_padded
        return Y_padded[half_band_width:-half_band_width]

    def _apply_overlap_add(self, x):
        l = x.shape[-1]
        kernel = self._get_kernel()
        H = jnp.fft.fft(kernel, self.fft_size)
        half_band_width = kernel.size // 2
        m = self.fft_size - 2 * half_band_width

        # pad x so that its size is a multiple of m
        x_padding = 0 if l % m == 0 else m - (l % m)
        x_padded = jnp.pad(x, (x_padding,), mode='constant')
        y = jnp.zeros(l + 2 * half_band_width)

        def func(j, y):
            i = j * m
            x_block_not_padded = lax.dynamic_slice(x_padded, (i,), (m,))
            x_block = jnp.pad(
                x_block_not_padded, (half_band_width, half_band_width), mode='constant'
            )
            X_block = jnp.fft.fft(x_block, self.fft_size)
            Y_block = X_block * H
            y_block = jnp.fft.ifft(Y_block).real
            y = lax.dynamic_update_slice(
                y, lax.dynamic_slice(y, (i,), (self.fft_size,)) + y_block, (i,)
            )
            return y

        y = lax.fori_loop(0, len(range(0, l, m)), func, y)
        return y[half_band_width:-half_band_width]

    def _apply_overlap_save(self, x):
        kernel = self._get_kernel()
        half_band_width = kernel.size // 2
        H = jnp.fft.fft(kernel, self.fft_size)
        l = x.shape[-1]
        overlap = 2 * half_band_width
        step_size = self.fft_size - overlap
        nblock = int(np.ceil((l + overlap) / step_size))
        total_length = (nblock - 1) * step_size + self.fft_size
        x_padding_start = overlap
        x_padding_end = total_length - overlap - l
        x_padded = jnp.pad(x, (x_padding_start, x_padding_end), mode='constant')
        y = jnp.zeros(l + x_padding_end)

        def func(iblock, y):
            position = iblock * step_size
            x_block = lax.dynamic_slice(x_padded, (position,), (self.fft_size,))
            X = jnp.fft.fft(x_block)
            y_block = jnp.fft.ifft(X * H).real
            y = lax.dynamic_update_slice(
                y, lax.dynamic_slice(y_block, (2 * half_band_width,), (step_size,)), (position,)
            )
            return y

        y = lax.fori_loop(0, nblock, func, y)
        return y[half_band_width : half_band_width + l]

    def _get_kernel(self) -> Float[Array, ' a']:
        """[4, 3, 2, 1] -> [1, 2, 3, 4, 3, 2, 1]"""
        return jnp.concatenate([self.band_values[-1:0:-1], self.band_values])

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        return self._get_func()(x)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self

    def as_matrix(self) -> Inexact[Array, 'a a']:
        raise dense_symmetric_band_toeplitz(self.shape[0], self.band_values)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct(self.shape, float)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure()


def dense_symmetric_band_toeplitz(n, band_values):
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


def _overlap_add_jax(x, H, fft_size, b):
    l = x.shape[0]
    if b % 2:
        raise NotImplementedError('Odd bandwidth size not implemented')
    m = fft_size - b

    # pad x so that its size is a multiple of m
    x_padding = 0 if l % m == 0 else m - (l % m)
    x_padded = jnp.pad(x, (x_padding,), mode='constant')
    y = jnp.zeros(l + b)

    def func(j, y):
        i = j * m
        x_block_not_padded = lax.dynamic_slice(x_padded, (i,), (m,))
        x_block = jnp.pad(x_block_not_padded, (b // 2, b // 2), mode='constant')
        X_block = jnp.fft.fft(x_block, fft_size)
        Y_block = X_block * H
        y_block = jnp.fft.ifft(Y_block).real
        y = lax.dynamic_update_slice(y, lax.dynamic_slice(y, (i,), (fft_size,)) + y_block, (i,))
        return y

    y = lax.fori_loop(0, len(range(0, l, m)), func, y)
    return y[b // 2 : -b // 2 - x_padding]


@lx.is_symmetric.register(SymmetricBandToeplitzOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return True


@lx.is_positive_semidefinite.register(SymmetricBandToeplitzOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return True


@lx.is_negative_semidefinite.register(SymmetricBandToeplitzOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.linearise.register(SymmetricBandToeplitzOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


@lx.conj.register(SymmetricBandToeplitzOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator
