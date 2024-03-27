from functools import partial
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

    Four methods are available:
        - direct, using a direct convolution: O(N^2)
        - fft, which applies the Discrete Fourier Transform on the whole input: O(NlogN)
        - overlap_save: O(NlogK)
        - overlap_add: O(NlogK)
    """

    METHODS: ClassVar[tuple[str]] = ['direct', 'fft', 'overlap_save', 'overlap_add']
    kernel: Float[Array, '...'] = eqx.field(static=True)
    shape: tuple[int, int]
    method: str
    fft_size: int

    #    def __hash__(self) -> int:
    #        return id(self)

    def __init__(
        self,
        kernel: Float[Array, ' a'],
        shape: tuple[int, int],
        *,
        method: str = 'overlap_save',
        fft_size: int | None = None,
    ):
        self.kernel = kernel
        self.shape = shape
        if fft_size is None:
            additional_power = 1
            fft_size = int(2 ** (additional_power + np.ceil(np.log2(self.kernel.size))))
        self.fft_size = fft_size
        self.method = method

    def _get_func(self):
        if self.method == 'direct':
            convolve = partial(_convolution, self.kernel)
        elif self.method == 'fft':
            H = jnp.fft.fft(self.kernel, self.shape[1] + self.kernel.size - 1)
            convolve = partial(_convolution_fft, H, self.kernel.size)
        elif self.method == 'overlap_save':
            H = jnp.fft.fft(self.kernel, self.fft_size)
            convolve = lambda x: _overlap_save_jax(x, H, self.fft_size, self.kernel.size)
        elif self.method == 'overlap_add':
            H = jnp.fft.fft(self.kernel, self.fft_size)
            convolve = lambda x: _overlap_add_jax(x, H, self.fft_size, self.kernel.size)
        else:
            raise NotImplementedError
        return jax.vmap(convolve)

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        return self._get_func()(x)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self

    def as_matrix(self) -> Inexact[Array, 'a a']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct(self.shape, float)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure()


def _convolution(h, x):
    b = h.shape[0]
    return jnp.convolve(jnp.pad(x, (b - 1, 0)), h, mode='valid')


def _convolution_fft(H, b, x):
    x_padded = jnp.pad(x, (b - 1, 0), mode='constant')
    X_padded = jnp.fft.fft(x_padded)
    Y_padded = jnp.fft.ifft(X_padded * H)
    return Y_padded[b - 1 :].real


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


def _overlap_save_jax(x, H, fft_size, b):
    l = x.size
    overlap = b - 1
    step_size = fft_size - overlap
    nblock = int(np.ceil((l + overlap) / step_size))
    total_length = (nblock - 1) * step_size + fft_size
    x_padding_start = overlap
    x_padding_end = total_length - overlap - l
    x_padded = jnp.pad(x, (x_padding_start, x_padding_end), mode='constant')
    y = jnp.zeros(l + x_padding_end, dtype=complex)

    def func(iblock, y):
        position = iblock * step_size
        x_block = lax.dynamic_slice(x_padded, (position,), (fft_size,))
        X = jnp.fft.fft(x_block)
        y_block = jnp.fft.ifft(X * H)
        y = lax.dynamic_update_slice(
            y, lax.dynamic_slice(y_block, (b - 1,), (step_size,)), (position,)
        )
        return y

    y = lax.fori_loop(0, nblock, func, y)
    return y[:l]


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
