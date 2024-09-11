from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from furax.operators import AbstractLinearOperator, PackOperator, square
from furax.operators.toeplitz import SymmetricBandToeplitzOperator

__all__ = [
    'GapFillingOperator',
]


@square
class GapFillingOperator(AbstractLinearOperator):
    """Class for filling masked time samples with a constrained noise realization."""

    cov: SymmetricBandToeplitzOperator
    pack: PackOperator
    rate: float  # sampling rate
    seed: int

    def __init__(
        self,
        cov: SymmetricBandToeplitzOperator,
        pack: PackOperator,
        *,
        rate: float = 1.0,
        seed: int = 0,
    ):
        self.cov = cov
        self.pack = pack
        self.rate = rate
        self.seed = seed

    @staticmethod
    def _get_default_fft_size(n: int) -> int:
        additional_power = 1
        return int(2 ** (additional_power + np.ceil(np.log2(n))))

    @staticmethod
    def _get_kernel(n_tt: Float[Array, ' _'], size: int) -> Float[Array, ' {size}']:
        kernel = jnp.concatenate((n_tt[-1:0:-1], n_tt))
        lagmax = kernel.size // 2
        nb_zeros = size - kernel.size
        kernel = jnp.pad(kernel, (0, nb_zeros), mode='constant')
        kernel = jnp.roll(kernel, -lagmax)
        return kernel

    def _get_psd(
        self, n_tt: Float[Array, ' _'], fft_size: int
    ) -> Float[Array, ' {fft_size // 2 + 1}']:
        kernel = self._get_kernel(n_tt, fft_size)
        psd = jnp.abs(jnp.fft.rfft(kernel, n=fft_size))
        # zero out DC value
        psd = psd.at[0].set(0)
        return psd

    def _generate_realization_for(self, x: Float[Array, '...'], seed: int) -> Float[Array, '...']:
        @partial(jnp.vectorize, signature='(n),(k),()->(n)')
        def func(x, n_tt, subkey):  # type: ignore[no-untyped-def]
            x_size = x.size
            fft_size = self._get_default_fft_size(x_size)
            npsd = fft_size // 2 + 1
            norm = self.rate * float(npsd - 1)

            # Get PSD values (size = fft_size // 2 + 1)
            psd = self._get_psd(n_tt, fft_size)
            scale = jnp.sqrt(norm * psd)

            # Gaussian Re/Im random numbers
            rngdata = jax.random.normal(subkey, shape=(fft_size,))

            fdata = jnp.empty(npsd, dtype=jnp.complex128)

            # Set DC and Nyquist frequency imaginary parts to zero
            fdata = fdata.at[0].set(rngdata[0] + 0.0j)
            fdata = fdata.at[-1].set(rngdata[npsd - 1] + 0.0j)

            # Repack the other values
            fdata = fdata.at[1:-1].set(rngdata[1 : npsd - 1] + 1j * rngdata[-1 : npsd - 1 : -1])

            # scale by PSD and inverse FFT
            tdata = jnp.fft.irfft(fdata * scale)

            # subtract the DC level for the samples we want
            offset = (fft_size - x_size) // 2
            xi = tdata[offset : offset + x_size]
            xi -= jnp.mean(xi)
            return xi

        key = jax.random.key(seed)
        subkeys = jax.random.split(key, x.shape[:-1])
        real: Float[Array, '...'] = func(x, self.cov.band_values, subkeys)
        return real

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        real = self._generate_realization_for(x, self.seed)
        p, u = self.pack, self.pack.T  # pack, unpack operators
        incomplete_cov = p @ self.cov @ u
        op = self.cov @ u @ incomplete_cov.I @ p
        y: Float[Array, '...'] = real + op(x - real)
        # copy valid samples from original vector
        y = y.at[p.mask].set(p(x))
        return y

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.pack.T.in_structure()
