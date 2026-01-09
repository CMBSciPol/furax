from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from furax.core import (
    AbstractLinearOperator,
    DiagonalOperator,
    FourierOperator,
    SymmetricBandToeplitzOperator,
)

from .config import NoiseFitConfig
from .utils import apodization_window, fit_white_noise_model
from .utils import fit_psd_model as fit_atmospheric_psd_model


@jax.tree_util.register_dataclass
@dataclass
class NoiseModel:
    """Dataclass for noise models used for ground observation data"""

    @property
    @abstractmethod
    def n_detectors(self) -> int: ...

    @abstractmethod
    def psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

    @abstractmethod
    def log_psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

    @abstractmethod
    def to_array(self) -> Float[Array, 'dets n']: ...

    @abstractmethod
    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator: ...

    @abstractmethod
    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator: ...

    def to_operator_fourier(
        self,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        sample_rate: float,
        inverse: bool = True,
    ) -> FourierOperator:
        """Fourier operator representation of the noise model"""
        func = (lambda f: 1.0 / self.psd(f)) if inverse else self.psd
        # do not use apodization -- sufficient padding is done by the FourierOperator
        return FourierOperator(func, in_structure, sample_rate=sample_rate, apodize=False)

    def l2_loss(
        self, f: Float[Array, ' a'], Pxx: Float[Array, 'dets a'], mask: Float[Array, ' a']
    ) -> Float[Array, '']:
        """l2 loss in log-log spacea with given frequency mask"""
        pred = self.log_psd(f)
        loss = jnp.trapezoid(((pred - jnp.log10(Pxx)) * mask[None, :]) ** 2, jnp.log10(f))
        return jnp.mean(loss)


@jax.tree_util.register_dataclass
@dataclass
class WhiteNoiseModel(NoiseModel):
    """Dataclass for the white noise model used for ground observation data"""

    sigma: Float[Array, ' dets']

    @property
    def n_detectors(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return self.sigma**2

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return 2 * jnp.log10(self.sigma)

    def to_array(self) -> Float[Array, 'dets n']:
        return self.sigma[:, None]

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        return DiagonalOperator(self.sigma[:, None] ** 2, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        inv_var = jnp.where(self.sigma > 0, 1.0 / (self.sigma**2), 0.0)
        return DiagonalOperator(inv_var[:, None], in_structure=in_structure)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
        hwp_freq: Array,
        config: NoiseFitConfig = NoiseFitConfig(),
    ) -> 'WhiteNoiseModel':
        """Fit a white noise model to data"""
        sigma = fit_white_noise_model(f, Pxx, hwp_freq=hwp_freq, config=config)['fit']
        return cls(sigma)


@jax.tree_util.register_dataclass
@dataclass
class AtmosphericNoiseModel(NoiseModel):
    """Dataclass for the 1/f noise model used for ground observation data"""

    sigma: Float[Array, ' dets']
    alpha: Float[Array, ' dets']
    fk: Float[Array, ' dets']
    f0: Float[Array, ' dets']

    @property
    def n_detectors(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return self.sigma**2 * (1 + ((f + self.f0) / self.fk) ** self.alpha)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return 2 * jnp.log10(self.sigma) + jnp.log10(1 + ((f + self.f0) / self.fk) ** self.alpha)

    def to_array(self) -> Float[Array, 'dets n']:
        return jnp.stack([self.sigma, self.alpha, self.fk, self.f0], axis=-1)

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))
        window = apodization_window(correlation_length)

        fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = self.psd(freq)
        invntt = jnp.fft.irfft(1.0 / eval_psd, n=fft_size)[..., :correlation_length]
        inv_band = invntt * window
        # pad only the last dimension
        pad_width = [(0, 0)] * (inv_band.ndim - 1) + [(0, fft_size - (2 * inv_band.shape[-1] - 1))]
        padded_band = jnp.pad(inv_band, pad_width)
        symmetrised_band = jnp.concatenate([padded_band, inv_band[..., -1:0:-1]], axis=-1)
        eff_inv_psd = jnp.fft.rfft(symmetrised_band, n=fft_size).real
        new_band = jnp.fft.irfft(1.0 / eff_inv_psd, n=fft_size)[:, :correlation_length] * window
        return SymmetricBandToeplitzOperator(new_band, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the inverse autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))

        fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(2 * correlation_length - 1)
        freq = jnp.fft.rfftfreq(fft_size, 1 / sample_rate)
        eval_psd = self.psd(freq)
        inv_psd = jnp.where(eval_psd > 0, 1 / eval_psd, 0.0)
        invntt = jnp.fft.irfft(inv_psd, n=fft_size)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(invntt * window, in_structure)

    def to_white_noise_model(self) -> WhiteNoiseModel:
        return WhiteNoiseModel(sigma=self.sigma)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
        hwp_freq: Array,
        config: NoiseFitConfig = NoiseFitConfig(),
    ) -> 'AtmosphericNoiseModel':
        """Fit a atmospheric (1/f) noise model to data"""
        result = fit_atmospheric_psd_model(f, Pxx, hwp_freq=hwp_freq, config=config)
        return cls(*result['fit'].T)
