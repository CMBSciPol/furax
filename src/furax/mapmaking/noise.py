from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

from furax.core import AbstractLinearOperator, DiagonalOperator, SymmetricBandToeplitzOperator

from .utils import apodization_window, run_lbfgs
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

    def l2_loss(self, f: Float[Array, ' a'], Pxx: Float[Array, 'dets a']) -> Float[Array, '']:
        """l2 loss in log-log space
        f[0] is assume to be 0
        """
        pred = self.log_psd(f[1:])
        loss = jnp.trapezoid((pred - jnp.log10(Pxx[:, 1:])) ** 2, jnp.log10(f[1:]))
        return jnp.mean(loss)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
        init_model: 'NoiseModel',
        max_iter: int,
        tol: float,
    ) -> 'NoiseModel':
        """Fit a noise model to the given psd using l2 loss in log-log space.
        Must provide an instance of a NoiseModel subclass as init_model,
        and the output is the fitted noise model of the same class.
        It is recommended to use the given noise model class's fit_psd_model
        function instead if available.
        """
        loss = jax.jit(lambda m: m.l2_loss(f, Pxx))
        opt = optax.lbfgs()
        final_model, _ = run_lbfgs(init_model, loss, opt, max_iter, tol)
        return final_model  # type: ignore[return-value]


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
    def fit_psd_model(  # type: ignore[override]
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
    ) -> 'WhiteNoiseModel':
        """Fit a white noise model to data"""
        # avoid f=0
        sigma = jnp.power(10, 0.5 * jnp.median(jnp.log10(Pxx[:, 1:]), axis=-1))
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

        nperseg: int = cast(int, kwargs.get('nperseg'))
        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))
        window = apodization_window(correlation_length)

        freq = jnp.fft.rfftfreq(nperseg, 1 / sample_rate)
        eval_psd = self.psd(freq)
        invntt = jnp.fft.irfft(1.0 / eval_psd, n=nperseg)[..., :correlation_length]
        inv_band = invntt * window
        # padded_band = jnp.pad(inv_band, [(0, 0), (0, nperseg // 2 + 1 - inv_band.shape[-1])])
        padded_band = jnp.pad(inv_band, [(0, 0), (0, nperseg + 1 - inv_band.shape[-1])])
        symmetrised_band = jnp.concatenate([padded_band, padded_band[..., -2:0:-1]], axis=-1)
        eff_inv_psd = jnp.fft.rfft(symmetrised_band, n=nperseg).real
        new_band = jnp.fft.irfft(1.0 / eff_inv_psd, n=nperseg)[:, :correlation_length] * window
        return SymmetricBandToeplitzOperator(new_band, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the inverse autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        nperseg: int = cast(int, kwargs.get('nperseg'))
        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))

        freq = jnp.fft.rfftfreq(nperseg, 1 / sample_rate)
        eval_psd = self.psd(freq)
        inv_psd = jnp.where(eval_psd > 0, 1 / eval_psd, 0.0)
        invntt = jnp.fft.irfft(inv_psd, n=nperseg)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(invntt * window, in_structure)

    def to_white_noise_model(self) -> WhiteNoiseModel:
        return WhiteNoiseModel(sigma=self.sigma)

    @classmethod
    def fit_psd_model(  # type: ignore[override]
        cls,
        f: Float[Array, ' freq'],
        Pxx: Float[Array, 'dets freq'],
    ) -> 'AtmosphericNoiseModel':
        """Fit a atmospheric (1/f) noise model to data"""
        params = fit_atmospheric_psd_model(f, Pxx)
        return cls(*params.T)
