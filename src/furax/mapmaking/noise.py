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


@jax.tree_util.register_dataclass
@dataclass
class NoiseModel:
    """Dataclass for noise models used for ground observation data"""

    @property
    @abstractmethod
    def n_dets(self) -> int: ...

    @abstractmethod
    def psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

    @abstractmethod
    def log_psd(self, f: Float[Array, ' a']) -> Float[Array, 'dets a']: ...

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
        loss = jnp.trapezoid((pred - jnp.log(Pxx[:, 1:])) ** 2, jnp.log(f[1:]))
        return jnp.mean(loss)

    @classmethod
    def fit_psd_model(
        cls,
        f: Float[Array, ' a'],
        Pxx: Float[Array, ' a'],
        init_model: 'NoiseModel',
        max_iter: int,
        tol: float,
    ) -> 'NoiseModel':
        """Fit a noise model to the given psd using l2 loss in log-log space.
        Must provide an instance of a NoiseModel subclass as init_model,
        and the output is the fitted noise model of the same class.
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
    def n_dets(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return self.sigma**2

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, ' dets']:
        return 2 * jnp.log(self.sigma)

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        return DiagonalOperator(self.sigma[:, None] ** 2, in_structure=in_structure)

    def inverse_operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        assert in_structure.ndim == 2, 'Dimensions assumed to be (ndets, nsamps)'
        return DiagonalOperator(1.0 / self.sigma[:, None] ** 2, in_structure=in_structure)


@jax.tree_util.register_dataclass
@dataclass
class AtmosphericNoiseModel(NoiseModel):
    """Dataclass for the 1/f noise model used for ground observation data"""

    sigma: Float[Array, ' dets']
    alpha: Float[Array, ' dets']
    fk: Float[Array, ' dets']
    f0: Float[Array, ' dets']

    @property
    def n_dets(self) -> int:
        return len(self.sigma)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return self.sigma**2 * (1 + ((f + self.f0) / self.fk) ** self.alpha)

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0), out_axes=1)
    def log_psd(self, f: Float[Array, '']) -> Float[Array, '']:
        return 2 * jnp.log(self.sigma) + jnp.log10(1 + ((f + self.f0) / self.fk) ** self.alpha)

    def operator(
        self, in_structure: PyTree[jax.ShapeDtypeStruct], **kwargs: Any
    ) -> AbstractLinearOperator:
        """Toeplitz operator from the autocorrelation function evaluated via
        inverse fft of the noise psd, which is apodized and cut at given length
        """

        nperseg: int = cast(int, kwargs.get('nperseg'))
        sample_rate: float = cast(float, kwargs.get('sample_rate'))
        correlation_length: int = cast(int, kwargs.get('correlation_length'))

        freq = jnp.fft.rfftfreq(nperseg, 1 / sample_rate)
        eval_psd = self.psd(freq)
        ntt = jnp.fft.irfft(eval_psd)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(ntt * window, in_structure)

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
        invntt = jnp.fft.irfft(1.0 / eval_psd)[..., :correlation_length]
        window = apodization_window(correlation_length)

        return SymmetricBandToeplitzOperator(invntt * window, in_structure)

    def to_white_noise_model(self) -> WhiteNoiseModel:
        return WhiteNoiseModel(sigma=self.sigma)
