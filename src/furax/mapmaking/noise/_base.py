"""The interface of the noise models, and the convention of their realizations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Key, PyTree

from furax import AbstractLinearOperator
from furax.core._lowrank import rfft_folds

if TYPE_CHECKING:
    from .white import WhiteNoiseModel


@jax.tree_util.register_dataclass
@dataclass
class NoiseModel(ABC):
    r"""The noise of a TOD: its covariance $N$, as a power spectral density per channel.

    The PSD is one-sided, in units² per Hz, as a Welch estimate gives it: a white noise of
    standard deviation $\sigma$ per sample, at the sample rate $f_s$, has the PSD
    $2 \sigma^2 / f_s$. Every model holds the sample rate of the TOD it describes, the same for
    all its detectors.

    A model may correlate the noise in time, across channels, or both:

    | channels    | white in time     | coloured in time                         |
    | ----------- | ----------------- | ---------------------------------------- |
    | independent | `WhiteNoiseModel` | `OneOverFNoiseModel`, `BinnedNoiseModel` |
    | coupled     |                   | `LowRankNoiseModel`                      |

    [`covariance`][], [`inverse_covariance`][] and [`draw`][] keep every correlation of the
    model. [`white`][] drops those in time, [`decoupled`][] those across channels.
    """

    @abstractmethod
    def psd(self, f: Float[Array, ' a']) -> Float[Array, '... a']:
        """The PSD of each channel at the frequencies `f`, in Hz."""

    def to_array(self) -> Float[Array, '*dets n']:
        """The parameters of each detector, stacked on the last axis (see [`detector_parameters`][])."""
        return jnp.stack(list(self.detector_parameters().values()), axis=-1)

    @abstractmethod
    def covariance(self, in_structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        """The covariance $N$ on TODs of structure `in_structure`, with all its correlations."""

    @abstractmethod
    def inverse_covariance(
        self, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        """The inverse covariance $N^{-1}$ on TODs of structure `in_structure`.

        It keeps all the correlations: it is the inverse of [`covariance`][], up to the
        approximations of the model.
        """

    @abstractmethod
    def draw(
        self, key: Key[Array, ''], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> PyTree[Array]:
        r"""A stationary realization of the noise, of PSD `psd`, periodic over the TOD.

        The rfft coefficient of each frequency $f$ of a TOD of $n$ samples has the variance
        $n f_s \, \mathrm{PSD}(f) / 2$, shared by its real and imaginary parts but at the zero and
        Nyquist frequencies, so the end of the TOD is correlated with its start.

        Args:
            key: The random key.
            in_structure: The structure of the TOD.
        """

    @abstractmethod
    def white(self) -> 'WhiteNoiseModel':
        """The model without its correlations in time, nor across channels.

        Each model says which white level stands for it; its sample rate is that of the model.
        """

    @abstractmethod
    def decoupled(self) -> 'NoiseModel':
        """The model without its correlations across channels, still coloured in time."""

    def detector_parameters(self) -> dict[str, Array]:
        """The parameters of each detector, by name, as the results record them.

        By default, those of the [`white`][] model.
        """
        return self.white().detector_parameters()


def _stationary_draw(
    key: Key[Array, ''], psd: Float[Array, '... freq'], n: int, sample_rate: Array | float
) -> Array:
    """A periodic realization of `n` samples of each channel, of PSD `psd` at the rfft frequencies."""
    return jnp.fft.irfft(_stationary_coefficients(key, _per_sample(psd, sample_rate), n), n=n)


def _stationary_coefficients(
    key: Key[Array, ''], variance: Float[Array, '... freq'], n: int
) -> Complex[Array, '... freq']:
    """The rfft coefficients of a draw of `n` samples, of `variance` per sample in each frequency."""
    k_real, k_imag = jax.random.split(key)
    coefficients = jax.lax.complex(
        jax.random.normal(k_real, variance.shape, variance.dtype),
        jax.random.normal(k_imag, variance.shape, variance.dtype),
    )
    return coefficients * _amplitude_scale(n, variance.shape[-1]) * jnp.sqrt(variance)


def _per_sample(psd: Array, sample_rate: Array | float) -> Array:
    """The one-sided PSD `psd`, in units² per Hz, as a variance per sample in each frequency."""
    return psd * sample_rate / 2


def _amplitude_scale(n: int, n_frequencies: int) -> Array:
    r"""$\sqrt{n / w_f}$ over the first frequencies of `n` samples."""
    return jnp.asarray(np.sqrt(n / rfft_folds(n, n_frequencies)))
