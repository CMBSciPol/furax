from dataclasses import dataclass
from functools import partial
from typing import Any, Union

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array, Float, Integer, PyTree, ScalarLike, Shaped

from .samplings import Sampling

# XXX Remove after https://github.com/google/jax/pull/19669 is accepted
NumberType = Union[jnp.float32, jnp.int32]  # to be completed with all jax scalar number types
ScalarType = Union[jnp.bool_, NumberType]
DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[Union[bool, int, float, complex, np.bool_, np.number[Any], ScalarType]],
    np.dtype[Any],
]


@dataclass(frozen=True)
class HealpixLandscape:
    nside: int
    stokes: str = 'I'

    @property
    def npixel(self) -> int:
        return 12 * self.nside**2

    def full(
        self, fill_value: ScalarLike, dtype: DTypeLike | None = None
    ) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        return {_: jnp.full(self.npixel, fill_value, dtype) for _ in self.stokes}

    def zeros(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        return self.full(0, dtype or float)

    def ones(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        return self.full(1, dtype or float)

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' {self.npixel}']:
        pixels = self.ang2pix(arg.theta, arg.phi)
        indices, counts = jnp.unique(pixels, return_counts=True)
        coverage = jnp.zeros(self.npixel, dtype=np.int64)
        coverage = coverage.at[indices].add(counts, indices_are_sorted=True, unique_indices=True)
        return coverage

    @partial(jax.jit, static_argnums=0)
    def ang2pix(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> Integer[Array, ' *dims']:
        r"""Convert angles to HEALPix index for HEALPix ring ordering scheme.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.
        """
        return jhp.ang2pix(self.nside, theta, phi)  # type: ignore[no-any-return]
