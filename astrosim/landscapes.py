from dataclasses import dataclass
from functools import partial
from typing import Union

import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import numpy.typing as npt
from jaxtyping import Array, Float

from .samplings import Sampling


@dataclass(frozen=True)
class HealpixLandscape:
    nside: int
    stokes: str = 'I'

    @property
    def npixel(self) -> int:
        return 12 * self.nside**2

    def full(self, fill_value: float, dtype=None):
        return {_: jnp.full(self.npixel, fill_value, dtype) for _ in self.stokes}

    def zeros(self, dtype=None):
        return self.full(0, dtype or float)

    def ones(self, dtype=None):
        return self.full(1, dtype or float)

    def get_coverage(self, arg: Union[npt.NDArray[np.int64], Sampling]) -> npt.NDArray[np.int64]:
        if isinstance(arg, Sampling):
            pixels = self.ang2pix(arg.theta, arg.phi)
        else:
            pixels = arg

        indices, counts = jnp.unique(pixels, return_counts=True)
        coverage = jnp.zeros(self.npixel, dtype=np.int64)
        coverage = coverage.at[indices].add(counts, indices_are_sorted=True, unique_indices=True)
        return hp.ud_grade(np.asarray(coverage), self.nside)

    @partial(jax.jit, static_argnums=0)
    def ang2pix(self, theta: Float[Array, '...'], phi: Float[Array, '...']):
        r"""Convert angles to HEALPix index for HEALPix ring ordering scheme.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.
        """
        return jhp.ang2pix(self.nside, theta, phi)
