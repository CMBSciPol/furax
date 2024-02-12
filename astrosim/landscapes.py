import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array, Float, Integer, PyTree, ScalarLike, Shaped

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from .samplings import Sampling

# XXX Remove after https://github.com/google/jax/pull/19669 is accepted
NumberType = Union[
    jnp.float32, jnp.int32, jnp.int16
]  # to be completed with all jax scalar number types
ScalarType = Union[jnp.bool_, NumberType]
DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[Union[bool, int, float, complex, ScalarType, np.bool_, np.number]],  # type: ignore[type-arg]  # noqa: E501
    np.dtype,  # type: ignore[type-arg]
]


@jdc.pytree_dataclass
class HealpixPyTree:
    stokes: ClassVar[str] = 'I'
    I: Array

    @property
    def nside(self) -> int:
        return jhp.npix2nside(self.npixel)  # type: ignore[no-any-return]

    @property
    def npixel(self) -> int:
        return self.I.size

    @property
    def size(self) -> int:
        return len(self.stokes) * self.npixel


@jdc.pytree_dataclass
class HealpixIQUPyTree(HealpixPyTree):
    stokes: ClassVar[str] = 'IQU'
    Q: Array
    U: Array


class Landscape(ABC):
    @abstractmethod
    def full(
        self, fill_value: ScalarLike, dtype: DTypeLike | None = None
    ) -> PyTree[Shaped[Array, '...']]: ...
    @abstractmethod
    def zeros(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, '...']]: ...
    @abstractmethod
    def ones(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, '...']]: ...


@dataclass(unsafe_hash=True)
class HealpixLandscape(Landscape):
    nside: int
    stokes: str = 'IQU'

    def __post_init__(self) -> None:
        self.pytree_cls = {
            'I': HealpixPyTree,
            'IQU': HealpixIQUPyTree,
        }[self.stokes]

    def _tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {'nside': self.nside, 'stokes': self.stokes}  # static values
        return (), aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children) -> Self:  # type: ignore[no-untyped-def]
        return cls(**aux_data)

    @property
    def npixel(self) -> int:
        return 12 * self.nside**2

    def zeros(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, ' 12*nside**2']]:
        return self.full(0, dtype or float)

    def ones(self, dtype: DTypeLike | None = None) -> PyTree[Shaped[Array, ' 12*nside**2']]:
        return self.full(1, dtype or float)

    def full(
        self, fill_value: ScalarLike, dtype: DTypeLike | None = None
    ) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        return self.pytree_cls(**{_: jnp.full(self.npixel, fill_value, dtype) for _ in self.stokes})

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' 12*nside**2']:
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


jax.tree_util.register_pytree_node(
    HealpixLandscape, HealpixLandscape._tree_flatten, HealpixLandscape._tree_unflatten
)
