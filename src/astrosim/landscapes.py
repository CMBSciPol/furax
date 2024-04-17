import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Literal, Union, cast, get_args

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

ValidStokesType = Literal['I', 'QU', 'IQU', 'IQUV']


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
class StokesPyTree:
    stokes: ClassVar[ValidStokesType] = 'I'

    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], getattr(self, self.stokes[0]).shape)

    @classmethod
    def shape_pytree(cls, shape: tuple[int, ...], dtype: DTypeLike) -> PyTree[jax.ShapeDtypeStruct]:
        stokes_arrays = len(cls.stokes) * [jax.ShapeDtypeStruct(shape, dtype)]
        return cls(*stokes_arrays)

    @property
    def dtype(self) -> DTypeLike:
        return cast(DTypeLike, getattr(self, self.stokes[0]).dtype)

    def __getitem__(self, index: Integer[Array, '...']) -> Self:
        arrays = [getattr(self, stoke)[index] for stoke in self.stokes]
        return type(self)(*arrays)

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: DTypeLike | float = float) -> Self:
        return cls.full(shape, 0, dtype)

    @classmethod
    def ones(cls, shape: tuple[int, ...], dtype: DTypeLike | float = float) -> Self:
        return cls.full(shape, 1, dtype)

    @classmethod
    def full(
        cls, shape: tuple[int, ...], fill_value: ScalarLike, dtype: DTypeLike | float = float
    ) -> Self:
        arrays = len(cls.stokes) * [jnp.full(shape, fill_value, dtype)]  # type: ignore[arg-type]
        return cls(*arrays)


@jdc.pytree_dataclass
class StokesIPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'I'
    I: Array

    @classmethod
    def from_iquv(cls, i, q, u, v) -> Self:
        return cls(i)


@jdc.pytree_dataclass
class StokesQUPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'QU'
    Q: Array
    U: Array

    @classmethod
    def from_iquv(cls, i, q, u, v) -> Self:
        return cls(q, u)


@jdc.pytree_dataclass
class StokesIQUPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'IQU'
    I: Array
    Q: Array
    U: Array

    @classmethod
    def from_iquv(cls, i, q, u, v) -> Self:
        return cls(i, q, u)


@jdc.pytree_dataclass
class StokesIQUVPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'IQUV'
    I: Array
    Q: Array
    U: Array
    V: Array

    @classmethod
    def from_iquv(cls, i, q, u, v) -> Self:
        return cls(i, q, u, v)


def stokes_pytree_cls(stokes: ValidStokesType) -> type[StokesPyTree]:
    if stokes not in get_args(ValidStokesType):
        raise ValueError(f'Invalid stokes parameters: {stokes!r}')
    return {
        'I': StokesIPyTree,
        'QU': StokesQUPyTree,
        'IQU': StokesIQUPyTree,
        'IQUV': StokesIQUVPyTree,
    }[stokes]


class Landscape(ABC):
    @abstractmethod
    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, '...']]: ...

    def zeros(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(0)

    def ones(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(1)


@jax.tree_util.register_pytree_node_class
@dataclass(unsafe_hash=True)
class HealpixLandscape(Landscape):
    nside: int
    stokes: ValidStokesType = 'IQU'
    dtype: DTypeLike = float

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'nside': self.nside,
            'stokes': self.stokes,
            'dtype': self.dtype,
        }  # static values
        return (), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:  # type: ignore[no-untyped-def]
        return cls(**aux_data)

    @property
    def npixel(self) -> int:
        return 12 * self.nside**2

    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = stokes_pytree_cls(self.stokes)
        return cls.full((self.npixel,), fill_value, self.dtype)

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
