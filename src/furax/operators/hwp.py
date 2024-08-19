import equinox
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree

from furax.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesPyTreeType,
    StokesQUPyTree,
    ValidStokesType,
)
from furax.operators import AbstractLinearOperator, diagonal, symmetric


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate."""

    shape: tuple[int, ...]
    dtype: DTypeLike = equinox.field(static=True)
    stokes: ValidStokesType = equinox.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        stokes: ValidStokesType,
        dtype: DTypeLike = float,
    ):
        self.shape = shape
        self.stokes = stokes
        self.dtype = np.dtype(dtype)

    @classmethod
    def create(cls, shape: tuple[int, ...], stokes: ValidStokesType):
        return cls(shape, stokes)

    def mv(self, x: StokesPyTreeType) -> StokesPyTree:
        if self.stokes != x.stokes:
            raise TypeError('Invalid input')

        if isinstance(x, StokesIPyTree):
            return x
        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(x.q, -x.u)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.i, x.q, -x.u)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.i, x.q, -x.u, -x.v)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)


@symmetric
class RotatingHWPOperator(AbstractLinearOperator):
    """Operator for an ideal Half-wave plate."""

    shape: tuple[int, ...]
    dtype: DTypeLike = equinox.field(static=True)
    stokes: ValidStokesType = equinox.field(static=True)
    cos_4angles: Float[Array, '...']
    sin_4angles: Float[Array, '...']

    def __init__(
        self,
        shape: tuple[int, ...],
        stokes: ValidStokesType,
        cos_angles: Float[Array, '...'],
        sin_angles: Float[Array, '...'],
        dtype: DTypeLike = float,
    ):
        self.shape = shape
        self.stokes = stokes
        self.dtype = np.dtype(dtype)
        self.cos_4angles = jnp.asarray(cos_angles, dtype=dtype)
        self.sin_4angles = jnp.asarray(sin_angles, dtype=dtype)

    @classmethod
    def create(cls, shape: tuple[int, ...], stokes: ValidStokesType, angles: Float[Array, '...']):
        cos_4angles = jnp.cos(4 * angles)
        sin_4angles = jnp.sin(4 * angles)
        return cls(shape, stokes, cos_4angles, sin_4angles)

    def mv(self, x: StokesPyTreeType) -> StokesPyTree:
        if self.stokes != x.stokes:
            raise TypeError('Invalid input')

        # we should try using cls(x).from_iquv
        if isinstance(x, StokesIPyTree):
            return x
        Q = x.q * self.cos_4angles - x.u * self.sin_4angles
        U = -(x.q * self.sin_4angles + x.u * self.cos_4angles)

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.i, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            V = -x.v
            return StokesIQUVPyTree(x.i, Q, U, V)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)
