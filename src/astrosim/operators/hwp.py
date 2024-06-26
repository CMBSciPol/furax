import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree

from astrosim.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesPyTreeType,
    StokesQUPyTree,
    ValidStokesType,
)
from astrosim.operators import AbstractLinearOperator, diagonal, symmetric


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate."""

    shape: tuple[int, ...]
    dtype: DTypeLike = eqx.field(static=True)
    stokes: ValidStokesType = eqx.field(static=True)

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
            return StokesQUPyTree(x.Q, -x.U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, x.Q, -x.U)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, x.Q, -x.U, -x.V)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)


@symmetric
class RotatingHWPOperator(AbstractLinearOperator):
    """Operator for an ideal Half-wave plate."""

    shape: tuple[int, ...]
    dtype: DTypeLike = eqx.field(static=True)
    stokes: ValidStokesType = eqx.field(static=True)
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
        Q = x.Q * self.cos_4angles - x.U * self.sin_4angles
        U = -(x.Q * self.sin_4angles + x.U * self.cos_4angles)

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            V = -x.V
            return StokesIQUVPyTree(x.I, Q, U, V)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)
