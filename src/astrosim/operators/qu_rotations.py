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
    StokesQUPyTree,
    ValidStokesType,
    stokes_pytree_cls,
)
from astrosim.operators import AbstractLinearOperator, positive_semidefinite


@positive_semidefinite
class QURotationOperator(AbstractLinearOperator):
    shape: tuple[int, ...]
    dtype: DTypeLike = eqx.field(static=True)
    stokes: str = eqx.field(static=True)
    cos_2angles: Float[Array, '...']
    sin_2angles: Float[Array, '...']

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
        self.cos_2angles = jnp.asarray(cos_angles, dtype=dtype)
        self.sin_2angles = jnp.asarray(sin_angles, dtype=dtype)

    @classmethod
    def create(cls, shape: tuple[int, ...], stokes: ValidStokesType, angles: Float[Array, '...']):
        cos_2angles = jnp.cos(2 * angles)
        sin_2angles = jnp.sin(2 * angles)
        return cls(shape, stokes, cos_2angles, sin_2angles)

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if self.stokes != x.stokes:
            raise TypeError('Invalid input')

        # we should try using cls(x).from_iquv
        if isinstance(x, StokesIPyTree):
            return x
        Q = x.Q * self.cos_2angles - x.U * self.sin_2angles
        U = x.Q * self.sin_2angles + x.U * self.cos_2angles

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, Q, U, x.V)
        raise NotImplementedError

    def transpose(self) -> AbstractLinearOperator:
        return QURotationOperatorT(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.stokes)
        return cls.shape_pytree(self.shape, self.dtype)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure()


@positive_semidefinite
class QURotationOperatorT(AbstractLinearOperator):
    operator: QURotationOperator

    def __init__(self, operator: QURotationOperator):
        self.operator = operator

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if self.operator.stokes != x.stokes:
            raise TypeError('Invalid input')

        # we should try using cls(x).from_iquv
        if isinstance(x, StokesIPyTree):
            return x
        Q = x.Q * self.operator.cos_2angles + x.U * self.operator.sin_2angles
        U = -x.Q * self.operator.sin_2angles + x.U * self.operator.cos_2angles

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, Q, U, x.V)
        raise NotImplementedError

    def transpose(self) -> AbstractLinearOperator:
        return self.operator

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()