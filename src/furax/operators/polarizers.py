import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from furax.landscapes import (
    DTypeLike,
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesPyTreeType,
    StokesQUPyTree,
    ValidStokesType,
)
from furax.operators import AbstractLazyTransposeOperator, AbstractLinearOperator


class LinearPolarizerOperator(AbstractLinearOperator):
    """Class that integrates the input Stokes parameters assuming a linear polarizer."""

    shape: tuple[int, ...]
    dtype: DTypeLike = equinox.field(static=True)
    stokes: ValidStokesType = equinox.field(static=True)
    theta: Float[Array, '...']

    def __init__(
        self,
        shape: tuple[int, ...],
        stokes: ValidStokesType,
        dtype: DTypeLike = float,
        theta: Float[Array, '...'] | float = 0.0,
    ):
        self.shape = shape
        self.stokes = stokes
        self.dtype = np.dtype(dtype)
        self.theta = jnp.asarray(theta, dtype=dtype)  # detector's polarizer angle

    def mv(self, x: StokesPyTreeType) -> Float[Array, ' {self.shape}']:

        if self.stokes != x.stokes:
            raise TypeError('Invalid input')
        if isinstance(x, StokesIPyTree):
            return 0.5 * x.i
        # broadcast on the samples. Is it efficient in Jax ?
        q = (x.q.T * jnp.cos(2 * self.theta)).T
        u = (x.u.T * jnp.sin(2 * self.theta)).T
        if isinstance(x, StokesQUPyTree):
            return 0.5 * (q + u)
        if isinstance(x, StokesIQUPyTree) or isinstance(x, StokesIQUVPyTree):
            return 0.5 * (x.i + q + u)
        raise NotImplementedError(f'HWPOperator not implemented for Stokes {self.stokes!r}')

    def transpose(self) -> AbstractLinearOperator:
        return LinearPolarizerTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.shape, self.dtype, self.stokes)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)


class LinearPolarizerTransposeOperator(AbstractLazyTransposeOperator):
    operator: LinearPolarizerOperator

    def mv(self, x: Float[Array, ' {self.shape}']) -> StokesPyTreeType:
        stokes = self.operator.stokes
        i = 0.5 * x
        if stokes == 'I':
            return StokesIPyTree(i)
        q = (i.T * jnp.cos(2 * self.operator.theta)).T
        u = (i.T * jnp.sin(2 * self.operator.theta)).T
        if stokes == 'QU':
            return StokesQUPyTree(q, u)
        if stokes == 'IQU':
            return StokesIQUPyTree(i, q, u)
        v = jnp.zeros_like(i)
        if stokes == 'IQUV':
            return StokesIQUVPyTree(i, q, u, v)
        raise NotImplementedError
