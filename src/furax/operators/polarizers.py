import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PyTree

from furax.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
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

    def mv(self, x: StokesPyTree) -> Float[Array, ' {self.shape}']:
        if self.stokes != x.stokes:
            raise TypeError('Invalid input')
        if isinstance(x, StokesIPyTree):
            return 0.5 * x.I
        # broadcast on the samples. Is it efficient in Jax ?
        Q = (x.Q.T * jnp.cos(2 * self.theta)).T
        U = (x.U.T * jnp.sin(2 * self.theta)).T
        if isinstance(x, StokesQUPyTree):
            return 0.5 * (Q + U)
        if isinstance(x, StokesIQUPyTree) or isinstance(x, StokesIQUVPyTree):
            return 0.5 * (x.I + Q + U)
        raise NotImplementedError(f'HWPOperator not implemented for Stokes {self.stokes!r}')

    def transpose(self) -> AbstractLinearOperator:
        return LinearPolarizerTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)


class LinearPolarizerTransposeOperator(AbstractLazyTransposeOperator):

    def mv(self, x: Float[Array, ' {self.shape}']) -> StokesPyTree:
        stokes = self.operator.stokes
        cls = StokesPyTree.class_for(stokes)
        I = 0.5 * x
        if stokes == 'I':
            return cls(I)
        Q = (I.T * jnp.cos(2 * self.operator.theta)).T
        U = (I.T * jnp.sin(2 * self.operator.theta)).T
        if stokes == 'QU':
            return cls(Q, U)
        if stokes == 'IQU':
            return cls(I, Q, U)
        V = jnp.zeros_like(I)
        if stokes == 'IQUV':
            return cls(I, Q, U, V)
        raise NotImplementedError
