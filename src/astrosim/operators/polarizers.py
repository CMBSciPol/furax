import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float, PyTree

from astrosim.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesQUPyTree,
    ValidStokesType,
    stokes_pytree_cls,
)
from astrosim.operators import AbstractLinearOperator


class LinearPolarizerOperator(AbstractLinearOperator):  # type: ignore[misc]
    """Class that integrates the input Stokes parameters assuming a linear polarizer."""

    shape: tuple[int, ...]
    dtype: DTypeLike = eqx.field(static=True)
    stokes: str = eqx.field(static=True)
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
        return LinearPolarizerOperatorT(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.stokes)
        return cls.shape_pytree(self.shape, self.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype)


class LinearPolarizerOperatorT(AbstractLinearOperator):  # type: ignore[misc]
    operator: LinearPolarizerOperator

    def __init__(self, operator: LinearPolarizerOperator):
        self.operator = operator

    def mv(self, x: Float[Array, ' {self.shape}']) -> StokesPyTree:
        stokes = self.operator.stokes
        cls = stokes_pytree_cls(stokes)
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

    def transpose(self) -> AbstractLinearOperator:
        return self.operator

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()
