import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Inexact, PyTree

from astrosim.landscapes import (
    Info,
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesQUPyTree,
    ValidStokesType,
    stokes_pytree_cls,
)


class LinearPolarizerOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    """Class that integrates the input Stokes parameters assuming a linear polarizer."""

    shape: tuple[int, ...]
    info: Info
    theta: Float[Array, '...']

    def __init__(
        self,
        shape: tuple[int, ...],
        stokes: ValidStokesType,
        theta: Float[Array, '...'] | float = 0.0,
    ):
        self.shape = shape
        self.info = Info(stokes=stokes, dtype=float)
        self.theta = jnp.asarray(theta)  # detector's polarizer angle

    def mv(self, x: StokesPyTree) -> Float[Array, ' {self.shape}']:
        if self.info.stokes != x.stokes:
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
        raise NotImplementedError(f'HWPOperator not implemented for Stokes {self.info.stokes!r}')

    def transpose(self) -> lx.AbstractLinearOperator:
        return LinearPolarizerOperatorT(self)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.info.stokes)
        return cls.shape_pytree(self.shape, self.info.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.info.dtype)


class LinearPolarizerOperatorT(lx.AbstractLinearOperator):  # type: ignore[misc]
    operator: LinearPolarizerOperator

    def __init__(self, operator: LinearPolarizerOperator):
        self.operator = operator

    def mv(self, x: Float[Array, ' {self.shape}']) -> StokesPyTree:
        stokes = self.operator.info.stokes
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

    def transpose(self) -> lx.AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


@lx.is_symmetric.register(LinearPolarizerOperator)
@lx.is_symmetric.register(LinearPolarizerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_positive_semidefinite.register(LinearPolarizerOperator)
@lx.is_positive_semidefinite.register(LinearPolarizerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_negative_semidefinite.register(LinearPolarizerOperator)
@lx.is_negative_semidefinite.register(LinearPolarizerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.linearise.register(LinearPolarizerOperator)
@lx.linearise.register(LinearPolarizerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


@lx.conj.register(LinearPolarizerOperator)
@lx.conj.register(LinearPolarizerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return operator
