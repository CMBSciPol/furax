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


class HWPOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    shape: tuple[int, ...]
    info: Info
    cos_2phi: Float[Array, '...']
    sin_2phi: Float[Array, '...']

    def __init__(self, shape: tuple[int, ...], stokes: ValidStokesType, pa: Float[Array, '...']):
        self.shape = shape
        self.info = Info(stokes=stokes, dtype=float)
        self.cos_2phi = jnp.cos(2 * pa)
        self.sin_2phi = jnp.sin(2 * pa)

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if self.info.stokes != x.stokes:
            raise TypeError('Invalid input')
        if isinstance(x, StokesIPyTree):
            return x
        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(x.Q * self.cos_2phi, x.U * self.sin_2phi)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, x.Q * self.cos_2phi, x.U * self.sin_2phi)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, x.Q * self.cos_2phi, x.U * self.sin_2phi, x.V * 0)
        raise NotImplementedError(f'HWPOperator not implemented for Stokes {self.info.stokes!r}')

    def transpose(self) -> lx.AbstractLinearOperator:
        return self

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.info.stokes)
        return cls.shape_pytree(self.shape, self.info.dtype)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure()


@lx.is_symmetric.register(HWPOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return True


@lx.is_positive_semidefinite.register(HWPOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return True


@lx.is_negative_semidefinite.register(HWPOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.linearise.register(HWPOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


@lx.conj.register(HWPOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator
