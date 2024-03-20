import jax
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


class BolometerOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    """Class that integrates the input Stokes parameters."""

    shape: tuple[int, ...]
    info: Info

    def __init__(self, shape: tuple[int, ...], stokes: ValidStokesType):
        self.shape = shape
        self.info = Info(stokes=stokes, dtype=float)

    def mv(self, x: StokesPyTree) -> Float[Array, ' {self.shape}']:
        if self.info.stokes != x.stokes:
            raise TypeError('Invalid input')
        if isinstance(x, StokesIPyTree):
            return x.I
        if isinstance(x, StokesQUPyTree):
            return x.Q + x.U
        if isinstance(x, StokesIQUPyTree):
            return x.I + x.Q + x.U
        if isinstance(x, StokesIQUVPyTree):
            return x.I + x.Q + x.U + x.V
        raise NotImplementedError(f'HWPOperator not implemented for Stokes {self.info.stokes!r}')

    def transpose(self) -> lx.AbstractLinearOperator:
        return BolometerOperatorT(self)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.info.stokes)
        return cls.shape_pytree(self.shape, self.info.dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.info.dtype)


class BolometerOperatorT(lx.AbstractLinearOperator):  # type: ignore[misc]
    operator: BolometerOperator

    def __init__(self, operator: BolometerOperator):
        self.operator = operator

    def mv(self, x: Float[Array, ' {self.shape}']) -> StokesPyTree:
        stokes = self.operator.info.stokes
        cls = stokes_pytree_cls(stokes)
        arrays = len(stokes) * [x]
        return cls(*arrays)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


@lx.is_symmetric.register(BolometerOperator)
@lx.is_symmetric.register(BolometerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_positive_semidefinite.register(BolometerOperator)
@lx.is_positive_semidefinite.register(BolometerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_negative_semidefinite.register(BolometerOperator)
@lx.is_negative_semidefinite.register(BolometerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.linearise.register(BolometerOperator)
@lx.linearise.register(BolometerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


@lx.conj.register(BolometerOperator)
@lx.conj.register(BolometerOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return operator
