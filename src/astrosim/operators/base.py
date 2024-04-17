from typing import TypeVar

import jax
import lineax as lx
from jaxtyping import Array, Inexact, PyTree


class AbstractLinearOperator(lx.AbstractLinearOperator):
    def __init_subclass__(cls, **keywords) -> None:
        _monkey_patch_operator(cls)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def transpose(self) -> lx.AbstractLinearOperator:
        raise NotImplementedError

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        raise NotImplementedError


def _monkey_patch_operator(cls: type[lx.AbstractLinearOperator]) -> None:
    for tag in [
        lx.is_diagonal,
        lx.is_lower_triangular,
        lx.is_upper_triangular,
        lx.is_tridiagonal,
        lx.is_symmetric,
        lx.is_positive_semidefinite,
        lx.is_negative_semidefinite,
    ]:
        if _already_registered(cls, tag):
            continue
        tag.register(cls)(lambda _: False)

    lx.linearise.register(cls)(lambda _: _)
    lx.conj.register(cls)(lambda _: _)

    cls.__call__ = _base_class__call__


def _already_registered(cls: type[lx.AbstractLinearOperator], tag) -> bool:
    return any(
        registered_cls is not object and issubclass(cls, registered_cls)
        for registered_cls in tag.registry
    )


def _base_class__call__(self, x: PyTree[jax.ShapeDtypeStruct]) -> PyTree[jax.ShapeDtypeStruct]:
    if isinstance(x, lx.AbstractLinearOperator):
        raise ValueError("Use '@' to compose operators")
    return self.mv(x)


_monkey_patch_operator(lx.ComposedLinearOperator)


T = TypeVar('T')


def diagonal(cls: type[T]) -> type[T]:
    lx.is_diagonal.register(cls)(lambda _: True)
    symmetric(cls)
    return cls


def lower_triangular(cls: type[T]) -> type[T]:
    lx.is_lower_triangular.register(cls)(lambda _: True)
    return cls


def upper_triangular(cls: type[T]) -> type[T]:
    lx.is_upper_triangular.register(cls)(lambda _: True)
    return cls


def symmetric(cls: type[T]) -> type[T]:
    lx.is_symmetric.register(cls)(lambda _: True)
    cls.out_structure = cls.in_structure
    cls.transpose = lambda self: self
    return cls


def positive_semidefinite(cls: type[T]) -> type[T]:
    lx.is_positive_semidefinite.register(cls)(lambda _: True)
    return cls


def negative_semidefinite(cls: type[T]) -> type[T]:
    lx.is_negative_semidefinite.register(cls)(lambda _: True)
    return cls
