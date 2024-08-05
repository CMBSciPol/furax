import functools as ft
from collections.abc import Sequence
from typing import cast

import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import PyTree

from .core import AbstractLinearOperator
from .rules import AbstractBinaryRule, NoReduction

__all__ = [
    'MoveAxisOperator',
]


class MoveAxisOperator(AbstractLinearOperator):
    """Operator to move axes of pytree leaves to new positions

    The operation is conceptually the same as:
        y = jnp.moveaxis(x, source, destination)
    """

    source: tuple[int]
    destination: tuple[int]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        source: int | Sequence[int],
        destination: int | Sequence[int],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        if isinstance(source, int):
            source = (source,)
        elif not isinstance(source, tuple):
            source = cast(tuple[int], tuple(source))
        if isinstance(destination, int):
            destination = (destination,)
        elif not isinstance(destination, tuple):
            destination = cast(tuple[int], tuple(destination))
        self.source = source
        self.destination = destination
        self._in_structure = in_structure

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array, '...']:
        return jax.tree.map(lambda leaf: jnp.moveaxis(leaf, self.source, self.destination), x)

    def transpose(self) -> AbstractLinearOperator:
        return MoveAxisOperator(self.destination, self.source, self.out_structure())

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        moveaxis = ft.partial(jnp.moveaxis, source=self.source, destination=self.destination)
        return jax.tree.map(
            lambda leaf: jax.eval_shape(moveaxis, leaf),
            self.in_structure(),
        )


class MoveAxisInverseRule(AbstractBinaryRule):
    """We cannot decorate MoveAxisOperator with :orthogonal: because it is not square."""

    operator_class = MoveAxisOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if not isinstance(left, MoveAxisOperator) or not isinstance(right, MoveAxisOperator):
            raise NoReduction
        if left.source != right.destination or left.destination != right.source:
            raise NoReduction
        return []


# Note: if an algebraic rule to compose MoveAxisOperators is to be implemented, it may be best
# to implement a class TransposeOperator wrapping jnp.transpose and transform MoveAxisOperator
# instances into TransposeOperator instances. That way, it would be easier to include reductions for
# new operators, such as SwapAxesOperator, etc.
