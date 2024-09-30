import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Bool, PyTree

from .core import AbstractLazyTransposeOperator, AbstractLinearOperator
from .rules import AbstractBinaryRule, NoReduction


class PackOperator(AbstractLinearOperator):
    """Class for packing the leaves of a PyTree according to a common mask.

    The operation is conceptually the same as:
        y = x[mask]
    """

    mask: Bool[Array, '...'] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, mask: Bool[Array, '...'], in_structure: PyTree[jax.ShapeDtypeStruct]):
        self.mask = mask
        self._in_structure = in_structure

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        return x[self.mask]

    def transpose(self) -> AbstractLinearOperator:
        return PackTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class PackTransposeOperator(AbstractLazyTransposeOperator):
    """Class for unpacking the leaves of a PyTree according to a common mask.

    The operation is conceptually the same as:
        y = jnp.zeros(out_structure)
        y[mask] = x
    """

    operator: PackOperator

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        y = jax.tree.map(
            lambda leaf: jnp.zeros(self.operator.mask.shape, leaf.dtype)
            .at[self.operator.mask]
            .set(leaf),
            x,
        )
        return y


class PackUnpackRule(AbstractBinaryRule):
    """Binary rule for `pack @ pack.T = I`."""

    left_operator_class = PackOperator
    right_operator_class = PackTransposeOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        assert isinstance(right, PackTransposeOperator)  # mypy assert
        if left is not right.operator:
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return []
