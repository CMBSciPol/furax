import equinox
import jax
from jax import Array
from jaxtyping import Bool, PyTree

from .core import AbstractLinearOperator, TransposeOperator
from .rules import AbstractBinaryRule


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

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class PackUnpackRule(AbstractBinaryRule):
    """Binary rule for `pack @ pack.T = I`."""

    left_operator_class = PackOperator
    right_operator_class = TransposeOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return []
