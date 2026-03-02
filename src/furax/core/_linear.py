from jax import Array
from jaxtyping import Bool, PyTree

from ._base import AbstractLinearOperator, TransposeOperator
from .rules import AbstractBinaryRule


class PackOperator(AbstractLinearOperator):
    """Class for packing the leaves of a PyTree according to a common mask.

    The operation is conceptually the same as:
        y = x[mask]
    """

    mask: Bool[Array, '...']

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        return x[self.mask]


class PackUnpackRule(AbstractBinaryRule):
    """Binary rule for `pack @ pack.T = I`."""

    left_operator_class = PackOperator
    right_operator_class = TransposeOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return []
