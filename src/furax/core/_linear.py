from jax import Array
from jaxtyping import Bool, PyTree

from ._base import AbstractLinearOperator, TransposeOperator
from .rules import AbstractBinaryRule


class PackOperator(AbstractLinearOperator):
    """Operator that extracts elements using a boolean mask: y = x[mask].

    This operator satisfies: Pack @ Pack.T = Identity (orthogonal on its range).

    Attributes:
        mask: Boolean array selecting elements to extract.
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
