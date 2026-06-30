from typing import Self

import jax
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, MaskOperator, symmetric


@symmetric
class WeightOperator(AbstractLinearOperator):
    """Masked noise weight `M W M` as a single operator."""

    weight: AbstractLinearOperator  # symmetric
    mask: MaskOperator

    @classmethod
    def create(cls, weight: AbstractLinearOperator, mask: MaskOperator) -> Self:
        return cls(weight, mask, in_structure=mask.in_structure)

    def with_mask(self, mask: MaskOperator) -> 'WeightOperator':
        """Rebuild the weight around a new mask."""
        return WeightOperator(self.weight, mask)

    def mv(self, x: PyTree[Inexact[jax.Array, '...']]) -> PyTree[Inexact[jax.Array, '...']]:
        W, M = self.weight, self.mask
        if W.is_diagonal:
            return W(M(x))
        return M(W(M(x)))
