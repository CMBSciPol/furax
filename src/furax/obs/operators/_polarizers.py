import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, Float

from furax import AbstractLinearOperator
from furax.core.rules import AbstractBinaryRule

from ..stokes import (
    Stokes,
    StokesI,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)
from ._hwp import HWPOperator
from ._qu_rotations import QURotationOperator


class LinearPolarizerOperator(AbstractLinearOperator):
    """Class that integrates the input Stokes parameters assuming a linear polarizer."""

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        polarizer = cls(in_structure=in_structure)
        if angles is None:
            return polarizer
        rot = QURotationOperator(angles=angles, in_structure=in_structure)
        rotated_polarizer: AbstractLinearOperator = polarizer @ rot
        return rotated_polarizer

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        if isinstance(x, StokesI):
            return 0.5 * x.i
        if isinstance(x, StokesQU):
            return 0.5 * x.q
        return 0.5 * (x.i + x.q)


class LinearPolarizerHWPRule(AbstractBinaryRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)`."""

    left_operator_class = LinearPolarizerOperator
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return [left]
