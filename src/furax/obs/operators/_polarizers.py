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
    """Operator for an ideal linear polarizer.

    Extracts the intensity seen by a linear polarizer aligned with the x-axis:
    d = 0.5 * (I + Q). For a polarizer at angle psi, use
    ``LinearPolarizerOperator.create(..., angles=psi)``.

    This implements: d = 0.5 * (I + Q*cos(2*psi) + U*sin(2*psi)).

    Algebraic rule: LinearPolarizer @ HWP = LinearPolarizer.
    """

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
    """Binary rule for LinPol @ HWP = LinPol`."""

    left_operator_class = LinearPolarizerOperator
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return [left]
