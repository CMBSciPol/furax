import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Float

from furax import AbstractLinearOperator, diagonal
from furax.core.rules import AbstractBinaryRule

from ..stokes import (
    Stokes,
    StokesI,
    StokesIQU,
    StokesIQUV,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)
from ._qu_rotations import QURotationOperator, QURotationTransposeOperator


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate."""

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
        hwp = cls(in_structure=in_structure)
        if angles is None:
            return hwp
        rot = QURotationOperator(angles=angles, in_structure=in_structure)
        rotated_hwp: AbstractLinearOperator = rot.T @ hwp @ rot
        return rotated_hwp

    def mv(self, x: StokesPyTreeType) -> Stokes:
        if isinstance(x, StokesI):
            return x
        if isinstance(x, StokesQU):
            return StokesQU(x.q, -x.u)
        if isinstance(x, StokesIQU):
            return StokesIQU(x.i, x.q, -x.u)
        if isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, x.q, -x.u, -x.v)
        raise NotImplementedError


class QURotationHWPRule(AbstractBinaryRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)`."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator):
            return [right, QURotationTransposeOperator(operator=left)]
        assert isinstance(left, QURotationTransposeOperator)
        return [right, left.operator]
