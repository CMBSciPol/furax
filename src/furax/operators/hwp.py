import sys

from furax._base.rules import AbstractBinaryRule
from furax.operators.qu_rotations import QURotationOperator, QURotationTransposeOperator

if sys.version_info < (3, 11):
    pass
else:
    pass

import equinox
import jax
import numpy as np
from jax import Array
from jaxtyping import Float, PyTree

from furax.landscapes import (
    DTypeLike,
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesPyTreeType,
    StokesQUPyTree,
    ValidStokesType,
)
from furax.operators import AbstractLinearOperator, diagonal


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal static Half-wave plate."""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = StokesPyTree.structure_for(shape, dtype, stokes)
        hwp = cls(in_structure)
        if angles is None:
            return hwp
        rot = QURotationOperator(angles, in_structure)
        rotated_hwp: AbstractLinearOperator = rot.T @ hwp @ rot
        return rotated_hwp

    def mv(self, x: StokesPyTreeType) -> StokesPyTree:
        if isinstance(x, StokesIPyTree):
            return x
        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(x.q, -x.u)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.i, x.q, -x.u)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.i, x.q, -x.u, -x.v)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class QURotationHWPRule(AbstractBinaryRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)`."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator):
            return [right, QURotationTransposeOperator(left)]
        assert isinstance(left, QURotationTransposeOperator)
        return [right, left.operator]
