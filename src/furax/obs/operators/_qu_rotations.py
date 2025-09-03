import equinox
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, orthogonal
from furax.core import AbstractLazyInverseOrthogonalOperator
from furax.core.rules import AbstractBinaryRule, NoReduction

from ..stokes import (
    Stokes,
    StokesI,
    StokesIQU,
    StokesIQUV,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)


@orthogonal
class QURotationOperator(AbstractLinearOperator):
    """Operator for QU rotations.

    The angles in the constructor are in radians.
    """

    angles: Float[Array, '...']
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        angles: Float[Array, '...'],
    ) -> AbstractLinearOperator:
        structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        return cls(angles, structure)

    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        if isinstance(x, StokesI):
            return x

        cos_2angles = jnp.cos(2 * self.angles)
        sin_2angles = jnp.sin(2 * self.angles)
        q = x.q * cos_2angles - x.u * sin_2angles
        u = x.q * sin_2angles + x.u * cos_2angles

        if isinstance(x, StokesQU):
            return StokesQU(q, u)
        if isinstance(x, StokesIQU):
            return StokesIQU(x.i, q, u)
        if isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, q, u, x.v)
        raise NotImplementedError

    def transpose(self) -> AbstractLinearOperator:
        return QURotationTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class QURotationTransposeOperator(AbstractLazyInverseOrthogonalOperator):
    operator: QURotationOperator

    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        if isinstance(x, StokesI):
            return x

        cos_2angles = jnp.cos(2 * self.operator.angles)
        sin_2angles = jnp.sin(2 * self.operator.angles)
        q = x.q * cos_2angles + x.u * sin_2angles
        u = -x.q * sin_2angles + x.u * cos_2angles

        if isinstance(x, StokesQU):
            return StokesQU(q, u)
        if isinstance(x, StokesIQU):
            return StokesIQU(x.i, q, u)
        if isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, q, u, x.v)
        raise NotImplementedError


class QURotationRule(AbstractBinaryRule):
    """Adds or subtracts QU rotation angles."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = (QURotationOperator, QURotationTransposeOperator)

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator):
            if isinstance(right, QURotationOperator):
                angles = left.angles + right.angles
            elif isinstance(right, QURotationTransposeOperator):
                angles = left.angles - right.operator.angles
            else:
                raise NoReduction
        else:
            assert isinstance(left, QURotationTransposeOperator)  # mypy assert
            if isinstance(right, QURotationOperator):
                angles = right.angles - left.operator.angles
            elif isinstance(right, QURotationTransposeOperator):
                angles = -left.operator.angles - right.operator.angles
            else:
                raise NoReduction
        return [QURotationOperator(angles, right.in_structure())]
