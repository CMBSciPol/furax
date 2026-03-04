import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float

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


def _rotate_qu(
    x: StokesPyTreeType,
    cos_2angles: Float[Array, '...'],
    sin_2angles: Float[Array, '...'],
) -> StokesPyTreeType:
    """Apply the QU rotation given precomputed cos(2a) and sin(2a)."""
    if isinstance(x, StokesI):
        return x
    q = x.q * cos_2angles + x.u * sin_2angles
    u = -x.q * sin_2angles + x.u * cos_2angles
    if isinstance(x, StokesQU):
        return StokesQU(q, u)
    if isinstance(x, StokesIQU):
        return StokesIQU(x.i, q, u)
    if isinstance(x, StokesIQUV):
        return StokesIQUV(x.i, q, u, x.v)
    raise NotImplementedError


@jax.jit
def rotate_qu(x: StokesPyTreeType, angles: Float[Array, '...']) -> StokesPyTreeType:
    """Rotate QU Stokes parameters by the given angles (in radians).

    The transpose rotation is obtained by passing ``-angles``.
    """
    return _rotate_qu(x, jnp.cos(2 * angles), jnp.sin(2 * angles))


@jax.jit
def rotate_qu_cs(
    x: StokesPyTreeType,
    cos_angles: Float[Array, '...'],
    sin_angles: Float[Array, '...'],
) -> StokesPyTreeType:
    """Rotate QU Stokes parameters given precomputed cos(a) and sin(a).

    The transpose rotation is obtained by negating ``sin_angles``.
    """
    # double angle formulas
    cos_2angles = cos_angles**2 - sin_angles**2
    sin_2angles = 2 * cos_angles * sin_angles
    return _rotate_qu(x, cos_2angles, sin_2angles)


@orthogonal
class QURotationOperator(AbstractLinearOperator):
    """Operator for QU rotations.

    The angles in the constructor are in radians.
    """

    angles: Float[Array, '...']

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
        return cls(angles=angles, in_structure=structure)

    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        return rotate_qu(x, self.angles)  # type: ignore[no-any-return]

    def transpose(self) -> AbstractLinearOperator:
        return QURotationTransposeOperator(operator=self)


class QURotationTransposeOperator(AbstractLazyInverseOrthogonalOperator):
    operator: QURotationOperator

    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        return rotate_qu(x, -self.operator.angles)  # type: ignore[no-any-return]


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
        return [QURotationOperator(angles=angles, in_structure=right.in_structure)]
