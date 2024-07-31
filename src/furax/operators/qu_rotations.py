import equinox
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree

from furax._base.rules import AbstractBinaryRule, NoReduction
from furax.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesQUPyTree,
    ValidStokesType,
)
from furax.operators import (
    AbstractLazyInverseOrthogonalOperator,
    AbstractLinearOperator,
    orthogonal,
)


@orthogonal
class QURotationOperator(AbstractLinearOperator):
    """Operator for QU rotations.

    The angles in the constructor are in radians.
    """

    angles: Float[Array, '...']
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        angles: Float[Array, '...'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.angles = angles
        self._in_structure = in_structure

    @classmethod
    def create(
        cls,
        angles: Float[Array, '...'],
        stokes: ValidStokesType,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
    ) -> AbstractLinearOperator:
        structure = StokesPyTree.structure_for(stokes, shape, dtype)
        return cls(angles, structure)

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if isinstance(x, StokesIPyTree):
            return x

        cos_2angles = jnp.cos(2 * self.angles)
        sin_2angles = jnp.sin(2 * self.angles)
        Q = x.Q * cos_2angles - x.U * sin_2angles
        U = x.Q * sin_2angles + x.U * cos_2angles

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, Q, U, x.V)
        raise NotImplementedError

    def transpose(self) -> AbstractLinearOperator:
        return QURotationTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class QURotationTransposeOperator(AbstractLazyInverseOrthogonalOperator):
    operator: QURotationOperator

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if isinstance(x, StokesIPyTree):
            return x

        cos_2angles = jnp.cos(2 * self.operator.angles)
        sin_2angles = jnp.sin(2 * self.operator.angles)
        Q = x.Q * cos_2angles + x.U * sin_2angles
        U = -x.Q * sin_2angles + x.U * cos_2angles

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            return StokesIQUVPyTree(x.I, Q, U, x.V)
        raise NotImplementedError


class QURotationRule(AbstractBinaryRule):
    """Adds or subtracts QU rotation angles."""

    operator_class = QURotationOperator

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
        elif isinstance(left, QURotationTransposeOperator):
            if isinstance(right, QURotationOperator):
                angles = right.angles - left.operator.angles
            elif isinstance(right, QURotationTransposeOperator):
                angles = -left.operator.angles - right.operator.angles
            else:
                raise NoReduction
        else:
            raise NoReduction
        return [QURotationOperator(angles, right.in_structure())]
