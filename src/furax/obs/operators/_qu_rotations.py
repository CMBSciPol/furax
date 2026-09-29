"""QU rotation operators and helper functions.

Angle conventions
-----------------
``R(theta)`` expresses how the polarisation state (Stokes vector) of an incident beam of
light transforms under a rotation of angle ``theta`` about the x axis. In other words,
if a polariser or a wave plate is rotated by an angle ``theta`` from the x axis, the
Mueller matrix ``M(theta)`` for the rotated component is

    M(theta) = R(-theta) @ M @ R(theta)

which encodes successively (from right to left):

- the rotation of the input Stokes vector into the local frame of the component;
- the effect of the "bare" component (in its local frame);
- the rotation back to the original frame.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Self, TypeVar

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float

from furax import AbstractLinearOperator, orthogonal
from furax.core import AbstractLazyInverseOrthogonalOperator
from furax.core.rules import AbstractCompositionRule, NoReduction

from ..stokes import Stokes, ValidStokesLiteral


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spin2Rotation:
    r"""The rotation $R(a)$ of [`QURotationOperator`][], stored as $(\cos 2a, \sin 2a)$.

    It expresses $(Q, U)$ in a basis rotated by $a$, $P = Q + iU \to P e^{-2ia}$:

    $$Q' = Q \cos 2a + U \sin 2a, \quad U' = -Q \sin 2a + U \cos 2a.$$

    Storing the doubled angle lets rotations compose and invert without trigonometric functions.
    The two arrays broadcast together, one rotation per element. Indexing a rotation indexes both,
    and it unpacks into the arguments of [`Stokes.rotate_qu`][]: `x.rotate_qu(*rotation)`.

    Attributes:
        cos_2angles: $\cos 2a$.
        sin_2angles: $\sin 2a$.
    """

    cos_2angles: Float[Array, '...']
    sin_2angles: Float[Array, '...']

    @classmethod
    def from_angles(cls, angles: Float[Array, '...']) -> Self:
        """The rotation by `angles`, in radians."""
        return cls(jnp.cos(2 * angles), jnp.sin(2 * angles))

    @classmethod
    def from_cos_sin(cls, cos: Float[Array, '...'], sin: Float[Array, '...']) -> Self:
        r"""The rotation by the angle $a$ of $(\cos a, \sin a)$, without evaluating $a$."""
        return cls(cos**2 - sin**2, 2 * cos * sin)

    def __iter__(self) -> Iterator[Float[Array, '...']]:
        return iter((self.cos_2angles, self.sin_2angles))

    def __getitem__(self, index: Any) -> Self:
        return type(self)(self.cos_2angles[index], self.sin_2angles[index])

    def compose(self, other: 'Spin2Rotation') -> Self:
        """The rotation by this angle then by that of `other`: the angles add."""
        c1, s1 = self
        c2, s2 = other
        return type(self)(c1 * c2 - s1 * s2, s1 * c2 + c1 * s2)

    def inverse(self) -> Self:
        """The rotation by the opposite angle, which is also the transpose."""
        return type(self)(self.cos_2angles, -self.sin_2angles)

    def broadcast_to(self, shape: tuple[int, ...]) -> Self:
        """The rotations broadcast to `shape`."""
        return type(self)(
            jnp.broadcast_to(self.cos_2angles, shape), jnp.broadcast_to(self.sin_2angles, shape)
        )


@jax.jit
def rotate_qu[StokesT: Stokes](x: StokesT, angles: Float[Array, '...']) -> StokesT:
    """Rotate QU Stokes parameters by the given angles (in radians).

    The transpose rotation is obtained by passing ``-angles``.
    """
    return x.rotate_qu(*Spin2Rotation.from_angles(angles))


@jax.jit
def rotate_qu_cs[StokesT: Stokes](
    x: StokesT,
    cos_angles: Float[Array, '...'],
    sin_angles: Float[Array, '...'],
) -> StokesT:
    """Rotate QU Stokes parameters given precomputed cos(a) and sin(a).

    The transpose rotation is obtained by negating ``sin_angles``.
    """
    return x.rotate_qu(*Spin2Rotation.from_cos_sin(cos_angles, sin_angles))


_StokesT = TypeVar('_StokesT', bound=Stokes)


@orthogonal
class QURotationOperator(AbstractLinearOperator):
    """Operator that rotates Q and U Stokes parameters by angle theta.

    Applies the rotation matrix R(theta) which transforms (Q, U) as:
        Q' = Q*cos(2*theta) + U*sin(2*theta)
        U' = -Q*sin(2*theta) + U*cos(2*theta)

    I and V components are unchanged. The operator is orthogonal:
    R.T = R.I = R(-theta).

    Consecutive rotations combine: R(a) @ R(b) = R(a+b).

    Attributes:
        angles: Rotation angles in radians.
        atomic: If True, prevents this operator from being merged with adjacent
            QU rotation operators during algebraic reduction.
    """

    angles: Float[Array, '...']
    atomic: bool = field(kw_only=True, default=False, metadata={'static': True})

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesLiteral = 'IQU',
        *,
        angles: Float[Array, '...'],
        atomic: bool = False,
    ) -> AbstractLinearOperator:
        structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        return cls(angles=angles, atomic=atomic, in_structure=structure)

    def mv(self, x: _StokesT) -> _StokesT:
        return rotate_qu(x, self.angles)

    def transpose(self) -> AbstractLinearOperator:
        return QURotationTransposeOperator(operator=self)


class QURotationTransposeOperator(AbstractLazyInverseOrthogonalOperator):
    operator: QURotationOperator

    def mv(self, x: _StokesT) -> _StokesT:
        return rotate_qu(x, -self.operator.angles)


class QURotationRule(AbstractCompositionRule):
    """Adds or subtracts QU rotation angles."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = (QURotationOperator, QURotationTransposeOperator)

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator) and not left.atomic:
            if isinstance(right, QURotationOperator) and not right.atomic:
                angles = left.angles + right.angles
            elif isinstance(right, QURotationTransposeOperator) and not right.operator.atomic:
                angles = left.angles - right.operator.angles
            else:
                raise NoReduction
        elif isinstance(left, QURotationTransposeOperator) and not left.operator.atomic:
            if isinstance(right, QURotationOperator) and not right.atomic:
                angles = right.angles - left.operator.angles
            elif isinstance(right, QURotationTransposeOperator) and not right.operator.atomic:
                angles = -left.operator.angles - right.operator.angles
            else:
                raise NoReduction
        else:
            raise NoReduction
        return [QURotationOperator(angles=angles, in_structure=right.in_structure)]
