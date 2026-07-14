import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float

from furax import AbstractLinearOperator, diagonal
from furax.core.rules import AbstractCompositionRule

from ..stokes import Stokes, StokesType, ValidStokesType
from ._qu_rotations import QURotationOperator, QURotationTransposeOperator
from ._transfer_matrix import Stack, mueller_matrix


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal half-wave plate (HWP).

    A HWP flips the sign of the U (and V) Stokes parameters while leaving
    I and Q unchanged. This models the Mueller matrix `diag(1, 1, -1, -1)`.

    The operator is diagonal and symmetric.

    Algebraic rule: `R(theta) @ HWP = HWP @ R(-theta)`.

    Examples:
        A fixed HWP:

        >>> import jax.numpy as jnp
        >>> from furax.obs.operators import HWPOperator
        >>> hwp = HWPOperator.create(shape=(5,), stokes='IQU')

        A rotating HWP at angle theta, computed as R(-theta) @ HWP @ R(theta):

        >>> theta = jnp.linspace(0, jnp.pi, 5)
        >>> hwp = HWPOperator.create(shape=(5,), stokes='IQU', angles=theta)
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
        """Build an HWP operator.

        Args:
            shape: Shape of each Stokes component the operator acts on.
            dtype: Floating-point dtype of the Stokes data.
            stokes: Stokes components the operator acts on.
            angles: HWP rotation angles in radians, broadcastable to ``shape``.
                If None, the HWP is fixed (no rotation).

        Returns:
            A fixed HWP, or ``R(-angles) @ HWP @ R(angles)`` if `angles` is given.
        """
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        hwp = cls(in_structure=in_structure)
        if angles is None:
            return hwp
        rot = QURotationOperator(angles=angles, in_structure=in_structure)
        rotated_hwp: AbstractLinearOperator = rot.T @ hwp @ rot
        return rotated_hwp

    def mv(self, x: StokesType) -> Stokes:
        # Canonical component order is I, Q, U, V: once 'U' is present, it and
        # every component after it (i.e. V) flip sign; I and Q are untouched.
        idx = x.stokes.find('U')
        if idx < 0:
            return x
        return x.from_array(x.data.at[idx:].multiply(-1.0))


def hwp_mueller_from_stack(
    stack: Stack,
    frequency: Array,
    angle_incidence: Array,
) -> Float[Array, '3 3']:
    """Compute the 3×3 IQU Mueller matrix for a physical HWP stack.

    Uses the built-in transfer matrix method to compute the transmitted
    Mueller matrix for a multilayer HWP stack at non-normal incidence.
    JIT-friendly: ``frequency`` and ``angle_incidence`` are JAX-traced.

    Args:
        stack: A [`Stack`][furax.obs.operators.Stack] instance.
        frequency: Frequency in Hz.
        angle_incidence: Angle of incidence in radians.

    Returns:
        3×3 JAX array — the IQU block of the transmitted Mueller matrix.

    Examples:
        Build a per-detector batched Mueller matrix (shape ``(n_det, 3, 3)``) by
        vmapping over angles:

        >>> import jax
        >>> muellers = jax.vmap(
        ...     lambda a: hwp_mueller_from_stack(stack, freq, a)
        ... )(focal_plane_angles)  # focal_plane_angles shape (n_det,)
    """
    return mueller_matrix(stack, frequency, angle_incidence)[:-1, :-1]


class NonIdealHWPOperator(AbstractLinearOperator):
    """HWP operator with full 3×3 IQU Mueller matrix, for non-normal incidence.

    Unlike the ideal HWP (which simply flips U sign), this operator applies a
    pre-computed frequency- and angle-dependent Mueller matrix. All IQU
    components can mix.

    Attributes:
        mueller: The 3×3 IQU Mueller matrix, or a batched ``(*batch, 3, 3)`` array
            where the batch dimensions form a leading prefix of each Stokes
            component's shape. The typical batched case is ``(n_det, 3, 3)``
            paired with Stokes components of shape ``(n_det, n_samples)``.

    Examples:
        Compute the Mueller matrix from a physical HWP stack, then build a
        fixed operator:

        >>> import jax.numpy as jnp
        >>> from furax.obs.operators import (
        ...     SO_MF_HWP_STACK, NonIdealHWPOperator, hwp_mueller_from_stack
        ... )
        >>> M = hwp_mueller_from_stack(SO_MF_HWP_STACK, frequency=150e9, angle_incidence=0.0)
        >>> hwp = NonIdealHWPOperator.create(shape=(5,), stokes='IQU', mueller=M)

        A rotating HWP at angle theta, computed as R(-theta) @ HWP @ R(theta):

        >>> theta = jnp.linspace(0, jnp.pi, 5)
        >>> hwp = NonIdealHWPOperator.create(shape=(5,), stokes='IQU', mueller=M, angles=theta)

    Note:
        For StokesIQUV input, V is passed through unchanged (not modelled).
    """

    mueller: Array

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        mueller: Array,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        """Build a non-ideal HWP operator.

        Args:
            shape: Shape of each Stokes component the operator acts on.
            dtype: Floating-point dtype of the Stokes data.
            stokes: Stokes components the operator acts on.
            mueller: The 3×3 IQU Mueller matrix, or a batched ``(*batch, 3, 3)``
                array; see the class docstring.
            angles: HWP rotation angles in radians, broadcastable to ``shape``.
                If None, the HWP is fixed (no rotation).

        Returns:
            A fixed HWP, or ``R(-angles) @ HWP @ R(angles)`` if `angles` is given.
        """
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        op = cls(mueller=mueller, in_structure=in_structure)
        if angles is None:
            return op
        rot = QURotationOperator(angles=angles, in_structure=in_structure)
        return rot.T @ op @ rot

    _MUELLER_SLICE = {
        'I': slice(0, 1),
        'QU': slice(1, 3),
        'IQU': slice(0, 3),
        'IQUV': slice(0, 3),
    }

    def mv(self, x: StokesType) -> StokesType:
        # Slice the Mueller matrix to keep only relevant components
        sl = self._MUELLER_SLICE[x.stokes]
        data = x.data[:3]  # drops V if x has it; no-op otherwise
        M = self.mueller[..., sl, sl]

        # M may have batch dims that are a leading prefix of data's trailing dims,
        # e.g. M shape (n_det, k, k) with data shape (k, n_det, n_samples).
        stk = jnp.moveaxis(data, 0, -1)  # (*batch, *extra, k)
        extra_ndim = stk.ndim - M.ndim + 1
        M_exp = M.reshape(M.shape[:-2] + (1,) * extra_ndim + M.shape[-2:])
        out = (M_exp @ stk[..., None]).squeeze(-1)  # (*batch, *extra, k)
        out = jnp.moveaxis(out, -1, 0)  # (k, *batch, *extra)

        if x.stokes == 'IQUV':
            return x.from_array(jnp.concatenate([out, x.data[3:]]))  # append V
        return x.from_array(out)

    def transpose(self) -> 'NonIdealHWPOperator':
        return NonIdealHWPOperator(
            mueller=jnp.swapaxes(self.mueller, -2, -1),
            in_structure=self.in_structure,
        )


class QURotationHWPRule(AbstractCompositionRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)."""

    left_operator_class = (QURotationOperator, QURotationTransposeOperator)
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, QURotationOperator):
            return [right, QURotationTransposeOperator(operator=left)]
        assert isinstance(left, QURotationTransposeOperator)
        return [right, left.operator]
