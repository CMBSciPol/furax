import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float

from furax import AbstractLinearOperator, diagonal
from furax.core.rules import AbstractCompositionRule

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
from ._transfer_matrix import Stack
from ._transfer_matrix import mueller_matrix as _mueller_matrix


@diagonal
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal half-wave plate (HWP).

    A HWP flips the sign of the U (and V) Stokes parameters while leaving
    I and Q unchanged. This models the Mueller matrix diag(1, 1, -1, -1).

    The operator is diagonal and symmetric. For a rotating HWP at angle theta,
    use ``HWPOperator.create(..., angles=theta)`` which computes R(-theta) @ HWP @ R(theta).

    Algebraic rule: R(theta) @ HWP = HWP @ R(-theta).
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
        stack: A :class:`~furax.obs.operators.Stack` instance.
        frequency: Frequency in Hz.
        angle_incidence: Angle of incidence in radians.

    Returns:
        3×3 JAX array — the IQU block of the transmitted Mueller matrix.

    To build a per-detector batched Mueller matrix (shape ``(n_det, 3, 3)``),
    vmap over angles::

        import jax
        muellers = jax.vmap(
            lambda a: hwp_mueller_from_stack(stack, freq, a)
        )(focal_plane_angles)   # focal_plane_angles shape (n_det,)
    """
    return _mueller_matrix(stack, frequency, angle_incidence)[:-1, :-1]


def _apply_mueller_iqu(
    M: Array,
    i: Array,
    q: Array,
    u: Array,
) -> tuple[Array, Array, Array]:
    """Apply a 3×3 Mueller matrix to IQU Stokes components.

    ``M`` may have batch dimensions that are a leading prefix of ``i``'s
    dimensions.  For example, ``M`` shape ``(n_det, 3, 3)`` and ``i`` shape
    ``(n_det, n_samples)`` is handled correctly via matmul broadcasting.
    """
    # stk: (*batch, *extra, 3)  —  extra is typically (n_samples,)
    stk = jnp.stack([i, q, u], axis=-1)
    # Insert (1,)*extra_ndim between batch dims and the (3,3) matrix dims so
    # that @ broadcasts M over the extra (sample) axes.
    extra_ndim = stk.ndim - M.ndim + 1
    M_exp = M.reshape(M.shape[:-2] + (1,) * extra_ndim + (3, 3))
    out = (M_exp @ stk[..., None]).squeeze(-1)  # (*batch, *extra, 3)
    return out[..., 0], out[..., 1], out[..., 2]


class NonIdealHWPOperator(AbstractLinearOperator):
    """HWP operator with full 3×3 IQU Mueller matrix, for non-normal incidence.

    Unlike the ideal HWP (which simply flips U sign), this operator applies a
    pre-computed frequency- and angle-dependent Mueller matrix. All IQU
    components can mix. The Mueller matrix is computed externally (e.g. using
    ``hwp_mueller_from_stack`` with a transfer matrix stack) and passed to
    ``create()``.

    The ``mueller`` field may be a single ``(3, 3)`` matrix shared across all
    detectors, or a batched ``(*batch, 3, 3)`` array where the batch dimensions
    form a leading prefix of the Stokes leaf shape.  The typical batched case
    is ``(n_det, 3, 3)`` paired with Stokes leaves of shape ``(n_det, n_samples)``.

    For a rotating HWP at angle theta, pass ``angles=theta`` which wraps the
    operator as R(-theta) @ HWP @ R(theta).

    Note:
        For StokesIQUV input, V is passed through unchanged (not modelled).
    """

    mueller: Array  # (*batch, 3, 3) — single or per-detector

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
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)
        op = cls(mueller=mueller, in_structure=in_structure)
        if angles is None:
            return op
        rot = QURotationOperator(angles=angles, in_structure=in_structure)
        return rot.T @ op @ rot

    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        M = self.mueller
        if isinstance(x, StokesI):
            i_out, _, _ = _apply_mueller_iqu(M, x.i, jnp.zeros_like(x.i), jnp.zeros_like(x.i))
            return StokesI(i_out)
        if isinstance(x, StokesQU):
            _, q_out, u_out = _apply_mueller_iqu(M, jnp.zeros_like(x.q), x.q, x.u)
            return StokesQU(q_out, u_out)
        if isinstance(x, StokesIQU):
            i_out, q_out, u_out = _apply_mueller_iqu(M, x.i, x.q, x.u)
            return StokesIQU(i_out, q_out, u_out)
        if isinstance(x, StokesIQUV):
            i_out, q_out, u_out = _apply_mueller_iqu(M, x.i, x.q, x.u)
            return StokesIQUV(i_out, q_out, u_out, x.v)
        raise NotImplementedError

    def transpose(self) -> 'NonIdealHWPOperator':
        return NonIdealHWPOperator(
            mueller=jnp.swapaxes(self.mueller, -2, -1),
            in_structure=self.in_structure,
        )


class QURotationHWPRule(AbstractCompositionRule):
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
