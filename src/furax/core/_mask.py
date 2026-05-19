import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Inexact, PyTree, UInt8

from ._base import AbstractLinearOperator, symmetric
from .rules import AbstractBinaryRule


@symmetric
class MaskOperator(AbstractLinearOperator):
    """Operator that zeros out values according to a boolean mask: M(x) = x * mask.

    The mask is symmetric and idempotent (M @ M = M). A True value means the
    data point is valid and preserved; False values are set to zero.

    The mask is bit-packed internally to save memory. Use ``from_boolean_mask()``
    to create from a standard boolean array.

    Attributes:
        mask: Bit-packed mask as a pytree matching ``in_structure``.
    """

    mask: PyTree[UInt8[Array, '...']]
    """Bit-packed sample mask as a pytree matching ``in_structure``."""

    def __post_init__(self) -> None:
        if not all(leaf.dtype == jnp.uint8 for leaf in jax.tree.leaves(self.mask)):
            msg = (
                'Expected boolean masks of unsigned byte data type. '
                'You might be looking for the `MaskOperator.from_boolean_mask()` factory.'
            )
            raise ValueError(msg)

    @classmethod
    def from_boolean_mask(
        cls,
        boolean_mask: PyTree[Bool[Array, '...']],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> 'MaskOperator':
        """Create a MaskOperator from a boolean mask.

        Args:
            boolean_mask: Either a single boolean array (broadcast to all leaves of
                ``in_structure``) or a pytree of boolean arrays matching the structure
                of ``in_structure``.
            in_structure: The pytree structure of the operator's input.
        """
        mask_treedef = jax.tree.structure(boolean_mask)
        struct_treedef = jax.tree.structure(in_structure)

        if mask_treedef == struct_treedef:  # type: ignore[operator]
            # Pytree of masks matching in_structure: pack each leaf
            packed_mask = jax.tree.map(_check_and_pack, boolean_mask, in_structure)
        else:
            # Single mask broadcast to all leaves
            packed_mask = jax.tree.map(
                lambda struct_leaf: _check_and_pack(boolean_mask, struct_leaf),
                in_structure,
            )

        return cls(mask=packed_mask, in_structure=in_structure)

    def to_boolean_mask(self) -> PyTree[Bool[Array, '...']]:
        """Return the unpacked boolean mask as a pytree matching ``in_structure``."""
        return jax.tree.map(
            lambda m, s: jnp.unpackbits(m, axis=-1, count=s.shape[-1]),
            self.mask,
            self.in_structure,
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        boolean_mask = self.to_boolean_mask()
        return jax.tree.map(lambda m, leaf: jnp.where(m, leaf, 0), boolean_mask, x)


def _check_and_pack(
    boolean_mask: Bool[Array, '...'], struct_leaf: jax.ShapeDtypeStruct
) -> UInt8[Array, '...']:
    """Check shape compatibility and pack a boolean mask."""
    try:
        _ = jnp.broadcast_shapes(boolean_mask.shape, struct_leaf.shape)
    except ValueError:
        msg = 'Boolean mask shape must be broadcastable to leaf shape'
        raise ValueError(msg)
    return jnp.packbits(boolean_mask, axis=-1)


class InverseBinaryRule(AbstractBinaryRule):
    """Binary rule for composition of MaskOperator's."""

    left_operator_class = MaskOperator
    right_operator_class = MaskOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        # type checking assert
        assert isinstance(left, MaskOperator) and isinstance(right, MaskOperator)

        # Apply bit-wise AND to combine masks (leaf-wise for pytree masks)
        mask = jax.tree.map(lambda l, r: l & r, left.mask, right.mask)

        # Left and right operators have the same structure, just take one
        return [MaskOperator(mask, in_structure=left.in_structure)]
