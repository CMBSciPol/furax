import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Shaped, UInt8

from ._base import AbstractLinearOperator, symmetric
from .rules import AbstractBinaryRule


@symmetric
class MaskOperator(AbstractLinearOperator):
    """Operator that sets values to zero according to a boolean mask.

    The mask is bit-packed to save memory and unpacked as needed when applying to a vector.

    A True value in the boolean mask means that the data point is valid.
    """

    mask: UInt8[Array, '...']
    """Sample mask, bit-packed along the last axis to save memory"""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    """The static input structure of the operator"""

    def __init__(
        self,
        packed_mask: UInt8[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        self._in_structure = in_structure
        self.mask = packed_mask

    @classmethod
    def from_boolean_mask(
        cls,
        boolean_mask: Bool[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> 'MaskOperator':
        # Check shape compatibility
        try:
            _ = jnp.broadcast_shapes(boolean_mask.shape, in_structure.shape)
        except ValueError:
            msg = 'Boolean mask shape must be broadcastable to operator shape'
            raise ValueError(msg)

        # Pack boolean mask along samples axis
        packed_mask = jnp.packbits(boolean_mask, axis=-1)
        return cls(packed_mask, in_structure=in_structure)

    def mv(self, x: Shaped[Array, '*dims']) -> Shaped[Array, '*dims']:
        # This will be a uint8 array but would be the same size with booleans
        boolean_mask = jnp.unpackbits(self.mask, axis=-1, count=x.shape[-1])
        # 1 = good, 0 = bad
        return jnp.where(boolean_mask, x, 0)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class InverseBinaryRule(AbstractBinaryRule):
    """Binary rule for composition of MaskOperator's."""

    operator_class = MaskOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        # type checking assert
        assert isinstance(left, MaskOperator) and isinstance(right, MaskOperator)

        # Apply bit-wise AND to combine masks
        # Since both are broadcastable to the same operator structure shape, they are broadcastable
        mask = left.mask & right.mask

        # Left and right operators have the same structure, just take one
        return [MaskOperator(mask, in_structure=left.in_structure())]
