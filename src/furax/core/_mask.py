import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, PyTree, Shaped, UInt8

from ._base import AbstractLinearOperator, symmetric
from .rules import AbstractBinaryRule


@symmetric
class MaskOperator(AbstractLinearOperator):
    """Operator that sets values to zero according to a boolean mask.

    The mask is bit-packed to save memory and unpacked in chunks when applying to a vector.

    A True value in the boolean mask means that the data point is valid.
    """

    mask: UInt8[Array, '...']
    """Sample mask, bit-packed along the last axis to save memory"""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    """The static input structure of the operator"""

    _chunk_size: int = equinox.field(static=True, default=8192)
    """Chunk size for unpacking (default: 8KB of mask bits)"""

    def __init__(
        self,
        packed_mask: UInt8[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        chunk_size: int = 8192,
    ) -> None:
        if packed_mask.dtype != jnp.uint8:
            msg = (
                'Expected an input array of unsigned byte data type.'
                'You might be looking for the `MaskOperator.from_boolean_mask()` factory.'
            )
            raise ValueError(msg)
        self.mask = packed_mask
        self._in_structure = in_structure
        self._chunk_size = chunk_size

    @classmethod
    def from_boolean_mask(
        cls,
        boolean_mask: Bool[Array, '...'],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        chunk_size: int = 8192,
    ) -> 'MaskOperator':
        # Check shape compatibility
        try:
            _ = jnp.broadcast_shapes(boolean_mask.shape, in_structure.shape)
        except ValueError:
            msg = 'Boolean mask shape must be broadcastable to operator shape'
            raise ValueError(msg)

        # Pack boolean mask along samples axis
        packed_mask = jnp.packbits(boolean_mask, axis=-1)
        return cls(packed_mask, in_structure=in_structure, chunk_size=chunk_size)

    def to_boolean_mask(self) -> Bool[Array, '...']:
        """Full unpacking - only use when necessary"""
        return jnp.unpackbits(self.mask, axis=-1, count=self._in_structure.shape[-1])

    def mv(self, x: Shaped[Array, '*dims']) -> Shaped[Array, '*dims']:
        """Matrix-vector multiplication with chunked mask unpacking"""
        n = self._in_structure.shape[-1]
        chunk_size = self._chunk_size

        # For small arrays, just unpack everything (overhead not worth it)
        if n <= chunk_size:
            boolean_mask = self.to_boolean_mask()
            return jnp.where(boolean_mask, x, 0)

        # Pad input to multiple of chunk_size to avoid dynamic shapes
        n_chunks = (n + chunk_size - 1) // chunk_size
        padded_n = n_chunks * chunk_size
        pad_amount = padded_n - n

        # Pad x with zeros on the right along last axis
        if pad_amount > 0:
            pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad_amount)]
            x_padded = jnp.pad(x, pad_width, mode='constant')
            # Also pad the mask - need to ensure we have enough bytes
            bytes_needed = (padded_n + 7) // 8
            bytes_current = self.mask.shape[-1]
            if bytes_needed > bytes_current:
                mask_pad_width = [(0, 0)] * (self.mask.ndim - 1) + [
                    (0, bytes_needed - bytes_current)
                ]
                mask_padded = jnp.pad(self.mask, mask_pad_width, mode='constant', constant_values=0)
            else:
                mask_padded = self.mask
        else:
            x_padded = x
            mask_padded = self.mask

        # Allocate output array (padded size)
        result = jnp.zeros_like(x_padded)

        # Calculate bytes needed per chunk (static)
        bytes_per_chunk = (chunk_size + 7) // 8

        def process_chunk(i, result):  # type: ignore[no-untyped-def]
            start = i * chunk_size
            byte_start = start // 8

            # Slice packed mask with static size
            mask_slice = jax.lax.dynamic_slice_in_dim(
                mask_padded, byte_start, bytes_per_chunk, axis=-1
            )

            # Slice input data with static size
            x_slice = jax.lax.dynamic_slice_in_dim(x_padded, start, chunk_size, axis=-1)

            # Unpack mask
            mask_unpacked = jnp.unpackbits(mask_slice, axis=-1, count=chunk_size)

            # Apply mask
            result_chunk = jnp.where(mask_unpacked, x_slice, 0)

            # Update result array with this chunk
            return jax.lax.dynamic_update_slice_in_dim(result, result_chunk, start, axis=-1)

        # Process all chunks in a loop
        result_padded = jax.lax.fori_loop(0, n_chunks, process_chunk, result)

        # Slice off padding to return original size
        return result_padded[..., :n]  # type: ignore[no-any-return]

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class InverseBinaryRule(AbstractBinaryRule):
    """Binary rule for composition of MaskOperator's."""

    left_operator_class = MaskOperator
    right_operator_class = MaskOperator

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
