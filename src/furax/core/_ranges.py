from collections.abc import Sequence

import equinox
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Array, Bool, Int32, PyTree

from ._base import AbstractLinearOperator


@jdc.pytree_dataclass
class Ranges:
    """A class to efficiently store and manipulate a collection of integer ranges.

    Internally stores ranges as an int32 (n, 2) array of (start, stop) pairs.
    Intervals are half-open.
    """

    data: Int32[Array, 'n 2']
    """Array of shape (n, 2) storing (start, stop) pairs"""

    @classmethod
    def from_list(cls, ranges: Sequence[tuple[int, int]]) -> 'Ranges':
        """Create a Ranges object from a list of ranges.

        Args:
            ranges: List of (start, stop) tuples.
                   If empty, creates an empty range collection.

        Returns:
            A new Ranges object.
        """
        if len(ranges) == 0:
            # Initialize empty ranges
            return cls(jnp.zeros((0, 2), dtype=jnp.int32))
        else:
            # Convert input ranges to JAX array and normalize
            data = cls._normalize_ranges(jnp.array(ranges, dtype=jnp.int32))
            return cls(data)

    @classmethod
    def from_mask(cls, mask: Bool[Array, ' n']) -> 'Ranges':
        """Create a Ranges object from a boolean mask array.

        Args:
            mask: 1D boolean array where True indicates the presence of a point.

        Returns:
            Ranges object representing the contiguous True segments in the mask.
        """
        if mask.ndim != 1:
            raise ValueError('Mask must be a 1D array.')

        # Find the indices where the mask changes
        diff = jnp.diff(
            jnp.concatenate(
                [
                    jnp.array([0], dtype=mask.dtype),
                    mask.astype(jnp.int32),
                    jnp.array([0], dtype=mask.dtype),
                ]
            )
        )
        starts = jnp.where(diff == 1)[0]
        stops = jnp.where(diff == -1)[0]
        data = jnp.stack([starts, stops], axis=-1)
        return cls(data)  # no need to normalize

    @staticmethod
    def _normalize_ranges(ranges: Int32[Array, 'n 2'], min_gap: int = 0) -> Int32[Array, 'm 2']:
        """Convert ranges to a normalized form (sorted, merged overlapping)."""
        if ranges.size == 0:
            return ranges

        # Make sure start <= stop for each range (sort each row)
        ranges = ranges.sort(axis=-1)

        # Now sort each column independently
        # This breaks the pairs but is fine because it preserves gaps
        ranges = ranges.sort(axis=0)

        # Find the gaps between ranges larger than min_gap
        gaps = ranges[1:, 0] > ranges[:-1, 1] + min_gap
        if jnp.all(gaps):
            return ranges  # No overlaps, already normalized

        # Merge overlapping ranges
        true = jnp.array([True])
        starts = ranges[:, 0][jnp.concatenate((true, gaps))]
        stops = ranges[:, 1][jnp.concatenate((gaps, true))]
        return jnp.stack([starts, stops], axis=-1)

    def to_mask(self, length: int) -> Bool[Array, ' n']:
        # Handle empty ranges case
        if len(self) == 0:
            return jnp.zeros(length, dtype=bool)

        # Create indices array [0, 1, 2, ..., length-1] with shape (length, 1)
        # This will allow broadcasting below
        indices = jnp.arange(length).reshape(-1, 1)

        # Check which indices fall within each range
        # This broadcasts to shape (length, n)
        in_ranges = (indices >= self.data[:, 0]) & (indices < self.data[:, 1])

        # Reduce over ranges dimension (any index in any range)
        return jnp.any(in_ranges, axis=1)

    def __len__(self) -> int:
        """Return the number of distinct ranges."""
        return self.data.shape[0]

    def is_empty(self) -> bool:
        """Check if the ranges collection is empty."""
        return len(self) == 0

    def buffer(self, amount: int) -> 'Ranges':
        """Expand each range by a specified amount on both sides."""
        if amount < 0:
            raise ValueError('Buffer amount must be non-negative.')

        # Compute the expanded ranges
        buffered_data = self.data.at[:].add([-amount, amount])
        buffered_data = jnp.clip(buffered_data, a_min=0)  # Ensure no negative starts
        return Ranges(buffered_data)

    def close_gaps(self, gap: int) -> 'Ranges':
        """Close gaps smaller than or equal to a specified size."""
        if gap < 0:
            raise ValueError('Gap size must be non-negative.')

        if len(self) == 0:
            return self

        # Normalize ranges with the specified minimum gap
        closed_data = self._normalize_ranges(self.data, min_gap=gap)
        return Ranges(closed_data)


class MaskOperator(AbstractLinearOperator):
    """Operator that sets to zero specified ranges of values in 2-d vectors."""

    ranges: list[Ranges]
    """List of Ranges objects, one for each detector to mask in the input vector"""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    """The static input structure of the operator"""

    def __init__(
        self,
        ranges: list[Ranges] | Ranges,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        if isinstance(ranges, Ranges):
            ranges = [ranges]
        self.ranges = ranges
        self._in_structure = in_structure

        # Check consistency
        if not len(ranges) == in_structure.shape[0]:
            raise ValueError

    def mv(self, x: Array) -> Array:
        mask = jnp.stack([range.to_mask(x.shape[-1]) for range in self.ranges])
        return jnp.where(mask, 0, x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


if __name__ == '__main__':
    # Create a test 2D array
    x = jnp.arange(20).astype(jnp.float32).reshape(2, 10)
    struct = jax.ShapeDtypeStruct(x.shape, x.dtype)
    print('Original 2D array:')
    print(x)

    # Create multiple Ranges objects, one for each column
    ranges1 = Ranges.from_list([(2, 4), (7, 8)])
    ranges2 = Ranges.from_list([(1, 3), (6, 9)])
    print('Ranges to mask:')
    print(ranges1)
    print(ranges2)

    # Create and apply the mask operator
    mask_op = MaskOperator([ranges1, ranges2], in_structure=struct)
    result = mask_op.mv(x)

    print('After masking:')
    print(result)

    # Test with JIT
    # Have to use a lambda to avoid "TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'"
    mv = lambda x: mask_op.mv(x)
    jit_result = jax.jit(mv)(x)

    print('After JIT masking:')
    print(jit_result)
    print('Are results identical?', jnp.allclose(result, jit_result))
