from types import EllipsisType

import equinox
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Inexact, Integer, PyTree

from ._base import AbstractLinearOperator, IdentityOperator, TransposeOperator
from ._diagonal import DiagonalOperator
from .rules import AbstractBinaryRule, NoReduction

__all__ = ['IndexOperator']


class IndexOperator(AbstractLinearOperator):
    """Class for indexing operations on pytrees.

    The operation is conceptually the same as y = x[indices]

    Usage:
    To extract the second element of the first axis:

        >>> op = IndexOperator(0, in_structure=jax.ShapeDtypeStruct((10, 4), jax.numpy.float32))

    To extract values from the penultimate axis given an array of indices:

        >>> indices = jax.numpy.array([2, 4, 4, 5, 7])
        >>> in_structure = jax.ShapeDtypeStruct((9, 8, 3), jax.numpy.float32)
        >>> op = IndexOperator((..., indices, slice(None)), in_structure=in_structure)

    In order to extract values using a boolean mask, it is required to specify an output structure:

        >>> indices = jax.numpy.array([True, False, True, False])
        >>> in_structure = jax.ShapeDtypeStruct((4,), jax.numpy.float32)
        >>> out_structure = jax.ShapeDtypeStruct((2,), jax.numpy.float32)
        >>> op = IndexOperator(indices, in_structure=in_structure, out_structure=out_structure)
    """

    indices: tuple[int | slice | Bool[Array, '...'] | Integer[Array, '...'] | EllipsisType, ...]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _out_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    unique_indices: bool = equinox.field(static=True)

    def __init__(
        self,
        indices: (
            int
            | slice
            | Bool[Array, '...']
            | Integer[Array, '...']
            | tuple[int | slice | Bool[Array, '...'] | Integer[Array, '...'] | EllipsisType, ...]
        ),
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        out_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
        unique_indices: bool | None = None,
    ) -> None:
        if not isinstance(indices, tuple):
            indices = (indices,)
        self._check_indices(indices)
        
        # Set these attributes first before computing _out_structure
        self.indices = indices
        self._in_structure = in_structure
        
        if all(
            isinstance(_, int | slice | EllipsisType) or isinstance(_, Array) and _.dtype == bool
            for _ in indices
        ):
            unique_indices = True
        elif unique_indices is None:
            unique_indices = False
        if out_structure is None and any(isinstance(_, Array) and _.dtype == bool for _ in indices):
            raise ValueError(
                'The output structure must be specified when the indices are determined using a '
                'boolean array.'
            )
        self.unique_indices = unique_indices
        if out_structure is None:
            # Compute output structure manually to avoid circular dependency
            # during initialization
            def temp_mv(x):
                return jax.tree.map(lambda leaf: leaf[self.indices], x)
            self._out_structure = jax.eval_shape(temp_mv, self._in_structure)
        else:
            self._out_structure = out_structure

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(lambda leaf: leaf[self.indices], x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def reduce(self) -> AbstractLinearOperator:
        if len(self.indexed_axes) == 0:
            return IdentityOperator(self.in_structure())
        return self

    @staticmethod
    def _check_indices(
        indices: tuple[
            int | slice | Bool[Array, '...'] | Integer[Array, '...'] | EllipsisType, ...
        ],
    ) -> None:
        ellipsis_count = sum(index is Ellipsis for index in indices)
        if ellipsis_count > 1:
            raise ValueError('more than one Ellipsis in specified in the indices.')

    @property
    def indexed_axes(self) -> list[int]:
        """Returns the list of axes for which an indexing is performed.

        Example: for an indexing of (slice(None), 3, ..., jnp.array([1, 2])),
            it returns [1, -1].
        """
        try:
            ellipsis_index = self.indices.index(Ellipsis)
        except ValueError:
            ellipsis_index = len(self.indices)

        axes = []
        for axis, index in enumerate(self.indices):
            if axis >= ellipsis_index:
                break
            if isinstance(index, slice) and index == slice(None):
                continue
            axes.append(axis)
        for axis, index in enumerate(self.indices[ellipsis_index + 1 :], ellipsis_index + 1):
            if index == slice(None):
                continue
            axes.append(axis - len(self.indices))
        return axes


class IndexTransposeRule(AbstractBinaryRule):
    """Binary rule for `index @ index.T = I`."""

    left_operator_class = IndexOperator
    right_operator_class = TransposeOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, IndexOperator)
        if not left.unique_indices:
            raise NoReduction
        return []


class TransposeIndexRule(AbstractBinaryRule):
    """Binary rule for `index.T @ index = I`."""

    left_operator_class = TransposeOperator
    right_operator_class = IndexOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(right, IndexOperator)
        indexed_axes = right.indexed_axes
        if len(indexed_axes) > 1:
            raise NoReduction
        if right.unique_indices:
            raise NoReduction

        dtype = right.out_promoted_dtype
        shapes = {leaf.shape for leaf in jax.tree.leaves(right.in_structure())}
        if len(shapes) > 1:
            raise NoReduction
        shape = shapes.pop()

        axis = indexed_axes[0]
        index = right.indices[axis]
        assert isinstance(index, Array)

        size_max = shape[axis]
        unique_indices, counts = jnp.unique(index, return_counts=True, size=size_max, fill_value=-1)
        coverage = jnp.zeros(size_max, dtype=dtype)
        coverage = coverage.at[unique_indices].add(
            counts, indices_are_sorted=True, unique_indices=True
        )
        diagonal_op = DiagonalOperator(
            coverage, axis_destination=axis, in_structure=right.in_structure()
        )
        return [diagonal_op]
