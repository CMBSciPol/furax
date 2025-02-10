from collections import Counter
from collections.abc import Sequence

import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Inexact, PyTree

from furax.tree import is_leaf

from ._base import AbstractLazyInverseOperator, AbstractLinearOperator, diagonal


class BroadcastDiagonalOperator(AbstractLinearOperator):
    """Class representing a generalized diagonal operation.

    This operator is not necessarily diagonal. It can be a block row operator with diagonal blocks
    if the broadcasting extends the dimensions of the inputs on the left (regular broadcasting), or
    a block diagonal operator with blocks being columns if the broadcasting applies on the right.

    Args:
        diagonal: The diagonal values for the generalized diagonal operator.
        axis_destination: The axes along which the generalized diagonal values are applied to the
            input. If the type is a sequence, there should be as many axes as the number of
            dimensions in the ``diagonal`` input. If the type is a non-negative scalar integer, the
            dimensions will be ``(axis, ..., axis + diagonal.ndim - 1)``. If the type is a negative
            scalar integer, the dimensions will be ``(axis - diagonal.ndim, ..., axis)``.
        in_structure: The expected structure of the operator input.

    Usage:
        >>> import furax as fx
        >>> import jax.numpy as jnp
        >>> from numpy.testing import assert_allclose
        >>> x = jnp.array([1, 2, 3])
        >>> values = jnp.array([[1, 1, 1], [2, 1, 0]])
        >>> op = BroadcastDiagonalOperator(
        ...     values, in_structure=fx.tree.as_structure(x), axis_destination=-1
        ... )
        >>> assert_allclose(op(x), jnp.array([[1, 2, 3], [2, 2, 0]]))
        >>> op.as_matrix()
        Array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1],
               [2, 0, 0],
               [0, 1, 0],
               [0, 0, 0]], dtype=int32)

        >>> x = jnp.array([1, 2])
        >>> values = jnp.array([[2, 3, 1], [1, 0, 1]])
        >>> op = BroadcastDiagonalOperator(
        ...     values, in_structure=fx.tree.as_structure(x), axis_destination=0
        ... )
        >>> assert_allclose(op(x), jnp.array([[2, 3, 1], [2, 0, 2]]))
        >>> op.as_matrix()
        Array([[2, 0],
               [3, 0],
               [1, 0],
               [0, 1],
               [0, 0],
               [0, 1]], dtype=int32)

        >>> x = jnp.array([[0, 1, 2], [2, 3, 4]])
        >>> values = jnp.array([2, 1])
        >>> op = BroadcastDiagonalOperator(
        ...     values, in_structure=fx.tree.as_structure(x), axis_destination=0
        ... )
        >>> assert_allclose(op(x), jnp.array([[0, 2, 4], [2, 3, 4]]))
        >>> op.as_matrix()
        Array([[2, 0, 0, 0, 0, 0],
               [0, 2, 0, 0, 0, 0],
               [0, 0, 2, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 1]], dtype=int32)
    """

    _diagonal: Inexact[Array, '...']
    axis_destination: tuple[int, ...] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        diagonal: Inexact[Array, '...'],
        *,
        axis_destination: int | Sequence[int] = -1,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        if not is_leaf(diagonal):
            raise ValueError(
                'the diagonal values cannot be a pytree. Use a BlockDiagonalOperator with'
                'DiagonalOperators or BroadcastDiagonalOperators instead.'
            )
        if diagonal.ndim == 0:
            raise ValueError('the diagonal values are scalar. Use HomothetyOperator instead.')
        if isinstance(axis_destination, int):
            if axis_destination >= 0:
                # if a positive int is specified, return (axis, axis + 1, axis + 2, ...)
                axis_destination = tuple(range(axis_destination, axis_destination + diagonal.ndim))
            else:
                # if a negative int is specified, return (..., axis - 2, axis - 1, axis)
                axis_destination = tuple(
                    range(axis_destination - diagonal.ndim + 1, axis_destination + 1)
                )
        if not isinstance(axis_destination, tuple):
            axis_destination = tuple(axis_destination)

        self._diagonal = diagonal
        self.axis_destination = axis_destination
        self._in_structure = in_structure

        # check dimensions
        _ = AbstractLinearOperator.out_structure(self)

    @property
    def diagonal(self) -> Inexact[Array, '...']:
        return self._diagonal

    def _reshape_leaves(
        self,
        input_leaf: Inexact[Array, '...'],
    ) -> tuple[Inexact[Array, '#b'], Inexact[Array, '#b']]:
        axes = self._normalize_axes(input_leaf.shape)
        reshaped_diagonal = self._reshape_diagonal(axes, input_leaf.ndim)
        reshaped_input_leaf = self._reshape_input_leaf(axes, input_leaf)
        self._check_leaf_shapes(
            reshaped_diagonal.shape, reshaped_input_leaf.shape, input_leaf.shape
        )
        return reshaped_diagonal, reshaped_input_leaf

    def _normalize_axes(self, input_leaf_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns positive axes according to the input leaf shape."""
        axes = tuple(
            axis if axis >= 0 else len(input_leaf_shape) + axis for axis in self.axis_destination
        )
        dups = [k for k, v in Counter(axes).items() if v > 1]
        if dups:
            raise ValueError(
                f'duplicated axis destination {list(self.axis_destination)} for leaf of shape '
                f'{input_leaf_shape}.'
            )
        return axes

    def _reshape_diagonal(
        self, axes: tuple[int, ...], input_leaf_ndim: int
    ) -> Inexact[Array, '...']:
        left_broadcast_dimensions = -min(0, min(axes))
        right_broadcast_dimensions = max(0, max(axes) - input_leaf_ndim + 1)
        reshaped_diagonal_leaf = self.diagonal.reshape(
            self._diagonal.shape
            + (
                left_broadcast_dimensions
                + right_broadcast_dimensions
                + input_leaf_ndim
                - self._diagonal.ndim
            )
            * (1,)
        )
        axes = tuple(axis + left_broadcast_dimensions for axis in axes)
        return jnp.moveaxis(reshaped_diagonal_leaf, range(len(axes)), axes)

    def _reshape_input_leaf(
        self, axes: tuple[int, ...], input_leaf: Inexact[Array, '...']
    ) -> Inexact[Array, '...']:
        right_broadcast_dimensions = max(0, max(axes) - input_leaf.ndim + 1)
        reshaped_input_leaf = input_leaf.reshape(
            input_leaf.shape + right_broadcast_dimensions * (1,)
        )
        return reshaped_input_leaf

    def _check_leaf_shapes(
        self,
        diagonal_shape: tuple[int, ...],
        leaf_shape: tuple[int, ...],
        input_shape: tuple[int, ...],
    ) -> None:
        _ = jnp.broadcast_shapes(diagonal_shape, leaf_shape)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        def func(input_leaf: Inexact[Array, '...']) -> Inexact[Array, '...']:
            reshaped_diagonal_leaf, reshaped_input_leaf = self._reshape_leaves(input_leaf)
            return reshaped_diagonal_leaf * reshaped_input_leaf

        return jax.tree.map(func, x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


@diagonal
class DiagonalOperator(BroadcastDiagonalOperator):
    """Class representing a diagonal operator.

    The axes to be used for the multiplication can be specified: given a 1-dimensional array
    ``diagonal`` and a 2-dimensional array ``x``:

        op_first_axis = DiagonalOperator(diagonal, axis_destination=0)
    op_first_axis(x) is conceptually equivalent to ``diagonal[:, None] * x``.

        op_last_axis = DiagonalOperator(diagonal, axis_destination=-1)
    op_last_axis(x) is conceptually equivalent to the usual broadcasting ``diagonal[None, :] * x``.


    Args:
        diagonal: The diagonal values for the diagonal operator.
        axis_destination: The axes along which the diagonal values are applied to the input.
            If the type is a sequence, there should be as many axes as the number of dimensions in
            the ``diagonal`` input. If the type is a non-negative scalar integer, the dimensions
            will be ``(axis, ..., axis + diagonal.ndim - 1)``. If the type is a negative scalar
            integer, the dimensions will be ``(axis - diagonal.ndim, ..., axis)``.
        in_structure: The expected structure of the operator input.

    Usage:
        >>> import furax as fx
        >>> from numpy.testing import assert_allclose
        >>> key_gain, key_tod, key_common = jax.random.split(jax.random.PRNGKey(0), 3)
        >>> detector_count = 3
        >>> sample_count = 10
        >>> x = {
        ...     'tod': jax.random.normal(key_tod, (detector_count, sample_count)),
        ...     'ground': jax.random.normal(key_common, (detector_count,)),
        ... }
        >>> detector_gains = jax.random.normal(key_gain, (detector_count,)) / 100 + 1
        >>> op = DiagonalOperator(
        ...     detector_gains, axis_destination=0, in_structure=fx.tree.as_structure(x)
        ... )
        >>> y = op(x)
        >>> assert_allclose(x['tod'] * detector_gains[:, None], y['tod'])
        >>> assert_allclose(x['ground'] * detector_gains, y['ground'])
    """

    def _check_leaf_shapes(
        self,
        diagonal_shape: tuple[int, ...],
        leaf_shape: tuple[int, ...],
        input_shape: tuple[int, ...],
    ) -> None:
        shape = jnp.broadcast_shapes(diagonal_shape, leaf_shape)
        if shape != input_shape:
            raise ValueError(
                f'the input shape {input_shape} cannot be changed to {shape} '
                f'by a DiagonalOperator. For broadcasting, use BroadcastDiagonalOperator.'
            )

    def inverse(self) -> 'AbstractLinearOperator':
        return DiagonalInverseOperator(self)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        diagonals = [
            jnp.broadcast_to(
                self._reshape_diagonal(self._normalize_axes(leaf.shape), leaf.ndim), leaf.shape
            ).ravel()
            for leaf in jax.tree.leaves(self.in_structure())
        ]
        matrix = jnp.diag(jnp.concatenate(diagonals, dtype=self.out_promoted_dtype))
        return matrix


class DiagonalInverseOperator(DiagonalOperator, AbstractLazyInverseOperator):
    def __init__(self, operator: DiagonalOperator) -> None:
        AbstractLazyInverseOperator.__init__(self, operator)
        DiagonalOperator.__init__(
            self,
            operator._diagonal,
            axis_destination=operator.axis_destination,
            in_structure=operator.in_structure(),
        )

    @property
    def diagonal(self) -> PyTree[Inexact[Array, '...']]:
        return jnp.where(self._diagonal != 0, 1 / self._diagonal, 0)

    def inverse(self) -> 'AbstractLinearOperator':
        return self.operator
