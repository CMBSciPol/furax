import functools as ft
from collections.abc import Sequence
from math import prod
from typing import cast

import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Inexact, PyTree

from .core import AbstractLazyTransposeOperator, AbstractLinearOperator
from .rules import AbstractBinaryRule, NoReduction

__all__ = [
    'MoveAxisOperator',
    'RavelOperator',
    'ReshapeOperator',
]


class MoveAxisOperator(AbstractLinearOperator):
    """Operator to move axes of pytree leaves to new positions.

    The operation is conceptually the same as:
        y = jnp.moveaxis(x, source, destination)

    Example:
        >>> in_structure = jax.ShapeDtypeStruct((2, 3), jnp.float32)
        >>> op = MoveAxisOperator(0, 1, in_structure)
        >>> op(jnp.array([[1., 1, 1], [2, 2, 2]]))
        Array([[1., 2.],
               [1., 2.],
               [1., 2.]], dtype=float32)
    """

    source: tuple[int]
    destination: tuple[int]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        source: int | Sequence[int],
        destination: int | Sequence[int],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        if isinstance(source, int):
            source = (source,)
        elif not isinstance(source, tuple):
            source = cast(tuple[int], tuple(source))
        if isinstance(destination, int):
            destination = (destination,)
        elif not isinstance(destination, tuple):
            destination = cast(tuple[int], tuple(destination))
        self.source = source
        self.destination = destination
        self._in_structure = in_structure

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array, '...']:
        return jax.tree.map(lambda leaf: jnp.moveaxis(leaf, self.source, self.destination), x)

    def transpose(self) -> AbstractLinearOperator:
        return MoveAxisOperator(self.destination, self.source, in_structure=self.out_structure())

    inverse = transpose

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        moveaxis = ft.partial(jnp.moveaxis, source=self.source, destination=self.destination)
        return jax.tree.map(
            lambda leaf: jax.eval_shape(moveaxis, leaf),
            self.in_structure(),
        )


class MoveAxisInverseRule(AbstractBinaryRule):
    """Binary rule for move_axis.T @ move_axis = I`.

    Note:
        We cannot simply decorate MoveAxisOperator with :orthogonal: because it is not square,
        in the sense that its input and output structures are different.
    """

    left_operator_class = MoveAxisOperator
    right_operator_class = MoveAxisOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, MoveAxisOperator)  # mypy assert
        assert isinstance(right, MoveAxisOperator)  # mypy assert
        if left.source != right.destination or left.destination != right.source:
            raise NoReduction
        return []


# Note: if an algebraic rule to compose MoveAxisOperators is to be implemented, it may be best
# to implement a class TransposeOperator wrapping jnp.transpose and transform MoveAxisOperator
# instances into TransposeOperator instances. That way, it would be easier to include reductions for
# new operators, such as SwapAxesOperator, etc.


class AbstractRavelOrReshapeOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, in_structure: PyTree[jax.ShapeDtypeStruct]):
        self._in_structure = in_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.eye(self.in_size(), dtype=self.out_promoted_dtype)

    def transpose(self) -> AbstractLinearOperator:
        return ReshapeTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(self.mv, self._in_structure)


class RavelOperator(AbstractRavelOrReshapeOperator):
    """Class for raveling dimensions of pytree leaves.

    When instantiated with the default values, this operation is conceptually the same as:
        y = x.ravel()

    The input dimensions can also be partially flattened between two axes.

    Attributes:
        first_axis: The first axis of the pytree leaves that will be flattened.
        last_axis: The last axis of the pytree leaves that will be flattened.

    Examples:
        To flatten the leaves of a pytree:

        >>> in_structure = jax.ShapeDtypeStruct((2, 3), jnp.float32)
        >>> op = RavelOperator(in_structure=in_structure)
        >>> op.out_structure()
        ShapeDtypeStruct(shape=(6,), dtype=float32)

        To flatten the first two axes of the leaves of a pytree:

        >>> import furax as fx
        >>> x = [jnp.ones((2, 2)), jnp.ones((2, 2, 8))]
        >>> op = RavelOperator(0, 1, in_structure=fx.tree.as_structure(x))
        >>> op.out_structure()
        [ShapeDtypeStruct(shape=(4,), dtype=float32),
        ShapeDtypeStruct(shape=(4, 8), dtype=float32)]


        To flatten the last two axes of the leaves of a pytree:

        >>> import furax as fx
        >>> x = [jnp.ones((2, 2, 3)), jnp.ones((2, 8))]
        >>> op = RavelOperator(-2, -1, in_structure=fx.tree.as_structure(x))
        >>> op.out_structure()
        [ShapeDtypeStruct(shape=(2, 6), dtype=float32),
        ShapeDtypeStruct(shape=(16,), dtype=float32)]
    """

    first_axis: int = equinox.field(static=True)
    last_axis: int = equinox.field(static=True)

    def __init__(
        self,
        first_axis: int = 0,
        last_axis: int = -1,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        if 0 <= last_axis < first_axis or last_axis < first_axis < 0:
            raise ValueError(
                f'the first axis ({first_axis}) to be flattened should be before the last one '
                f'({last_axis}).'
            )

        if first_axis < 0 <= last_axis or last_axis < 0 <= first_axis:
            for leaf in jax.tree.leaves(in_structure):
                first = leaf.ndim + first_axis if first_axis < 0 else first_axis
                last = leaf.ndim + last_axis if last_axis < 0 else last_axis
                if first > last:
                    raise ValueError(
                        f'there are no dimensions between {first_axis} and {last_axis} '
                        f'to be flattened in leaf of shape {leaf.shape}.'
                    )

        super().__init__(in_structure)
        self.first_axis = first_axis
        self.last_axis = last_axis

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        def func(leaf: Inexact[Array, ' _a']) -> Inexact[Array, ' _b']:
            first_axis = leaf.ndim + self.first_axis if self.first_axis < 0 else self.first_axis
            last_axis = leaf.ndim + self.last_axis if self.last_axis < 0 else self.last_axis
            if first_axis > last_axis:
                assert False, 'unreachable'
            if first_axis == last_axis:
                return leaf
            new_shape = leaf.shape[:first_axis] + (-1,) + leaf.shape[last_axis + 1 :]
            return leaf.reshape(new_shape)

        return jax.tree.map(func, x)


class ReshapeOperator(AbstractRavelOrReshapeOperator):
    """Class for reshaping pytree leaves.

    The operation is conceptually the same as:
        y = x.reshape(new_shape)

    Attributes:
        shape: The new shape of the input pytree leaves.
    """

    shape: tuple[int, ...] = equinox.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self._check_shape(shape, in_structure)
        super().__init__(in_structure)
        self.shape = shape

    def _check_shape(
        self, shape: tuple[int, ...], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> None:
        for leaf in jax.tree.leaves(in_structure):
            new_shape = self._normalize_shape(shape, leaf.shape)
            if leaf.size != prod(new_shape):
                raise ValueError(f'invalid new shape {shape} for leaf of shape {leaf.shape}.')

    @staticmethod
    def _normalize_shape(shape: tuple[int, ...], leaf_shape: tuple[int, ...]) -> tuple[int, ...]:
        if any(_ < -1 for _ in shape):
            raise ValueError(f'reshape new sizes should be all positive, got {shape}.')
        try:
            index = shape.index(-1)
        except ValueError:
            return shape

        before = shape[:index]
        after = shape[index + 1 :]
        if -1 in after:
            raise ValueError('can only specify one unknown dimension.')
        unknown_dimension = -prod(leaf_shape) / prod(shape)
        if unknown_dimension != int(unknown_dimension):
            raise ValueError(f'cannot reshape array of shape {leaf_shape} into shape {shape}.')
        return before + (int(unknown_dimension),) + after

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(lambda leaf: leaf.reshape(self.shape), x)


class ReshapeTransposeOperator(AbstractLazyTransposeOperator):
    operator: ReshapeOperator | RavelOperator

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(
            lambda leaf, out_structure_leaf: leaf.reshape(out_structure_leaf.shape),
            x,
            self.out_structure(),
        )


class ReshapeInverseRule(AbstractBinaryRule):
    """Binary rule for reshape.T @ reshape = I and reshape @ reshape.T = I`.

    Note:
        We cannot simply decorate ReshapeOperator with :orthogonal: because it is not square,
        in the sense that its input and output structures are different.
    """

    left_operator_class = (AbstractRavelOrReshapeOperator, ReshapeTransposeOperator)
    right_operator_class = (AbstractRavelOrReshapeOperator, ReshapeTransposeOperator)

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if isinstance(left, AbstractRavelOrReshapeOperator):
            if not isinstance(right, ReshapeTransposeOperator):
                raise NoReduction
            if right.operator is not left:
                raise NoReduction
            return []
        elif isinstance(left, ReshapeTransposeOperator):
            if not isinstance(right, AbstractRavelOrReshapeOperator):
                raise NoReduction
            if left.operator is not right:
                raise NoReduction
            return []
        else:
            assert False, 'unreachable'
