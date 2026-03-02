import functools
from abc import ABC
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import Array
from jax.tree_util import PyTreeDef
from jaxtyping import Inexact, PyTree

from ..tree import add
from ._base import AbstractLinearOperator, AdditionOperator, IdentityOperator
from .rules import AbstractBinaryRule


class AbstractBlockOperator(AbstractLinearOperator, ABC):
    blocks: PyTree[AbstractLinearOperator]

    def __init__(self, blocks: PyTree[AbstractLinearOperator]) -> None:
        object.__setattr__(self, 'blocks', blocks)
        super().__init__(in_structure=self._tree_map(lambda op: op.in_structure))

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._tree_map(lambda op: op.out_structure)

    def reduce(self) -> AbstractLinearOperator:
        return type(self)(self._tree_map(lambda op: op.reduce()))

    @property
    def block_leaves(self) -> list[AbstractLinearOperator]:
        """Returns the flat list of operators."""
        return jax.tree.leaves(self.blocks, is_leaf=lambda x: isinstance(x, AbstractLinearOperator))

    def _tree_map(self, f: Callable[..., Any], *args: Any) -> Any:
        return jax.tree.map(
            f,
            self.blocks,
            *args,
            is_leaf=lambda x: isinstance(x, AbstractLinearOperator),
        )


class BlockRowOperator(AbstractBlockOperator):
    """A block row operator, where each block is an operator.

    The blocks are stored in a pytree structure, with each leaf being an AbstractLinearOperator.

    Examples:
        >>> x = jnp.array([1, 2], jnp.float32)
        >>> I = IdentityOperator(in_structure=jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> op_list = BlockRowOperator([I, 2*I, 3*I])
        >>> op_list.as_matrix()
        Array([[1., 0., 2., 0., 3., 0.],
               [0., 1., 0., 2., 0., 3.]], dtype=float32)
        >>> op_list([x, x, x])
        Array([ 6., 12.], dtype=float32)

        >>> op_dict = BlockRowOperator({'a': I, 'b': 2*I, 'c': 3*I})
        >>> op_dict({'a': x, 'b': x, 'c': x})
        Array([ 6., 12.], dtype=float32)
    """

    def __init__(self, blocks: PyTree[AbstractLinearOperator]) -> None:
        super().__init__(blocks)
        try:
            operators = self.block_leaves
            ref_structure = operators[0].out_structure
        except (AttributeError, TypeError):
            # During JAX/equinox tree operations, operators may have boolean placeholders
            return
        invalid_structures = [
            structure
            for operator in operators[1:]
            if (structure := operator.out_structure) != ref_structure
        ]
        if len(invalid_structures) > 0:
            structures_as_str = '\n - '.join(str(structure) for structure in invalid_structures)
            raise ValueError(
                f'The operators in a BlockRowOperator must have the same output structure:\n'
                f' - {ref_structure}\n'
                f' - {structures_as_str}'
            )

    def mv(self, x: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        treedef: PyTreeDef = jax.tree.structure(
            self.blocks, is_leaf=lambda op: isinstance(op, AbstractLinearOperator)
        )
        output_leaves = (
            block(leaf) for block, leaf in zip(self.block_leaves, treedef.flatten_up_to(x))
        )
        return functools.reduce(lambda a, b: add(a, b), output_leaves)

    def transpose(self) -> AbstractLinearOperator:
        return BlockColumnOperator(self._tree_map(lambda op: op.T))

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.block_leaves[0].out_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.hstack([op.as_matrix() for op in self.block_leaves])


class BlockDiagonalOperator(AbstractBlockOperator):
    """A block diagonal operator, where each block is an operator.

    The blocks are stored in a pytree structure, with each leaf being an AbstractLinearOperator.

    Example:
        >>> x = jnp.array([1, 2], jnp.float32)
        >>> H = DenseBlockDiagonalOperator(
        ...     jnp.array([[0, 1], [1, 0]]),
        ...     jax.ShapeDtypeStruct((2,), jnp.float32)
        ... )
        >>> H.as_matrix()
        Array([[0., 1.],
               [1., 0.]], dtype=float32)
        >>> op_list = BlockDiagonalOperator([H, 2*H, 3*H])
        >>> op_list.as_matrix()
        Array([[0., 1., 0., 0., 0., 0.],
               [1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 2., 0., 0.],
               [0., 0., 2., 0., 0., 0.],
               [0., 0., 0., 0., 0., 3.],
               [0., 0., 0., 0., 3., 0.]], dtype=float32)
        >>> op_list([x, x, x])
        [Array([2., 1.], dtype=float32),
         Array([4., 2.], dtype=float32),
         Array([6., 3.], dtype=float32)]

        >>> op_dict = BlockDiagonalOperator({'a': H, 'b': 2*H, 'c': 3*H})
        >>> op_dict({'a': x, 'b': x, 'c': x})
        {'a': Array([2., 1.], dtype=float32),
         'b': Array([4., 2.], dtype=float32),
         'c': Array([6., 3.], dtype=float32)}
    """

    def __init__(self, blocks: PyTree[AbstractLinearOperator]) -> None:
        # required: otherwise, the parent constructor would not be called by the dataclass-generated constructor
        super().__init__(blocks)

    def mv(self, vector: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        return self._tree_map(lambda op, vect: op.mv(vect), vector)

    def transpose(self) -> AbstractLinearOperator:
        return BlockDiagonalOperator(self._tree_map(lambda op: op.T))

    def inverse(self) -> AbstractLinearOperator:
        # if some of the blocks are not square, let's defer to the default inverse method
        if not jax.tree.all(self._tree_map(lambda op: op.in_structure == op.out_structure)):
            return super().inverse()
        return BlockDiagonalOperator(self._tree_map(lambda op: op.I))

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jsl.block_diag(*[op.as_matrix() for op in self.block_leaves])  # type: ignore[no-any-return]  # noqa: E501

    def reduce(self) -> AbstractLinearOperator:
        """BlockDiagonalOperator([I, I, ...]) -> I."""
        op = super().reduce()
        assert isinstance(op, BlockDiagonalOperator)
        if all(isinstance(block, IdentityOperator) for block in op.block_leaves):
            return IdentityOperator(in_structure=self.in_structure)
        return op


class BlockColumnOperator(AbstractBlockOperator):
    """A block column operator, where each block is an operator.

    The blocks are stored in a pytree structure, with each leaf being an AbstractLinearOperator.

    Examples:
        >>> x = jnp.array([1, 2], jnp.float32)
        >>> I = IdentityOperator(in_structure=jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> op_list = BlockColumnOperator([I, I, I])
        >>> op_list.as_matrix()
        Array([[1., 0.],
               [0., 1.],
               [1., 0.],
               [0., 1.],
               [1., 0.],
               [0., 1.]], dtype=float32)
        >>> op_list(x)
        [Array([1., 2.], dtype=float32),
         Array([1., 2.], dtype=float32),
         Array([1., 2.], dtype=float32)]

        >>> op_dict = BlockColumnOperator({'a': I, 'b': I, 'c': I})
        >>> op_dict(x)
        {'a': Array([1., 2.], dtype=float32),
         'b': Array([1., 2.], dtype=float32),
         'c': Array([1., 2.], dtype=float32)}
    """

    def __init__(self, blocks: PyTree[AbstractLinearOperator]) -> None:
        super().__init__(blocks)
        try:
            operators = self.block_leaves
            ref_structure = operators[0].in_structure
        except (AttributeError, TypeError):
            # During JAX/equinox tree operations, operators may have boolean placeholders
            return
        object.__setattr__(self, 'in_structure', ref_structure)
        invalid_structures = [
            structure
            for operator in operators[1:]
            if (structure := operator.in_structure) != ref_structure
        ]
        if len(invalid_structures) > 0:
            structures_as_str = '\n - '.join(str(structure) for structure in invalid_structures)
            raise ValueError(
                f'The operators in a BlockColumnOperator must have the same input structure:\n'
                f' - {ref_structure}\n'
                f' - {structures_as_str}'
            )

    def mv(self, vector: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        return self._tree_map(lambda op: op.mv(vector))

    def transpose(self) -> AbstractLinearOperator:
        return BlockRowOperator(self._tree_map(lambda op: op.T))

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.vstack([op.as_matrix() for op in self.block_leaves])


class AbstractBlockDiagonalRule(AbstractBinaryRule):
    reduced_class: type[AbstractBlockOperator] | type[AdditionOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractBlockOperator)  # mypy assert
        assert isinstance(right, AbstractBlockOperator)  # mypy assert
        return [self.reduced_class(left._tree_map(lambda l, r: l @ r, right.blocks)).reduce()]


class BlockRowBlockDiagonalRule(AbstractBlockDiagonalRule):
    """Binary rule for the composition of a block row and a block diagonal operator.

    BlockRow(A_i, ...) @ BlockDiagonal(B_i, ...) = BlockRow(A_i @ B_i, ...)
    """

    left_operator_class = BlockRowOperator
    right_operator_class = BlockDiagonalOperator
    reduced_class = BlockRowOperator


class BlockDiagonalBlockColumnRule(AbstractBlockDiagonalRule):
    """Binary rule for the composition of a block diagonal and a block column operator.

    BlockDiagonal(A_i, ...) @ BlockColumn(B_i, ...) = BlockColumn(A_i @ B_i, ...)
    """

    left_operator_class = BlockDiagonalOperator
    right_operator_class = BlockColumnOperator
    reduced_class = BlockColumnOperator


class BlockDiagonalBlockDiagonalRule(AbstractBlockDiagonalRule):
    """Binary rule for the composition of two block diagonal operators.

    BlockDiagonal(A_i, ...) @ BlockDiagonal(B_i, ...) = BlockDiagonal(A_i @ B_i, ...)
    """

    left_operator_class = BlockDiagonalOperator
    right_operator_class = BlockDiagonalOperator
    reduced_class = BlockDiagonalOperator


class BlockRowBlockColumnRule(AbstractBlockDiagonalRule):
    """Binary rule for the composition of a block row and a block column operator.

    BlockRow(A_i, ...) @ BlockColumn(B_i, ...) = Σ A_i @ B_i
    """

    left_operator_class = BlockRowOperator
    right_operator_class = BlockColumnOperator
    reduced_class = AdditionOperator
