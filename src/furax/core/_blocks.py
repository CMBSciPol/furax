import functools
from abc import ABC
from collections.abc import Callable
from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import Array
from jaxtyping import Inexact, PyTree

from ..tree import add
from ._base import (
    AbstractLinearOperator,
    AdditionOperator,
    IdentityOperator,
    TransposeOperator,
    _group_by_signature,
    structure_equal,
)
from ._ensembles import Ensemble, ListEnsemble, StackedEnsemble, lift, map_reduce
from .rules import AbstractCompositionRule, NoReduction


class AbstractBlockOperator(AbstractLinearOperator, ABC):
    """Base class for the block composites.

    `blocks` is a pytree of operators or an [`Ensemble`][furax.core.Ensemble] of operators. An
    ensemble closes into the composite through the engine, with the composite deciding which side
    carries the member index: its stacked inputs and outputs are pytrees whose array leaves carry
    a leading member axis.
    """

    blocks: PyTree[AbstractLinearOperator]

    def __init__(self, blocks: PyTree[AbstractLinearOperator] | Ensemble) -> None:
        object.__setattr__(self, 'blocks', blocks)
        if isinstance(blocks, Ensemble):
            structure = self._ensemble_in_structure(blocks.fetch(0))
        else:
            structure = self._tree_map(lambda op: op.in_structure)
        super().__init__(in_structure=structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        if isinstance(self.blocks, Ensemble):
            return self._ensemble_out_structure(self.blocks.fetch(0))
        return self._tree_map(lambda op: op.out_structure)

    def reduce(self) -> AbstractLinearOperator:
        if isinstance(self.blocks, Ensemble):
            return type(self)(lift(lambda op: op.reduce(), self.blocks))
        return type(self)(self._tree_map(lambda op: op.reduce()))

    def _ensemble_in_structure(self, member: AbstractLinearOperator) -> PyTree[Any]:
        """The input structure of the composite over an ensemble, from one member's."""
        return _stacked_structure(member.in_structure, self.blocks.size)

    def _ensemble_out_structure(self, member: AbstractLinearOperator) -> PyTree[Any]:
        """The output structure of the composite over an ensemble, from one member's."""
        return _stacked_structure(member.out_structure, self.blocks.size)

    @property
    def block_leaves(self) -> list[AbstractLinearOperator]:
        """Returns the flat list of operators."""
        if isinstance(self.blocks, Ensemble):
            return [self.blocks.fetch(i) for i in range(self.blocks.size)]
        return jax.tree.leaves(self.blocks, is_leaf=lambda x: isinstance(x, AbstractLinearOperator))

    def _tree_map(self, f: Callable[..., Any], *args: Any) -> Any:
        return jax.tree.map(
            f,
            self.blocks,
            *args,
            is_leaf=lambda x: isinstance(x, AbstractLinearOperator),
        )


class BlockRowOperator(AbstractBlockOperator):
    """Operator that horizontally concatenates block operators: [A | B | C].

    Applies each block to the corresponding part of a pytree input and sums
    the results. All blocks must have the same output structure.

    Blocks that only differ by the values of their arrays (same class, static fields, array
    shapes and dtypes) are applied through a single traced body, a loop that selects one block
    and its input at a time, as [`AdditionOperator`][] does with `sequential=True`.

    Transpose: BlockRowOperator.T = BlockColumnOperator

    Attributes:
        blocks: A pytree of operators (all with identical output structure).

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

    def __init__(self, blocks: PyTree[AbstractLinearOperator] | Ensemble) -> None:
        super().__init__(blocks)
        if isinstance(blocks, Ensemble):
            # the members of an ensemble share their structures
            return
        try:
            operators = self.block_leaves
            ref_structure = operators[0].out_structure
        except (AttributeError, TypeError):
            # During JAX/equinox tree operations, operators may have boolean placeholders
            return
        invalid_structures = [
            structure
            for operator in operators[1:]
            if not structure_equal((structure := operator.out_structure), ref_structure)
        ]
        if len(invalid_structures) > 0:
            structures_as_str = '\n - '.join(str(structure) for structure in invalid_structures)
            raise ValueError(
                f'The operators in a BlockRowOperator must have the same output structure:\n'
                f' - {ref_structure}\n'
                f' - {structures_as_str}'
            )

    def mv(self, x: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        if isinstance(self.blocks, Ensemble):
            zipped = _zip_input(self.blocks, x)
            return map_reduce(lambda member: member[0](member[1]), zipped, 'sum')[0]
        treedef = jax.tree.structure(
            self.blocks, is_leaf=lambda op: isinstance(op, AbstractLinearOperator)
        )
        blocks = self.block_leaves
        leaves = treedef.flatten_up_to(x)
        contributions = (
            blocks[group[0]](leaves[group[0]])
            if len(group) == 1
            else map_reduce(
                lambda member: member[0](member[1]),
                ListEnsemble([(blocks[i], leaves[i]) for i in group]),
                'sum',
            )[0]
            for group in _group_by_signature(blocks)
        )
        return functools.reduce(lambda a, b: add(a, b), contributions)

    def transpose(self) -> AbstractLinearOperator:
        if isinstance(self.blocks, Ensemble):
            return BlockColumnOperator(lift(lambda op: op.T, self.blocks))
        return BlockColumnOperator(self._tree_map(lambda op: op.T))

    def _ensemble_out_structure(self, member: AbstractLinearOperator) -> PyTree[Any]:
        return member.out_structure  # summed over the members

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        if isinstance(self.blocks, Ensemble):
            return self.blocks.fetch(0).out_structure
        return self.block_leaves[0].out_structure

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.hstack([op.as_matrix() for op in self.block_leaves])


class BlockDiagonalOperator(AbstractBlockOperator):
    """Operator with independent diagonal blocks: diag(A, B, C).

    Applies each block independently to the corresponding part of a pytree input.
    No constraints on block input/output structures.

    The inverse is the block diagonal of individual inverses (if all blocks are square).

    Attributes:
        blocks: A pytree of operators.

    Examples:
        >>> from furax import DenseBlockDiagonalOperator
        >>> x = jnp.array([1, 2], jnp.float32)
        >>> H = DenseBlockDiagonalOperator(
        ...     jnp.array([[0, 1], [1, 0]]),
        ...     in_structure=jax.ShapeDtypeStruct((2,), jnp.float32),
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

    def mv(self, x: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        if isinstance(self.blocks, Ensemble):
            zipped = _zip_input(self.blocks, x)
            return map_reduce(lambda member: member[0](member[1]), zipped, 1)[0]
        return self._tree_map(lambda op, vect: op.mv(vect), x)

    def transpose(self) -> AbstractLinearOperator:
        if isinstance(self.blocks, Ensemble):
            return BlockDiagonalOperator(lift(lambda op: op.T, self.blocks))
        return BlockDiagonalOperator(self._tree_map(lambda op: op.T))

    def inverse(self) -> AbstractLinearOperator:
        if isinstance(self.blocks, Ensemble):
            member = self.blocks.fetch(0)
            if not structure_equal(member.in_structure, member.out_structure):
                return super().inverse()
            return BlockDiagonalOperator(lift(lambda op: op.I, self.blocks))
        # if some of the blocks are not square, let's defer to the default inverse method
        if not jax.tree.all(
            self._tree_map(lambda op: structure_equal(op.in_structure, op.out_structure))
        ):
            return super().inverse()
        return BlockDiagonalOperator(self._tree_map(lambda op: op.I))

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jsl.block_diag(*[op.as_matrix() for op in self.block_leaves])

    def reduce(self) -> AbstractLinearOperator:
        """BlockDiagonalOperator([I, I, ...]) -> I."""
        op = super().reduce()
        assert isinstance(op, BlockDiagonalOperator)
        witnesses = [op.blocks.fetch(0)] if isinstance(op.blocks, Ensemble) else op.block_leaves
        if all(isinstance(block, IdentityOperator) for block in witnesses):
            return IdentityOperator(in_structure=self.in_structure)
        return op


class BlockColumnOperator(AbstractBlockOperator):
    """Operator that vertically stacks block operators: [A; B; C].

    Applies each block to the same input and returns a pytree of outputs.
    All blocks must have the same input structure.

    Transpose: BlockColumnOperator.T = BlockRowOperator

    Attributes:
        blocks: A pytree of operators (all with identical input structure).

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

    def __init__(self, blocks: PyTree[AbstractLinearOperator] | Ensemble) -> None:
        super().__init__(blocks)
        if isinstance(blocks, Ensemble):
            # the members of an ensemble share their structures
            return
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
            if not structure_equal((structure := operator.in_structure), ref_structure)
        ]
        if len(invalid_structures) > 0:
            structures_as_str = '\n - '.join(str(structure) for structure in invalid_structures)
            raise ValueError(
                f'The operators in a BlockColumnOperator must have the same input structure:\n'
                f' - {ref_structure}\n'
                f' - {structures_as_str}'
            )

    def mv(self, x: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        if isinstance(self.blocks, Ensemble):
            return map_reduce(lambda op, v: op(v), self.blocks, 1, x)[0]
        return self._tree_map(lambda op: op.mv(x))

    def transpose(self) -> AbstractLinearOperator:
        if isinstance(self.blocks, Ensemble):
            return BlockRowOperator(lift(lambda op: op.T, self.blocks))
        return BlockRowOperator(self._tree_map(lambda op: op.T))

    def _ensemble_in_structure(self, member: AbstractLinearOperator) -> PyTree[Any]:
        return member.in_structure  # every member takes the input whole

    def as_matrix(self) -> Inexact[Array, 'a b']:
        return jnp.vstack([op.as_matrix() for op in self.block_leaves])


class AbstractBlockDiagonalRule(AbstractCompositionRule):
    reduced_class: type[AbstractBlockOperator] | type[AdditionOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractBlockOperator)  # ty assert
        assert isinstance(right, AbstractBlockOperator)  # ty assert
        if isinstance(left.blocks, Ensemble) or isinstance(right.blocks, Ensemble):
            # the rules keep their statement over ensembles, composed member-wise by lifting
            if type(left.blocks) is not type(right.blocks):
                raise NoReduction
            if left.blocks.size != right.blocks.size:
                raise NoReduction
            composed = lift(lambda l, r: l @ r, left.blocks, right.blocks)
            return [self.reduced_class(composed).reduce()]
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


class BlockSelectOperator(AbstractLinearOperator):
    """Operator that selects one block of a block-structured input: y = x[key].

    Attributes:
        key: Index or key of the selected block, in the top-level node of the input structure.

    Examples:
        >>> structure = [jax.ShapeDtypeStruct((2,), jnp.float32)] * 3
        >>> op = BlockSelectOperator(1, in_structure=structure)
        >>> op([jnp.zeros(2), jnp.ones(2), 2 * jnp.ones(2)])
        Array([1., 1.], dtype=float32)
        >>> op.as_matrix()
        Array([[0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.]], dtype=float32)

        The transpose pads the block back with zeros:

        >>> BlockSelectOperator(0, in_structure=structure[:2]).T(jnp.ones(2))
        [Array([1., 1.], dtype=float32), Array([0., 0.], dtype=float32)]

        Dictionary-structured inputs are selected by key:

        >>> op = BlockSelectOperator('b', in_structure={'a': structure[0], 'b': structure[1]})
        >>> op({'a': jnp.zeros(2), 'b': jnp.ones(2)})
        Array([1., 1.], dtype=float32)
    """

    key: Any = field(metadata={'static': True})

    def __init__(self, key: Any, *, in_structure: PyTree[jax.ShapeDtypeStruct]) -> None:
        object.__setattr__(self, 'key', key)
        super().__init__(in_structure=in_structure)

    def mv(self, x: PyTree[Inexact[Array, ' _b']]) -> PyTree[Inexact[Array, ' _a']]:
        return x[self.key]

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.in_structure[self.key]


class BlockSelectTransposeRule(AbstractCompositionRule):
    """Binary rule for `select @ select.T = I`."""

    left_operator_class = BlockSelectOperator
    right_operator_class = TransposeOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return []


class BlockSelectBlockDiagonalRule(AbstractCompositionRule):
    """Binary rule for `select_k @ BlockDiagonal(A_i, ...) = A_k @ select_k`."""

    left_operator_class = BlockSelectOperator
    right_operator_class = BlockDiagonalOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, BlockSelectOperator)  # ty assert
        assert isinstance(right, BlockDiagonalOperator)  # ty assert
        if isinstance(right.blocks, Ensemble):
            raise NoReduction
        try:
            block = right.blocks[left.key]
        except (IndexError, KeyError, TypeError):
            raise NoReduction
        if not isinstance(block, AbstractLinearOperator):
            # the key selects a subtree of blocks, not a single one
            raise NoReduction
        return [block, BlockSelectOperator(left.key, in_structure=right.in_structure)]


def _zip_input(ensemble: Ensemble, x: PyTree[Any]) -> Ensemble:
    """Members paired with their slice of a stacked input: member i becomes `(member, x[i])`."""
    if isinstance(ensemble, StackedEnsemble):
        strides = tuple(ensemble._leaves_and_strides()[1]) + (1,) * len(jax.tree.leaves(x))
        return StackedEnsemble(
            ensemble.size,
            (ensemble.stacked, x),
            strides,
            valid=ensemble.valid,
            policy=ensemble.policy,
        )
    assert isinstance(ensemble, ListEnsemble)  # ty assert
    members = [
        (member, jax.tree.map(lambda leaf, i=i: leaf[i], x))
        for i, member in enumerate(ensemble.members)
    ]
    return ListEnsemble(members, valid=ensemble.valid, policy=ensemble.policy)


def _stacked_structure(structure: PyTree[jax.ShapeDtypeStruct], n: int) -> PyTree[Any]:
    """A structure with a leading member axis on every array leaf."""
    return jax.tree.map(lambda s: jax.ShapeDtypeStruct((n, *s.shape), s.dtype), structure)
