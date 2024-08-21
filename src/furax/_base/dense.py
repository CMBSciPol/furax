import functools as ft

import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Num, PyTree

from furax._base.core import AbstractLinearOperator
from furax._base.utils import is_leaf


class DenseBlockDiagonalOperator(AbstractLinearOperator):
    """Operator for block diagonal dense matrix operations involving pytrees.

    Only the diagonal blocks are stored by the operator.

    Example:
        For a matrix made of three 2x4 diagonal blocks, and input block columns of three blocks of
        four elements each, the operator can be written as:
        >>> blocks = jnp.arange(24).reshape(3, 2, 4)
        >>> op = DenseBlockDiagonalOperator(
        ...     blocks, jax.ShapeDtypeStruct((3, 4), jnp.int32), 'imn,in->im')
        >>> op.as_matrix()
        Array([[ 0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 4,  5,  6,  7,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  8,  9, 10, 11,  0,  0,  0,  0],
               [ 0,  0,  0,  0, 12, 13, 14, 15,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0, 16, 17, 18, 19],
               [ 0,  0,  0,  0,  0,  0,  0,  0, 20, 21, 22, 23]], dtype=int32)

        The axes along which the operator is block diagonal can be non-leading dimensions.
        As a matter of fact, by default, the diagonal axes are assumed to be "on the right".
        The notion of block diagonality should be understood in a tensor context. The representation
        of this operator as a 2d matrix, which relies on the row-major layout, may not be block
        diagonal.
        >>> blocks = jnp.arange(24).reshape(3, 2, 4)
        >>> op = DenseBlockDiagonalOperator(blocks, jax.ShapeDtypeStruct((2, 4), jnp.int32))
        >>> op.as_matrix()
        Array([[ 0,  0,  0,  0,  4,  0,  0,  0],
               [ 0,  1,  0,  0,  0,  5,  0,  0],
               [ 0,  0,  2,  0,  0,  0,  6,  0],
               [ 0,  0,  0,  3,  0,  0,  0,  7],
               [ 8,  0,  0,  0, 12,  0,  0,  0],
               [ 0,  9,  0,  0,  0, 13,  0,  0],
               [ 0,  0, 10,  0,  0,  0, 14,  0],
               [ 0,  0,  0, 11,  0,  0,  0, 15],
               [16,  0,  0,  0, 20,  0,  0,  0],
               [ 0, 17,  0,  0,  0, 21,  0,  0],
               [ 0,  0, 18,  0,  0,  0, 22,  0],
               [ 0,  0,  0, 19,  0,  0,  0, 23]], dtype=int32)
    """

    blocks: Num[Array, '...'] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    subscripts: str = equinox.field(static=True)

    def __init__(
        self,
        blocks: PyTree[Num[Array, '...']],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        subscripts='ij...,j...->i...',
    ) -> None:
        subscripts = subscripts.replace(' ', '')
        if not jax.tree.all(jax.tree.map(lambda leaf: len(leaf.shape) >= 2, blocks)):
            raise ValueError('The blocks should at least have 2 dimensions.')
        self._parse_subscripts(subscripts)
        self.blocks = blocks
        self._in_structure = in_structure
        self.subscripts = subscripts

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        if is_leaf(x):
            return jnp.einsum(self.subscripts, self.blocks, x)
        leaves, treedef = jax.tree.flatten(x)
        if is_leaf(self.blocks):
            return jax.tree.unflatten(
                treedef, [jnp.einsum(self.subscripts, self.blocks, leaf) for leaf in leaves]
            )
        return jax.tree.map(ft.partial(jnp.einsum, self.subscripts), self.blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return DenseBlockDiagonalOperator(
            self.blocks,
            self.out_structure(),
            self._get_transposed_subscripts(self.subscripts),
        )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        einsum = ft.partial(jnp.einsum, self.subscripts)
        if is_leaf(self.blocks):
            return jax.tree.map(
                lambda leaf: jax.eval_shape(einsum, self.blocks, leaf), self.in_structure()
            )

        return jax.tree.map(
            lambda block, leaf: jax.eval_shape(einsum, block, leaf),
            self.blocks,
            self.in_structure(),
        )

    @staticmethod
    def _parse_subscripts(subscripts: str) -> tuple[str, str, str]:
        split_subscripts = subscripts.split(',')
        if len(split_subscripts) != 2:
            raise ValueError(f'There should be a single comma in the subscripts: {subscripts!r}."')
        left_subscripts, subscripts = split_subscripts
        split_subscripts = subscripts.split('->')
        if len(split_subscripts) != 2:
            raise ValueError('Explicit mode (with `->) is required for the einsum subscripts.')
        right_subscripts, result_subscripts = split_subscripts
        return left_subscripts, right_subscripts, result_subscripts

    @staticmethod
    def _get_transposed_subscripts(subscripts: str) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns the einsum subscripts for the transpose operation.

        Examples:
            ij...,j...->i...    gives ji...,j...->i...
            hij...,hj...->hi... gives hji...,hj...->hi...
            ikj,kj->ki          gives jki,kj->ki
        """
        lefts, rights, results = DenseBlockDiagonalOperator._parse_subscripts(subscripts)
        lefts_as_set = set(lefts.replace('...', ''))
        rights_as_set = set(rights.replace('...', ''))
        results_as_set = set(results.replace('...', ''))

        # the sum axis is in the subscripts left and right but not in result
        sum_axis_as_set = lefts_as_set & rights_as_set - results_as_set
        if len(sum_axis_as_set) != 1:
            raise ValueError(f'The summation should be performed in one axis {subscripts!r}.')
        sum_axis = sum_axis_as_set.pop()

        # the transpose axis is in the subscripts left and result but not in right
        transpose_axis_as_set = lefts_as_set & results_as_set - rights_as_set
        if len(transpose_axis_as_set) == 0:
            raise ValueError(f'No transposition axis has been specified {subscripts!r}.')
        if len(transpose_axis_as_set) > 1:
            raise ValueError(f'Several transposition axes have been specified: {subscripts!r}.')
        transpose_axis = transpose_axis_as_set.pop()

        # we swap the transpose and sum axes
        sum_axis_number = lefts.index(sum_axis)
        transpose_axis_number = lefts.index(transpose_axis)
        lefts_as_list = list(lefts)
        lefts_as_list[sum_axis_number] = transpose_axis
        lefts_as_list[transpose_axis_number] = sum_axis
        lefts = ''.join(lefts_as_list)

        transpose_axis_number = results.index(transpose_axis)
        results_as_list = list(results)
        results_as_list[transpose_axis_number] = sum_axis
        expected_results = ''.join(results_as_list)
        if expected_results != rights:
            raise ValueError(
                f'The dimensions of the inputs {rights!r} cannot be reordered '
                f'into {expected_results!r}.'
            )

        return f'{lefts},{rights}->{results}'
