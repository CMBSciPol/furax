import functools as ft
from dataclasses import field

import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Inexact, PyTree

from furax.tree import is_leaf

from ._base import AbstractLinearOperator


class DenseBlockDiagonalOperator(AbstractLinearOperator):
    """Operator that applies block diagonal dense matrices via einsum.

    Only the diagonal blocks are stored, making this more memory-efficient than
    a full dense matrix. The operation is defined by einsum subscripts.

    Attributes:
        blocks: The dense blocks as an array (at least 2D).
        subscripts: Einsum subscripts defining the operation (default: 'ij...,j...->i...').

    Examples:
        For a matrix made of three 2x4 diagonal blocks, and input block columns of three blocks of
        four elements each, the operator can be written as:

        >>> blocks = jnp.arange(24).reshape(3, 2, 4)
        >>> op = DenseBlockDiagonalOperator(
        ...     blocks, in_structure=jax.ShapeDtypeStruct((3, 4), jnp.int32), subscripts='imn,in->im'
        ... )
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
        >>> op = DenseBlockDiagonalOperator(
        ...     blocks, in_structure=jax.ShapeDtypeStruct((2, 4), jnp.int32)
        ... )
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

    blocks: Inexact[Array, '...']
    subscripts: str = field(default='ij...,j...->i...', metadata={'static': True})

    def __post_init__(self) -> None:
        subscripts = self.subscripts.replace(' ', '')
        if subscripts != self.subscripts:
            object.__setattr__(self, 'subscripts', subscripts)

        if not jax.tree.all(jax.tree.map(lambda leaf: len(leaf.shape) >= 2, self.blocks)):
            raise ValueError('The blocks should at least have 2 dimensions.')
        self._parse_subscripts(subscripts)

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
            blocks=self.blocks,
            in_structure=self.out_structure,
            subscripts=self._get_transposed_subscripts(self.subscripts),
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
    def _get_transposed_subscripts(subscripts: str) -> str:
        """Returns the einsum subscripts for the transpose operation.

        The transpose of `einsum('L,R->O', blocks, x)` is `einsum('L,O->R', blocks, y)`: the blocks
        are reused as they are, and the roles of the input and output subscripts are swapped.

        Examples:
            ij...,j...->i...     gives ij...,i...->j...
            hij...,hj...->hi...  gives hij...,hi...->hj...
            fstqp,tfp->sfq       gives fstqp,sfq->tfp
        """
        lefts, rights, results = DenseBlockDiagonalOperator._parse_subscripts(subscripts)
        rights_as_list = list(rights.replace('...', ''))
        if len(set(rights_as_list)) != len(rights_as_list):
            raise ValueError(f'The input subscripts should not be repeated: {subscripts!r}.')

        # an input axis summed over without the blocks cannot be restored by the transpose,
        # which would have to broadcast along it
        missing_axes = set(rights_as_list) - set(lefts) - set(results)
        if '...' in rights and '...' not in lefts + results:
            missing_axes.add('...')
        if missing_axes:
            raise ValueError(
                f'The input axes {sorted(missing_axes)} are neither in the blocks nor in the '
                f'output, so the transpose cannot be expressed as an einsum: {subscripts!r}.'
            )

        return f'{lefts},{results}->{rights}'
