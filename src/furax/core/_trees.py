from typing import Any

import equinox
import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import PyTreeDef
from jaxtyping import Inexact, PyTree

from ..tree import _dense_to_tree, _get_outer_treedef, _tree_to_dense, matmat, matvec
from ._base import AbstractLinearOperator
from .rules import AbstractBinaryRule


class TreeOperator(AbstractLinearOperator):
    """Operator applying a generalized matrix, where the matrix is a pytree of pytrees.

    Using a Tree Operator rather than a dense operator can be beneficial when the matrix structure
    allows for optimization, such as exploiting symmetries, shared elements, zero values, or
    broadcastable components. XLA will then apply Common Subexpression Elimination and Dead Code
    Elimination.

    The structure of the generalized matrix is the tree product of an outer tree structure
    whose leaves represent the rows and an inner tree structure whose leaves represents the columns.
    A tree product of two tree structures is formed by replacing each leaf of the first tree with a
    copy of the second. The leaves of the generalized matrix must be broadcastable when they belong
    to same inner tree (elements of the same row). There is no such requirement for leaves of
    different inner trees (elements of different rows).

    The inner structure of the generalized matrix is given by the `in_structure` argument.

    Attributes:
        outer_treedef: The PyTreeDef of the outer tree structure.
        inner_treedef: The PyTreeDef of the inner tree structure.
        tree_shape: The shape of the generalized matrix, defined by the number of leaves in
            the outer structure followed by those in the inner structure.

    Example:
        To represent the Mueller Matrix of a quarter-wave plate with a vertical fast-axis:

        >>> from furax.obs.stokes import StokesIQUV
        >>> op = TreeOperator(
        ...     StokesIQUV(
                    StokesIQUV(1, 0, 0,  0),
                    StokesIQUV(0, 1, 0,  0),
                    StokesIQUV(0, 0, 0, -1),
                    StokesIQUV(0, 0, 1,  0),
                ),
                in_structure=StokesIQUV.structure_for((), jnp.float32)
        ... )
        >>> op.as_matrix()
        Array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  0., -1.],
               [ 0.,  0.,  1.,  0.]], dtype=float32)
        >>> op(StokesIQUV(1., 1., 1., 1.))
        StokesIQUV(i=1.0, q=1.0, u=-1.0, v=1.0)
    """

    tree: PyTree[Array]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    inner_treedef: PyTreeDef = equinox.field(static=True)
    outer_treedef: PyTreeDef = equinox.field(static=True)

    def __init__(
        self,
        tree: PyTree[PyTree[Any]],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.tree = tree
        self.inner_treedef = jax.tree.structure(in_structure)
        self.outer_treedef = _get_outer_treedef(in_structure, tree)
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    @property
    def tree_shape(self) -> tuple[int, int]:
        """Return the number of leaves of the outer and inner structures."""
        return self.outer_treedef.num_leaves, self.inner_treedef.num_leaves

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return matvec(self.outer_treedef, self.tree, x)

    def transpose(self) -> AbstractLinearOperator:
        transposed_tree = jax.tree.transpose(self.outer_treedef, self.inner_treedef, self.tree)
        return TreeOperator(transposed_tree, in_structure=self.out_structure())

    def inverse(self) -> AbstractLinearOperator:
        dense = _tree_to_dense(self.outer_treedef, self.inner_treedef, self.tree)
        dense_pinv = jnp.linalg.pinv(dense)
        tree = _dense_to_tree(self.inner_treedef, self.outer_treedef, dense_pinv)
        return TreeOperator(tree, in_structure=self.out_structure())


class TreeMultiplicationRule(AbstractBinaryRule):
    """Binary rule for `tree_left @ tree_right."""

    left_operator_class = TreeOperator
    right_operator_class = TreeOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, TreeOperator)
        assert isinstance(right, TreeOperator)
        return [
            TreeOperator(
                matmat(left.outer_treedef, left.tree, right.outer_treedef, right.tree),
                in_structure=right.in_structure(),
            )
        ]
