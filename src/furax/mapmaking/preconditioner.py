from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from furax import AbstractLinearOperator, TreeOperator, symmetric
from furax.obs.stokes import Stokes
from furax.tree import _dense_to_tree, _get_outer_treedef, _tree_to_dense, zeros_like


@symmetric
class BJPreconditioner(TreeOperator):
    """Class representing a block-diagonal Jacobi preconditioner."""

    def __init__(
        self,
        tree: PyTree[PyTree[Any]],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):

        inner_treedef = jax.tree.structure(in_structure)
        outer_treedef = _get_outer_treedef(in_structure, tree)
        object.__setattr__(self, 'tree', tree)
        object.__setattr__(self, 'in_structure', in_structure)
        object.__setattr__(self, 'inner_treedef', inner_treedef)
        object.__setattr__(self, 'outer_treedef', outer_treedef)

        # Check that we have a (square) Stokes-pytree of Stokes-pytrees
        pytree_is_stokes = isinstance(tree, Stokes)
        subtrees_are_the_same_stokes = jax.tree.map(
            lambda x: isinstance(x, type(tree)),
            tree,
            is_leaf=lambda x: x is not tree,
        )
        if not (pytree_is_stokes and subtrees_are_the_same_stokes):
            raise ValueError('tree must be a square Stokes-pytree matrix')

    @classmethod
    def create(cls, op: AbstractLinearOperator) -> 'BJPreconditioner':
        """Creates the dense preconditioner from a symmetric operator acting on Stokes pytrees.

        The operator is assumed to be diagonal with respect with all dimensions of the pytree.
        """
        # Check the input and output structure of the operator
        in_struct = op.in_structure
        if not isinstance(in_struct, Stokes):
            raise ValueError('operator must act on Stokes pytrees (sky maps)')
        if not in_struct == op.out_structure:
            raise ValueError('operator must be square')

        # Create the preconditioner by evaluating the operator
        basis = zeros_like(in_struct)
        basis_leaves, treedef = jax.tree.flatten(basis)
        n = len(basis_leaves)
        stokes = in_struct.stokes
        tree_cls = Stokes.class_for(stokes)

        out_leaves = []
        for j in range(n):
            probe_leaves = [
                jnp.ones_like(leaf) if i == j else jnp.zeros_like(leaf)
                for i, leaf in enumerate(basis_leaves)
            ]
            probe = jax.tree.unflatten(treedef, probe_leaves)
            result = op(probe)
            out_leaves.append(jax.tree.leaves(result))

        tree = tree_cls(
            **{
                stoke: jax.tree.unflatten(treedef, [out_leaves[j][i] for j in range(n)])
                for i, stoke in enumerate(stokes.lower())
            }
        )
        return cls(tree, in_structure=in_struct)

    def inverse(self) -> 'BJPreconditioner':
        # Override the parent inverse so the result stays a BJPreconditioner and
        # keeps the @symmetric tag; the generic TreeOperator inverse would
        # downgrade to a plain operator.
        dense = _tree_to_dense(self.outer_treedef, self.inner_treedef, self.tree)
        dense_inv = jnp.linalg.inv(dense)
        tree = _dense_to_tree(self.inner_treedef, self.outer_treedef, dense_inv)
        return BJPreconditioner(tree, in_structure=self.in_structure)

    def get_blocks(self) -> Array:
        """Convert the preconditioner blocks as a dense matrix."""
        return _tree_to_dense(self.outer_treedef, self.inner_treedef, self.tree)
