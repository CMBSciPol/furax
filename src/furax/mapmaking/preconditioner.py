from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from furax import AbstractLinearOperator, TreeOperator, symmetric
from furax.obs.stokes import Stokes
from furax.tree import _tree_to_dense, zeros_like


@symmetric
class BJPreconditioner(TreeOperator):
    """Class representing a block-diagonal Jacobi preconditioner."""

    def __init__(
        self,
        tree: PyTree[PyTree[Any]],
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        super().__init__(tree, in_structure=in_structure)

        # Check that we have a (square) Stokes-pytree of Stokes-pytrees
        pytree_is_stokes = isinstance(tree, Stokes)
        subtrees_are_the_same_stokes = jax.tree.map(
            lambda x: isinstance(x, type(tree)), tree, is_leaf=lambda x: x is not tree
        )
        if not (pytree_is_stokes and subtrees_are_the_same_stokes):
            raise ValueError('tree must be a square Stokes-pytree matrix')

    @classmethod
    def create(cls, op: AbstractLinearOperator) -> 'BJPreconditioner':
        """Creates the dense preconditioner from a symmetric operator acting on Stokes pytrees.

        The operator is assumed to be diagonal with respect with all dimensions of the pytree.
        """
        # Check the input and output structure of the operator
        in_struct = op.in_structure()
        if not isinstance(in_struct, Stokes):
            raise ValueError('operator must act on Stokes pytrees (sky maps)')
        if not in_struct == op.out_structure():
            raise ValueError('operator must be square')

        # Create the preconditioner by evaluating the operator
        in_pytree = zeros_like(in_struct)
        in_leaves_ref, in_treedef = jax.tree.flatten(in_pytree)

        stokes = in_struct.stokes
        tree_cls = Stokes.class_for(stokes)

        reduced_op = op.reduce()
        f = jax.jit(lambda x: reduced_op.mv(x))

        def _compute_for_stokes(i):  # type: ignore[no-untyped-def]
            zeros = in_leaves_ref.copy()
            zeros[i] = jnp.ones_like(in_leaves_ref[i])
            in_pytree = jax.tree.unflatten(in_treedef, zeros)
            return f(in_pytree)

        tree = tree_cls(**{stoke: _compute_for_stokes(i) for i, stoke in enumerate(stokes.lower())})
        return cls(tree, in_structure=in_struct)

    def get_blocks(self) -> Array:
        """Convert the preconditioner blocks as a dense matrix."""
        return _tree_to_dense(self.outer_treedef, self.inner_treedef, self.tree)
