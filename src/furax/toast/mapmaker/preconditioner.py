import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

from furax._base.dense import DenseBlockDiagonalOperator
from furax.landscapes import StokesPyTree
from furax.operators import AbstractLinearOperator, symmetric
from furax.tree import is_leaf, zeros_like


@symmetric
class BJPreconditioner(DenseBlockDiagonalOperator):
    """Class representing a block-diagonal Jacobi preconditioner.

    Each block is a small (n,n) matrix where n is the number of Stokes parameters of the map.
    Any other axes must be diagonal and must be placed "on the left".
    """

    def __init__(self, blocks: Inexact[Array, '...'], in_structure: jax.ShapeDtypeStruct) -> None:
        # initialize with non-diagonal axes on the right
        super().__init__(blocks, in_structure, '...ij,...j->...i')

        # do not accept a pytree for the blocks
        if not is_leaf(self.blocks):
            raise NotImplementedError('This operator does not support having Pytrees as blocks.')

        # at least check that the blocks are square
        if not (shape := self.blocks.shape)[-1] == shape[-2]:
            raise ValueError('The blocks must be square in the last two dimensions.')

        # enforce that the operator acts on StokesPyTrees
        if not isinstance(self._in_structure, StokesPyTree):
            raise ValueError('The operator should act on StokesPyTrees (sky maps).')

    @classmethod
    def create(cls, op: AbstractLinearOperator) -> 'AbstractLinearOperator':
        """Creates the dense preconditioner from a symmetric operator acting on StokesPyTrees.

        The operator is assumed to be diagonal with respect with all dimensions of the pytree.
        """
        in_struct = op.in_structure()
        if not in_struct == op.out_structure():
            raise ValueError('Operator must be square.')

        # adapt AbstractLinearOperator.as_matrix() method
        in_pytree = zeros_like(in_struct)
        in_leaves_ref, in_treedef = jax.tree.flatten(in_pytree)

        stokes, shape = in_struct.stokes, in_struct.shape
        blocks = jnp.empty(shape + (n := len(stokes), n), dtype=op.out_promoted_dtype)

        for i, leaf in enumerate(in_leaves_ref):
            # looping through the Stokes parameters
            zeros = in_leaves_ref.copy()
            zeros[i] = jnp.ones_like(leaf)
            in_pytree = jax.tree.unflatten(in_treedef, zeros)
            out_pytree = op.mv(in_pytree)
            out_leaves = jax.tree.leaves(out_pytree)
            # we have extracted column i of each block
            blocks = blocks.at[..., i, :].set(jnp.stack(out_leaves, axis=-1))

        return cls(blocks, in_struct)

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array, '...']:
        in_leaves, treedef = jax.tree.flatten(x)
        # the "stokes" dimension must be on the right
        y = jnp.einsum(self.subscripts, self.blocks, jnp.stack(in_leaves, axis=-1))
        out_leaves = jnp.split(y, treedef.num_leaves, axis=-1)  # type: ignore[attr-defined]
        # squeeze the last dimension to match input shape
        return jax.tree.unflatten(treedef, [jnp.squeeze(leaf, axis=-1) for leaf in out_leaves])

    # def solve(self, b: Inexact[Array, ' *shape']) -> Inexact[Array, ' *shape']:
    #     if len(b.shape) == 2:
    #         # avoid warning "jnp.linalg.solve: batched 1D solves with b.ndim > 1 are deprecated"
    #         x = jnp.linalg.solve(self.blocks, b[..., None])
    #         return x[..., 0]  # type: ignore[no-any-return]
    #     return jnp.linalg.solve(self.blocks, b)  # type: ignore[no-any-return]

    def inverse(self) -> 'AbstractLinearOperator':
        # compute the inverse blocks
        inv_blocks = jnp.linalg.inv(self.blocks)

        # set preconditioner to zero for pixels that are not observed at all
        hits = self.blocks[..., 0, 0]
        inv_blocks = inv_blocks.at[hits == 0].set(0)

        # check the condition number and set bad pixels to zero
        # they must be handled appropriately in the acquisition model
        cond = jnp.linalg.cond(self.blocks)
        inv_blocks = inv_blocks.at[cond > 1e1].set(0)

        return BJPreconditioner(inv_blocks, self._in_structure)
