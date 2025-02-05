import equinox
import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.obs.stokes import Stokes
from furax.tree import is_leaf, zeros_like


@symmetric
class BJPreconditioner(AbstractLinearOperator):
    """Class representing a block-diagonal Jacobi preconditioner.

    Each block is a small (n,n) matrix where n is the number of Stokes parameters of the map.
    Any other axes must be diagonal and must be placed "on the left".
    """

    blocks: Inexact[Array, '...']
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, blocks: Inexact[Array, '...'], in_structure: jax.ShapeDtypeStruct) -> None:
        # Impose some restrictions on the input
        # 1. the blocks must be an Array of shape (..., n, n)
        # 2. the input structure must be a Stokes pytree

        if not is_leaf(blocks):
            raise NotImplementedError('This operator does not support having Pytrees as blocks.')

        match blocks.shape:
            case (*_, m, n) if m == n:
                pass
            case _:
                raise ValueError('The blocks must be an Array of shape (..., n, n).')

        if not isinstance(in_structure, Stokes):
            raise ValueError('The operator should act on Stokes pytrees (sky maps).')

        self.blocks = blocks
        self._in_structure = in_structure

    @classmethod
    def create(cls, op: AbstractLinearOperator) -> 'AbstractLinearOperator':
        """Creates the dense preconditioner from a symmetric operator acting on Stokes pytrees.

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

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array, '...']:
        in_leaves, treedef = jax.tree.flatten(x)
        # the "stokes" dimension must be on the right
        y = jnp.matvec(self.blocks, jnp.stack(in_leaves, axis=-1))
        out_leaves = jnp.split(y, treedef.num_leaves, axis=-1)  # type: ignore[attr-defined]
        # squeeze the last dimension to match input shape
        return jax.tree.unflatten(treedef, [jnp.squeeze(leaf, axis=-1) for leaf in out_leaves])

    def inverse(self) -> 'AbstractLinearOperator':
        # A pseudo-inverse of each block is computed

        # This avoids numerical issues with ill-conditioned pixels
        # BUT does not mean that those pixels will be estimated correctly by the solver
        pinv_blocks = jnp.linalg.pinv(self.blocks)
        return BJPreconditioner(pinv_blocks, self._in_structure)
