from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

import furax.linalg
from furax import AbstractLinearOperator, TreeOperator, symmetric
from furax.core import AbstractLazyInverseOperator
from furax.obs.stokes import Stokes
from furax.tree import _dense_to_tree, _get_outer_treedef, _tree_to_dense, zeros_like

# Pass op as an explicit argument so JAX traces its arrays as inputs rather than
# capturing them as XLA constants (which would happen with jit(op.mv) or jit(op)).
_apply = jax.jit(lambda op, x: op(x))


@symmetric
class CGInverseOperator(AbstractLazyInverseOperator):
    """Lazy ``A⁻¹`` realised by furax preconditioned conjugate gradient.

    Unlike :class:`furax.core.InverseOperator` (which solves via lineax
    ``linear_solve``), this uses :func:`furax.linalg.cg`, which is sharding-aware
    (works on map/TOD vectors distributed across observations) and records the
    per-iteration residual norm. ``A`` (``operator``) is assumed symmetric positive
    definite, so ``A⁻¹`` is symmetric — hence the ``@symmetric`` tag.

    Apply it like any operator (``A_inv(x)`` solves ``A y = x``). The optional
    ``preconditioner`` ``M ≈ A⁻¹`` (e.g. a block-Jacobi inverse) is passed straight
    to the CG solver.
    """

    preconditioner: AbstractLinearOperator | None = field(default=None, kw_only=True)
    rtol: float = field(default=1e-6, kw_only=True, metadata={'static': True})
    atol: float = field(default=0.0, kw_only=True, metadata={'static': True})
    max_steps: int = field(default=1_000, kw_only=True, metadata={'static': True})

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return furax.linalg.cg(
            self.operator,
            x,
            preconditioner=self.preconditioner,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        ).solution


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
            result = _apply(op, probe)
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
