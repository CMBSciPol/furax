"""Direct solver for operators that couple only the same position of different leaves."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Num, PyTree

from furax import AbstractLinearOperator

__all__ = ['BlockDiagonalResult', 'BlockDiagonalSolver']


class BlockDiagonalResult(NamedTuple):
    """Result of the block-diagonal solver.

    Attributes:
        solution: The solution x to Ax = b.
    """

    solution: PyTree[Num[Array, '...']]


@dataclass(frozen=True)
class BlockDiagonalSolver:
    r"""Direct solver for operators that couple only the same position of different leaves.

    The operator acts on a pytree whose $n$ leaves all have the same shape, and the entry at a
    given position of an output leaf depends only on the entries at the same position of the input
    leaves. The operator is then made of independent $n \times n$ blocks, one per position. This is
    the case of the normal matrix $A^T N^{-1} A$ of parametric component separation without beams,
    with one block per pixel and Stokes parameter, of size the number of components.

    The blocks are recovered with one application of the operator per leaf and inverted directly,
    so the cost does not depend on their conditioning, unlike [`CGSolver`][]. The structure is not
    checked: an operator that couples different positions, for example through a beam, gives a
    wrong solution without error.

    Pass it where a lineax solver is accepted, e.g. `A.I(solver=BlockDiagonalSolver())` or
    `Config(solver=BlockDiagonalSolver())`. The initial guess and the preconditioner are ignored.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax import TreeOperator
        >>> from furax.tree import as_structure
        >>> from furax.linalg import BlockDiagonalSolver
        >>> b = {'a': jnp.array([3., 6.]), 'b': jnp.array([2., 4.])}
        >>> A = TreeOperator(
        ...     {
        ...         'a': {'a': jnp.array([2., 4.]), 'b': jnp.array([1., 2.])},
        ...         'b': {'a': jnp.array([1., 2.]), 'b': jnp.array([1., 2.])},
        ...     },
        ...     in_structure=as_structure(b),
        ... )
        >>> x = BlockDiagonalSolver()(A, b).solution
        >>> x['a'].tolist(), x['b'].tolist()
        ([1.0, 1.0], [1.0, 1.0])
    """

    def __call__(
        self,
        A: AbstractLinearOperator,
        b: PyTree[Num[Array, '...']],
        x0: PyTree[Num[Array, '...']] | None = None,
        *,
        preconditioner: AbstractLinearOperator | None = None,
    ) -> BlockDiagonalResult:
        """Solve `A x = b` by inverting the blocks of `A`."""
        del x0, preconditioner
        leaves, treedef = jax.tree.flatten(A.in_structure)
        columns = []
        for column in range(len(leaves)):
            # unit probe on one leaf: its image holds this column of every block
            probe = [jnp.zeros(leaf.shape, leaf.dtype) for leaf in leaves]
            probe[column] = jnp.ones(leaves[column].shape, leaves[column].dtype)
            image = jax.tree.leaves(A(jax.tree.unflatten(treedef, probe)))
            columns.append(jnp.stack(image, axis=-1))
        blocks = jnp.stack(columns, axis=-1)  # (..., n_out, n_in)
        rhs = jnp.stack(jax.tree.leaves(b), axis=-1)
        x = jnp.einsum('...ij,...j->...i', jnp.linalg.inv(blocks), rhs)
        solution = jax.tree.unflatten(treedef, [x[..., i] for i in range(len(leaves))])
        return BlockDiagonalResult(solution)
