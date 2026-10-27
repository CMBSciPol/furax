"""Low-rank approximation for PyTree-aware linear operators."""

from typing import Any, Literal, NamedTuple, get_args

import jax
from jax import Array
from jaxtyping import Float, Key, Num, PyTree

from furax import AbstractLinearOperator, symmetric, tree

from ._lanczos import lanczos_eigh, lanczos_tr
from ._lobpcg import lobpcg_standard

LowRankMethod = Literal['lanczos', 'lanczos_tr', 'lobpcg', 'nystrom']


class LowRankTerms(NamedTuple):
    """Low-rank approximation of a Hermitian operator: A ≈ U @ diag(S) @ U^T.

    Attributes:
        eigenvalues: The k eigenvalues (S), shape (k,).
        eigenvectors: The k eigenvectors (U) as a block PyTree with leading dimension k.
    """

    eigenvalues: Float[Array, ' k']
    eigenvectors: PyTree[Num[Array, ' k ...']]


def low_rank(
    A: AbstractLinearOperator,
    rank: int,
    key: Key[Array, ''],
    *,
    method: LowRankMethod = 'lanczos_tr',
    **kwargs: Any,
) -> LowRankTerms:
    """Compute a low-rank approximation of a Hermitian operator.

    The approximation is of the form A ≈ U @ diag(S) @ U^T, where U contains
    the k eigenvectors and S contains the corresponding eigenvalues.

    The solvers available via `method` are:

    - `'lanczos'`: single-shot m-step Lanczos ([`lanczos_eigh`][furax.linalg.lanczos_eigh]).
      Builds one Krylov subspace and returns the k best-converged Ritz pairs by residual norm.
      Extra kwargs: `m`.
    - `'lanczos_tr'` (default): thick-restart Lanczos ([`lanczos_tr`][furax.linalg.lanczos_tr]).
      Restarts until convergence and targets specific eigenpairs via `which` (`'LM'`, `'SM'`,
      `'LA'`, `'SA'`, `'BE'`; default `'LM'`).
      Extra kwargs: `m`, `which`, `max_restarts`, `tol`, `preconditioner`. A `preconditioner`
      $M$ changes the operator: the eigenpairs are those of $M A$.
    - `'lobpcg'`: block preconditioned eigensolver
      ([`lobpcg_standard`][furax.linalg.lobpcg_standard]) from a random block of k vectors.
      Targets the largest eigenvalues, or the smallest with `largest=False`.
      Extra kwargs: `largest`, `maxiter`, `tol`, `preconditioner`. A `preconditioner` only
      accelerates convergence: the eigenpairs are those of $A$.
    - `'nystrom'`: randomized Nyström approximation
      ([`randomized_nystrom`][furax.linalg.randomized_nystrom]) of the largest eigenpairs, for a
      positive semi-definite operator, in a single pass of independent products with A.
      Extra kwargs: `oversampling`.

    Args:
        A: A Hermitian linear operator.
        rank: Number of eigenvalues/eigenvectors to compute.
        key: Random key for initialization of the eigenvalue solver.
        method: Eigenvalue solver to use (see above). Defaults to `'lanczos_tr'`.
        **kwargs: Additional keyword arguments passed to the selected solver.

    Returns:
        LowRankTerms containing eigenvalues and eigenvectors. Convergence information is not
        returned: to check it, call the selected solver directly.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, rank=2, key=jax.random.key(0), method='lanczos_tr')
        >>> terms.eigenvalues  # Should be approximately [4, 5] (which='LM' by default)
        Array([4., 5.], dtype=float32)
    """
    if method == 'nystrom':
        from ._nystrom import randomized_nystrom  # circular: _nystrom builds on LowRankTerms

        return randomized_nystrom(A, rank, key=key, **kwargs)
    if method == 'lobpcg':
        X = tree.stacked_normal_like(A.in_structure, rank, key)
        kwargs.setdefault('largest', True)
        result = lobpcg_standard(A, X, **kwargs)
    elif method in ('lanczos', 'lanczos_tr'):
        solver = lanczos_eigh if method == 'lanczos' else lanczos_tr
        result = solver(A, key=key, k=rank, **kwargs)
    else:
        raise ValueError(f'Unknown method: {method!r}. Use one of {get_args(LowRankMethod)}.')
    return LowRankTerms(
        eigenvalues=result.eigenvalues,
        eigenvectors=result.eigenvectors,
    )


def low_rank_mv(terms: LowRankTerms, x: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
    """Apply low-rank approximation as a matrix-vector product.

    Computes y = U @ diag(S) @ U^T @ x efficiently without forming the full matrix.

    Args:
        terms: Low-rank terms containing eigenvalues and eigenvectors.
        x: Input PyTree with structure matching the eigenvectors (without leading k dimension).

    Returns:
        Output PyTree y = U @ diag(S) @ U^T @ x.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, rank=5, key=jax.random.key(0))  # Full rank
        >>> x = jnp.array([1., 0., 0., 0., 0.])
        >>> y = low_rank_mv(terms, x)
        >>> # Should be close to A @ x = [1, 0, 0, 0, 0]
    """
    U = terms.eigenvectors
    coeffs = tree.stacked_dot(U, x)  # U^T x
    return tree.stacked_combine(U, terms.eigenvalues * coeffs)  # U (S U^T x)


@symmetric
class LowRankOperator(AbstractLinearOperator):
    """Linear operator from low-rank terms: A = U @ diag(S) @ U^T.

    This wraps a LowRankTerms as an AbstractLinearOperator, enabling
    composition with other operators.

    Args:
        terms: Low-rank terms containing eigenvalues and eigenvectors.
        in_structure: The expected structure of the operator input. If None,
            inferred from the eigenvectors structure.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, rank=2, key=jax.random.key(0))
        >>> B = LowRankOperator(terms)
        >>> x = jnp.ones(5)
        >>> y = B(x)  # Applies low-rank approximation
    """

    terms: LowRankTerms

    def __init__(
        self,
        terms: LowRankTerms,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
    ):
        if in_structure is None:
            # Infer structure from eigenvectors by removing leading dimension
            in_structure = jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype),
                terms.eigenvectors,
            )
        object.__setattr__(self, 'terms', terms)
        object.__setattr__(self, 'in_structure', in_structure)

    def mv(self, x: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
        return low_rank_mv(self.terms, x)
