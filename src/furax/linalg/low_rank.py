"""Low-rank approximation for PyTree-aware linear operators."""

from typing import Any, Literal, NamedTuple, TypeAlias, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Num, PRNGKeyArray, PyTree

from furax import AbstractLinearOperator, symmetric
from furax.tree import dot, normal_like

from ._lanczos import lanczos_eigh, lanczos_tr

LowRankMethod: TypeAlias = Literal['lanczos', 'lanczos_tr']


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
    key: PRNGKeyArray,
    *,
    method: LowRankMethod = 'lanczos_tr',
    **kwargs: Any,
) -> LowRankTerms:
    """Compute a low-rank approximation of a Hermitian operator.

    The approximation is of the form A ≈ U @ diag(S) @ U^T, where U contains
    the k eigenvectors and S contains the corresponding eigenvalues.

    Two solvers are available via ``method``:

    - 'lanczos': single-shot m-step Lanczos. Builds one Krylov subspace and returns the k
      best-converged Ritz pairs by residual norm.
      Extra kwargs: ``m``.
    - 'lanczos_tr' (default): thick-restart Lanczos. Restarts until convergence and targets
      specific eigenpairs via ``which`` ('LM', 'SM', 'LA', 'SA', 'BE'; default 'LM').
      Extra kwargs: ``m``, ``which``, ``max_restarts``, ``tol``.

    Args:
        A: A Hermitian linear operator.
        rank: Number of eigenvalues/eigenvectors to compute.
        key: Random key for initialization of the eigenvalue solver.
        method: Eigenvalue solver to use (see above). Defaults to 'lanczos_tr'.
        **kwargs: Additional keyword arguments passed to the selected solver.

    Returns:
        LowRankTerms containing eigenvalues and eigenvectors.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, rank=2, key=jax.random.PRNGKey(0), method='lanczos_tr')
        >>> terms.eigenvalues  # Should be approximately [4, 5] (which='LM' by default)
        Array([4., 5.], dtype=float32)
    """
    solvers = {'lanczos': lanczos_eigh, 'lanczos_tr': lanczos_tr}
    if method not in solvers:
        raise ValueError(f'Unknown method: {method!r}. Use one of {get_args(LowRankMethod)}.')
    v0 = normal_like(A.in_structure, key)
    result = solvers[method](A, v0, k=rank, **kwargs)
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
        >>> terms = low_rank(A, rank=5, key=jax.random.PRNGKey(0))  # Full rank
        >>> x = jnp.array([1., 0., 0., 0., 0.])
        >>> y = low_rank_mv(terms, x)
        >>> # Should be close to A @ x = [1, 0, 0, 0, 0]
    """
    U = terms.eigenvectors
    coeffs = jax.vmap(dot, (0, None), 0)(U, x)  # U^T x
    scaled = terms.eigenvalues * coeffs  # S (U^T x)
    return jax.tree.map(lambda a: jnp.einsum('k,k...->...', scaled, a), U)  # U (S U^T x)


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
        >>> terms = low_rank(A, rank=2, key=jax.random.PRNGKey(0))
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
