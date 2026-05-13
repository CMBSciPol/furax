"""Low-rank approximation for PyTree-aware linear operators."""

from typing import Any, Literal, NamedTuple, TypeAlias, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Num, PRNGKeyArray, PyTree

from furax.core import AbstractLinearOperator
from furax.core._base import symmetric
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
    method: LowRankMethod = 'lanczos',
    **kwargs: Any,
) -> LowRankTerms:
    """Compute a low-rank approximation of a Hermitian operator.

    The approximation is of the form A ≈ U @ diag(S) @ U^T, where U contains
    the k eigenvectors and S contains the corresponding eigenvalues.

    Args:
        A: A Hermitian linear operator.
        rank: Number of eigenvalues/eigenvectors to compute.
        key: Random key for initialization of the eigenvalue solver.
        method: Eigenvalue solver to use: 'lanczos' or 'lanczos_tr'.
        **kwargs: Additional keyword arguments passed to the solver
            (e.g., m, max_restarts, tol for lanczos_tr).

    Returns:
        LowRankTerms containing eigenvalues and eigenvectors.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, rank=2, key=jax.random.PRNGKey(0), method='lanczos_tr')
        >>> terms.eigenvalues  # Should be approximately [1, 2]
        Array([1., 2.], dtype=float32)
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

    Example:
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

    # Compute coeffs = U^T @ x (shape: (k,))
    coeffs = jax.vmap(dot, (0, None), 0)(terms.eigenvectors, x)

    # Scale by eigenvalues: S * coeffs
    scaled = terms.eigenvalues * coeffs  # (k,)

    # Compute U @ scaled_coeffs = sum_i scaled[i] * u_i
    def compute_result(u_leaf: Array) -> Array:
        # u_leaf: (k, ...), scaled: (k,)
        # Result: sum over k of scaled[i] * u_leaf[i]
        return jnp.tensordot(scaled, u_leaf, axes=(0, 0))

    return jax.tree.map(compute_result, terms.eigenvectors)


@symmetric
class LowRankOperator(AbstractLinearOperator):
    """Linear operator from low-rank terms: A = U @ diag(S) @ U^T.

    This wraps a LowRankTerms as an AbstractLinearOperator, enabling
    composition with other operators.

    Args:
        terms: Low-rank terms containing eigenvalues and eigenvectors.
        in_structure: The expected structure of the operator input. If None,
            inferred from the eigenvectors structure.

    Example:
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
