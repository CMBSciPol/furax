"""Low-rank approximation for PyTree-aware linear operators."""

from typing import Any, Literal, NamedTuple

import equinox
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Num, PRNGKeyArray, PyTree

from furax.core import AbstractLinearOperator
from furax.core._base import symmetric

from ._lanczos import lanczos_eigh
from ._lobpcg import lobpcg_standard


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
    k: int,
    *,
    method: Literal['lanczos', 'lobpcg'] = 'lanczos',
    largest: bool = True,
    key: PRNGKeyArray | None = None,
    **solver_kwargs: Any,
) -> LowRankTerms:
    """Compute a low-rank approximation of a Hermitian operator.

    The approximation is of the form A ≈ U @ diag(S) @ U^T, where U contains
    the k eigenvectors and S contains the corresponding eigenvalues.

    Args:
        A: A Hermitian linear operator.
        k: Number of eigenvalues/eigenvectors to compute.
        method: Eigenvalue solver to use, either 'lanczos' or 'lobpcg'.
        largest: If True (default), compute the k largest eigenvalues.
            For low-rank approximations, largest eigenvalues typically
            capture most of the operator's action.
        key: Random key for initialization of the eigenvalue solver.
        **solver_kwargs: Additional keyword arguments passed to the solver
            (e.g., tol, max_iters for lobpcg; m for lanczos).

    Returns:
        LowRankTerms containing eigenvalues and eigenvectors.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> # Create operator with known eigenvalues [1, 2, 3, 4, 5]
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = low_rank(A, k=2, key=jax.random.PRNGKey(0))
        >>> terms.eigenvalues  # Should be approximately [4, 5]
        Array([4., 5.], dtype=float32)
    """
    if method == 'lanczos':
        lanczos_result = lanczos_eigh(A, k=k, largest=largest, key=key, **solver_kwargs)
        return LowRankTerms(
            eigenvalues=lanczos_result.eigenvalues,
            eigenvectors=lanczos_result.eigenvectors,
        )
    elif method == 'lobpcg':
        lobpcg_result = lobpcg_standard(A, k=k, largest=largest, key=key, **solver_kwargs)
        return LowRankTerms(
            eigenvalues=lobpcg_result.eigenvalues,
            eigenvectors=lobpcg_result.eigenvectors,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'lanczos' or 'lobpcg'.")


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
        >>> terms = low_rank(A, k=5, key=jax.random.PRNGKey(0))  # Full rank
        >>> x = jnp.array([1., 0., 0., 0., 0.])
        >>> y = low_rank_mv(terms, x)
        >>> # Should be close to A @ x = [1, 0, 0, 0, 0]
    """

    # Compute coeffs = U^T @ x (shape: (k,))
    # For each eigenvector u_i and input x, compute dot(u_i, x)
    def leaf_dots(u_leaf: Array, x_leaf: Array) -> Array:
        # u_leaf: (k, ...), x_leaf: (...)
        # Flatten trailing dims and compute dot for each of k vectors
        k = u_leaf.shape[0]
        u_flat = u_leaf.reshape(k, -1)  # (k, n)
        x_flat = x_leaf.ravel()  # (n,)
        return u_flat @ jnp.conj(x_flat)  # (k,)

    leaf_dots_list = jax.tree.leaves(jax.tree.map(leaf_dots, terms.eigenvectors, x))
    coeffs = leaf_dots_list[0]
    for leaf in leaf_dots_list[1:]:
        coeffs = coeffs + leaf

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
        >>> terms = low_rank(A, k=2, key=jax.random.PRNGKey(0))
        >>> B = LowRankOperator(terms)
        >>> x = jnp.ones(5)
        >>> y = B(x)  # Applies low-rank approximation
    """

    terms: LowRankTerms
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        terms: LowRankTerms,
        in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
    ):
        self.terms = terms
        if in_structure is None:
            # Infer structure from eigenvectors by removing leading k dimension
            self._in_structure = jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape[1:], leaf.dtype),
                terms.eigenvectors,
            )
        else:
            self._in_structure = in_structure

    def mv(self, x: PyTree[Num[Array, '...']]) -> PyTree[Num[Array, '...']]:
        return low_rank_mv(self.terms, x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure
