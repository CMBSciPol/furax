"""Randomized Nyström approximation and preconditioner for PyTree-aware linear operators."""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Key

from furax import tree
from furax.core import AbstractLinearOperator, IdentityOperator
from furax.tree import _svqb, _svqb_coefficients

from ._dense import _eigh_live, _hermitian_part, _matmul
from .low_rank import LowRankOperator, LowRankTerms


def randomized_nystrom(
    A: AbstractLinearOperator,
    rank: int,
    *,
    key: Key[Array, ''],
    oversampling: int = 10,
) -> LowRankTerms:
    r"""Low-rank approximation of a positive semi-definite operator by randomized Nyström.

    Applies $A$ to $\ell$ = `rank + oversampling` random orthonormal test vectors $\Omega$ and
    returns the eigenpairs of the Nyström approximation
    $A \approx (A \Omega) (\Omega^H A \Omega)^{-1} (A \Omega)^H$, truncated to `rank`. The
    approximation is accurate for the largest eigenvalues, and improves with the oversampling
    and the decay of the spectrum.

    Each of the $\ell$ products with $A$ is independent: unlike a Krylov method, the cost is a
    single pass over $A$.

    Note:
        The method is the numerically stable Nyström approximation of Tropp et al. (2017), as
        stated by Frangella, Tropp & Udell (2023, Algorithm 2), which sketches $A + \nu I$ for a
        small shift $\nu$ and subtracts $\nu$ from the eigenvalues. It differs as follows.

        - The shift is $\nu = \epsilon \|A \Omega\|_F$, the order of the rounding errors of
          $\Omega^H A \Omega$, instead of $\sqrt{n}\, \epsilon \|A \Omega\|_2$. The shift adds
          a flat spectrum of trace $n \nu$ to the sketched operator, which with the larger
          value degrades the approximation of large single-precision operators: about 10% on
          the largest eigenvalue for $n = 3 \times 10^6$.
        - $\nu$ is kept positive for a zero operator, which otherwise gives NaN.
        - The SVD of $B = A \Omega\, L^{-H}$, with $\Omega^H (A + \nu I) \Omega = L L^H$, is
          replaced by Gram-matrix operations, so the vectors may be sharded: an orthonormal
          basis $Q$ of the span of $B$ (SVQB), then the eigendecomposition of the small matrix
          $(Q^H B)(Q^H B)^H$. Using $B^H B$ instead would square the condition number of $B$,
          and leave the eigenvectors of the eigenvalues clipped to 0 far from orthonormal.

    Args:
        A: A Hermitian positive semi-definite linear operator.
        rank: Number of eigenpairs to return, at most the operator size.
        key: Random key for the test vectors.
        oversampling: Number of test vectors beyond `rank`. The total is capped at the operator
            size.

    Returns:
        [`LowRankTerms`][furax.linalg.LowRankTerms] with the approximate eigenvalues, sorted
        ascending, and orthonormal eigenvectors. When the rank of $A$ is below `rank`, some
        eigenvalues are 0 and their eigenvectors may be zero instead.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> from furax.tree import as_structure
        >>> d = jnp.array([1., 2., 3., 4., 5.])
        >>> A = DiagonalOperator(d, in_structure=as_structure(d))
        >>> terms = randomized_nystrom(A, rank=2, key=jax.random.key(0))
        >>> terms.eigenvalues.round(4)
        Array([4., 5.], dtype=float32)
    """
    n = A.in_size
    if not 1 <= rank <= n:
        raise ValueError(f'rank ({rank}) must be between 1 and the operator size ({n})')
    n_samples = min(rank + oversampling, n)

    # A single orthonormalization pass is enough for a Gaussian block, which is well conditioned.
    Omega = _svqb(tree.stacked_normal_like(A.in_structure, n_samples, key))
    Y = tree.stacked_map(A, Omega)
    # Sketch A + ν I, so that Ωᴴ Y is positive definite in spite of rounding (see the note).
    finfo = jnp.finfo(jax.tree.leaves(Y)[0].dtype)
    nu = jnp.maximum(finfo.eps * jnp.linalg.norm(tree.stacked_norm(Y)), finfo.tiny)
    Y = tree.add(Y, tree.mul(nu, Omega))

    # With Ωᴴ Y = L Lᴴ, the approximation is B Bᴴ for B = Y L⁻ᴴ. B is formed explicitly: products
    # such as L⁻¹ Yᴴ Y L⁻ᴴ would amplify the rounding errors of Yᴴ Y by up to ||A|| / ν.
    L = jnp.linalg.cholesky(_hermitian_part(tree.stacked_gram(Omega, Y)))
    identity = jnp.eye(n_samples, dtype=L.dtype)
    B = tree.stacked_combine(Y, jax.scipy.linalg.solve_triangular(L.conj().T, identity))

    # The eigenvectors of B Bᴴ are Q W, with C Cᴴ = W Σ² Wᴴ for C = Qᴴ B and Q an orthonormal
    # basis of the span of B. Q = Q₁ T is obtained by two SVQB passes, the second one restoring the
    # orthonormality that the first loses for an ill-conditioned B; Q itself is not formed, only
    # Q₁ and the eigenvectors Q₁ T W. Directions lost to the rank deficiency of B, of zero
    # columns in T, are never selected.
    Q1 = _svqb(B)
    T, live = _svqb_coefficients(tree.stacked_gram(Q1, Q1))
    C = _matmul(T.conj().T, tree.stacked_gram(Q1, B))
    sigma2, W = _eigh_live(_hermitian_part(_matmul(C, C.conj().T)), live, largest=True)
    sigma2, W = sigma2[-rank:], W[:, -rank:]
    return LowRankTerms(
        eigenvalues=jnp.maximum(sigma2 - nu, 0),
        eigenvectors=tree.stacked_combine(Q1, _matmul(T, W)),
    )


def make_nystrom_preconditioner(terms: LowRankTerms, *, mu: float = 0.0) -> AbstractLinearOperator:
    r"""Nyström preconditioner for $A + \mu I$, from a low-rank approximation of $A$.

    With the eigenpairs $(\Lambda, U)$ of a Nyström approximation of $A$, e.g. from
    [`randomized_nystrom`][furax.linalg.randomized_nystrom], the preconditioner of Frangella,
    Tropp & Udell (2023) is

    $$
    P^{-1} = (\lambda_\text{min} + \mu)\, U (\Lambda + \mu I)^{-1} U^H + (I - U U^H),
    $$

    where $\lambda_\text{min}$ is the smallest eigenvalue of $\Lambda$. It maps the
    approximated eigenvalues of $A + \mu I$ to $\lambda_\text{min} + \mu$ and leaves the
    complement of $U$ unchanged, so that the preconditioned operator
    $P^{-1} (A + \mu I)$ has a condition number close to $(\lambda_\text{min} + \mu) / \mu$ for a
    rank large enough to capture the eigenvalues above $\lambda_\text{min}$.

    Eigenvalues of $\Lambda$ that are zero up to rounding (below $\epsilon \max \Lambda$, as
    [`randomized_nystrom`][furax.linalg.randomized_nystrom] returns for an operator of rank below
    the one requested) carry no information on $A$: their eigenvectors are left in the
    complement, and $\lambda_\text{min}$ is the smallest of the other eigenvalues.

    Args:
        terms: Eigenvalues $\Lambda$ and orthonormal eigenvectors $U$ of the approximation of
            $A$.
        mu: Regularization $\mu \ge 0$.

    Returns:
        The Hermitian positive definite operator $P^{-1}$, to be applied as the preconditioner
        of a solve with $A + \mu I$.
    """
    eigenvalues, U = terms.eigenvalues, terms.eigenvectors
    captured = eigenvalues > jnp.finfo(eigenvalues.dtype).eps * jnp.max(eigenvalues)
    lambda_min = jnp.min(jnp.where(captured, eigenvalues, jnp.inf))
    safe = jnp.where(captured, eigenvalues, 1)
    scale = jnp.where(captured, (lambda_min + mu) / (safe + mu) - 1, 0)
    correction = LowRankOperator(LowRankTerms(scale, U))
    return IdentityOperator(in_structure=correction.in_structure) + correction
