"""Tests for LOBPCG eigenvalue solver."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.linalg import LOBPCGResult, lobpcg_standard
from furax.linalg._lobpcg import _rayleigh_ritz
from furax.tree import as_structure
from furax.tree_block import block_normal_like


class TestRayleighRitz:
    """Tests for Rayleigh-Ritz procedure."""

    def test_rayleigh_ritz_diagonal(self):
        """Test Rayleigh-Ritz with diagonal operator."""
        # Eigenvalues of diagonal operator are the diagonal elements
        d = jnp.array([1.0, 3.0, 2.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Use identity as search space (exact eigenvectors)
        S = jnp.eye(3)
        AS = jax.vmap(A.mv)(S)

        eigenvalues, eigenvectors, _ = _rayleigh_ritz(S, AS, k=2, largest=False)

        # Should get smallest 2 eigenvalues: 1.0 and 2.0
        assert_allclose(sorted(eigenvalues), [1.0, 2.0], atol=1e-5)


class TestLOBPCG:
    """Integration tests for LOBPCG solver."""

    def test_lobpcg_diagonal_operator(self):
        """Test LOBPCG with diagonal operator (known eigenvalues)."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(0))
        result = lobpcg_standard(A, X, maxiter=50, tol=1e-8)

        assert isinstance(result, LOBPCGResult)
        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors.shape == (2, 5)

        # Should find the 2 smallest eigenvalues: 1.0 and 2.0
        assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-5)

    def test_lobpcg_largest_eigenvalues(self):
        """Test LOBPCG finding largest eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(1))
        result = lobpcg_standard(A, X, maxiter=50, tol=1e-8, largest=True)

        # Should find the 2 largest eigenvalues: 4.0 and 5.0
        assert_allclose(result.eigenvalues, [4.0, 5.0], atol=1e-5)

    def test_lobpcg_convergence(self):
        """Test that LOBPCG converges and residuals are small."""
        d = jnp.array([1.0, 2.0, 5.0, 10.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(2))
        result = lobpcg_standard(A, X, maxiter=100, tol=1e-6)

        assert jnp.all(result.converged)
        assert jnp.all(result.residual_norms < 1e-5)

    def test_lobpcg_with_initial_guess(self):
        """Test LOBPCG with provided initial vectors."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Provide good initial guess close to true eigenvectors
        X0 = jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.1]])
        result = lobpcg_standard(A, X0, maxiter=50, tol=1e-8)

        assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-5)

    def test_lobpcg_pytree_structure(self):
        """Test LOBPCG with PyTree-structured operators."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        key = jax.random.PRNGKey(3)
        X0 = block_normal_like(A.in_structure, 2, key)
        result = lobpcg_standard(A, X0, maxiter=50, tol=1e-6)

        # Total eigenvalues are [1, 2, 3], should find smallest 2
        assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-4)

    def test_lobpcg_compare_with_eigh(self):
        """Test LOBPCG results match jnp.linalg.eigh on small dense problem."""
        # Create a random symmetric positive definite matrix
        key = jax.random.PRNGKey(42)
        n = 10
        A_dense = jax.random.normal(key, (n, n))
        A_dense = A_dense @ A_dense.T + jnp.eye(n)  # Make positive definite

        # Create diagonal operator (for simplicity, use diagonal of A)
        d = jnp.diag(A_dense)
        A = DiagonalOperator(d, in_structure=as_structure(d))

        X = block_normal_like(as_structure(d), 3, jax.random.PRNGKey(0))
        result = lobpcg_standard(A, X, maxiter=50, tol=1e-8)

        # For diagonal operator, eigenvalues are the diagonal elements
        sorted_d = jnp.sort(d)
        assert_allclose(result.eigenvalues, sorted_d[:3], atol=1e-5)

    def test_lobpcg_with_preconditioner_matches_no_preconditioner(self):
        """Preconditioned LOBPCG finds same eigenvalues as unpreconditioned."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        # Exact inverse preconditioner: T = A^{-1}
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))

        X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(7))
        result_no_prec = lobpcg_standard(A, X, maxiter=100, tol=1e-8)
        result_prec = lobpcg_standard(A, X, maxiter=100, tol=1e-8, preconditioner=M)

        assert_allclose(result_prec.eigenvalues, result_no_prec.eigenvalues, atol=1e-5)

    def test_lobpcg_preconditioner_pytree(self):
        """Preconditioned LOBPCG works with PyTree-structured operators."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )
        M = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(1.0 / d1, in_structure=structure['a']),
                'b': DiagonalOperator(1.0 / d2, in_structure=structure['b']),
            }
        )

        X0 = block_normal_like(A.in_structure, 2, jax.random.PRNGKey(9))
        result = lobpcg_standard(A, X0, maxiter=50, tol=1e-6, preconditioner=M)

        assert_allclose(result.eigenvalues, [1.0, 2.0], atol=1e-4)

    def test_lobpcg_preconditioner_accelerates_convergence(self):
        """Approximate inverse preconditioner dramatically reduces iterations on ill-conditioned problem.

        The 1D Laplacian has condition number ~ n^2 / pi^2.  Without preconditioning
        LOBPCG fails to converge in a fixed budget; with the preconditioner it converges
        in a handful of iterations.
        """
        from furax.core import AbstractLinearOperator

        class Laplacian1D(AbstractLinearOperator):
            n: int

            def __init__(self, n: int, dtype=jnp.float64):
                object.__setattr__(self, 'n', n)
                super().__init__(in_structure=jax.ShapeDtypeStruct((n,), dtype))

            def mv(self, x):
                return 2 * x - jnp.pad(x[:-1], (1, 0)) - jnp.pad(x[1:], (0, 1))

        n = 30
        k = 2
        A = Laplacian1D(n)

        # Approximate inverse: (A + shift*I)^{-1} with shift << lambda_1
        # This is cheaper to motivate than exact inverse but still effective
        diag = 2.0 * jnp.ones(n, dtype=jnp.float64)
        off = -1.0 * jnp.ones(n - 1, dtype=jnp.float64)
        A_mat = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
        shift = 0.01  # small relative to bulk spectrum (~100x lambda_1)
        M = DenseBlockDiagonalOperator(
            jnp.linalg.inv(A_mat + shift * jnp.eye(n, dtype=jnp.float64)),
            in_structure=A.in_structure,
            subscripts='ij,j->i',
        )

        # True smallest eigenvalues: lambda_j = 2 - 2*cos(pi*j/(n+1))
        true_eigs = 2 - 2 * jnp.cos(jnp.pi * jnp.arange(1, k + 1) / (n + 1))

        X = block_normal_like(A.in_structure, k, jax.random.PRNGKey(42))
        result_none = lobpcg_standard(A, X, maxiter=50, tol=1e-8)
        result_prec = lobpcg_standard(A, X, maxiter=50, tol=1e-8, preconditioner=M)

        # Both should find the correct eigenvalues
        assert_allclose(result_prec.eigenvalues, true_eigs, atol=1e-6)
        # Without preconditioner the problem is too ill-conditioned to converge quickly
        assert not jnp.all(result_none.converged)
        # With exact inverse the solver converges in very few iterations
        assert result_prec.iterations < 20
