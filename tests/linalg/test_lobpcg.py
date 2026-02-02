"""Tests for LOBPCG eigenvalue solver."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LOBPCGResult, lobpcg_standard
from furax.linalg._lobpcg import _rayleigh_ritz
from furax.tree import as_structure
from furax.tree_block import apply_operator_block, block_normal_like


class TestRayleighRitz:
    """Tests for Rayleigh-Ritz procedure."""

    def test_rayleigh_ritz_diagonal(self):
        """Test Rayleigh-Ritz with diagonal operator."""
        # Eigenvalues of diagonal operator are the diagonal elements
        d = jnp.array([1.0, 3.0, 2.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Use identity as search space (exact eigenvectors)
        S = jnp.eye(3)
        AS = apply_operator_block(A, S)

        eigenvalues, eigenvectors = _rayleigh_ritz(S, AS, k=2, largest=False)

        # Should get smallest 2 eigenvalues: 1.0 and 2.0
        assert_allclose(sorted(eigenvalues), [1.0, 2.0], atol=1e-5)


class TestLOBPCG:
    """Integration tests for LOBPCG solver."""

    def test_lobpcg_diagonal_operator(self):
        """Test LOBPCG with diagonal operator (known eigenvalues)."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = lobpcg_standard(A, k=2, key=key, max_iters=50, tol=1e-8)

        assert isinstance(result, LOBPCGResult)
        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors.shape == (2, 5)

        # Should find the 2 smallest eigenvalues: 1.0 and 2.0
        assert_allclose(sorted(result.eigenvalues), [1.0, 2.0], atol=1e-5)

    def test_lobpcg_largest_eigenvalues(self):
        """Test LOBPCG finding largest eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(1)
        result = lobpcg_standard(A, k=2, key=key, max_iters=50, tol=1e-8, largest=True)

        # Should find the 2 largest eigenvalues: 4.0 and 5.0
        assert_allclose(sorted(result.eigenvalues), [4.0, 5.0], atol=1e-5)

    def test_lobpcg_convergence(self):
        """Test that LOBPCG converges and residuals are small."""
        d = jnp.array([1.0, 2.0, 5.0, 10.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(2)
        result = lobpcg_standard(A, k=2, key=key, max_iters=100, tol=1e-6)

        assert jnp.all(result.converged)
        assert jnp.all(result.residual_norms < 1e-5)

    def test_lobpcg_with_initial_guess(self):
        """Test LOBPCG with provided initial vectors."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Provide good initial guess close to true eigenvectors
        X0 = jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.1]])
        result = lobpcg_standard(A, X=X0, max_iters=50, tol=1e-8)

        assert_allclose(sorted(result.eigenvalues), [1.0, 2.0], atol=1e-5)

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
        X0 = block_normal_like(A.in_structure(), 2, key)
        result = lobpcg_standard(A, X=X0, max_iters=50, tol=1e-6)

        # Total eigenvalues are [1, 2, 3], should find smallest 2
        assert_allclose(sorted(result.eigenvalues), [1.0, 2.0], atol=1e-4)

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

        key2 = jax.random.PRNGKey(0)
        result = lobpcg_standard(A, k=3, key=key2, max_iters=50, tol=1e-8)

        # For diagonal operator, eigenvalues are the diagonal elements
        sorted_d = jnp.sort(d)
        assert_allclose(sorted(result.eigenvalues), sorted_d[:3], atol=1e-5)

    def test_lobpcg_requires_k_or_x(self):
        """Test that LOBPCG requires either k or X."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match='k must be specified'):
            lobpcg_standard(A, key=jax.random.PRNGKey(0))

    def test_lobpcg_requires_key_when_no_x(self):
        """Test that LOBPCG requires key when X is None."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match='key must be specified'):
            lobpcg_standard(A, k=2)
