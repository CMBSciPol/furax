"""Tests for Lanczos eigenvalue solver."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LanczosResult, lanczos_eigh, lanczos_tridiag
from furax.linalg._lanczos import _tridiag_eigh
from furax.tree import as_structure
from furax.tree_block import batched_dot


class TestTridiagEigh:
    """Tests for tridiagonal eigenvalue decomposition."""

    def test_tridiag_eigh_diagonal(self):
        """Test with diagonal matrix (zero off-diagonal)."""
        alpha = jnp.array([1.0, 3.0, 2.0])
        beta = jnp.array([0.0, 0.0])
        eigenvalues, eigenvectors = _tridiag_eigh(alpha, beta)
        assert_allclose(eigenvalues, jnp.array([1.0, 2.0, 3.0]), atol=1e-5)

    def test_tridiag_eigh_simple(self):
        """Test with simple tridiagonal matrix."""
        # Construct a 3x3 tridiagonal matrix
        alpha = jnp.array([2.0, 2.0, 2.0])
        beta = jnp.array([-1.0, -1.0])
        eigenvalues, eigenvectors = _tridiag_eigh(alpha, beta)

        # Build full matrix and verify
        T = jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
        expected_eigenvalues, _ = jnp.linalg.eigh(T)
        assert_allclose(eigenvalues, expected_eigenvalues, atol=1e-5)

    def test_tridiag_eigh_orthonormal(self):
        """Test that eigenvectors are orthonormal."""
        alpha = jnp.array([1.0, 2.0, 3.0, 4.0])
        beta = jnp.array([0.5, 0.5, 0.5])
        eigenvalues, eigenvectors = _tridiag_eigh(alpha, beta)
        G = eigenvectors.T @ eigenvectors
        assert_allclose(G, jnp.eye(4), atol=1e-5)


class TestLanczosTridiag:
    """Tests for Lanczos tridiagonalization."""

    def test_lanczos_tridiag_diagonal_operator(self):
        """Test Lanczos on diagonal operator."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        alpha, beta, V = lanczos_tridiag(A, v0, m=5)

        assert alpha.shape == (5,)
        assert beta.shape == (4,)
        assert V.shape == (5, 5)

    def test_lanczos_tridiag_orthonormal_vectors(self):
        """Test that Lanczos produces orthonormal vectors."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        v0 = jax.random.normal(key, (4,))
        alpha, beta, V = lanczos_tridiag(A, v0, m=4)

        # Check orthonormality
        G = batched_dot(V, V)
        assert_allclose(G, jnp.eye(4), atol=1e-5)

    def test_lanczos_tridiag_pytree(self):
        """Test Lanczos with PyTree structure."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0, 4.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        v0 = {'a': jnp.array([1.0, 0.0]), 'b': jnp.array([0.0, 1.0])}
        alpha, beta, V = lanczos_tridiag(A, v0, m=4)

        assert alpha.shape == (4,)
        assert beta.shape == (3,)
        assert V['a'].shape == (4, 2)
        assert V['b'].shape == (4, 2)

    def test_lanczos_tridiag_eigenvalue_relation(self):
        """Test that tridiagonal matrix has same eigenvalues as A restricted to Krylov space."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = jnp.array([1.0, 1.0, 1.0])
        alpha, beta, V = lanczos_tridiag(A, v0, m=3)

        # Build tridiagonal matrix
        T = jnp.diag(alpha) + jnp.diag(beta, k=1) + jnp.diag(beta, k=-1)
        eigenvalues_T, _ = jnp.linalg.eigh(T)

        # For diagonal operator with full Krylov space, eigenvalues should match
        assert_allclose(jnp.sort(eigenvalues_T), jnp.sort(d), atol=1e-4)


class TestLanczosEigh:
    """Integration tests for Lanczos eigenvalue solver."""

    def test_lanczos_diagonal_operator(self):
        """Test Lanczos with diagonal operator (known eigenvalues)."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = lanczos_eigh(A, k=2, key=key)

        assert isinstance(result, LanczosResult)
        assert result.eigenvalues.shape == (2,)
        assert result.eigenvectors.shape == (2, 5)

        # Should find the 2 smallest eigenvalues: 1.0 and 2.0
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0]), atol=1e-4)

    def test_lanczos_largest_eigenvalues(self):
        """Test Lanczos finding largest eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(1)
        result = lanczos_eigh(A, k=2, key=key, largest=True)

        # Should find the 2 largest eigenvalues: 4.0 and 5.0
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([4.0, 5.0]), atol=1e-4)

    def test_lanczos_with_initial_vector(self):
        """Test Lanczos with provided initial vector."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Provide initial vector
        v0 = jnp.array([1.0, 1.0, 1.0])
        result = lanczos_eigh(A, v0=v0, k=2)

        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0]), atol=1e-4)

    def test_lanczos_pytree_structure(self):
        """Test Lanczos with PyTree-structured operators."""
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
        result = lanczos_eigh(A, k=2, key=key)

        # Total eigenvalues are [1, 2, 3], should find smallest 2
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0]), atol=1e-3)

    def test_lanczos_single_eigenvalue(self):
        """Test Lanczos finding single eigenvalue."""
        d = jnp.array([1.0, 5.0, 10.0, 15.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(4)
        result = lanczos_eigh(A, k=1, m=4, key=key)

        assert result.eigenvalues.shape == (1,)
        assert_allclose(result.eigenvalues[0], 1.0, atol=1e-4)

    def test_lanczos_custom_krylov_dimension(self):
        """Test Lanczos with custom Krylov subspace dimension."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(5)
        result = lanczos_eigh(A, k=2, m=5, key=key)

        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0]), atol=1e-4)

    def test_lanczos_requires_key_when_no_v0(self):
        """Test that Lanczos requires key when v0 is None."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match='key must be specified'):
            lanczos_eigh(A, k=2)

    def test_lanczos_m_less_than_k_raises(self):
        """Test that m < k raises an error."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(6)
        with pytest.raises(ValueError, match='m .* must be >= k'):
            lanczos_eigh(A, k=3, m=2, key=key)

    def test_lanczos_eigenvectors_orthonormal(self):
        """Test that returned eigenvectors are approximately orthonormal."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(7)
        result = lanczos_eigh(A, k=2, key=key)

        # Check orthonormality of eigenvectors
        G = batched_dot(result.eigenvectors, result.eigenvectors)
        assert_allclose(G, jnp.eye(2), atol=1e-4)

    def test_lanczos_vs_lobpcg(self):
        """Test that Lanczos and LOBPCG give similar results."""
        from furax.linalg import lobpcg_standard

        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(8)
        lanczos_result = lanczos_eigh(A, k=2, key=key)

        key2 = jax.random.PRNGKey(9)
        lobpcg_result = lobpcg_standard(A, k=2, key=key2, tol=1e-8)

        assert_allclose(
            jnp.sort(lanczos_result.eigenvalues),
            jnp.sort(lobpcg_result.eigenvalues),
            atol=1e-4,
        )
