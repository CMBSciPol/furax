"""Tests for Lanczos eigenvalue solver."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LanczosResult, lanczos_eigh, lanczos_tridiag
from furax.linalg._lanczos import _compute_residual_norms, _tridiag_eigh
from furax.tree import as_structure
from furax.tree_block import block_normal_like, gram


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
        alpha, beta, V, _ = lanczos_tridiag(A, v0, m=5)

        assert alpha.shape == (5,)
        assert beta.shape == (4,)
        assert V.shape == (5, 5)

    def test_lanczos_tridiag_orthonormal_vectors(self):
        """Test that Lanczos produces orthonormal vectors."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        v0 = jax.random.normal(key, (4,))
        alpha, beta, V, _ = lanczos_tridiag(A, v0, m=4)

        # Check orthonormality
        G = gram(V, V)
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
        alpha, beta, V, _ = lanczos_tridiag(A, v0, m=4)

        assert alpha.shape == (4,)
        assert beta.shape == (3,)
        assert V['a'].shape == (4, 2)
        assert V['b'].shape == (4, 2)

    def test_lanczos_tridiag_eigenvalue_relation(self):
        """Test that tridiagonal matrix has same eigenvalues as A restricted to Krylov space."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = jnp.array([1.0, 1.0, 1.0])
        alpha, beta, V, _ = lanczos_tridiag(A, v0, m=3)

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

        v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(0))[0]
        result = lanczos_eigh(A, v0, rank=5)

        assert isinstance(result, LanczosResult)
        assert result.eigenvalues.shape == (5,)
        assert result.eigenvectors.shape == (5, 5)

        assert_allclose(jnp.sort(result.eigenvalues), d, atol=1e-4)

    def test_lanczos_with_initial_vector(self):
        """Test Lanczos with provided initial vector."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Provide initial vector
        v0 = jnp.array([1.0, 1.0, 1.0])
        result = lanczos_eigh(A, v0, rank=3)

        assert_allclose(jnp.sort(result.eigenvalues), d, atol=1e-4)

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

        v0_block = block_normal_like(A.in_structure, 1, jax.random.PRNGKey(3))
        v0 = jax.tree.map(lambda leaf: leaf[0], v0_block)
        result = lanczos_eigh(A, v0, rank=3)

        # Total eigenvalues are [1, 2, 3]
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0, 3.0]), atol=1e-3)

    def test_lanczos_eigenvectors_orthonormal(self):
        """Test that returned eigenvectors are approximately orthonormal."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(7))[0]
        result = lanczos_eigh(A, v0, rank=2)

        # Check orthonormality of eigenvectors
        G = gram(result.eigenvectors, result.eigenvectors)
        assert_allclose(G, jnp.eye(2), atol=1e-4)

    def test_lanczos_residual_norms(self):
        """Test that cheap residual norms match actual residual norms."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(42))[0]
        result = lanczos_eigh(A, v0, rank=5)

        actual_norms = _compute_residual_norms(A, result.eigenvectors, result.eigenvalues)
        assert_allclose(result.residual_norms, actual_norms, atol=1e-5)

    def test_lanczos_vs_lobpcg(self):
        """Test that Lanczos and LOBPCG give similar results."""
        from furax.linalg import lobpcg_standard

        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = block_normal_like(as_structure(d), 1, jax.random.PRNGKey(8))[0]
        lanczos_result = lanczos_eigh(A, v0, rank=5)

        X = block_normal_like(as_structure(d), 2, jax.random.PRNGKey(9))
        lobpcg_result = lobpcg_standard(A, X, tol=1e-8)

        # Compare the 2 smallest eigenvalues found by each method
        assert_allclose(
            jnp.sort(lanczos_result.eigenvalues)[:2],
            jnp.sort(lobpcg_result.eigenvalues),
            atol=1e-4,
        )
