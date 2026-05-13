"""Tests for Lanczos eigenvalue solver."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg._lanczos import (
    LanczosResult,
    lanczos_eigh,
    lanczos_tr,
    lanczos_tridiag,
)
from furax.tree import as_structure, normal_like


class TestLanczosTridiag:
    """Tests for Lanczos tridiagonalization."""

    def test_lanczos_tridiag_diagonal_operator(self):
        """Test Lanczos on diagonal operator."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        alpha, beta, V, _, _ = lanczos_tridiag(A, v0, m=5)

        assert alpha.shape == (5,)
        assert beta.shape == (4,)
        assert V.shape == (5, 5)

    def test_lanczos_tridiag_orthonormal_vectors(self):
        """Test that Lanczos produces orthonormal vectors."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        v0 = jax.random.normal(key, (4,))
        alpha, beta, V, _, _ = lanczos_tridiag(A, v0, m=4)

        # Check orthonormality
        G = V @ V.T
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
        alpha, beta, V, _, _ = lanczos_tridiag(A, v0, m=4)

        assert alpha.shape == (4,)
        assert beta.shape == (3,)
        assert V['a'].shape == (4, 2)
        assert V['b'].shape == (4, 2)

    def test_lanczos_tridiag_eigenvalue_relation(self):
        """Test that tridiagonal matrix has same eigenvalues as A restricted to Krylov space."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = jnp.array([1.0, 1.0, 1.0])
        alpha, beta, V, _, _ = lanczos_tridiag(A, v0, m=3)

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

        v0 = normal_like(as_structure(d), jax.random.PRNGKey(0))
        result = lanczos_eigh(A, v0, k=5)

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
        result = lanczos_eigh(A, v0, k=3)

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

        v0 = normal_like(A.in_structure, jax.random.PRNGKey(3))
        result = lanczos_eigh(A, v0, k=3)

        # Total eigenvalues are [1, 2, 3]
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0, 3.0]), atol=1e-3)

    def test_lanczos_eigenvectors_orthonormal(self):
        """Test that returned eigenvectors are approximately orthonormal."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        v0 = normal_like(as_structure(d), jax.random.PRNGKey(7))
        result = lanczos_eigh(A, v0, k=2)

        # Check orthonormality of eigenvectors
        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(2), atol=1e-4)


class TestLanczosThickRestart:
    """Tests for the thick-restart Lanczos method."""

    def test_tr_smallest_eigenvalues(self):
        """TR finds k smallest eigenvalues of a diagonal operator."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.PRNGKey(0))

        result = lanczos_tr(A, v0, k=2, m=4, which='smallest')

        assert isinstance(result, LanczosResult)
        assert result.eigenvalues.shape == (2,)
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([1.0, 2.0]), atol=1e-4)

    def test_tr_largest_eigenvalues(self):
        """TR finds k largest eigenvalues of a diagonal operator."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.PRNGKey(0))

        result = lanczos_tr(A, v0, k=2, m=4, which='largest')

        assert result.eigenvalues.shape == (2,)
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([4.0, 5.0]), atol=1e-4)

    def test_tr_best_converges(self):
        """TR with which='best' converges to k valid eigenpairs."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.PRNGKey(42))
        tol = 1e-6

        result = lanczos_tr(A, v0, k=3, m=6, tol=tol, which='best')

        assert result.eigenvalues.shape == (3,)
        assert jnp.all(result.residual_norms < tol)
        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - d[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(3), atol=1e-4)

    def test_tr_eigenvectors_orthonormal(self):
        """TR returns orthonormal eigenvectors."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.PRNGKey(0))

        result = lanczos_tr(A, v0, k=3, m=4)

        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(3), atol=1e-4)

    def test_tr_pytree_operator(self):
        """TR works with PyTree-structured operators."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0, 4.0, 5.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )
        v0 = {
            'a': jax.random.normal(jax.random.PRNGKey(0), (2,)),
            'b': jax.random.normal(jax.random.PRNGKey(1), (3,)),
        }
        tol = 1e-6

        result = lanczos_tr(A, v0, k=2, m=4, tol=tol)

        assert result.eigenvalues.shape == (2,)
        assert jnp.all(result.residual_norms < tol)
        true_eigs = jnp.concatenate([d1, d2])
        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigs[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(2), atol=1e-3)

    def test_tr_requires_m_greater_than_k(self):
        """TR raises when m <= k."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = jnp.ones(3)

        with pytest.raises(ValueError, match='m .* must be > k'):
            lanczos_tr(A, v0, k=3, m=3)
