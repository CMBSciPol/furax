"""Tests for Lanczos eigenvalue solver."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.linalg._lanczos import lanczos_eigh, lanczos_tr, lanczos_tridiag
from furax.tree import as_structure, normal_like


def _random_spd_operator(n: int, key: jax.Array):
    """Random symmetric positive definite operator of size n.

    Returns (A, v0, eigenvalues) where eigenvalues are sorted ascending.
    """
    key_mat, key_v0 = jax.random.split(key)
    B = jax.random.normal(key_mat, (n, n))
    mat = B @ B.T + n * jnp.eye(n)
    eigenvalues = jnp.linalg.eigvalsh(mat)
    v0 = jax.random.normal(key_v0, (n,))
    A = DenseBlockDiagonalOperator(mat, in_structure=as_structure(v0))
    return A, v0, eigenvalues


class TestLanczosTridiag:
    """Tests for Lanczos tridiagonalization."""

    def test_lanczos_tridiag_orthonormal_vectors(self):
        """Lanczos basis is orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_spd_operator(30, jax.random.key(0))
        _, _, V, _, _ = lanczos_tridiag(A, v0, m=12)
        assert_allclose(V @ V.T, jnp.eye(12), atol=1e-10)

    def test_lanczos_tridiag_eigenvalue_relation(self):
        """Tridiagonal T has same eigenvalues as A when Krylov space is full (m == n)."""
        A, v0, true_eigenvalues = _random_spd_operator(15, jax.random.key(1))
        alpha, beta, _, _, _ = lanczos_tridiag(A, v0, m=15)
        eigenvalues = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
        assert_allclose(eigenvalues, true_eigenvalues, atol=1e-10)


class TestLanczosEigh:
    """Integration tests for Lanczos eigenvalue solver."""

    def test_lanczos_eigh_eigenvalues(self):
        """lanczos_eigh returns accurate eigenvalues when Krylov space is full (m == n)."""
        A, v0, true_eigenvalues = _random_spd_operator(15, jax.random.key(0))
        result = lanczos_eigh(A, v0, k=5, m=15)

        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigenvalues[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(5), atol=1e-10)

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

        v0 = normal_like(A.in_structure, jax.random.key(3))
        result = lanczos_eigh(A, v0, k=3)

        assert_allclose(result.eigenvalues, jnp.array([1.0, 2.0, 3.0]), atol=1e-10)

    def test_lanczos_eigenvectors_orthonormal(self):
        """Ritz vectors are orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_spd_operator(30, jax.random.key(7))
        result = lanczos_eigh(A, v0, k=4, m=12)

        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(4), atol=1e-10)


class TestLanczosThickRestart:
    """Tests for the thick-restart Lanczos method."""

    def test_tr_smallest_eigenvalues(self):
        """TR finds k smallest eigenvalues of a random SPD operator."""
        A, v0, true_eigenvalues = _random_spd_operator(20, jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=6, which='smallest')
        assert_allclose(result.eigenvalues, true_eigenvalues[:2], atol=1e-10)

    def test_tr_largest_eigenvalues(self):
        """TR finds k largest eigenvalues of a random SPD operator."""
        A, v0, true_eigenvalues = _random_spd_operator(20, jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=6, which='largest')
        assert_allclose(result.eigenvalues, true_eigenvalues[-2:], atol=1e-10)

    def test_tr_best_converges(self):
        """TR with which='best' converges to k valid eigenpairs."""
        A, v0, true_eigenvalues = _random_spd_operator(20, jax.random.key(42))
        tol = 1e-6

        result = lanczos_tr(A, v0, k=3, m=8, tol=tol, which='best')

        assert jnp.all(result.residual_norms < tol)
        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigenvalues[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(3), atol=1e-10)

    def test_tr_eigenvectors_orthonormal(self):
        """TR Ritz vectors are orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_spd_operator(30, jax.random.key(0))
        result = lanczos_tr(A, v0, k=3, m=8)

        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(3), atol=1e-10)

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
            'a': jax.random.normal(jax.random.key(0), (2,)),
            'b': jax.random.normal(jax.random.key(1), (3,)),
        }
        tol = 1e-6

        result = lanczos_tr(A, v0, k=2, m=4, tol=tol)

        assert jnp.all(result.residual_norms < tol)
        true_eigs = jnp.concatenate([d1, d2])
        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigs[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(2), atol=1e-10)

    def test_tr_requires_m_greater_than_k(self):
        """TR raises when m <= k."""
        A, v0, _ = _random_spd_operator(5, jax.random.key(0))

        with pytest.raises(ValueError, match='m .* must be > k'):
            lanczos_tr(A, v0, k=5, m=5)
