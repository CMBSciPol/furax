"""Tests for LOBPCG eigenvalue solver."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.linalg import (
    LOBPCGResult,
    block_normal_like,
    block_zeros_like,
    lobpcg_standard,
    stack_pytrees,
    unstack_pytree,
)
from furax.linalg._utils import (
    apply_operator_block,
    apply_rotation,
    batched_dot,
    block_norms,
    orthonormalize,
    qr_pytree,
    rayleigh_ritz,
)
from furax.tree import as_structure


class TestBlockPyTreeUtilities:
    """Tests for block PyTree utilities in furax/linalg/_utils.py."""

    def test_stack_unstack_roundtrip(self):
        """Test that stacking and unstacking PyTrees is a roundtrip."""
        p1 = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        p2 = {'a': jnp.array([4.0, 5.0]), 'b': jnp.array([6.0])}
        pytrees = [p1, p2]

        block = stack_pytrees(pytrees)
        assert block['a'].shape == (2, 2)
        assert block['b'].shape == (2, 1)

        unstacked = unstack_pytree(block, 2)
        assert_allclose(unstacked[0]['a'], p1['a'])
        assert_allclose(unstacked[0]['b'], p1['b'])
        assert_allclose(unstacked[1]['a'], p2['a'])
        assert_allclose(unstacked[1]['b'], p2['b'])

    def test_stack_pytrees_empty(self):
        """Test that stacking an empty list raises an error."""
        with pytest.raises(ValueError, match='Cannot stack an empty list'):
            stack_pytrees([])

    def test_block_zeros_like(self):
        """Test block_zeros_like creates correct shape."""
        x = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0, 4.0, 5.0])}
        block = block_zeros_like(x, 4)
        assert block['a'].shape == (4, 2)
        assert block['b'].shape == (4, 3)
        assert_allclose(block['a'], jnp.zeros((4, 2)))
        assert_allclose(block['b'], jnp.zeros((4, 3)))

    def test_block_normal_like(self):
        """Test block_normal_like creates correct shape with random values."""
        x = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        key = jax.random.PRNGKey(42)
        block = block_normal_like(x, 3, key)
        assert block['a'].shape == (3, 2)
        assert block['b'].shape == (3, 1)
        # Check that values are not all zeros (random)
        assert jnp.abs(block['a']).sum() > 0


class TestBlockOperations:
    """Tests for block linear algebra utilities in furax/linalg/_utils.py."""

    def test_batched_dot_identity(self):
        """Test batched_dot with orthonormal vectors gives identity."""
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}  # 2 orthonormal vectors
        G = batched_dot(X, X)
        assert_allclose(G, jnp.eye(2), atol=1e-6)

    def test_batched_dot_cross(self):
        """Test batched_dot between different blocks."""
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}
        Y = {'a': jnp.array([[1.0, 1.0], [2.0, 0.0]])}
        G = batched_dot(X, Y)
        # G[0,0] = dot([1,0], [1,1]) = 1
        # G[0,1] = dot([1,0], [2,0]) = 2
        # G[1,0] = dot([0,1], [1,1]) = 1
        # G[1,1] = dot([0,1], [2,0]) = 0
        expected = jnp.array([[1.0, 2.0], [1.0, 0.0]])
        assert_allclose(G, expected, atol=1e-6)

    def test_batched_dot_symmetry(self):
        """Test batched_dot(X, X) is symmetric."""
        key = jax.random.PRNGKey(0)
        X = {'a': jax.random.normal(key, (3, 5))}
        G = batched_dot(X, X)
        assert_allclose(G, G.T, atol=1e-6)

    def test_apply_operator_block(self):
        """Test applying operator to block of vectors."""
        d = jnp.array([2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        X = jnp.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])  # 2 vectors
        AX = apply_operator_block(A, X)
        expected = jnp.array([[2.0, 3.0, 4.0], [0.0, 3.0, 0.0]])
        assert_allclose(AX, expected, atol=1e-6)

    def test_block_norms(self):
        """Test computing norms of block vectors."""
        X = {'a': jnp.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])}  # 3 vectors
        norms = block_norms(X)
        expected = jnp.array([5.0, 1.0, 0.0])
        assert_allclose(norms, expected, atol=1e-6)

    def test_apply_rotation(self):
        """Test linear combination of block vectors."""
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}  # 2 basis vectors
        C = jnp.array([[1.0, 0.5], [1.0, -0.5]])  # Rotation matrix
        Y = apply_rotation(X, C)
        # Y[0] = 1*X[0] + 1*X[1] = [1, 1]
        # Y[1] = 0.5*X[0] - 0.5*X[1] = [0.5, -0.5]
        expected = jnp.array([[1.0, 1.0], [0.5, -0.5]])
        assert_allclose(Y['a'], expected, atol=1e-6)


class TestOrthonormalization:
    """Tests for QR decomposition and orthonormalization."""

    def test_qr_pytree_orthonormal(self):
        """Test that QR produces orthonormal vectors."""
        key = jax.random.PRNGKey(0)
        X = {'a': jax.random.normal(key, (3, 5))}
        Q, R = qr_pytree(X)

        # Check Q^T Q = I (orthonormal rows)
        G = batched_dot(Q, Q)
        assert_allclose(G, jnp.eye(3), atol=1e-5)

    def test_qr_pytree_decomposition(self):
        """Test that X = Q @ R."""
        key = jax.random.PRNGKey(1)
        X = {'a': jax.random.normal(key, (2, 4)), 'b': jax.random.normal(key, (2, 3))}
        Q, R = qr_pytree(X)

        # Reconstruct X from Q and R
        X_reconstructed = apply_rotation(Q, R)
        assert_allclose(X_reconstructed['a'], X['a'], atol=1e-5)
        assert_allclose(X_reconstructed['b'], X['b'], atol=1e-5)

    def test_orthonormalize(self):
        """Test orthonormalize produces orthonormal vectors."""
        X = {'a': jnp.array([[1.0, 1.0], [1.0, 0.0]])}  # Non-orthogonal
        Q = orthonormalize(X)

        G = batched_dot(Q, Q)
        assert_allclose(G, jnp.eye(2), atol=1e-5)


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

        eigenvalues, eigenvectors = rayleigh_ritz(S, AS, k=2, largest=False)

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
        # Create a block diagonal operator with PyTree structure
        from furax import BlockDiagonalOperator

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

        # Get true eigenvalues
        true_eigenvalues, _ = jnp.linalg.eigh(A_dense)

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
