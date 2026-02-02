"""Tests for block PyTree utilities."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.tree import as_structure
from furax.tree_block import (
    apply_operator_block,
    apply_rotation,
    batched_dot,
    block_normal_like,
    block_norms,
    block_zeros_like,
    orthonormalize,
    qr_pytree,
    stack_pytrees,
    unstack_pytree,
)


class TestStackUnstack:
    """Tests for stack_pytrees and unstack_pytree."""

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


class TestBlockCreation:
    """Tests for block_zeros_like and block_normal_like."""

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


class TestBatchedDot:
    """Tests for batched_dot."""

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


class TestApplyOperatorBlock:
    """Tests for apply_operator_block."""

    def test_apply_operator_block(self):
        """Test applying operator to block of vectors."""
        d = jnp.array([2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        X = jnp.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])  # 2 vectors
        AX = apply_operator_block(A, X)
        expected = jnp.array([[2.0, 3.0, 4.0], [0.0, 3.0, 0.0]])
        assert_allclose(AX, expected, atol=1e-6)


class TestBlockNorms:
    """Tests for block_norms."""

    def test_block_norms(self):
        """Test computing norms of block vectors."""
        X = {'a': jnp.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])}  # 3 vectors
        norms = block_norms(X)
        expected = jnp.array([5.0, 1.0, 0.0])
        assert_allclose(norms, expected, atol=1e-6)


class TestApplyRotation:
    """Tests for apply_rotation."""

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
