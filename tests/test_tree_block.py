"""Tests for block PyTree utilities."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax.tree_block import (
    block_from_array,
    block_norm,
    block_normal_like,
    block_to_array,
    block_zeros_like,
    gram,
    matvec,
    orthonormalize,
    qr,
    stack,
    unstack,
    vecmat,
)


class TestStackUnstack:
    """Tests for stack and unstack."""

    def test_stack_unstack_roundtrip(self):
        """Test that stacking and unstacking PyTrees is a roundtrip."""
        p1 = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        p2 = {'a': jnp.array([4.0, 5.0]), 'b': jnp.array([6.0])}
        pytrees = [p1, p2]

        block = stack(pytrees)
        assert block['a'].shape == (2, 2)
        assert block['b'].shape == (2, 1)

        unstacked = unstack(block)
        assert_allclose(unstacked[0]['a'], p1['a'])
        assert_allclose(unstacked[0]['b'], p1['b'])
        assert_allclose(unstacked[1]['a'], p2['a'])
        assert_allclose(unstacked[1]['b'], p2['b'])

    def test_stack_pytrees_empty(self):
        """Test that stacking an empty list raises an error."""
        with pytest.raises(ValueError, match='Need at least one Pytree to stack'):
            stack([])

    def test_stack_axis(self):
        """Test that stack respects the axis argument."""
        p1 = {'a': jnp.array([1.0, 2.0])}
        p2 = {'a': jnp.array([3.0, 4.0])}
        block = stack([p1, p2], axis=1)
        assert block['a'].shape == (2, 2)
        assert_allclose(block['a'], jnp.array([[1.0, 3.0], [2.0, 4.0]]))

    def test_unstack_axis(self):
        """Test that unstack respects the axis argument."""
        p1 = {'a': jnp.array([1.0, 2.0])}
        p2 = {'a': jnp.array([3.0, 4.0])}
        block = stack([p1, p2], axis=1)
        unstacked = unstack(block, axis=1)
        assert_allclose(unstacked[0]['a'], p1['a'])
        assert_allclose(unstacked[1]['a'], p2['a'])

    def test_stack_unstack_roundtrip_axis(self):
        """Test roundtrip with non-default axis."""
        p1 = {'a': jnp.array([1.0, 2.0]), 'b': jnp.array([3.0])}
        p2 = {'a': jnp.array([4.0, 5.0]), 'b': jnp.array([6.0])}
        for axis in [0, 1]:
            block = stack([p1, p2], axis=axis)
            unstacked = unstack(block, axis=axis)
            assert_allclose(unstacked[0]['a'], p1['a'])
            assert_allclose(unstacked[1]['b'], p2['b'])


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


class TestGram:
    """Tests for gram."""

    def test_gram_identity(self):
        """Test gram with orthonormal vectors gives identity."""
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}  # 2 orthonormal vectors
        G = gram(X, X)
        assert_allclose(G, jnp.eye(2), atol=1e-6)

    def test_gram_cross(self):
        """Test gram between different blocks."""
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}
        Y = {'a': jnp.array([[1.0, 1.0], [2.0, 0.0]])}
        G = gram(X, Y)
        # G[0,0] = dot([1,0], [1,1]) = 1
        # G[0,1] = dot([1,0], [2,0]) = 2
        # G[1,0] = dot([0,1], [1,1]) = 1
        # G[1,1] = dot([0,1], [2,0]) = 0
        expected = jnp.array([[1.0, 2.0], [1.0, 0.0]])
        assert_allclose(G, expected, atol=1e-6)

    def test_gram_symmetry(self):
        """Test gram(X, X) is symmetric."""
        key = jax.random.PRNGKey(0)
        X = {'a': jax.random.normal(key, (3, 5))}
        G = gram(X, X)
        assert_allclose(G, G.T, atol=1e-6)


class TestBlockNorm:
    """Tests for block_norm."""

    def test_block_norm(self):
        """Test computing norms of block vectors."""
        X = {'a': jnp.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])}  # 3 vectors
        norms = block_norm(X)
        expected = jnp.array([5.0, 1.0, 0.0])
        assert_allclose(norms, expected, atol=1e-6)


class TestMatvecVecmat:
    """Tests for matvec and vecmat."""

    def test_matvec(self):
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}  # 2 basis vectors
        C = jnp.array([[1.0, 1.0], [0.5, -0.5]])  # (k, m)
        Y = matvec(C, X)
        # Y[0] = 1*X[0] + 1*X[1] = [1, 1]
        # Y[1] = 0.5*X[0] - 0.5*X[1] = [0.5, -0.5]
        assert_allclose(Y['a'], jnp.array([[1.0, 1.0], [0.5, -0.5]]), atol=1e-6)

    def test_vecmat(self):
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}  # 2 basis vectors
        C = jnp.array([[1.0, 0.5], [1.0, -0.5]])  # (m, k)
        Y = vecmat(X, C)
        # Y[0] = 1*X[0] + 1*X[1] = [1, 1]
        # Y[1] = 0.5*X[0] - 0.5*X[1] = [0.5, -0.5]
        assert_allclose(Y['a'], jnp.array([[1.0, 1.0], [0.5, -0.5]]), atol=1e-6)

    def test_matvec_vecmat_consistent(self):
        X = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]])}
        C = jnp.array([[1.0, 1.0], [0.5, -0.5]])  # (k, m)
        assert_allclose(matvec(C, X)['a'], vecmat(X, C.T)['a'], atol=1e-6)


class TestBlockToFromArray:
    """Tests for block_to_array and block_from_array."""

    def test_block_to_array_shape(self):
        """Test that block_to_array produces correct output shape."""
        X = {'a': jnp.ones((3, 2)), 'b': jnp.ones((3, 1))}
        X_flat, treedef, shapes = block_to_array(X)

        assert X_flat.shape == (3, 3)

    def test_block_to_array_values(self):
        """Test that block_to_array concatenates leaves correctly."""
        X = {'a': jnp.array([[1.0, 2.0], [3.0, 4.0]]), 'b': jnp.array([[5.0], [6.0]])}
        X_flat, treedef, shapes = block_to_array(X)

        expected = jnp.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
        assert_allclose(X_flat, expected)

    def test_block_from_array_roundtrip(self):
        """Test that block_to_array followed by block_from_array is a roundtrip."""
        X = {'a': jnp.array([[1.0, 2.0], [3.0, 4.0]]), 'b': jnp.array([[5.0], [6.0]])}
        X_flat, treedef, shapes = block_to_array(X)
        X2 = block_from_array(X_flat, treedef, shapes)

        assert_allclose(X2['a'], X['a'])
        assert_allclose(X2['b'], X['b'])

    def test_block_to_array_single_leaf(self):
        """Test block_to_array with a single-leaf PyTree."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        X_flat, treedef, shapes = block_to_array(X)

        assert X_flat.shape == (2, 3)
        assert_allclose(X_flat, X)

    def test_block_from_array_different_k(self):
        """Test block_from_array with a different leading dimension than original."""
        X = {'a': jnp.ones((3, 2)), 'b': jnp.ones((3, 1))}
        _, treedef, shapes = block_to_array(X)

        # Reconstruct with k=5 instead of k=3
        X_flat_new = jnp.zeros((5, 3))
        X_new = block_from_array(X_flat_new, treedef, shapes)

        assert X_new['a'].shape == (5, 2)
        assert X_new['b'].shape == (5, 1)

    def test_block_to_array_preserves_treedef(self):
        """Test that treedef from block_to_array can reconstruct tree structure."""
        X = {'a': jnp.ones((2, 4)), 'b': jnp.ones((2, 3))}
        X_flat, treedef, shapes = block_to_array(X)
        X2 = block_from_array(X_flat, treedef, shapes)

        assert set(X2.keys()) == {'a', 'b'}
        assert X2['a'].shape == (2, 4)
        assert X2['b'].shape == (2, 3)


class TestOrthonormalization:
    """Tests for QR decomposition and orthonormalization."""

    def test_qr_orthonormal(self):
        """Test that QR produces orthonormal vectors."""
        key = jax.random.PRNGKey(0)
        X = {'a': jax.random.normal(key, (3, 5))}
        Q, R = qr(X)

        # Check Q^T Q = I (orthonormal rows)
        G = gram(Q, Q)
        assert_allclose(G, jnp.eye(3), atol=1e-5)

    def test_qr_decomposition(self):
        """Test that X = Q @ R."""
        key = jax.random.PRNGKey(1)
        X = {'a': jax.random.normal(key, (2, 4)), 'b': jax.random.normal(key, (2, 3))}
        Q, R = qr(X)

        # Reconstruct X from Q and R
        X_reconstructed = vecmat(Q, R)
        assert_allclose(X_reconstructed['a'], X['a'], atol=1e-5)
        assert_allclose(X_reconstructed['b'], X['b'], atol=1e-5)

    def test_orthonormalize(self):
        """Test orthonormalize produces orthonormal vectors."""
        X = {'a': jnp.array([[1.0, 1.0], [1.0, 0.0]])}  # Non-orthogonal
        Q = orthonormalize(X)

        G = gram(Q, Q)
        assert_allclose(G, jnp.eye(2), atol=1e-5)
