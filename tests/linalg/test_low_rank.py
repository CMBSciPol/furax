"""Tests for low-rank approximation module."""

import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LowRankOperator, LowRankTerms, low_rank, low_rank_mv
from furax.tree import as_structure


class TestLowRankTerms:
    """Tests for LowRankTerms container."""

    def test_lowrankterms_creation(self):
        """Test LowRankTerms can be created."""
        eigenvalues = jnp.array([1.0, 2.0])
        eigenvectors = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        terms = LowRankTerms(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        assert_allclose(terms.eigenvalues, eigenvalues)
        assert_allclose(terms.eigenvectors, eigenvectors)

    def test_lowrankterms_pytree_eigenvectors(self):
        """Test LowRankTerms with PyTree eigenvectors."""
        eigenvalues = jnp.array([1.0, 2.0])
        eigenvectors = {'a': jnp.array([[1.0, 0.0], [0.0, 1.0]]), 'b': jnp.array([[0.5], [0.5]])}
        terms = LowRankTerms(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        assert terms.eigenvectors['a'].shape == (2, 2)
        assert terms.eigenvectors['b'].shape == (2, 1)


class TestLowRank:
    """Tests for low_rank function."""

    def test_low_rank_diagonal_lanczos(self):
        """Test low_rank with diagonal operator using lanczos method."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        assert isinstance(terms, LowRankTerms)
        assert terms.eigenvalues.shape == (2,)
        # Should find largest 2 eigenvalues: 4.0 and 5.0
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_diagonal_lobpcg(self):
        """Test low_rank with diagonal operator using lobpcg method."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lobpcg', largest=True, key=key)

        assert isinstance(terms, LowRankTerms)
        # Should find largest 2 eigenvalues: 4.0 and 5.0
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_smallest_eigenvalues(self):
        """Test low_rank finding smallest eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(1)
        terms = low_rank(A, k=2, method='lanczos', largest=False, key=key)

        # Should find smallest 2 eigenvalues: 1.0 and 2.0
        assert_allclose(sorted(terms.eigenvalues), [1.0, 2.0], atol=1e-4)

    def test_low_rank_methods_agree(self):
        """Test that lanczos and lobpcg produce similar results."""
        d = jnp.array([1.0, 2.0, 5.0, 10.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(42)
        terms_lanczos = low_rank(A, k=2, method='lanczos', largest=True, key=key)
        terms_lobpcg = low_rank(A, k=2, method='lobpcg', largest=True, key=key)

        # Both should find [5.0, 10.0]
        assert_allclose(
            sorted(terms_lanczos.eigenvalues), sorted(terms_lobpcg.eigenvalues), atol=1e-4
        )

    def test_low_rank_invalid_method(self):
        """Test low_rank raises error for invalid method."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match="Unknown method: 'invalid'"):
            low_rank(A, k=2, method='invalid', key=jax.random.PRNGKey(0))

    def test_low_rank_pytree_structure(self):
        """Test low_rank with PyTree-structured operators."""
        d1 = jnp.array([1.0, 5.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        key = jax.random.PRNGKey(3)
        terms = low_rank(A, k=2, method='lobpcg', largest=True, key=key, max_iters=100)

        # Total eigenvalues are [1, 3, 5], largest 2 are [3, 5]
        assert_allclose(sorted(terms.eigenvalues), [3.0, 5.0], atol=1e-3)
        assert 'a' in terms.eigenvectors
        assert 'b' in terms.eigenvectors


class TestLowRankMv:
    """Tests for low_rank_mv function."""

    def test_low_rank_mv_identity_approx(self):
        """Test low_rank_mv with full rank gives exact result."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Get all eigenvalues (full rank)
        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=3, method='lanczos', largest=True, key=key)

        x = jnp.array([1.0, 2.0, 3.0])
        y_approx = low_rank_mv(terms, x)
        y_exact = A.mv(x)

        assert_allclose(y_approx, y_exact, atol=1e-4)

    def test_low_rank_mv_partial_rank(self):
        """Test low_rank_mv with partial rank."""
        d = jnp.array([0.1, 0.2, 10.0, 20.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        # Get only largest 2 eigenvalues
        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        # Input aligned with large eigenvalue directions should be well approximated
        x = jnp.array([0.0, 0.0, 1.0, 0.0])
        y_approx = low_rank_mv(terms, x)
        y_exact = A.mv(x)

        # Should capture the large eigenvalue component well
        assert_allclose(y_approx[2], y_exact[2], atol=1e-3)

    def test_low_rank_mv_pytree(self):
        """Test low_rank_mv with PyTree structure."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=3, method='lobpcg', largest=True, key=key, max_iters=100)

        x = {'a': jnp.array([1.0, 1.0]), 'b': jnp.array([1.0])}
        y_approx = low_rank_mv(terms, x)
        y_exact = A.mv(x)

        assert_allclose(y_approx['a'], y_exact['a'], atol=1e-3)
        assert_allclose(y_approx['b'], y_exact['b'], atol=1e-3)


class TestLowRankJit:
    """Tests for JIT compatibility of low_rank module."""

    def test_low_rank_jit_lanczos(self):
        """Test low_rank with lanczos is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        @jax.jit
        def compute(key):
            return low_rank(A, k=2, method='lanczos', largest=True, key=key)

        terms = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_jit_lobpcg(self):
        """Test low_rank with lobpcg is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        @jax.jit
        def compute(key):
            return low_rank(A, k=2, method='lobpcg', largest=True, key=key)

        terms = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_mv_jit(self):
        """Test low_rank_mv is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        terms = low_rank(A, k=3, method='lanczos', largest=True, key=jax.random.PRNGKey(0))

        @jax.jit
        def apply(x):
            return low_rank_mv(terms, x)

        x = jnp.ones(5)
        y = apply(x)
        assert y.shape == x.shape

    def test_lowrankoperator_jit(self):
        """Test LowRankOperator is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        terms = low_rank(A, k=2, method='lanczos', largest=True, key=jax.random.PRNGKey(0))
        B = LowRankOperator(terms)

        @jax.jit
        def apply(x):
            return B(x)

        x = jnp.ones(5)
        y = apply(x)
        assert y.shape == x.shape


class TestLowRankOperator:
    """Tests for LowRankOperator class."""

    def test_lowrankoperator_creation(self):
        """Test LowRankOperator can be created."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        assert B.in_structure() == B.out_structure()

    def test_lowrankoperator_mv_matches_low_rank_mv(self):
        """Test LowRankOperator.mv matches low_rank_mv."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        y_op = B.mv(x)
        y_func = low_rank_mv(terms, x)

        assert_allclose(y_op, y_func, atol=1e-8)

    def test_lowrankoperator_symmetric(self):
        """Test LowRankOperator is registered as symmetric."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        assert lx.is_symmetric(B)

    def test_lowrankoperator_transpose_is_self(self):
        """Test LowRankOperator transpose returns itself."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        assert B.T is B

    def test_lowrankoperator_call(self):
        """Test LowRankOperator can be applied with call syntax."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        x = jnp.array([1.0, 2.0, 3.0])

        # B(x) should work
        y = B(x)
        assert y.shape == x.shape

    def test_lowrankoperator_pytree(self):
        """Test LowRankOperator with PyTree structure."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lobpcg', largest=True, key=key, max_iters=100)

        B = LowRankOperator(terms)
        x = {'a': jnp.array([1.0, 1.0]), 'b': jnp.array([1.0])}

        y = B(x)
        assert 'a' in y
        assert 'b' in y

    def test_lowrankoperator_explicit_structure(self):
        """Test LowRankOperator with explicitly provided structure."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, k=2, method='lanczos', largest=True, key=key)

        explicit_structure = as_structure(d)
        B = LowRankOperator(terms, in_structure=explicit_structure)

        assert B.in_structure() == explicit_structure
