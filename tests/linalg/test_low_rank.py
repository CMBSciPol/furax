"""Tests for low-rank approximation module."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LowRankOperator, LowRankTerms, low_rank, low_rank_mv
from furax.tree import as_structure


class TestLowRank:
    """Tests for low_rank function."""

    def test_low_rank_diagonal_lanczos(self):
        """Test low_rank with diagonal operator using lanczos method."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, rank=5, method='lanczos', key=key)

        assert isinstance(terms, LowRankTerms)
        assert terms.eigenvalues.shape == (5,)
        assert_allclose(sorted(terms.eigenvalues), sorted(d), atol=1e-4)

    def test_low_rank_diagonal_lobpcg(self):
        """Test low_rank with diagonal operator using lobpcg method."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, rank=2, method='lobpcg', largest=True, key=key)

        assert isinstance(terms, LowRankTerms)
        # Should find largest 2 eigenvalues: 4.0 and 5.0
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_methods_agree(self):
        """Test that lanczos and lobpcg produce similar results."""
        d = jnp.array([1.0, 2.0, 5.0, 10.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(42)
        terms_lanczos = low_rank(A, rank=4, method='lanczos', key=key)
        terms_lobpcg = low_rank(A, rank=2, method='lobpcg', largest=True, key=key)

        # lobpcg finds largest 2; lanczos (full space) should contain them
        assert_allclose(
            sorted(terms_lanczos.eigenvalues)[-2:], sorted(terms_lobpcg.eigenvalues), atol=1e-4
        )

    def test_low_rank_diagonal_nystrom(self):
        """Test low_rank with diagonal PSD operator using nystrom method."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, rank=2, method='nystrom', key=key)

        assert isinstance(terms, LowRankTerms)
        assert terms.eigenvalues.shape == (2,)
        # Nystrom returns largest eigenvalues
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-3)

    def test_low_rank_nystrom_requires_key(self):
        """Test low_rank nystrom raises error when key is not provided."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match="'nystrom' method requires a random key"):
            low_rank(A, rank=2, method='nystrom', key=None)

    def test_low_rank_invalid_method(self):
        """Test low_rank raises error for invalid method."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        with pytest.raises(ValueError, match="Unknown method: 'invalid'"):
            low_rank(A, rank=2, method='invalid', key=jax.random.PRNGKey(0))

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
        terms = low_rank(A, rank=2, method='lobpcg', largest=True, key=key, maxiter=100)

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
        terms = low_rank(A, rank=3, method='lanczos', largest=True, key=key)

        x = jnp.array([1.0, 2.0, 3.0])
        y_approx = low_rank_mv(terms, x)
        y_exact = A.mv(x)

        assert_allclose(y_approx, y_exact, atol=1e-4)

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
        terms = low_rank(A, rank=3, method='lobpcg', largest=True, key=key, maxiter=100)

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
            return low_rank(A, rank=5, method='lanczos', key=key)

        terms = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(terms.eigenvalues), sorted(d), atol=1e-4)

    def test_low_rank_jit_lobpcg(self):
        """Test low_rank with lobpcg is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        @jax.jit
        def compute(key):
            return low_rank(A, rank=2, method='lobpcg', largest=True, key=key)

        terms = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-4)

    def test_low_rank_jit_nystrom(self):
        """Test low_rank with nystrom is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        @jax.jit
        def compute(key):
            return low_rank(A, rank=2, method='nystrom', key=key)

        terms = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(terms.eigenvalues), [4.0, 5.0], atol=1e-3)


class TestLowRankOperator:
    """Tests for LowRankOperator class."""

    def test_lowrankoperator_mv_matches_low_rank_mv(self):
        """Test LowRankOperator.mv matches low_rank_mv."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        terms = low_rank(A, rank=2, method='lanczos', largest=True, key=key)

        B = LowRankOperator(terms)
        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        y_op = B.mv(x)
        y_func = low_rank_mv(terms, x)

        assert_allclose(y_op, y_func, atol=1e-8)
