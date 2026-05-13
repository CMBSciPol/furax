"""Tests for low-rank approximation wrapper."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import LowRankOperator, LowRankTerms, low_rank, low_rank_mv
from furax.tree import as_structure


def _diagonal_op(d):
    return DiagonalOperator(d, in_structure=as_structure(d))


class TestLowRank:
    """Tests for low_rank dispatch and output structure."""

    def test_low_rank_output_shape(self):
        """low_rank returns LowRankTerms with correct shapes."""
        d = jnp.arange(1.0, 6.0)
        A = _diagonal_op(d)
        terms = low_rank(A, rank=3, key=jax.random.key(0))
        assert isinstance(terms, LowRankTerms)
        assert terms.eigenvalues.shape == (3,)
        assert LowRankOperator(terms).in_structure == A.in_structure

    def test_low_rank_invalid_method(self):
        """low_rank raises for unknown method."""
        A = _diagonal_op(jnp.arange(1.0, 4.0))
        with pytest.raises(ValueError, match="Unknown method: 'invalid'"):
            low_rank(A, rank=2, method='invalid', key=jax.random.key(0))

    def test_low_rank_pytree_output_structure(self):
        """low_rank preserves PyTree structure in eigenvectors."""
        d1, d2 = jnp.array([1.0, 5.0]), jnp.array([3.0])
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=as_structure(d1)),
                'b': DiagonalOperator(d2, in_structure=as_structure(d2)),
            }
        )
        terms = low_rank(A, rank=2, method='lanczos', key=jax.random.key(0))
        assert LowRankOperator(terms).in_structure == A.in_structure


class TestLowRankMv:
    """Tests for low_rank_mv formula."""

    def test_low_rank_mv_formula(self):
        """low_rank_mv computes U @ diag(S) @ U^T @ x."""
        U = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (k=2, n=3)
        S = jnp.array([2.0, 3.0])
        terms = LowRankTerms(eigenvalues=S, eigenvectors=U)
        x = jnp.array([1.0, 1.0, 1.0])
        # U @ x = [1, 1], S * [1, 1] = [2, 3], U^T @ [2, 3] = [2, 3, 0]
        assert_allclose(low_rank_mv(terms, x), jnp.array([2.0, 3.0, 0.0]))

    def test_low_rank_mv_pytree(self):
        """low_rank_mv handles PyTree input/output."""
        U = {'a': jnp.array([[1.0]]), 'b': jnp.array([[0.0]])}  # (k=1, n=1) per leaf
        S = jnp.array([2.0])
        terms = LowRankTerms(eigenvalues=S, eigenvectors=U)
        x = {'a': jnp.array([3.0]), 'b': jnp.array([5.0])}
        # U @ x = [1*3 + 0*5] = [3], S * [3] = [6], U^T @ [6] = {'a': [6], 'b': [0]}
        result = low_rank_mv(terms, x)
        assert_allclose(result['a'], jnp.array([6.0]))
        assert_allclose(result['b'], jnp.array([0.0]))


class TestLowRankJit:
    """Tests for JIT compatibility."""

    def test_low_rank_jit_lanczos(self):
        """low_rank with lanczos is JIT-compatible."""
        A = _diagonal_op(jnp.arange(1.0, 6.0))
        key = jax.random.key(0)
        expected = low_rank(A, rank=3, method='lanczos', key=key)
        result = jax.jit(lambda k: low_rank(A, rank=3, method='lanczos', key=k))(key)
        assert_allclose(result.eigenvalues, expected.eigenvalues)

    def test_low_rank_jit_tr(self):
        """low_rank with lanczos_tr is JIT-compatible."""
        A = _diagonal_op(jnp.arange(1.0, 6.0))
        key = jax.random.key(0)
        expected = low_rank(A, rank=2, method='lanczos_tr', key=key)
        result = jax.jit(lambda k: low_rank(A, rank=2, method='lanczos_tr', key=k))(key)
        assert_allclose(result.eigenvalues, expected.eigenvalues)


class TestLowRankOperator:
    """Tests for LowRankOperator."""

    def test_lowrankoperator_infers_in_structure(self):
        """LowRankOperator infers in_structure from eigenvectors shape."""
        U = jnp.eye(5)[:3]  # shape (3, 5)
        terms = LowRankTerms(eigenvalues=jnp.ones(3), eigenvectors=U)
        op = LowRankOperator(terms)
        assert op.in_structure.shape == (5,)

    def test_lowrankoperator_mv_matches_low_rank_mv(self):
        """LowRankOperator.mv matches low_rank_mv."""
        U = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        S = jnp.array([2.0, 3.0])
        terms = LowRankTerms(eigenvalues=S, eigenvectors=U)
        op = LowRankOperator(terms)
        x = jnp.array([1.0, 2.0, 3.0])
        assert_allclose(op.mv(x), low_rank_mv(terms, x))
