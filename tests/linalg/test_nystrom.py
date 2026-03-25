"""Tests for randomized Nyström approximation."""

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator
from furax.linalg import NystromPreconditioner, NystromResult, randomized_nystrom
from furax.tree import as_structure
from furax.tree_block import gram


class TestNystromResult:
    """Tests for NystromResult container."""

    def test_nystromresult_creation(self):
        """Test NystromResult can be created."""
        eigenvalues = jnp.array([3.0, 2.0, 1.0])
        eigenvectors = jnp.eye(3)
        result = NystromResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        assert_allclose(result.eigenvalues, eigenvalues)
        assert_allclose(result.eigenvectors, eigenvectors)

    def test_nystromresult_pytree_eigenvectors(self):
        """Test NystromResult with PyTree eigenvectors."""
        eigenvalues = jnp.array([2.0, 1.0])
        eigenvectors = {'a': jnp.ones((2, 3)), 'b': jnp.ones((2, 1))}
        result = NystromResult(eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        assert result.eigenvectors['a'].shape == (2, 3)
        assert result.eigenvectors['b'].shape == (2, 1)


class TestRandomizedNystrom:
    """Tests for randomized_nystrom function."""

    def test_nystrom_basic_shapes(self):
        """Test that randomized_nystrom returns correct shapes."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=3, key=key)

        assert isinstance(result, NystromResult)
        assert result.eigenvalues.shape == (3,)
        assert result.eigenvectors.shape == (3, 5)

    def test_nystrom_eigenvalues_positive(self):
        """Test that eigenvalues of a PSD operator are non-negative."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=3, key=key)

        assert jnp.all(result.eigenvalues >= 0)

    def test_nystrom_largest_eigenvalues(self):
        """Test that nystrom captures the largest eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=2, key=key)

        assert_allclose(sorted(result.eigenvalues), [4.0, 5.0], atol=1e-3)

    def test_nystrom_full_rank_all_eigenvalues(self):
        """Test that full-rank nystrom recovers all eigenvalues."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=5, key=key, oversampling=0)

        assert_allclose(sorted(result.eigenvalues), sorted(d), atol=1e-3)

    def test_nystrom_eigenvectors_orthonormal(self):
        """Test that returned eigenvectors are orthonormal."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=3, key=key)

        G = gram(result.eigenvectors, result.eigenvectors)
        assert_allclose(G, jnp.eye(3), atol=1e-5)

    def test_nystrom_oversampling(self):
        """Test nystrom with custom oversampling parameter."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=2, key=key, oversampling=5)

        assert result.eigenvalues.shape == (2,)
        assert_allclose(sorted(result.eigenvalues), [4.0, 5.0], atol=1e-3)

    def test_nystrom_pytree_structure(self):
        """Test nystrom with PyTree-structured operator."""
        d1 = jnp.array([1.0, 5.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        key = jax.random.PRNGKey(0)
        result = randomized_nystrom(A, rank=2, key=key)

        assert result.eigenvalues.shape == (2,)
        assert 'a' in result.eigenvectors
        assert 'b' in result.eigenvectors
        assert result.eigenvectors['a'].shape == (2, 2)
        assert result.eigenvectors['b'].shape == (2, 1)
        assert_allclose(sorted(result.eigenvalues), [3.0, 5.0], atol=1e-3)

    def test_nystrom_jit(self):
        """Test that randomized_nystrom is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        @jax.jit
        def compute(key):
            return randomized_nystrom(A, rank=2, key=key)

        result = compute(jax.random.PRNGKey(0))
        assert_allclose(sorted(result.eigenvalues), [4.0, 5.0], atol=1e-3)


class TestNystromPreconditioner:
    """Tests for NystromPreconditioner class."""

    def test_preconditioner_creation(self):
        """Test NystromPreconditioner can be created."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        assert M_inv.in_structure == M_inv.out_structure

    def test_preconditioner_structure_inferred(self):
        """Test that in_structure is correctly inferred from eigenvectors."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        assert M_inv.in_structure.shape == (3,)

    def test_preconditioner_explicit_structure(self):
        """Test NystromPreconditioner with explicitly provided structure."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        explicit_structure = as_structure(d)
        M_inv = NystromPreconditioner(result, in_structure=explicit_structure)

        assert M_inv.in_structure == explicit_structure

    def test_preconditioner_symmetric(self):
        """Test NystromPreconditioner is symmetric."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        assert M_inv.is_symmetric

    def test_preconditioner_transpose_is_self(self):
        """Test NystromPreconditioner transpose returns itself."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        assert M_inv.T is M_inv

    def test_preconditioner_identity_operator(self):
        """Test preconditioner on identity operator returns the input unchanged."""
        # For A = I, all eigenvalues = 1, lambda_k = 1, M^{-1} = I
        d = jnp.array([1.0, 1.0, 1.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=3, key=jax.random.PRNGKey(0), oversampling=0)
        M_inv = NystromPreconditioner(result)

        x = jnp.array([1.0, 2.0, 3.0])
        y = M_inv.mv(x)
        assert_allclose(y, x, atol=1e-4)

    def test_preconditioner_full_rank_scales_by_lambda_k(self):
        """Test that M^{-1} A x ≈ lambda_k * x for full-rank nystrom."""
        # For full-rank nystrom on a diagonal operator:
        # M^{-1} A = lambda_k * I, where lambda_k = min(eigenvalues)
        d = jnp.array([1.0, 2.0, 4.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=3, key=jax.random.PRNGKey(0), oversampling=0)
        M_inv = NystromPreconditioner(result)

        lambda_k = jnp.min(result.eigenvalues)
        x = jnp.array([1.0, 1.0, 1.0])
        y = M_inv.mv(A.mv(x))
        assert_allclose(y, lambda_k * x, atol=1e-4)

    def test_preconditioner_mv_output_shape(self):
        """Test that preconditioner mv preserves input shape."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=3, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        x = jnp.ones(5)
        y = M_inv.mv(x)
        assert y.shape == x.shape

    def test_preconditioner_pytree(self):
        """Test NystromPreconditioner with PyTree structure."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        x = {'a': jnp.array([1.0, 1.0]), 'b': jnp.array([1.0])}
        y = M_inv.mv(x)
        assert 'a' in y
        assert 'b' in y
        assert y['a'].shape == (2,)
        assert y['b'].shape == (1,)

    def test_preconditioner_jit(self):
        """Test NystromPreconditioner is JIT-compatible."""
        d = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=3, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        @jax.jit
        def apply(x):
            return M_inv.mv(x)

        x = jnp.ones(5)
        y = apply(x)
        assert y.shape == x.shape

    def test_preconditioner_call_syntax(self):
        """Test NystromPreconditioner can be applied with call syntax."""
        d = jnp.array([1.0, 2.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))

        result = randomized_nystrom(A, rank=2, key=jax.random.PRNGKey(0))
        M_inv = NystromPreconditioner(result)

        x = jnp.ones(3)
        y = M_inv(x)
        assert y.shape == x.shape
