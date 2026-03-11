"""Tests for the Conjugate Gradient solver."""

import jax
import jax.numpy as jnp
from equinox import tree_equal
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DiagonalOperator, IdentityOperator
from furax.linalg import CGResult, cg
from furax.tree import as_structure


def _diagonal_system(n: int = 5):
    """Return a diagonal SPD operator and a simple RHS."""
    d = jnp.arange(1, n + 1, dtype=float)
    A = DiagonalOperator(d, in_structure=as_structure(d))
    b = jnp.ones(n, dtype=float)
    x_true = b / d
    return A, b, x_true


class TestCGBasic:
    def test_solves_diagonal_system(self):
        A, b, x_true = _diagonal_system()
        result = cg(A, b, max_iter=20)
        assert isinstance(result, CGResult)
        assert_allclose(result.solution, x_true, rtol=1e-4)

    def test_residuals(self):
        A, b, _ = _diagonal_system(n=20)
        result = cg(A, b, max_iter=20, rtol=0.0, atol=0.0)
        assert result.residuals.shape == (20,)
        assert_allclose(result.residuals[0], jnp.linalg.norm(b), rtol=1e-5)

    def test_custom_x0(self):
        A, b, x_true = _diagonal_system()
        x0 = jax.random.normal(jax.random.key(0), b.shape, dtype=b.dtype)
        result_default = cg(A, b, max_iter=20)
        result_x0 = cg(A, b, x0, max_iter=20)
        assert_allclose(result_x0.solution, result_default.solution, rtol=1e-5)

    def test_iterations_count_when_x0_is_solution(self):
        A, b, x_true = _diagonal_system(n=5)
        result = cg(A, b, x_true, max_iter=20, rtol=1e-10, atol=0.0)
        assert int(result.iterations) == 0

    def test_iterations_count_when_tol_is_zero(self):
        # iterates for max_iter even with true solution as starting vector
        A, b, x_true = _diagonal_system(n=10)
        result = cg(A, b, x_true, max_iter=5, rtol=0.0, atol=0.0)
        assert int(result.iterations) == 5


class TestCGPreconditioner:
    def test_identity_preconditioner_same_result(self):
        A, b, _ = _diagonal_system()
        M = IdentityOperator(in_structure=as_structure(b))
        result = cg(A, b, max_iter=20)
        result_m = cg(A, b, max_iter=20, preconditioner=M)
        assert tree_equal(result, result_m)

    def test_exact_preconditioner_converges_in_one_step(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)
        # M = A^{-1} makes M A = I → converges in 1 iteration
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        result = cg(A, b, max_iter=20, preconditioner=M, rtol=1e-8, atol=0.0)
        assert int(result.iterations) == 1


class TestCGPyTree:
    def test_pytree_rhs(self):
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0, 4.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )
        b = {'a': jnp.ones(2), 'b': jnp.ones(2)}
        x_true = {'a': jnp.array([1.0, 0.5]), 'b': jnp.array([1.0 / 3, 0.25])}

        result = cg(A, b, max_iter=20)
        assert tree_equal(result.solution, x_true, rtol=1e-4)


class TestCGJit:
    def test_jit_basic(self):
        A, b, x_true = _diagonal_system()

        @jax.jit
        def solve(b):
            return cg(A, b, max_iter=20)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-4)

    def test_jit_with_preconditioner(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        b = jnp.ones(5)
        x_true = b / d

        @jax.jit
        def solve(b):
            return cg(A, b, max_iter=20, preconditioner=M)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-4)

    def test_jit_with_stabilise(self):
        A, b, x_true = _diagonal_system(n=10)

        @jax.jit
        def solve(b):
            return cg(A, b, max_iter=30, stabilise_every=3)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-4)

    def test_jit_residuals_shape_preserved(self):
        A, b, _ = _diagonal_system()

        @jax.jit
        def solve(b):
            return cg(A, b, max_iter=13)

        result = solve(b)
        assert result.residuals.shape == (13,)

    def test_jit_pytree(self):
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0, 4.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )
        b = {'a': jnp.ones(2), 'b': jnp.ones(2)}

        @jax.jit
        def solve(b):
            return cg(A, b, max_iter=20)

        result = solve(b)
        x_true = {'a': jnp.array([1.0, 0.5]), 'b': jnp.array([1.0 / 3, 0.25])}
        assert tree_equal(result.solution, x_true, rtol=1e-4)
