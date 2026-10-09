"""Tests for the Conjugate Gradient solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import tree_equal
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

from furax import (
    BlockDiagonalOperator,
    DenseBlockDiagonalOperator,
    DiagonalOperator,
    IdentityOperator,
    IndexOperator,
)
from furax.linalg import CGResult, cg
from furax.tree import as_structure


def _diagonal_system(n: int = 5):
    """Return a diagonal SPD operator and a simple RHS."""
    d = jnp.arange(1, n + 1, dtype=float)
    A = DiagonalOperator(d, in_structure=as_structure(d))
    b = jnp.ones(n, dtype=float)
    x_true = b / d
    return A, b, x_true


def _dense_spd_system(
    n: int, condition: float, seed: int
) -> tuple[DenseBlockDiagonalOperator, jax.Array, jax.Array, jax.Array]:
    key = jax.random.key(seed)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n), dtype=jnp.float64))
    eigenvalues = jnp.geomspace(1.0 / condition, 1.0, n)
    matrix = (q * eigenvalues) @ q.T
    b = jax.random.normal(jax.random.key(seed + 1000), (n,), dtype=jnp.float64)
    A = DenseBlockDiagonalOperator(matrix, in_structure=as_structure(b))
    x_true = jnp.linalg.solve(matrix, b)
    return A, matrix, b, x_true


class TestCGBasic:
    def test_solves_diagonal_system(self):
        A, b, x_true = _diagonal_system()
        result = cg(A, b, max_steps=20)
        assert isinstance(result, CGResult)
        assert_allclose(result.solution, x_true, rtol=1e-10)

    def test_residuals(self):
        A, b, _ = _diagonal_system(n=20)
        result = cg(A, b, max_steps=20, rtol=0.0, atol=0.0)
        assert result.residuals.shape == (20,)
        assert_allclose(result.residuals[0], jnp.linalg.norm(b), rtol=1e-5)

    def test_custom_x0(self):
        A, b, _ = _diagonal_system()
        x0 = jax.random.normal(jax.random.key(0), b.shape, dtype=b.dtype)
        result_default = cg(A, b, max_steps=20)
        result_x0 = cg(A, b, x0, max_steps=20)
        assert_allclose(result_x0.solution, result_default.solution, rtol=1e-10)

    def test_iterations_count_when_x0_is_solution(self):
        A, b, x_true = _diagonal_system(n=5)
        result = cg(A, b, x_true, max_steps=20, rtol=1e-10, atol=0.0)
        assert int(result.num_steps) == 0

    def test_iterations_count_when_tol_is_zero(self):
        # iterates for max_iter even with true solution as starting vector
        A, b, x_true = _diagonal_system(n=10)
        result = cg(A, b, x_true, max_steps=5, rtol=0.0, atol=0.0)
        assert int(result.num_steps) == 5


class TestCGPreconditioner:
    def test_identity_preconditioner_same_result(self):
        A, b, _ = _diagonal_system()
        M = IdentityOperator(in_structure=as_structure(b))
        result = cg(A, b, max_steps=20)
        result_m = cg(A, b, max_steps=20, preconditioner=M)
        assert tree_equal(result, result_m)

    def test_exact_preconditioner_converges_in_one_step(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)
        # M = A^{-1} makes M A = I → converges in 1 iteration
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        result = cg(A, b, max_steps=20, preconditioner=M, rtol=1e-8, atol=0.0)
        assert int(result.num_steps) == 1


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

        result = cg(A, b, max_steps=20)
        assert tree_equal(result.solution, x_true, rtol=1e-10)


class TestCGStabilisation:
    def test_over_iteration_with_stabilisation_is_inert_after_convergence(self):
        A, _, b, x_true = _dense_spd_system(n=5, condition=1e3, seed=7)

        result = cg(A, b, max_steps=1000, rtol=0.0, atol=0.0)

        rel_error = jnp.linalg.norm(result.solution - x_true) / jnp.linalg.norm(x_true)
        assert_allclose(rel_error, 0.0, atol=1e-12)

    def test_stabilisation_still_tracks_true_residual_on_long_solve(self):
        A, matrix, b, _ = _dense_spd_system(n=50, condition=1e3, seed=0)

        recursive = cg(A, b, max_steps=200, rtol=0.0, atol=0.0, stabilise_every=0)
        stabilised = cg(A, b, max_steps=200, rtol=0.0, atol=0.0)

        norm_b = jnp.linalg.norm(b)
        recursive_true = jnp.linalg.norm(b - matrix @ recursive.solution) / norm_b
        stabilised_true = jnp.linalg.norm(b - matrix @ stabilised.solution) / norm_b
        recursive_reported = recursive.residuals[-1] / norm_b
        stabilised_reported = stabilised.residuals[-1] / norm_b

        # The recursive residual drifts away from the true one; the stabilised one does not.
        assert jnp.abs(stabilised_reported - stabilised_true) < jnp.abs(
            recursive_reported - recursive_true
        )


class TestCGJit:
    def test_jit_basic(self):
        A, b, x_true = _diagonal_system()

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=20)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-10)

    def test_jit_with_preconditioner(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        b = jnp.ones(5)
        x_true = b / d

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=20, preconditioner=M)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-10)

    def test_jit_with_stabilise(self):
        A, b, x_true = _diagonal_system(n=10)

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=30, stabilise_every=3)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-10)

    def test_jit_residuals_shape_preserved(self):
        A, b, _ = _diagonal_system()

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=13)

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
            return cg(A, b, max_steps=20)

        result = solve(b)
        x_true = {'a': jnp.array([1.0, 0.5]), 'b': jnp.array([1.0 / 3, 0.25])}
        assert tree_equal(result.solution, x_true, rtol=1e-10)


class TestCGGrad:
    """Gradient tests for the CG solver.

    With the default implicit differentiation, both modes work on the default loop. Unrolled
    differentiation is forward-mode only unless loop_kind='bounded' or 'checkpointed'.

    For f(b) = h(cg(A, b).solution), the Jacobian is A^{-1}: each column of
    jacfwd(solve)(b) is A^{-1} applied to the corresponding standard basis
    vector, by the implicit function theorem.
    """

    @pytest.mark.parametrize('differentiation', ['implicit', 'unrolled'])
    def test_jvp_wrt_b(self, differentiation):
        """Directional derivative of the solution w.r.t. b equals A^{-1} v."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)
        v = jax.random.normal(jax.random.key(42), b.shape)

        def solve(b):
            return cg(A, b, max_steps=20, differentiation=differentiation).solution

        _, jvp_val = jax.jvp(solve, (b,), (v,))
        # d(A^{-1} b)/db · v = A^{-1} v = v / d
        expected = v / d
        assert_allclose(jvp_val, expected, rtol=1e-10)

    @pytest.mark.parametrize('differentiation', ['implicit', 'unrolled'])
    def test_jacfwd_wrt_b_equals_inverse(self, differentiation):
        """Full Jacobian of the solution w.r.t. b is A^{-1}."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def solve(b):
            return cg(A, b, max_steps=20, differentiation=differentiation).solution

        jac = jax.jacfwd(solve)(b)
        expected = jnp.diag(1.0 / d)
        assert_allclose(jac, expected, atol=1e-10)

    @pytest.mark.parametrize('differentiation', ['implicit', 'unrolled'])
    def test_jvp_with_preconditioner(self, differentiation):
        """JVP still equals A^{-1} v when using a preconditioner."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        b = jnp.ones(5)
        v = jax.random.normal(jax.random.key(7), b.shape)

        def solve(b):
            return cg(
                A, b, max_steps=20, preconditioner=M, differentiation=differentiation
            ).solution

        _, jvp_val = jax.jvp(solve, (b,), (v,))
        expected = v / d
        assert_allclose(jvp_val, expected, rtol=1e-10)

    @pytest.mark.parametrize(
        'differentiation, loop_kind', [('implicit', 'lax'), ('unrolled', 'bounded')]
    )
    def test_grad_wrt_b(self, differentiation, loop_kind):
        """Reverse-mode AD: grad sum(A^{-1} b) = A^{-T} 1 = 1/d."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def f(b):
            result = cg(A, b, max_steps=20, loop_kind=loop_kind, differentiation=differentiation)
            return jnp.sum(result.solution)

        grad = jax.grad(f)(b)
        assert_allclose(grad, 1.0 / d, rtol=1e-10)

    @pytest.mark.parametrize(
        'differentiation, loop_kind', [('implicit', 'lax'), ('unrolled', 'bounded')]
    )
    def test_jacrev_wrt_b_equals_inverse(self, differentiation, loop_kind):
        """Full reverse-mode Jacobian of the solution w.r.t. b is A^{-1}."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def solve(b):
            return cg(
                A, b, max_steps=20, loop_kind=loop_kind, differentiation=differentiation
            ).solution

        jac = jax.jacrev(solve)(b)
        assert_allclose(jac, jnp.diag(1.0 / d), atol=1e-10)

    @pytest.mark.parametrize(
        'differentiation, loop_kind', [('implicit', 'lax'), ('unrolled', 'bounded')]
    )
    def test_grad_wrt_operator(self, differentiation, loop_kind):
        """The gradient w.r.t. the operator's matrix matches a dense solve on symmetric matrices.

        CG only sees the symmetric part of A, so the gradients are compared after projecting
        them onto symmetric matrices.
        """
        A, matrix, b, _ = _dense_spd_system(6, condition=100.0, seed=0)

        def f(matrix):
            op = DenseBlockDiagonalOperator(matrix, in_structure=A.in_structure)
            result = cg(
                op,
                b,
                rtol=1e-12,
                max_steps=50,
                loop_kind=loop_kind,
                differentiation=differentiation,
            )
            return jnp.sum(result.solution**2)

        grad = jax.grad(f)(matrix)
        expected = jax.grad(lambda m: jnp.sum(jnp.linalg.solve(m, b) ** 2))(matrix)
        assert_allclose(grad + grad.T, expected + expected.T, rtol=1e-8)

    @pytest.mark.parametrize('differentiated', ['operator', 'rhs'])
    def test_implicit_grad_wrt_one_block(self, differentiated):
        """Differentiating one block of a block system leaves the other block constant."""
        d1, d2 = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])
        b1, b2 = jnp.ones(3), jnp.ones(2)

        def f(d1, b1):
            A = BlockDiagonalOperator(
                {
                    'a': DiagonalOperator(d1, in_structure=as_structure(d1)),
                    'b': DiagonalOperator(d2, in_structure=as_structure(d2)),
                }
            )
            x = cg(A, {'a': b1, 'b': b2}, rtol=1e-12).solution
            return jnp.sum(x['a']) + jnp.sum(x['b'])

        if differentiated == 'operator':
            grad, expected = jax.grad(f, argnums=0)(d1, b1), -b1 / d1**2
        else:
            grad, expected = jax.grad(f, argnums=1)(d1, b1), 1 / d1
        assert_allclose(grad, expected, rtol=1e-10)

    def test_implicit_grad_with_non_array_operator_leaves(self):
        """Operator leaves that are not JAX values, like an `Ellipsis` index, are supported."""
        d = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([1.0, -1.0, 2.0])
        permutation = jnp.array([2, 0, 1])
        index = IndexOperator((..., permutation), in_structure=as_structure(d))

        def f(d):
            A = index.T @ DiagonalOperator(d, in_structure=as_structure(d)) @ index
            return jnp.sum(cg(A, b, rtol=1e-12).solution ** 2)

        def expected(d):
            # `index.T @ D @ index` is diagonal, with `d` permuted back.
            return jnp.sum((b / d[jnp.argsort(permutation)]) ** 2)

        assert_allclose(jax.grad(f)(d), jax.grad(expected)(d), rtol=1e-10)

    def test_implicit_hessian_wrt_b(self):
        """The implicit rule is itself differentiable: Hess ||A^{-1} b||^2 = 2 A^{-2}."""
        A, matrix, b, _ = _dense_spd_system(6, condition=100.0, seed=0)

        def f(b):
            return jnp.sum(cg(A, b, rtol=1e-12, max_steps=50).solution ** 2)

        inverse = jnp.linalg.inv(matrix)
        assert_allclose(jax.hessian(f)(b), 2 * inverse @ inverse, rtol=1e-8)

    def test_implicit_ignores_x0_and_preconditioner(self):
        """Implicit derivatives are those of A^{-1} b, which depends on neither."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def f(x0, p):
            M = DiagonalOperator(p, in_structure=as_structure(d))
            return jnp.sum(cg(A, b, x0, preconditioner=M, max_steps=2).solution)

        grad_x0, grad_p = jax.grad(f, argnums=(0, 1))(jnp.zeros(5), 1.0 / d)
        assert_allclose(grad_x0, 0.0)
        assert_allclose(grad_p, 0.0)

    def test_unrolled_grad_raises_with_lax_loop(self):
        """Unrolled differentiation of the default loop_kind='lax' is forward-mode only."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def f(b):
            return jnp.sum(cg(A, b, max_steps=20, differentiation='unrolled').solution)

        with pytest.raises(ValueError, match='[Rr]everse-mode'):
            jax.grad(f)(b)

    def test_invalid_differentiation_raises(self):
        A, b, _ = _diagonal_system()
        with pytest.raises(ValueError, match='differentiation'):
            cg(A, b, differentiation='finite')


class TestCGCurvature:
    """Negative curvature p^T A p < 0, which a positive definite A never produces."""

    def test_no_check_by_default(self):
        # Default assumes A positive definite and does not check; negative curvature is silent.
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(-d, in_structure=as_structure(d))  # negative definite
        b = jnp.ones(5)
        result = jax.block_until_ready(cg(A, b, max_steps=20))
        assert result.solution.shape == b.shape

    def test_error_mode_raises(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(-d, in_structure=as_structure(d))  # negative definite
        b = jnp.ones(5)
        with pytest.raises(Exception, match='negative curvature'):
            jax.block_until_ready(cg(A, b, max_steps=20, negative_curvature='error'))

    def test_truncate_stops_instead_of_raising(self):
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(-d, in_structure=as_structure(d))  # negative definite
        b = jnp.ones(5)
        # Bad curvature hits on the first direction: no step taken, solution stays at x0 (zeros).
        result = cg(A, b, max_steps=20, negative_curvature='truncate')
        assert int(result.num_steps) == 1
        assert_allclose(result.solution, jnp.zeros(5), atol=0.0)

    def test_truncate_matches_default_when_positive_definite(self):
        A, b, x_true = _diagonal_system()
        result = cg(A, b, max_steps=20, negative_curvature='truncate')
        assert_allclose(result.solution, x_true, rtol=1e-10)

    def test_invalid_negative_curvature_raises(self):
        A, b, _ = _diagonal_system()
        with pytest.raises(ValueError, match='negative_curvature'):
            cg(A, b, max_steps=20, negative_curvature='nope')


_AXIS_TYPES = {'explicit': AxisType.Explicit, 'auto': AxisType.Auto}


def _sharded_layout(ndim: int) -> tuple[tuple[int, ...], tuple[str, ...], P]:
    """Mesh shape, axis names and partition spec sharding the leading axes over all devices."""
    n = jax.device_count()
    if ndim == 1:
        return (n,), ('i',), P('i', None)
    if ndim == 2:
        assert n % 2 == 0, n
        return (2, n // 2), ('i', 'j'), P('i', 'j', None)
    raise ValueError(ndim)


@pytest.mark.distributed
class TestCGDistributed:
    """Multi-device CG over a vector sharded along its contracting axis."""

    K = 3  # replicated trailing dimension (amplitudes per block)

    @pytest.mark.parametrize('ndim', [1, 2], ids=['mesh1d', 'mesh2d'])
    @pytest.mark.parametrize('axis_type', ['explicit', 'auto'])
    def test_solves_vector_sharded_on_contracting_axis(self, axis_type: str, ndim: int) -> None:
        axis = _AXIS_TYPES[axis_type]
        mesh_shape, axis_names, spec = _sharded_layout(ndim)

        # SPD diagonal system over a vector of shape (*mesh_shape, K).
        vec_shape = (*mesh_shape, self.K)
        size = int(np.prod(vec_shape))
        diag = (1.0 + jnp.arange(size, dtype=float)).reshape(vec_shape)
        b = jnp.ones(vec_shape)
        x_true = b / diag

        # Single-device reference (no mesh): plain replicated solve.
        reference = cg(
            DiagonalOperator(diag, in_structure=as_structure(b)), b, max_steps=50, rtol=1e-10
        )
        assert_allclose(np.asarray(reference.solution), x_true, rtol=1e-9)

        mesh = jax.make_mesh(mesh_shape, axis_names, axis_types=(axis,) * ndim)
        with jax.set_mesh(mesh):
            shard = NamedSharding(mesh, spec)
            b_s = jax.device_put(b, shard)
            diag_s = jax.device_put(diag, shard)
            A = DiagonalOperator(diag_s, in_structure=as_structure(b_s))
            M = DiagonalOperator(1.0 / diag_s, in_structure=as_structure(b_s))

            plain = jax.jit(lambda b: cg(A, b, max_steps=50, rtol=1e-10))(b_s)
            # preconditioner exercises M(r) under sharding; exact M -> converges in one step.
            precond = jax.jit(lambda b: cg(A, b, max_steps=50, rtol=1e-10, preconditioner=M))(b_s)

        for result in (plain, precond):
            assert_allclose(np.asarray(result.solution), x_true, rtol=1e-9)
        assert int(precond.num_steps) == 1, int(precond.num_steps)

        # Under explicit axes the sharding is part of the type and propagates to the solution;
        # under auto axes JAX is free to choose, so only assert the spec in the explicit case.
        if axis is AxisType.Explicit:
            assert plain.solution.sharding.spec == spec, plain.solution.sharding.spec
