"""Tests for the Conjugate Gradient solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import tree_equal
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
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
        result = cg(A, b, max_steps=20)
        assert isinstance(result, CGResult)
        assert_allclose(result.solution, x_true, rtol=1e-4)

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
        assert_allclose(result_x0.solution, result_default.solution, rtol=1e-5)

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
        assert tree_equal(result.solution, x_true, rtol=1e-4)


class TestCGJit:
    def test_jit_basic(self):
        A, b, x_true = _diagonal_system()

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=20)

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
            return cg(A, b, max_steps=20, preconditioner=M)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-4)

    def test_jit_with_stabilise(self):
        A, b, x_true = _diagonal_system(n=10)

        @jax.jit
        def solve(b):
            return cg(A, b, max_steps=30, stabilise_every=3)

        result = solve(b)
        assert_allclose(result.solution, x_true, rtol=1e-4)

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
        assert tree_equal(result.solution, x_true, rtol=1e-4)


class TestCGGrad:
    """Gradient tests for the CG solver.

    Forward-mode (jvp/jacfwd) works with the default loop; reverse-mode (grad/vjp)
    needs loop_kind='bounded'.

    For f(b) = h(cg(A, b).solution), the Jacobian is A^{-1}: each column of
    jacfwd(solve)(b) is A^{-1} applied to the corresponding standard basis
    vector, by the implicit function theorem.
    """

    def test_jvp_wrt_b(self):
        """Directional derivative of the solution w.r.t. b equals A^{-1} v."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)
        v = jax.random.normal(jax.random.key(42), b.shape)

        def solve(b):
            return cg(A, b, max_steps=20).solution

        _, jvp_val = jax.jvp(solve, (b,), (v,))
        # d(A^{-1} b)/db · v = A^{-1} v = v / d
        expected = v / d
        assert_allclose(jvp_val, expected, rtol=1e-4)

    def test_jacfwd_wrt_b_equals_inverse(self):
        """Full Jacobian of the solution w.r.t. b is A^{-1}."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def solve(b):
            return cg(A, b, max_steps=20).solution

        jac = jax.jacfwd(solve)(b)
        expected = jnp.diag(1.0 / d)
        assert_allclose(jac, expected, atol=1e-4)

    def test_jvp_with_preconditioner(self):
        """JVP still equals A^{-1} v when using a preconditioner."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        M = DiagonalOperator(1.0 / d, in_structure=as_structure(d))
        b = jnp.ones(5)
        v = jax.random.normal(jax.random.key(7), b.shape)

        def solve(b):
            return cg(A, b, max_steps=20, preconditioner=M).solution

        _, jvp_val = jax.jvp(solve, (b,), (v,))
        expected = v / d
        assert_allclose(jvp_val, expected, rtol=1e-4)

    def test_grad_wrt_b(self):
        """Reverse-mode AD with loop_kind='bounded': grad sum(A^{-1} b) = A^{-T} 1 = 1/d."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def f(b):
            return jnp.sum(cg(A, b, max_steps=20, loop_kind='bounded').solution)

        grad = jax.grad(f)(b)
        assert_allclose(grad, 1.0 / d, rtol=1e-4)

    def test_jacrev_wrt_b_equals_inverse(self):
        """Full reverse-mode Jacobian of the solution w.r.t. b is A^{-1}."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def solve(b):
            return cg(A, b, max_steps=20, loop_kind='bounded').solution

        jac = jax.jacrev(solve)(b)
        assert_allclose(jac, jnp.diag(1.0 / d), atol=1e-4)

    def test_grad_raises_with_default_lax_loop(self):
        """The default loop_kind='lax' is forward-mode only; reverse-mode raises."""
        d = jnp.arange(1.0, 6.0)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        b = jnp.ones(5)

        def f(b):
            return jnp.sum(cg(A, b, max_steps=20).solution)

        with pytest.raises(ValueError, match='[Rr]everse-mode'):
            jax.grad(f)(b)


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
        assert_allclose(result.solution, x_true, rtol=1e-4)

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
