"""Tests for enlarged conjugate gradients and their block Krylov search space."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from equinox import tree_equal
from jax.flatten_util import ravel_pytree
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    DenseBlockDiagonalOperator,
    DiagonalOperator,
    HomothetyOperator,
    tree,
)
from furax.linalg import ECGResult, cg, ecg
from furax.linalg._ecg import _gram, _orthonormalize, _partition_labels
from furax.obs.stokes import StokesIQU


def _system(n=12, condition=10.0, seed=0, dtype=jnp.float64):
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (n, n), dtype=dtype))
    matrix = (q * jnp.geomspace(1.0 / condition, 1.0, n, dtype=dtype)) @ q.T
    b = jax.random.normal(jax.random.key(seed + 1), (n,), dtype=dtype)
    return DenseBlockDiagonalOperator(matrix, in_structure=tree.as_structure(b)), matrix, b


class _CoupledPyTreeOperator(AbstractLinearOperator):
    """Dense reference operator coupling every coordinate across arbitrary leaves."""

    matrix: jax.Array

    def mv(self, x):
        vector, unravel = ravel_pytree(x)
        return unravel(self.matrix @ vector)


class TestECG:
    @pytest.mark.parametrize('enlargement', [1, 2, 4, 16])
    @pytest.mark.parametrize('stabilise_every', [0, 10])
    def test_dense_spd(self, enlargement, stabilise_every):
        A, matrix, b = _system()
        result = ecg(
            A,
            b,
            enlargement=enlargement,
            stabilise_every=stabilise_every,
            max_steps=100,
            rtol=1e-10,
        )
        assert isinstance(result, ECGResult)
        assert_allclose(result.solution, jnp.linalg.solve(matrix, b), rtol=1e-8, atol=1e-9)
        assert tree.norm(tree.sub(b, A(result.solution))) <= 1e-10 * tree.norm(b)

    @pytest.mark.parametrize('x0_kind', ['zero', 'random', 'solution'])
    def test_initial_guess(self, x0_kind):
        d = jnp.arange(1.0, 9.0)
        b = jnp.ones(8)
        A = DiagonalOperator(d, in_structure=tree.as_structure(b))
        x0 = {'zero': jnp.zeros(8), 'random': -b, 'solution': b / d}[x0_kind]
        result = ecg(A, b, x0, max_steps=20, rtol=1e-10)
        assert_allclose(result.solution, b / d, rtol=1e-9)
        if x0_kind == 'solution':
            assert result.num_steps == 0

    @pytest.mark.parametrize('max_steps', [0, 1, 20])
    @pytest.mark.parametrize('rtol', [0.0, 1e-10])
    def test_zero_residual(self, max_steps, rtol):
        b = jnp.zeros(8)
        A = DiagonalOperator(jnp.ones(8), in_structure=tree.as_structure(b))
        result = ecg(A, b, max_steps=max_steps, rtol=rtol)
        assert_array_equal(result.solution, b)
        assert_array_equal(result.residuals, jnp.zeros(max_steps))
        assert result.num_steps == (max_steps if rtol == 0 else 0)

    @pytest.mark.parametrize('enlargement', [2, 4])
    def test_fixed_iterations_are_inert_after_convergence(self, enlargement):
        A, matrix, b = _system(n=8, condition=100.0)
        result = ecg(A, b, enlargement=enlargement, rtol=0, atol=0, max_steps=100)
        assert result.num_steps == 100
        assert_allclose(result.solution, jnp.linalg.solve(matrix, b), rtol=1e-10)
        assert_allclose(result.residuals[-10:], result.residuals[-1], atol=1e-13)

    def test_cg_equivalence(self):
        A, _, b = _system()
        kwargs = {'max_steps': 30, 'rtol': 0.0, 'stabilise_every': 3}
        enlarged = ecg(A, b, enlargement=1, **kwargs)
        classic = cg(A, b, **kwargs)
        assert tree_equal(tuple(enlarged), tuple(classic))

    @pytest.mark.parametrize('custom_partition', [False, True])
    def test_coupled_pytree(self, custom_partition):
        _, matrix, flat_b = _system(n=9)
        b = {'a': flat_b[:6].reshape(2, 3), 'b': [flat_b[6:8], flat_b[8]]}
        A = _CoupledPyTreeOperator(matrix, in_structure=tree.as_structure(b))
        labels = {'a': jnp.zeros((2, 3), dtype=int), 'b': [jnp.ones(2, dtype=int), jnp.array(3)]}
        result = jax.jit(
            lambda b: ecg(
                A, b, partition=labels if custom_partition else None, max_steps=40, rtol=1e-10
            )
        )(b)
        solution, _ = ravel_pytree(result.solution)
        assert_allclose(solution, jnp.linalg.solve(matrix, flat_b), rtol=1e-8)
        assert tree.as_structure(result.solution) == tree.as_structure(b)

    def test_partition_crosses_leaf_boundaries(self):
        b = {'a': jnp.ones((2, 3)), 'b': [jnp.ones(2), jnp.array(1.0)]}
        labels = _partition_labels(b, None, 3)
        assert_array_equal(labels['a'], [[0, 0, 0], [1, 1, 1]])
        assert_array_equal(labels['b'][0], [2, 2])
        assert labels['b'][1] == 2

    @pytest.mark.parametrize('size, width', [(10, 4), (3, 8), (9, 1)])
    def test_balanced_partition_with_remainders(self, size, width):
        labels = _partition_labels(jnp.ones(size), None, width)
        counts = np.bincount(labels, minlength=width)
        assert counts.max() - counts.min() <= 1
        assert np.all(np.diff(labels) >= 0)
        assert counts.sum() == size

    def test_stokes_container(self):
        b = StokesIQU(jnp.ones(4), jnp.arange(4.0), -jnp.ones(4))
        A = DiagonalOperator(jnp.arange(1.0, 5.0), in_structure=tree.as_structure(b))
        result = jax.jit(lambda b: ecg(A, b, max_steps=20, rtol=1e-10))(b)
        assert isinstance(result.solution, StokesIQU)
        assert_allclose(result.solution.data, b.data / jnp.arange(1.0, 5.0), atol=1e-10)

    def test_mixed_dtypes_and_empty_leaf(self):
        b = {'a': jnp.ones((2, 3), dtype=jnp.float32), 'b': jnp.array(1.0), 'c': jnp.zeros(0)}
        A = BlockDiagonalOperator(
            {
                key: HomothetyOperator(
                    jnp.array(2.0, dtype=value.dtype), in_structure=tree.as_structure(value)
                )
                for key, value in b.items()
            }
        )
        result = jax.jit(lambda b: ecg(A, b, max_steps=20, rtol=1e-5))(b)
        assert jax.tree.structure(result.solution) == jax.tree.structure(b)
        for actual, original in zip(jax.tree.leaves(result.solution), jax.tree.leaves(b)):
            assert actual.shape == original.shape
            assert actual.dtype == original.dtype
        for value in jax.tree.leaves(result.solution):
            assert_allclose(value, 0.5, atol=1e-6)

    def test_scalar_weak_type_does_not_define_the_vector_space(self):
        b = jnp.array(1.0)
        x0 = jnp.array(0.0, dtype=b.dtype)
        A = HomothetyOperator(2.0, in_structure=tree.as_structure(x0))
        assert_allclose(ecg(A, b, x0).solution, 0.5)

    @pytest.mark.parametrize('partition', [jnp.zeros(8, dtype=int), jnp.arange(8) % 4])
    def test_empty_and_converging_groups(self, partition):
        d = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
        b = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        A = DiagonalOperator(d, in_structure=tree.as_structure(b))
        result = ecg(A, b, partition=partition, rtol=1e-10, max_steps=30)
        assert_allclose(result.solution, b / d, atol=1e-10)

    def test_callbacks_and_history(self):
        A, _, b = _system(n=8)
        records = []

        def callback(step, norm):
            records.append((int(step), float(norm)))

        result = jax.jit(
            lambda b: ecg(
                A, b, enlargement=2, max_steps=20, rtol=1e-10, iteration_callback=callback
            )
        )(b)
        jax.block_until_ready(result)
        jax.effects_barrier()
        records.sort()
        assert len(records) == int(result.num_steps)
        assert [step for step, _ in records] == list(range(int(result.num_steps)))
        assert_allclose(result.residuals[0], tree.norm(b))
        assert_allclose(result.residuals[1 : len(records) + 1], [norm for _, norm in records])
        assert_allclose(result.residuals[len(records)], tree.norm(tree.sub(b, A(result.solution))))


class TestECGMathematics:
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
    @pytest.mark.parametrize(
        'case', ['orthogonal', 'near_dependent', 'rank_deficient', 'zero', 'projected']
    )
    def test_block_normalization(self, dtype, case):
        vectors = jnp.eye(4, 8, dtype=dtype)
        diagonal = jnp.arange(1, 9, dtype=dtype)
        rank_rtol = 100 * jnp.finfo(dtype).eps
        expected_rank = jnp.ones(4, dtype=bool)
        if case == 'near_dependent':
            delta = 1e-3 if dtype == jnp.float32 else 1e-8
            vectors = vectors.at[1].set(vectors[0] + delta * vectors[1])
        elif case == 'rank_deficient':
            vectors = vectors.at[1].set(vectors[0])
            expected_rank = expected_rank.at[1].set(False)
        reference_norms = jax.vmap(tree.norm)(vectors)
        if case in ('zero', 'projected'):
            vectors = vectors * (0 if case == 'zero' else rank_rtol / 10)
            expected_rank = jnp.zeros(4, dtype=bool)
        q, aq, active = jax.jit(
            lambda v: _orthonormalize(v, v * diagonal, reference_norms, rank_rtol)
        )(vectors)
        tolerance = 1e-5 if dtype == jnp.float32 else 1e-12
        assert_array_equal(active, expected_rank)
        assert_allclose(q @ aq.T, jnp.diag(expected_rank.astype(dtype)), atol=tolerance)
        assert_allclose(aq, q * diagonal, atol=tolerance)
        assert q.dtype == aq.dtype == dtype

    @pytest.mark.parametrize('condition', [100.0, 1e4])
    def test_long_recurrence_on_ill_conditioned_system(self, condition):
        A, matrix, b = _system(n=32, condition=condition)
        result = ecg(A, b, enlargement=4, max_steps=100, stabilise_every=0, rtol=1e-8)
        assert tree.norm(b - A(result.solution)) <= 1e-8 * tree.norm(b)
        assert_allclose(result.solution, jnp.linalg.solve(matrix, b), rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('steps', [1, 2, 3])
    def test_enlarged_krylov_galerkin_solution(self, steps):
        A, matrix, b = _system(n=12)
        x0 = jnp.linspace(-1.0, 1.0, 12)
        r0 = np.asarray(b - matrix @ x0)
        labels = np.arange(12) * 2 // 12
        block = r0[:, None] * (labels[:, None] == np.arange(2))
        powers = [block]
        for _ in range(steps - 1):
            powers.append(np.asarray(matrix) @ powers[-1])
        basis, _ = np.linalg.qr(np.concatenate(powers, axis=1))
        expected = np.asarray(x0) + basis @ np.linalg.solve(
            basis.T @ np.asarray(matrix) @ basis, basis.T @ r0
        )
        result = ecg(A, b, x0, enlargement=2, max_steps=steps, stabilise_every=0, atol=0, rtol=0)
        assert_allclose(result.solution, expected, rtol=1e-10, atol=1e-11)
        assert_allclose(basis.T @ (b - matrix @ result.solution), 0, atol=1e-11)

    def test_rank_deficient_block_orthonormalization(self):
        A, _, b = _system(n=8)
        vectors = jnp.stack([b, 2 * b, jnp.zeros_like(b), jnp.arange(8.0)])
        q, aq, active = _orthonormalize(
            vectors,
            jax.vmap(A)(vectors),
            jax.vmap(tree.norm)(vectors),
            100 * jnp.finfo(b.dtype).eps,
        )
        assert_array_equal(active, [True, False, False, True])
        assert_allclose(_gram(q, aq), jnp.diag(active.astype(float)), atol=1e-12)
        assert_allclose(aq, jax.vmap(A)(q), atol=1e-12)

    def test_periodic_restart_tracks_true_residual(self):
        A, matrix, b = _system(n=16, condition=100.0)
        result = ecg(A, b, max_steps=101, stabilise_every=5, atol=0, rtol=0)
        assert_allclose(result.residuals[100], tree.norm(b - matrix @ result.solution), atol=1e-12)
        assert_allclose(result.solution, jnp.linalg.solve(matrix, b), rtol=1e-8)


class TestECGValidation:
    @pytest.mark.parametrize(
        'b', [jnp.ones(2, dtype=int), jnp.ones(2, dtype=complex), jnp.zeros(0)]
    )
    def test_invalid_rhs(self, b):
        A = HomothetyOperator(1.0, in_structure=tree.as_structure(b))
        with pytest.raises(ValueError, match='floating-point|scalar coordinate'):
            ecg(A, b)

    def test_incompatible_operator(self):
        A, _, b = _system(n=8)
        with pytest.raises(ValueError, match='structure of b'):
            ecg(A, b[:7])

    def test_incompatible_initial_guess(self):
        A, _, b = _system(n=8)
        with pytest.raises(ValueError, match='x0 must match'):
            ecg(A, b, b[:7])

    def test_inconsistent_singular_system(self):
        b = jnp.ones(8)
        A = DiagonalOperator(jnp.arange(8.0), in_structure=tree.as_structure(b))
        with pytest.raises(Exception, match='zero curvature'):
            ecg(A, b, enlargement=8).solution.block_until_ready()

    @pytest.mark.parametrize(
        'kwargs, message',
        [
            ({'enlargement': 0}, 'enlargement'),
            ({'enlargement': 1.5}, 'enlargement'),
            ({'max_steps': -1}, 'max_steps'),
            ({'stabilise_every': -1}, 'stabilise_every'),
            ({'rtol': -1}, 'nonnegative'),
            ({'atol': -1}, 'nonnegative'),
            ({'loop_kind': 'invalid'}, 'loop_kind'),
            ({'partition': {'a': jnp.zeros(8, dtype=int)}}, 'PyTree'),
            ({'partition': jnp.zeros(7, dtype=int)}, 'shapes'),
            ({'partition': jnp.zeros(8)}, 'integer'),
        ],
    )
    def test_invalid_arguments(self, kwargs, message):
        A, _, b = _system(n=8)
        with pytest.raises(ValueError, match=message):
            ecg(A, b, **kwargs)

    @pytest.mark.parametrize('label', [-1, 4])
    def test_invalid_partition_labels_under_jit(self, label):
        A, _, b = _system(n=8)
        with pytest.raises(Exception, match='partition labels'):
            jax.jit(lambda labels: ecg(A, b, partition=labels))(
                jnp.full(8, label, dtype=int)
            ).solution.block_until_ready()

    @pytest.mark.parametrize(
        'diagonal, message',
        [
            (-jnp.ones(8), 'negative curvature'),
            (jnp.zeros(8), 'zero curvature'),
        ],
    )
    def test_non_spd(self, diagonal, message):
        b = jnp.ones(8)
        A = DiagonalOperator(diagonal, in_structure=tree.as_structure(b))
        with pytest.raises(Exception, match=message):
            ecg(A, b).solution.block_until_ready()


class TestECGAutodiff:
    def test_operator_derivative(self):
        _, matrix, b = _system(n=8)

        def loss(scale):
            A = DenseBlockDiagonalOperator(scale * matrix, in_structure=tree.as_structure(b))
            return jnp.sum(
                ecg(A, b, enlargement=2, max_steps=12, rtol=1e-10, loop_kind='bounded').solution
            )

        assert_allclose(jax.grad(loss)(1.0), -jnp.sum(jnp.linalg.solve(matrix, b)), rtol=1e-8)

    @pytest.mark.parametrize('loop_kind', ['lax', 'bounded'])
    def test_jvp(self, loop_kind):
        A, matrix, b = _system(n=8)
        v = jnp.arange(8.0)
        solve = lambda b: (
            ecg(A, b, enlargement=2, max_steps=12, rtol=1e-10, loop_kind=loop_kind).solution
        )
        _, actual = jax.jvp(solve, (b,), (v,))
        assert_allclose(actual, jnp.linalg.solve(matrix, v), rtol=1e-8, atol=1e-9)

    @pytest.mark.parametrize('loop_kind', ['bounded', 'checkpointed'])
    def test_reverse_mode(self, loop_kind):
        A, matrix, b = _system(n=8)

        def loss(b):
            return jnp.sum(
                ecg(A, b, enlargement=2, max_steps=12, rtol=1e-10, loop_kind=loop_kind).solution
            )

        assert_allclose(jax.grad(loss)(b), jnp.linalg.solve(matrix, jnp.ones(8)), rtol=1e-8)


@pytest.mark.insubprocess
def test_float32_without_x64():
    jax.config.update('jax_enable_x64', False)
    A, matrix, b = _system(n=8, dtype=jnp.float32)
    result = jax.jit(lambda b: ecg(A, b, max_steps=30, rtol=1e-5))(b)
    assert result.solution.dtype == result.residuals.dtype == jnp.float32
    assert_allclose(result.solution, jnp.linalg.solve(matrix, b), rtol=1e-4, atol=1e-4)
    assert tree.norm(b - A(result.solution)) <= 1e-5 * tree.norm(b)


@pytest.mark.distributed
@pytest.mark.parametrize('axis_type', [AxisType.Explicit, AxisType.Auto])
@pytest.mark.parametrize('ndim', [1, 2])
def test_sharded_contracting_axis(axis_type, ndim):
    n = jax.device_count()
    mesh_shape = (n,) if ndim == 1 else (2, n // 2)
    axes = ('i',) if ndim == 1 else ('i', 'j')
    b = jnp.ones((*mesh_shape, 4))
    diagonal = (1 + jnp.arange(n * 4, dtype=float)).reshape(b.shape)
    mesh = jax.make_mesh(mesh_shape, axes, axis_types=(axis_type,) * ndim)
    with jax.set_mesh(mesh):
        sharding = NamedSharding(mesh, P(*axes, None))
        b = jax.device_put(b, sharding)
        diagonal = jax.device_put(diagonal, sharding)
        A = DiagonalOperator(diagonal, in_structure=tree.as_structure(b))
        result = jax.jit(lambda b: ecg(A, b, max_steps=80, rtol=1e-10))(b)
        assert_allclose(result.solution, b / diagonal, rtol=1e-8)
        if axis_type is AxisType.Explicit:
            assert result.solution.sharding.spec == sharding.spec
