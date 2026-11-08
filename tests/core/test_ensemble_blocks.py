"""Ensembles as block containers: the composites close an ensemble through the engine."""

import dataclasses
import re

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator, HomothetyOperator
from furax.core import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    Ensemble,
    IndexOperator,
    ListEnsemble,
    Policy,
    StackedEnsemble,
    lift,
)

N, M = 6, 4


def _diagonals() -> jax.Array:
    return jnp.arange(float(N * M)).reshape(N, M) + 1.0


def _ensemble(representation: str, scale: float = 1.0) -> Ensemble:
    diagonals = scale * _diagonals()
    if representation == 'list':
        return ListEnsemble([DiagonalOperator(d) for d in diagonals])
    return lift(lambda m: DiagonalOperator(m['d']), StackedEnsemble(N, {'d': diagonals}, 1))


def _x() -> jax.Array:
    return jnp.linspace(1.0, 2.0, M)


def _xs() -> jax.Array:
    return jnp.outer(jnp.arange(1.0, N + 1), _x())


@pytest.mark.parametrize('representation', ['list', 'stacked'])
class TestClosures:
    def test_addition(self, representation):
        operator = AdditionOperator(_ensemble(representation))
        assert_allclose(operator(_x()), (_diagonals() * _x()).sum(axis=0))

    def test_block_row(self, representation):
        operator = BlockRowOperator(_ensemble(representation))
        assert_allclose(operator(_xs()), (_diagonals() * _xs()).sum(axis=0))

    def test_block_column(self, representation):
        operator = BlockColumnOperator(_ensemble(representation))
        assert_allclose(operator(_x()), _diagonals() * _x())

    def test_block_diagonal(self, representation):
        operator = BlockDiagonalOperator(_ensemble(representation))
        assert_allclose(operator(_xs()), _diagonals() * _xs())

    def test_transpose(self, representation):
        row = BlockRowOperator(_ensemble(representation))
        assert isinstance(row.T, BlockColumnOperator)
        assert isinstance(row.T.blocks, Ensemble)
        assert_allclose(row.T(_x()), _diagonals() * _x())
        assert isinstance(row.T.T, BlockRowOperator)

    def test_structures(self, representation):
        row = BlockRowOperator(_ensemble(representation))
        in_leaves = jax.tree.leaves(row.in_structure)
        assert in_leaves[0].shape == (N, M)  # per member, stacked
        assert jax.tree.leaves(row.out_structure)[0].shape == (M,)  # summed
        column = BlockColumnOperator(_ensemble(representation))
        assert jax.tree.leaves(column.in_structure)[0].shape == (M,)
        assert jax.tree.leaves(column.out_structure)[0].shape == (N, M)

    def test_as_matrix(self, representation):
        operator = AdditionOperator(_ensemble(representation))
        assert_allclose(operator.as_matrix(), np.diag(_diagonals().sum(axis=0)))


@pytest.mark.parametrize('representation', ['list', 'stacked'])
class TestMemberAxis:
    """The composites over an ensemble whose stacked sides carry the member axis at 1."""

    def test_block_row(self, representation):
        operator = BlockRowOperator(_ensemble(representation), axes=1)
        assert_allclose(operator(_xs().T), (_diagonals() * _xs()).sum(axis=0))

    def test_block_column(self, representation):
        operator = BlockColumnOperator(_ensemble(representation), axes=1)
        assert_allclose(operator(_x()), (_diagonals() * _x()).T)

    def test_block_diagonal(self, representation):
        operator = BlockDiagonalOperator(_ensemble(representation), axes=1)
        assert_allclose(operator(_xs().T), (_diagonals() * _xs()).T)

    def test_structures(self, representation):
        row = BlockRowOperator(_ensemble(representation), axes=1)
        assert jax.tree.leaves(row.in_structure)[0].shape == (M, N)
        assert jax.tree.leaves(row.out_structure)[0].shape == (M,)
        column = BlockColumnOperator(_ensemble(representation), axes=1)
        assert jax.tree.leaves(column.in_structure)[0].shape == (M,)
        assert jax.tree.leaves(column.out_structure)[0].shape == (M, N)

    def test_transpose_and_reduce_keep_the_axes(self, representation):
        row = BlockRowOperator(_ensemble(representation), axes=1)
        assert isinstance(row.T, BlockColumnOperator)
        assert row.T.axes == 1
        assert_allclose(row.T(_x()), (_diagonals() * _x()).T)
        assert row.T.T.axes == 1
        diagonal = BlockDiagonalOperator(_ensemble(representation), axes=1)
        reduced = diagonal.reduce()
        assert isinstance(reduced, BlockDiagonalOperator)
        assert reduced.axes == 1
        inverse = diagonal.inverse()
        assert isinstance(inverse, BlockDiagonalOperator)
        assert inverse.axes == 1
        assert_allclose(inverse(diagonal(_xs().T)), _xs().T, atol=1e-12)

    def test_as_matrix(self, representation):
        # the member entries interleave in the flattened space: built from the basis
        diagonal = BlockDiagonalOperator(_ensemble(representation), axes=1)
        x = _xs().T
        assert_allclose(diagonal.as_matrix() @ x.ravel(), diagonal(x).ravel())

    def test_composition_rules_keep_the_axes(self, representation):
        row = BlockRowOperator(_ensemble(representation), axes=1)
        diagonal = BlockDiagonalOperator(_ensemble(representation, scale=2.0), axes=1)
        reduced = (row @ diagonal).reduce()
        assert isinstance(reduced, BlockRowOperator)
        assert reduced.axes == 1
        assert_allclose(reduced(_xs().T), row(diagonal(_xs().T)))


def test_blocks_with_different_axes_do_not_reduce():
    diagonals = jnp.arange(16.0).reshape(4, 4) + 1.0
    ensemble = lift(lambda m: DiagonalOperator(m['d']), StackedEnsemble(4, {'d': diagonals}, 1))
    left = BlockDiagonalOperator(ensemble, axes=1)
    right = BlockDiagonalOperator(ensemble)  # same shapes, different layout
    reduced = (left @ right).reduce()
    assert not isinstance(reduced, BlockDiagonalOperator)
    assert_allclose(reduced(_diagonals()[:4].T), left(right(_diagonals()[:4].T)))


def test_axes_need_an_ensemble():
    with pytest.raises(ValueError, match='ensemble'):
        BlockDiagonalOperator([DiagonalOperator(jnp.ones(3))], axes=1)


def test_addition_tags_come_from_one_member():
    operator = AdditionOperator(_ensemble('stacked'))
    assert operator.is_symmetric and operator.is_diagonal


def test_addition_with_another_operand_nests():
    operator = AdditionOperator(_ensemble('stacked'))
    homothety = HomothetyOperator(2.0, in_structure=jax.ShapeDtypeStruct((M,), jnp.float64))
    total = operator + homothety
    assert_allclose(total(_x()), operator(_x()) + 2 * _x())
    total = homothety + operator
    assert_allclose(total(_x()), operator(_x()) + 2 * _x())


def test_a_scalar_varying_over_the_members_is_an_array():
    structure = jax.ShapeDtypeStruct((M,), jnp.float64)
    with pytest.raises(ValueError, match=r'\.value member 1 holds 3.0 and member 0 holds 2.0'):
        ListEnsemble([HomothetyOperator(v, in_structure=structure) for v in (2.0, 3.0)])
    members = [HomothetyOperator(jnp.asarray(v), in_structure=structure) for v in (2.0, 3.0)]
    assert_allclose(AdditionOperator(ListEnsemble(members))(_x()), 5 * _x())


def test_the_policy_travels_with_the_ensemble():
    ensemble = _ensemble('stacked')
    batched = dataclasses.replace(ensemble, policy=Policy(batch=N))
    assert_allclose(AdditionOperator(batched)(_x()), AdditionOperator(ensemble)(_x()))


def test_reduce_is_member_wise():
    operator = AdditionOperator(_ensemble('stacked'))
    reduced = operator.reduce()
    assert isinstance(reduced, AdditionOperator)
    assert isinstance(reduced.operands, Ensemble)
    assert_allclose(reduced(_x()), operator(_x()))


def test_block_diagonal_inverse_is_member_wise():
    operator = BlockDiagonalOperator(_ensemble('stacked'))
    inverse = operator.inverse()
    assert isinstance(inverse, BlockDiagonalOperator)
    assert isinstance(inverse.blocks, Ensemble)
    assert_allclose(inverse(_xs()), _xs() / _diagonals())


class TestRules:
    def test_block_row_times_block_diagonal(self):
        row = BlockRowOperator(_ensemble('stacked'))
        diagonal = BlockDiagonalOperator(_ensemble('stacked', scale=2.0))
        reduced = (row @ diagonal).reduce()
        assert isinstance(reduced, BlockRowOperator)
        assert isinstance(reduced.blocks, Ensemble)
        assert_allclose(reduced(_xs()), (2 * _diagonals() ** 2 * _xs()).sum(axis=0))

    def test_block_row_times_block_column(self):
        row = BlockRowOperator(_ensemble('stacked'))
        column = BlockColumnOperator(_ensemble('stacked', scale=2.0))
        reduced = (row @ column).reduce()
        assert isinstance(reduced, AdditionOperator)
        assert isinstance(reduced.operands, Ensemble)
        assert_allclose(reduced(_x()), (2 * _diagonals() ** 2 * _x()).sum(axis=0))

    def test_block_diagonal_times_block_diagonal(self):
        left = BlockDiagonalOperator(_ensemble('stacked'))
        right = BlockDiagonalOperator(_ensemble('stacked', scale=2.0))
        reduced = (left @ right).reduce()
        assert isinstance(reduced, BlockDiagonalOperator)
        assert_allclose(reduced(_xs()), 2 * _diagonals() ** 2 * _xs())

    def test_mixed_containers_do_not_reduce(self):
        row = BlockRowOperator(_ensemble('stacked'))
        diagonal = BlockDiagonalOperator(_ensemble('list'))
        reduced = (row @ diagonal).reduce()
        assert_allclose(reduced(_xs()), (_diagonals() ** 2 * _xs()).sum(axis=0))


class TestPytree:
    def test_jit_roundtrip(self):
        operator = AdditionOperator(_ensemble('stacked'))
        got = jax.jit(lambda op, v: op(v))(operator, _x())
        assert_allclose(got, operator(_x()))

    def test_static_signature_ignores_the_values(self):
        left = AdditionOperator(_ensemble('stacked'))
        right = AdditionOperator(_ensemble('stacked', scale=3.0))
        assert left._static_signature == right._static_signature

    def test_one_body_whatever_the_ensemble_size(self):
        def hlo(n: int) -> str:
            diagonals = jnp.arange(float(n * M)).reshape(n, M) + 1.0
            ensemble = lift(
                lambda m: DiagonalOperator(m['d']), StackedEnsemble(n, {'d': diagonals}, 1)
            )
            operator = AdditionOperator(ensemble)
            return jax.jit(lambda op, v: op(v)).lower(operator, _x()).compile().as_text()

        counts = [len(re.findall(r'\bmultiply\(', hlo(n))) for n in (2, 16)]
        assert counts[0] == counts[1] == 1


@pytest.mark.distributed
def test_composites_close_a_sharded_ensemble():
    n = jax.device_count()
    diagonals = jnp.arange(float(n * M)).reshape(n, M) + 1.0
    ensemble = lift(lambda m: DiagonalOperator(m['d']), StackedEnsemble(n, {'d': diagonals}, 1))
    x = _x()
    with jax.set_mesh(jax.make_mesh((n,), ('obs',))):
        # the composites and their algebra never materialize a member: they are built, transposed
        # and reduced on an ensemble whose member leaves are laid out over the devices
        column = BlockColumnOperator(ensemble.shard('obs'))
        assert_allclose(column(x), diagonals * x)
        normal = (column.T @ column).reduce()
        assert isinstance(normal, AdditionOperator)
        assert_allclose(normal(x), (diagonals**2 * x).sum(axis=0))


@pytest.mark.distributed
def test_least_squares_solve_over_a_sharded_ensemble():
    # the toy multi-observation problem: each observation is a pure pointing y_i = x[p_i], and
    # solving the reduced normal equations (sum_i H_i^T H_i) x = sum_i H_i^T y_i recovers x
    n_pix, n_samp = 16, 64
    n_obs = 2 * jax.device_count()
    key_x, key_p = jax.random.split(jax.random.key(0))
    x_true = jax.random.normal(key_x, (n_pix,))
    pointings = jax.random.randint(key_p, (n_obs, n_samp), 0, n_pix)
    pointings = pointings.at[:, :n_pix].set(jnp.arange(n_pix))  # every pixel is observed
    structure = jax.ShapeDtypeStruct((n_pix,), x_true.dtype)

    with jax.set_mesh(jax.make_mesh((jax.device_count(),), ('obs',))):
        hs = lift(
            lambda m: IndexOperator(m['p'], in_structure=structure),
            StackedEnsemble(n_obs, {'p': pointings}, policy=Policy(batch=2)),
        ).shard('obs')
        h = BlockColumnOperator(hs)
        y0 = h(x_true)
        assert y0.sharding.spec[0] == 'obs'  # the stacked TOD stays laid out over the devices
        normal = (h.T @ h).reduce()
        b = h.T(y0)  # summed across the devices, held by all of them
        x_hat, _ = jax.scipy.sparse.linalg.cg(normal, b, tol=1e-12)

    assert_allclose(x_hat, x_true, atol=1e-10)


def test_tree_transformations_keep_the_strides():
    # eqx.partition replaces the arrays with placeholders: the flattening aux data must not
    # depend on the leaf values, or the two sides of the partition disagree on the strides
    operator = AdditionOperator(_ensemble('stacked'))
    dynamic, static = eqx.partition(operator, eqx.is_array)
    combined = eqx.combine(dynamic, static)
    assert_allclose(combined(_x()), operator(_x()))


def test_lineax_solve_over_an_ensemble():
    # the inverse of an ensemble closure through lineax: the solver traverses the ensemble
    # with equinox tree transformations
    operator = AdditionOperator(_ensemble('stacked'))
    x = _x()
    b = operator(x)
    solver = lx.CG(rtol=1e-12, atol=1e-12, max_steps=100)
    got = operator.I(solver=solver)(b)
    assert_allclose(got, x, atol=1e-10)
