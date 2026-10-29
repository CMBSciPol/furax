"""Ensembles as block containers: the composites close an ensemble through the engine."""

import dataclasses
import re

import jax
import jax.numpy as jnp
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
