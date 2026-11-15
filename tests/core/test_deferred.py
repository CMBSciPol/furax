from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.core import (
    AdditionOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    DeferredOperator,
    StackedEnsemble,
    lift,
)

N, M = 6, 4


def _column(d):
    return BlockColumnOperator({'a': DiagonalOperator(d), 'b': DiagonalOperator(2 * d)})


def _scaled_diagonal(d, p):
    return DiagonalOperator(d * p)


@dataclass(frozen=True)
class _Scaled:
    scale: float

    def __call__(self, d):
        return DiagonalOperator(self.scale * d)


def test_applies_and_transposes_what_build_builds():
    d = jnp.arange(1.0, M + 1)
    op = DeferredOperator(_column, d)  # not square: from (M,) to a dict
    expected = _column(d)
    assert op.in_structure == expected.in_structure
    assert op.out_structure == expected.out_structure
    assert_allclose(op.as_matrix(), expected.as_matrix())
    assert_allclose(op.T.as_matrix(), expected.as_matrix().T)
    assert op.T.T.build is op.build


def test_inverts_and_tags_as_what_build_builds():
    d = jnp.arange(1.0, M + 1)
    op = DeferredOperator(DiagonalOperator, d)
    assert op.tags == DiagonalOperator(d).tags
    assert op.is_diagonal and op.is_symmetric
    # the exact inverse of the built operator, not an iterative solve
    assert isinstance(op.I.build(d), DiagonalOperator)
    assert_allclose(op.I(d), jnp.ones(M))
    assert op.I.I.build is op.build


def test_structures_of_abstract_members():
    # as in the template of an ensemble: the members' leaves have a shape but no values
    d = p = jax.ShapeDtypeStruct((M,), jnp.float32)
    op = DeferredOperator(_scaled_diagonal, d, p)
    assert op.in_structure == op.out_structure == jax.ShapeDtypeStruct((M,), jnp.float32)


def test_lift_keeps_the_arrays_and_strides_of_the_ensemble():
    # `build` computes from the leaves, yet the lift reuses them: nothing is vectorized or copied
    d = jnp.arange(float(N * M)).reshape(N, M)
    p = jnp.arange(float(N // 2 * M)).reshape(N // 2, M) + 1
    ensemble = StackedEnsemble(N, {'d': d, 'p': p}, {'d': 1, 'p': 2})
    lifted = lift(lambda m: DeferredOperator(_scaled_diagonal, m['d'], m['p']), ensemble)
    assert isinstance(lifted, StackedEnsemble)
    assert lifted.stacked.members[0] is d and lifted.stacked.members[1] is p
    assert lifted.strides == ensemble.strides
    assert_allclose(lifted.fetch(3)(jnp.ones(M)), d[3] * p[1])


def test_the_engine_builds_each_member_of_a_lifted_ensemble():
    # the members read their segment's `p`: the built operators are those of the dense reference
    d = jnp.arange(1.0, N * M + 1).reshape(N, M)
    p = jnp.arange(1.0, N // 2 * M + 1).reshape(N // 2, M)
    owner = jnp.arange(N) // 2
    ensemble = StackedEnsemble(N, {'d': d, 'p': p}, {'d': 1, 'p': 'segment'}, owner=owner)
    ops = lift(lambda m: DeferredOperator(_scaled_diagonal, m['d'], m['p']), ensemble)
    diagonals = d * p[owner]
    x = jnp.linspace(1.0, 2.0, M)
    assert_allclose(AdditionOperator(ops)(x), (diagonals * x).sum(axis=0))
    assert_allclose(BlockDiagonalOperator(ops).I(jnp.ones((N, M))), 1 / diagonals)


def test_jit_compiles_once_for_equal_builds():
    traces = 0

    @jax.jit
    def apply(op, x):
        nonlocal traces
        traces += 1
        return op(x)

    for d in (jnp.ones(M), jnp.arange(float(M))):
        apply(DeferredOperator(_Scaled(2.0), d), jnp.ones(M))
    assert traces == 1


def test_build_must_be_hashable():
    @dataclass  # mutable, so not hashable
    class Unhashable:
        def __call__(self, d):
            return DiagonalOperator(d)

    with pytest.raises(ValueError, match='hashable'):
        DeferredOperator(Unhashable(), jnp.ones(M))
