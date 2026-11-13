from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.core import BlockColumnOperator, DeferredOperator, StackedEnsemble, lift

N, M = 6, 4


def _column(d):
    return BlockColumnOperator({'a': DiagonalOperator(d), 'b': DiagonalOperator(2 * d)})


def _scaled_diagonal(member):
    return DiagonalOperator(member['d'] * member['p'])


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


def test_structures_of_an_abstract_member():
    # as in the template of an ensemble: the member's leaves have a shape but no values
    member = {
        'd': jax.ShapeDtypeStruct((M,), jnp.float32),
        'p': jax.ShapeDtypeStruct((M,), jnp.float32),
    }
    op = DeferredOperator(_scaled_diagonal, member)
    assert op.in_structure == op.out_structure == jax.ShapeDtypeStruct((M,), jnp.float32)


def test_lift_keeps_the_arrays_and_strides_of_the_ensemble():
    # `build` computes from the leaves, yet the lift reuses them: nothing is vectorized or copied
    d = jnp.arange(float(N * M)).reshape(N, M)
    p = jnp.arange(float(N // 2 * M)).reshape(N // 2, M) + 1
    ensemble = StackedEnsemble(N, {'d': d, 'p': p}, {'d': 1, 'p': 2})
    lifted = lift(lambda m: DeferredOperator(_scaled_diagonal, m), ensemble)
    assert isinstance(lifted, StackedEnsemble)
    assert lifted.stacked.member['d'] is d and lifted.stacked.member['p'] is p
    assert lifted.strides == ensemble.strides
    assert_allclose(lifted.fetch(3)(jnp.ones(M)), d[3] * p[1])


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
