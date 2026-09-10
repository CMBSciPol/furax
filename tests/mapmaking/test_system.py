"""The per-bucket sum of a normal system, and that it stays the map it replaces."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, HomothetyOperator
from furax.core import AdditionOperator
from furax.mapmaking._system import BucketSumOperator


class _MatOp(AbstractLinearOperator):
    matrix: Inexact[Array, 'n n']

    def mv(self, x: PyTree[Inexact[Array, ' n']]) -> PyTree[Inexact[Array, ' n']]:
        return self.matrix @ x

    def transpose(self) -> AbstractLinearOperator:
        return _MatOp(self.matrix.T, in_structure=self.in_structure)


def _terms(n_terms: int, n: int = 4) -> list[AbstractLinearOperator]:
    structure = jax.ShapeDtypeStruct((n,), jnp.float64)
    key = jax.random.key(0)
    return [
        _MatOp(jax.random.normal(k, (n, n)), in_structure=structure)
        for k in jax.random.split(key, n_terms)
    ]


@pytest.mark.parametrize('n_terms', [1, 2, 3, 5])
def test_matches_addition(n_terms: int) -> None:
    """Same linear map as the AdditionOperator it stands in for."""
    terms = _terms(n_terms)
    x = jnp.arange(4.0)
    assert_allclose(BucketSumOperator(terms)(x), AdditionOperator(terms)(x), rtol=1e-12)


@pytest.mark.parametrize('n_terms', [1, 4])
def test_matches_addition_as_matrix(n_terms: int) -> None:
    terms = _terms(n_terms)
    assert_allclose(
        BucketSumOperator(terms).as_matrix(), AdditionOperator(terms).as_matrix(), rtol=1e-12
    )


def test_under_jit() -> None:
    """The fori_loop/switch lowering must survive tracing, which is where it is used."""
    terms = _terms(3)
    x = jnp.arange(4.0)
    expected = AdditionOperator(terms)(x)
    assert_allclose(jax.jit(lambda v: BucketSumOperator(terms)(v))(x), expected, rtol=1e-12)


def test_transpose() -> None:
    terms = _terms(3)
    op = BucketSumOperator(terms)
    assert isinstance(op.T, BucketSumOperator)
    assert_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-12)


def test_symmetric_tag_propagates_from_every_term() -> None:
    """CG is handed this operator, so the symmetry tag must survive the substitution."""
    structure = jax.ShapeDtypeStruct((4,), jnp.float64)
    symmetric = [HomothetyOperator(v, in_structure=structure) for v in (2.0, 3.0)]
    assert BucketSumOperator(symmetric).is_symmetric
    # one untagged term is enough to lose it, as for AdditionOperator
    assert not BucketSumOperator([*symmetric, _terms(1)[0]]).is_symmetric


def test_rejects_no_operands() -> None:
    with pytest.raises(ValueError, match='at least one operand'):
        BucketSumOperator([])
