"""The one-operand-at-a-time application of AdditionOperator, and that it stays the same map."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, HomothetyOperator
from furax.core import AdditionOperator, CompositionOperator


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
def test_matches_plain_addition(n_terms: int) -> None:
    terms = _terms(n_terms)
    x = jnp.arange(4.0)
    fused = AdditionOperator(terms)(x)
    sequential = AdditionOperator(terms, sequential=True)(x)
    assert_allclose(sequential, fused, rtol=1e-12)


@pytest.mark.parametrize('n_terms', [1, 4])
def test_matches_plain_addition_as_matrix(n_terms: int) -> None:
    terms = _terms(n_terms)
    fused = AdditionOperator(terms)
    sequential = AdditionOperator(terms, sequential=True)
    assert_allclose(fused.as_matrix(), sequential.as_matrix(), rtol=1e-12)


def test_under_jit() -> None:
    terms = _terms(3)
    x = jnp.arange(4.0)
    expected = AdditionOperator(terms)(x)
    op = AdditionOperator(terms, sequential=True)
    assert_allclose(jax.jit(lambda v: op(v))(x), expected, rtol=1e-12)


@pytest.mark.parametrize('sequential', [False, True])
def test_flag_survives_transpose_and_negation(sequential: bool) -> None:
    op = AdditionOperator(_terms(3), sequential=sequential)
    assert op.T.sequential is sequential
    assert (-op).sequential is sequential


def test_flag_survives_further_addition() -> None:
    a, b, c = _terms(3)
    op = AdditionOperator([a, b], sequential=True)
    assert (op + c).sequential
    assert (c + op).sequential
    assert (op - c).sequential
    assert (op + AdditionOperator([c])).sequential
    assert not (AdditionOperator([a, b]) + c).sequential


def test_reduce_keeps_the_operands_apart() -> None:
    structure = jax.ShapeDtypeStruct((3,), jnp.float32)
    h = HomothetyOperator(3.0, in_structure=structure)
    terms = [h, h, CompositionOperator([h, h])]

    reduced = AdditionOperator(terms, sequential=True).reduce()
    assert isinstance(reduced, AdditionOperator)
    assert reduced.sequential
    assert len(reduced.operand_leaves) == 3
    # the operands themselves are still reduced, only their pairwise fusion is skipped
    assert isinstance(reduced.operand_leaves[2], HomothetyOperator)

    # without the flag, the three homotheties fuse into one
    assert isinstance(AdditionOperator(terms).reduce(), HomothetyOperator)


def test_rejects_no_operands() -> None:
    with pytest.raises(ValueError, match='at least one operand'):
        AdditionOperator([])
