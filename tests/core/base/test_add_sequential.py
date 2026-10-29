"""The one-operand-at-a-time application of AdditionOperator, and that it stays the same map."""

import re

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


# ---------------------------------------------------------------------------
# Sequential sums apply operands sharing a `_static_signature` through a
# single traced body (one compute region per group instead of one per
# operand), selecting each operand's arrays in turn.
# ---------------------------------------------------------------------------


def _diag(values: list[float]) -> AbstractLinearOperator:
    from furax import DiagonalOperator

    return DiagonalOperator(jnp.asarray(values))


def test_homogeneous_group_bit_exact() -> None:
    """Three diagonal operators of same shape: the grouped path must agree numerically."""
    from furax import DiagonalOperator

    d1 = DiagonalOperator(jnp.arange(1.0, 5.0))
    d2 = DiagonalOperator(jnp.arange(5.0, 9.0))
    d3 = DiagonalOperator(jnp.arange(9.0, 13.0))
    x = jnp.arange(1.0, 5.0)

    # expected: pointwise sum of three diagonal multiplications
    expected = (d1.diagonal + d2.diagonal + d3.diagonal) * x
    got = AdditionOperator([d1, d2, d3], sequential=True)(x)
    assert_allclose(got, expected, rtol=1e-12)


def test_mixed_groups_bit_exact() -> None:
    """Two diagonals + one homothety: two groups, each applied once."""
    structure = jax.ShapeDtypeStruct((4,), jnp.float64)
    d1 = _diag([1.0, 2.0, 3.0, 4.0])
    d2 = _diag([10.0, 20.0, 30.0, 40.0])
    h = HomothetyOperator(2.0, in_structure=structure)
    x = jnp.ones(4)

    expected = (jnp.asarray([1.0, 2.0, 3.0, 4.0]) + jnp.asarray([10.0, 20.0, 30.0, 40.0])) * x
    expected = expected + 2.0 * x
    got = AdditionOperator([d1, d2, h], sequential=True)(x)
    assert_allclose(got, expected, rtol=1e-12)


def test_singleton_group_preserved() -> None:
    """A group of size one is applied directly, not through the loop."""
    structure = jax.ShapeDtypeStruct((4,), jnp.float64)
    h = HomothetyOperator(5.0, in_structure=structure)
    x = jnp.arange(4.0)
    got = AdditionOperator([h], sequential=True)(x)
    assert_allclose(got, 5.0 * x, rtol=1e-12)


@pytest.mark.parametrize(
    'make_indices',
    [
        lambda idx: (..., idx),
        lambda idx: (slice(None), idx),
        lambda idx: (idx, slice(None)),
    ],
    ids=['ellipsis', 'leading-slice', 'trailing-slice'],
)
def test_index_operators_with_non_array_leaves(make_indices) -> None:
    """Index operators carry `...` and `slice` pytree leaves, which are shared, not selected."""
    from furax import IndexOperator

    structure = jax.ShapeDtypeStruct((8, 8), jnp.float64)
    terms = []
    for key in jax.random.split(jax.random.key(0), 3):
        H = IndexOperator(make_indices(jax.random.randint(key, (5,), 0, 8)), in_structure=structure)
        terms.append((H.T @ H).reduce())
    x = jax.random.normal(jax.random.key(1), (8, 8))

    expected = sum(term(x) for term in terms)
    got = AdditionOperator(terms, sequential=True)(x)
    assert_allclose(got, expected, rtol=1e-12)


def test_sequential_peak_memory_does_not_grow_with_operands() -> None:
    """With traced operands, the sequential path keeps one operand's buffers live at a time.

    The operator is passed to `jit` as an argument, so its arrays are inputs rather than foldable
    constants: a stacked copy of them would show up as a temporary growing with N.
    """
    from furax import DiagonalOperator

    def temp_bytes(n: int) -> int:
        diags = [jax.random.normal(k, (4096,)) for k in jax.random.split(jax.random.key(0), n)]
        op = AdditionOperator([DiagonalOperator(d) for d in diags], sequential=True)
        x = jnp.ones(4096)
        compiled = jax.jit(lambda op, v: op(v)).lower(op, x).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    small, large = temp_bytes(2), temp_bytes(16)
    assert large < 2 * small, f'temporaries scale with N: {small} -> {large} bytes'


@pytest.mark.parametrize('n_terms', [2, 16])
def test_sequential_compiles_the_body_once(n_terms: int) -> None:
    """The diagonal multiply appears once in the compiled program, whatever the operand count.

    Unrolled, each operand would bring its own multiply; the grouped loop traces one body and
    only adds a trivial selection branch per operand.
    """
    from furax import DiagonalOperator

    diags = [DiagonalOperator(jnp.arange(k * 4.0, (k + 1) * 4.0)) for k in range(n_terms)]
    op = AdditionOperator(diags, sequential=True)
    hlo = jax.jit(lambda v: op(v)).lower(jnp.ones(4)).compile().as_text()
    assert len(re.findall(r'\bmultiply\(', hlo)) == 1
