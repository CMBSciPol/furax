import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator
from furax.core import CompositionOperator
from furax.mapmaking._scan_blocks import (
    ScanAdditionOperator,
    ScanBlockColumnOperator,
    ScanBlockDiagonalOperator,
    ScanBlockRowOperator,
)

# ---------------------------------------------------------------------------
# Test fixture: a minimal AbstractLinearOperator that supports stacking via a
# leading batch axis on its ``matrix`` field.
# ---------------------------------------------------------------------------


class _TestOp(AbstractLinearOperator):
    matrix: Inexact[Array, '...']

    def __init__(
        self,
        matrix: Inexact[Array, '...'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'matrix', matrix)
        super().__init__(in_structure=in_structure)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jnp.einsum('...ij,...j->...i', self.matrix, x)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        # Per-slice metadata regardless of stacking.
        return jax.ShapeDtypeStruct((self.matrix.shape[-2],), self.matrix.dtype)

    def transpose(self) -> AbstractLinearOperator:
        return _TestOp(
            jnp.swapaxes(self.matrix, -1, -2),
            in_structure=jax.ShapeDtypeStruct((self.matrix.shape[-2],), self.matrix.dtype),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


N_OBS = 3
N_IN = 4
N_OUT = 5


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


def _stacked_test_op(rng: np.random.Generator, n_obs: int, n_out: int, n_in: int) -> _TestOp:
    matrices = rng.standard_normal((n_obs, n_out, n_in)).astype(np.float32)
    return _TestOp(
        jnp.asarray(matrices),
        in_structure=jax.ShapeDtypeStruct((n_in,), jnp.float32),
    )


def _per_obs_matrices(op: _TestOp) -> list[Array]:
    return [op.matrix[i] for i in range(op.matrix.shape[0])]


# ---------------------------------------------------------------------------
# 1. Forward correctness
# ---------------------------------------------------------------------------


def test_diagonal_mv(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockDiagonalOperator(blocks, N_OBS)
    x_stacked = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    expected = jnp.stack([m @ x_stacked[i] for i, m in enumerate(_per_obs_matrices(blocks))])
    assert_allclose(op.mv(x_stacked), expected, rtol=1e-5)


def test_column_mv(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockColumnOperator(blocks, N_OBS)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    expected = jnp.stack([m @ x for m in _per_obs_matrices(blocks)])
    assert_allclose(op.mv(x), expected, rtol=1e-5)


def test_row_mv(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockRowOperator(blocks, N_OBS)
    x_stacked = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    expected = sum(m @ x_stacked[i] for i, m in enumerate(_per_obs_matrices(blocks)))
    assert_allclose(op.mv(x_stacked), expected, rtol=1e-5)


def test_sum_mv(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanAdditionOperator(blocks, N_OBS)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    expected = sum(m @ x for m in _per_obs_matrices(blocks))
    assert_allclose(op.mv(x), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Transpose correctness — class swap and numerical match
# ---------------------------------------------------------------------------


def test_diagonal_transpose(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockDiagonalOperator(blocks, N_OBS)
    op_T = op.T
    assert isinstance(op_T, ScanBlockDiagonalOperator)
    y_stacked = jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float32))
    expected = jnp.stack([m.T @ y_stacked[i] for i, m in enumerate(_per_obs_matrices(blocks))])
    assert_allclose(op_T.mv(y_stacked), expected, rtol=1e-5)


def test_column_transpose_is_row(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockColumnOperator(blocks, N_OBS)
    op_T = op.T
    assert isinstance(op_T, ScanBlockRowOperator)
    y_stacked = jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float32))
    expected = sum(m.T @ y_stacked[i] for i, m in enumerate(_per_obs_matrices(blocks)))
    assert_allclose(op_T.mv(y_stacked), expected, rtol=1e-5)


def test_row_transpose_is_column(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockRowOperator(blocks, N_OBS)
    op_T = op.T
    assert isinstance(op_T, ScanBlockColumnOperator)
    y = jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float32))
    expected = jnp.stack([m.T @ y for m in _per_obs_matrices(blocks)])
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


def test_sum_transpose(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanAdditionOperator(blocks, N_OBS)
    op_T = op.T
    assert isinstance(op_T, ScanAdditionOperator)
    y = jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float32))
    expected = sum(m.T @ y for m in _per_obs_matrices(blocks))
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. Roundtrip op.T.T == op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'cls',
    [
        ScanBlockDiagonalOperator,
        ScanBlockColumnOperator,
        ScanBlockRowOperator,
        ScanAdditionOperator,
    ],
)
def test_transpose_roundtrip(cls: type, rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = cls(blocks, N_OBS)
    op_TT = op.T.T
    assert isinstance(op_TT, cls)
    if cls is ScanBlockDiagonalOperator:
        x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    elif cls is ScanBlockColumnOperator or cls is ScanAdditionOperator:
        x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    else:  # ScanBlockRow
        x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    assert_allclose(op_TT.mv(x), op.mv(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# 4. as_matrix consistency
# ---------------------------------------------------------------------------


def test_diagonal_as_matrix(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockDiagonalOperator(blocks, N_OBS)
    mats = _per_obs_matrices(blocks)
    import jax.scipy.linalg as jsl

    expected = jsl.block_diag(*mats)
    assert_allclose(op.as_matrix(), expected, rtol=1e-5)


def test_column_as_matrix(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockColumnOperator(blocks, N_OBS)
    mats = _per_obs_matrices(blocks)
    expected = jnp.vstack(mats)
    assert_allclose(op.as_matrix(), expected, rtol=1e-5)


def test_row_as_matrix(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockRowOperator(blocks, N_OBS)
    mats = _per_obs_matrices(blocks)
    expected = jnp.hstack(mats)
    assert_allclose(op.as_matrix(), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Fusion rules
# ---------------------------------------------------------------------------


def test_fusion_diag_diag(rng: np.random.Generator) -> None:
    left_blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)  # square
    right_blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    left = ScanBlockDiagonalOperator(left_blocks, N_OBS)
    right = ScanBlockDiagonalOperator(right_blocks, N_OBS)

    composed = left @ right
    reduced = composed.reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)

    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    assert_allclose(reduced.mv(x), composed.mv(x), rtol=1e-5)


def test_fusion_diag_column(rng: np.random.Generator) -> None:
    left_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_OUT)
    right_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    left = ScanBlockDiagonalOperator(left_blocks, N_OBS)
    right = ScanBlockColumnOperator(right_blocks, N_OBS)

    composed = left @ right
    reduced = composed.reduce()
    assert isinstance(reduced, ScanBlockColumnOperator)

    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    assert_allclose(reduced.mv(x), composed.mv(x), rtol=1e-5)


def test_fusion_row_diag(rng: np.random.Generator) -> None:
    left_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    right_blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    left = ScanBlockRowOperator(left_blocks, N_OBS)
    right = ScanBlockDiagonalOperator(right_blocks, N_OBS)

    composed = left @ right
    reduced = composed.reduce()
    assert isinstance(reduced, ScanBlockRowOperator)

    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    assert_allclose(reduced.mv(x), composed.mv(x), rtol=1e-5)


def test_fusion_row_column(rng: np.random.Generator) -> None:
    left_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    right_blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    left = ScanBlockRowOperator(left_blocks, N_OBS)
    right = ScanBlockColumnOperator(right_blocks, N_OBS)

    composed = left @ right
    reduced = composed.reduce()
    assert isinstance(reduced, ScanAdditionOperator)

    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    assert_allclose(reduced.mv(x), composed.mv(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# 6. n_obs mismatch declines fusion
# ---------------------------------------------------------------------------


def test_mismatched_leading_dim_rejected_by_matmul(rng: np.random.Generator) -> None:
    """Different leading-axis sizes give mismatched ``in_structure`` shapes, so
    ``__matmul__`` rejects the composition before any fusion rule runs."""
    blocks_a = _stacked_test_op(rng, 3, N_IN, N_IN)
    blocks_b = _stacked_test_op(rng, 4, N_IN, N_IN)
    left = ScanBlockDiagonalOperator(blocks_a, 3)
    right = ScanBlockDiagonalOperator(blocks_b, 4)
    with pytest.raises(ValueError, match='Incompatible'):
        _ = left @ right


# ---------------------------------------------------------------------------
# 7. Multi-step chain fusion: T.T W T  (Diag @ Diag @ Diag → single Diag)
# ---------------------------------------------------------------------------


def test_chain_fusion_diag(rng: np.random.Generator) -> None:
    a = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    b = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    c = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    A = ScanBlockDiagonalOperator(a, N_OBS)
    B = ScanBlockDiagonalOperator(b, N_OBS)
    C = ScanBlockDiagonalOperator(c, N_OBS)
    chain = A @ B @ C
    reduced = chain.reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)

    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    assert_allclose(reduced.mv(x), chain.mv(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# 8. Row @ Diag @ Column reduces all the way to _ScanSumOperator
# ---------------------------------------------------------------------------


def test_chain_fusion_row_diag_column(rng: np.random.Generator) -> None:
    """``H.T @ W @ H`` fuses to a single :class:`_ScanSumOperator` (= ``Σ H_i.T W_i H_i``)."""
    H_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)
    W_blocks = _stacked_test_op(rng, N_OBS, N_OUT, N_OUT)

    H = ScanBlockColumnOperator(H_blocks, N_OBS)
    W = ScanBlockDiagonalOperator(W_blocks, N_OBS)
    HT = H.T
    assert isinstance(HT, ScanBlockRowOperator)

    chain = HT @ W @ H
    reduced = chain.reduce()
    assert isinstance(reduced, ScanAdditionOperator)

    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float32))
    assert_allclose(reduced.mv(x), chain.mv(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# Sanity: CompositionOperator's __matmul__ structure check rejects mismatched shapes
# ---------------------------------------------------------------------------


def test_composition_rejects_mismatched_shapes(rng: np.random.Generator) -> None:
    a = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)  # in n_in, out n_out
    b = _stacked_test_op(rng, N_OBS, N_OUT, N_IN)  # same
    A = ScanBlockDiagonalOperator(a, N_OBS)
    B = ScanBlockDiagonalOperator(b, N_OBS)
    # A.in = (n_obs, n_in), B.out = (n_obs, n_out): not compatible
    with pytest.raises(ValueError, match='Incompatible'):
        _ = A @ B


# ---------------------------------------------------------------------------
# Smoke: reduce() on a Scan op delegates to inner blocks.reduce()
# ---------------------------------------------------------------------------


def test_reduce_delegates_to_blocks(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    op = ScanBlockDiagonalOperator(blocks, N_OBS)
    reduced = op.reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)
    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float32))
    assert_allclose(reduced.mv(x), op.mv(x), rtol=1e-5)


# ---------------------------------------------------------------------------
# Smoke: a Scan op composed with itself reduces via the appropriate rule
# (not just via the CompositionOperator wrapper)
# ---------------------------------------------------------------------------


def test_self_composition_reduces(rng: np.random.Generator) -> None:
    blocks = _stacked_test_op(rng, N_OBS, N_IN, N_IN)
    op = ScanBlockDiagonalOperator(blocks, N_OBS)
    composed = op @ op
    assert isinstance(composed, CompositionOperator)
    reduced = composed.reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)
