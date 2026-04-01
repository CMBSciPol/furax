"""Tests for BJPreconditioner."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import asoperator
from furax.core import BlockDiagonalOperator
from furax.core._diagonal import DiagonalOperator
from furax.mapmaking.preconditioner import BJPreconditioner
from furax.obs.stokes import StokesI, StokesIQU, StokesQU

N_PIX = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diagonal_op_i(weights: jax.Array) -> BlockDiagonalOperator:
    """Diagonal operator on StokesI: scales the I map by weights."""
    struct = jax.ShapeDtypeStruct(weights.shape, weights.dtype)
    return BlockDiagonalOperator(StokesI(DiagonalOperator(weights, in_structure=struct)))


def _diagonal_op_qu(weights_q: jax.Array, weights_u: jax.Array) -> BlockDiagonalOperator:
    """Diagonal operator on StokesQU: independently scales Q and U."""
    struct_q = jax.ShapeDtypeStruct(weights_q.shape, weights_q.dtype)
    struct_u = jax.ShapeDtypeStruct(weights_u.shape, weights_u.dtype)
    return BlockDiagonalOperator(
        StokesQU(
            DiagonalOperator(weights_q, in_structure=struct_q),
            DiagonalOperator(weights_u, in_structure=struct_u),
        )
    )


def _diagonal_op_iqu(
    weights_i: jax.Array, weights_q: jax.Array, weights_u: jax.Array
) -> BlockDiagonalOperator:
    """Diagonal operator on StokesIQU: independently scales I, Q, U."""
    struct = lambda w: jax.ShapeDtypeStruct(w.shape, w.dtype)  # noqa: E731
    return BlockDiagonalOperator(
        StokesIQU(
            DiagonalOperator(weights_i, in_structure=struct(weights_i)),
            DiagonalOperator(weights_q, in_structure=struct(weights_q)),
            DiagonalOperator(weights_u, in_structure=struct(weights_u)),
        )
    )


# ---------------------------------------------------------------------------
# Tests for StokesI
# ---------------------------------------------------------------------------


class TestBJPreconditionerI:
    def test_create_blocks_match_diagonal(self) -> None:
        """create() on a diagonal I operator gives blocks equal to the diagonal."""
        w = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        op = _diagonal_op_i(w)
        BJ = BJPreconditioner.create(op)
        blocks = BJ.get_blocks()  # (N_PIX, 1, 1)
        assert blocks.shape == (N_PIX, 1, 1)
        assert_allclose(blocks[:, 0, 0], w)

    def test_apply_matches_diagonal(self) -> None:
        """BJ(x) == w * x for a diagonal operator."""
        w = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        op = _diagonal_op_i(w)
        BJ = BJPreconditioner.create(op)
        x = StokesI(jnp.ones(N_PIX))
        result = BJ(x)
        assert_allclose(result.i, w)

    def test_inverse_undoes_operator(self) -> None:
        """BJ.I(BJ(x)) ≈ x."""
        w = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        op = _diagonal_op_i(w)
        BJ = BJPreconditioner.create(op)
        x = StokesI(jnp.ones(N_PIX))
        assert_allclose(BJ.I(BJ(x)).i, x.i, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tests for StokesQU
# ---------------------------------------------------------------------------


class TestBJPreconditionerQU:
    def test_create_blocks_diagonal(self) -> None:
        """create() on a diagonal QU operator gives 2×2 diagonal blocks."""
        wq = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wu = 2 * wq
        op = _diagonal_op_qu(wq, wu)
        BJ = BJPreconditioner.create(op)
        blocks = BJ.get_blocks()  # (N_PIX, 2, 2)
        assert blocks.shape == (N_PIX, 2, 2)
        assert_allclose(blocks[:, 0, 0], wq)
        assert_allclose(blocks[:, 1, 1], wu)
        assert_allclose(blocks[:, 0, 1], 0.0)
        assert_allclose(blocks[:, 1, 0], 0.0)

    def test_apply_matches_per_component_weights(self) -> None:
        wq = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wu = 2 * wq
        op = _diagonal_op_qu(wq, wu)
        BJ = BJPreconditioner.create(op)
        x = StokesQU(jnp.ones(N_PIX), jnp.ones(N_PIX))
        result = BJ(x)
        assert_allclose(result.q, wq)
        assert_allclose(result.u, wu)


# ---------------------------------------------------------------------------
# Tests for StokesIQU
# ---------------------------------------------------------------------------


class TestBJPreconditionerIQU:
    def test_create_blocks_diagonal(self) -> None:
        """create() on a diagonal IQU operator gives 3×3 diagonal blocks."""
        wi = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wq = 2 * wi
        wu = 3 * wi
        op = _diagonal_op_iqu(wi, wq, wu)
        BJ = BJPreconditioner.create(op)
        blocks = BJ.get_blocks()  # (N_PIX, 3, 3)
        assert blocks.shape == (N_PIX, 3, 3)
        assert_allclose(blocks[:, 0, 0], wi)
        assert_allclose(blocks[:, 1, 1], wq)
        assert_allclose(blocks[:, 2, 2], wu)
        # Off-diagonal blocks must be zero for a diagonal operator
        assert_allclose(blocks[:, 0, 1], 0.0)
        assert_allclose(blocks[:, 0, 2], 0.0)
        assert_allclose(blocks[:, 1, 0], 0.0)
        assert_allclose(blocks[:, 1, 2], 0.0)
        assert_allclose(blocks[:, 2, 0], 0.0)
        assert_allclose(blocks[:, 2, 1], 0.0)

    def test_apply_matches_per_component_weights(self) -> None:
        wi = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wq = 2 * wi
        wu = 3 * wi
        op = _diagonal_op_iqu(wi, wq, wu)
        BJ = BJPreconditioner.create(op)
        x = StokesIQU(jnp.ones(N_PIX), jnp.ones(N_PIX), jnp.ones(N_PIX))
        result = BJ(x)
        assert_allclose(result.i, wi)
        assert_allclose(result.q, wq)
        assert_allclose(result.u, wu)

    def test_off_diagonal_coupling(self) -> None:
        """create() correctly captures off-diagonal (cross-Stokes) coupling.

        Build an operator that couples I and Q via a known 2×2 block (per pixel)
        plus an independent U component, then check that get_blocks() reproduces
        the full matrix.
        """
        # Per-pixel 3×3 matrix: [[2, 1, 0], [1, 3, 0], [0, 0, 4]] (constant over pixels)
        # Implement as a function operator
        a, b, c = 2.0, 1.0, 3.0
        d = 4.0

        def coupled(x: StokesIQU) -> StokesIQU:
            return StokesIQU(
                i=a * x.i + b * x.q,
                q=b * x.i + c * x.q,
                u=d * x.u,
            )

        in_struct = StokesIQU.structure_for((N_PIX,), jnp.float64)
        op = asoperator(coupled, in_structure=in_struct)

        BJ = BJPreconditioner.create(op)
        blocks = BJ.get_blocks()  # (N_PIX, 3, 3)

        expected = jnp.array([[a, b, 0.0], [b, c, 0.0], [0.0, 0.0, d]])
        for pix in range(N_PIX):
            assert_allclose(blocks[pix], expected, atol=1e-12)

    def test_blocks_are_symmetric(self) -> None:
        """For a symmetric operator the recovered blocks must be symmetric."""
        wi = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wq = 2 * wi
        wu = 3 * wi
        op = _diagonal_op_iqu(wi, wq, wu)
        BJ = BJPreconditioner.create(op)
        blocks = BJ.get_blocks()
        assert_allclose(blocks, jnp.swapaxes(blocks, -1, -2), atol=1e-12)

    def test_inverse_undoes_operator(self) -> None:
        """BJ.I(BJ(x)) ≈ x for a full IQU preconditioner."""
        wi = jnp.arange(1, N_PIX + 1, dtype=jnp.float64)
        wq = 2 * wi
        wu = 3 * wi
        op = _diagonal_op_iqu(wi, wq, wu)
        BJ = BJPreconditioner.create(op)
        x = StokesIQU(jnp.ones(N_PIX), 2 * jnp.ones(N_PIX), 3 * jnp.ones(N_PIX))
        recovered = BJ.I(BJ(x))
        assert_allclose(recovered.i, x.i, rtol=1e-12)
        assert_allclose(recovered.q, x.q, rtol=1e-12)
        assert_allclose(recovered.u, x.u, rtol=1e-12)


# ---------------------------------------------------------------------------
# Error-handling
# ---------------------------------------------------------------------------


def test_create_raises_for_non_stokes_operator() -> None:
    """create() raises ValueError when the operator does not act on a Stokes pytree."""
    x = jnp.zeros(N_PIX)
    op = asoperator(lambda v: v, in_structure=jax.ShapeDtypeStruct(x.shape, x.dtype))
    with pytest.raises(ValueError, match='Stokes'):
        BJPreconditioner.create(op)


def test_create_raises_for_non_square_operator() -> None:
    """create() raises ValueError when in_structure != out_structure."""
    in_struct = StokesIQU.structure_for((N_PIX,), jnp.float64)

    def rectangular(x: StokesIQU) -> StokesQU:
        return StokesQU(x.q, x.u)

    op = asoperator(rectangular, in_structure=in_struct)
    with pytest.raises(ValueError, match='square'):
        BJPreconditioner.create(op)
