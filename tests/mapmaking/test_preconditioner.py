"""Tests for the mapmaking preconditioners."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, asoperator
from furax.linalg import LowRankTerms
from furax.mapmaking.preconditioner import BJPreconditioner, make_two_level_preconditioner
from furax.obs.stokes import StokesI, StokesIQU, StokesQU

N_PIX = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diagonal_op_i(weights: jax.Array) -> AbstractLinearOperator:
    """Diagonal operator on StokesI: scales the I map by weights."""
    in_struct = StokesI.structure_for(weights.shape, weights.dtype)
    return asoperator(lambda x: StokesI(weights * x.i), in_structure=in_struct)


def _diagonal_op_qu(weights_q: jax.Array, weights_u: jax.Array) -> AbstractLinearOperator:
    """Diagonal operator on StokesQU: independently scales Q and U."""
    in_struct = StokesQU.structure_for(weights_q.shape, weights_q.dtype)
    return asoperator(lambda x: StokesQU(weights_q * x.q, weights_u * x.u), in_structure=in_struct)


def _diagonal_op_iqu(
    weights_i: jax.Array, weights_q: jax.Array, weights_u: jax.Array
) -> AbstractLinearOperator:
    """Diagonal operator on StokesIQU: independently scales I, Q, U."""
    in_struct = StokesIQU.structure_for(weights_i.shape, weights_i.dtype)
    return asoperator(
        lambda x: StokesIQU(weights_i * x.i, weights_q * x.q, weights_u * x.u),
        in_structure=in_struct,
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
        blocks = BJ.blocks  # (N_PIX, 1, 1)
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
        blocks = BJ.blocks  # (N_PIX, 2, 2)
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
        blocks = BJ.blocks  # (N_PIX, 3, 3)
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
        blocks = BJ.blocks  # (N_PIX, 3, 3)

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
        blocks = BJ.blocks
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
    with pytest.raises(TypeError, match='Stokes'):
        BJPreconditioner.create(op)


def test_create_raises_for_non_square_operator() -> None:
    """create() raises ValueError when in_structure != out_structure."""
    in_struct = StokesIQU.structure_for((N_PIX,), jnp.float64)

    def rectangular(x: StokesIQU) -> StokesQU:
        return StokesQU(x.q, x.u)

    op = asoperator(rectangular, in_structure=in_struct)
    with pytest.raises(ValueError, match='square'):
        BJPreconditioner.create(op)


# ---------------------------------------------------------------------------
# Two-level preconditioner
# ---------------------------------------------------------------------------


def _two_level_setup(rank: int, preconditioned: bool):
    """SPD matrix A, Jacobi preconditioner M, the two-level M2 and its deflated subspace Z."""
    n = 12
    B = jax.random.normal(jax.random.key(0), (n, n))
    A = B @ B.T + 0.1 * jnp.eye(n)
    d = jnp.diag(A)
    M = asoperator(lambda x: x / d, in_structure=jax.ShapeDtypeStruct((n,), A.dtype))
    if preconditioned:
        # eigenpairs of M A, M⁻¹-orthonormal: Z = M^1/2 Y, with Y those of M^1/2 A M^1/2
        theta, Y = jnp.linalg.eigh(A / jnp.sqrt(d) / jnp.sqrt(d)[:, None])
        Z = Y / jnp.sqrt(d)[:, None]
    else:
        theta, Z = jnp.linalg.eigh(A)
    theta, Z = theta[:rank], Z[:, :rank]
    M2 = make_two_level_preconditioner(M, LowRankTerms(theta, Z.T), preconditioned=preconditioned)
    return A, M, M2, Z


@pytest.mark.parametrize('preconditioned', [False, True])
@pytest.mark.parametrize('rank', [1, 4])
def test_two_level_is_identity_on_deflated_subspace(rank: int, preconditioned: bool) -> None:
    A, _, M2, Z = _two_level_setup(rank, preconditioned)
    for z in Z.T:
        assert_allclose(M2(A @ z), z, atol=1e-10)


@pytest.mark.parametrize('preconditioned', [False, True])
@pytest.mark.parametrize('rank', [1, 4])
def test_two_level_matches_first_level_on_complement(rank: int, preconditioned: bool) -> None:
    A, M, M2, Z = _two_level_setup(rank, preconditioned)
    x = jax.random.normal(jax.random.key(1), (A.shape[0],))
    # A-orthogonal complement of the deflated subspace
    x = x - Z @ jnp.linalg.solve(Z.T @ A @ Z, Z.T @ A @ x)
    assert_allclose(M2(A @ x), M(A @ x), atol=1e-10)
