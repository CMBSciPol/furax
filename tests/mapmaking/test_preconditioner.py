"""Tests for the mapmaking preconditioners."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax import asoperator
from furax.linalg import LowRankTerms
from furax.mapmaking.preconditioner import BJPreconditioner, make_two_level_preconditioner
from furax.obs.stokes import Stokes, StokesIQU, StokesQU

N_PIX = 8


@pytest.mark.parametrize('stokes', ['I', 'QU', 'IQU'])
def test_block_jacobi_recovers_the_per_pixel_blocks(stokes: str) -> None:
    # an operator coupling the Stokes components of each pixel through its own SPD block
    n = len(stokes)
    a = jax.random.normal(jax.random.key(0), (N_PIX, n, n))
    matrices = a @ jnp.swapaxes(a, -1, -2) + jnp.eye(n)
    cls = Stokes.class_for(stokes)
    op = asoperator(
        lambda x: cls.from_array(jnp.einsum('pij,jp->ip', matrices, x.data)),
        in_structure=cls.structure_for((N_PIX,), jnp.float64),
    )

    BJ = BJPreconditioner.create(op)
    assert_allclose(BJ.blocks, matrices, rtol=1e-12, atol=1e-12)
    x = cls.from_array(jax.random.normal(jax.random.key(1), (n, N_PIX)))
    assert_allclose(BJ(x).data, op(x).data, rtol=1e-12)
    assert_allclose(BJ.I(BJ(x)).data, x.data, rtol=1e-10)


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
