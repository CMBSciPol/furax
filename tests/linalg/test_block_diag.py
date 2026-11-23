"""Tests for the block-diagonal direct solver."""

import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, Config, DiagonalOperator, HomothetyOperator, TreeOperator
from furax._config import verbose_solver_callback
from furax.linalg import BlockDiagonalResult, BlockDiagonalSolver
from furax.obs.landscapes import HealpixLandscape
from furax.obs.operators import CMBOperator, DustOperator, SynchrotronOperator, mixing_matrix
from furax.obs.stokes import ValidStokesLiteral
from furax.tree import as_structure, dot, ravel, unravel

NSIDE = 2
FREQUENCIES = jnp.array([30.0, 90.0, 150.0, 280.0])


def _tree_system(n: int = 6):
    """Return a pytree-matrix operator with one symmetric 2x2 block per position."""
    positions = jnp.arange(n, dtype=float)
    diag_a = 2.0 + positions
    diag_b = 3.0 + positions
    off = 0.5 * jnp.cos(positions)
    b = {'a': jnp.sin(positions), 'b': jnp.cos(positions)}
    A = TreeOperator(
        {'a': {'a': diag_a, 'b': off}, 'b': {'a': off, 'b': diag_b}}, in_structure=as_structure(b)
    )
    det = diag_a * diag_b - off**2
    x_true = {
        'a': (diag_b * b['a'] - off * b['b']) / det,
        'b': (diag_a * b['b'] - off * b['a']) / det,
    }
    return A, b, x_true


def _normal_matrix(stokes: ValidStokesLiteral, beta_dust: jax.Array) -> AbstractLinearOperator:
    """Return $A^T N^{-1} A$ of a component separation with two dust patches."""
    structure = HealpixLandscape(NSIDE, stokes).structure
    patches = jnp.arange(12 * NSIDE**2) % 2
    A = mixing_matrix(
        cmb=CMBOperator(FREQUENCIES, in_structure=structure),
        dust=DustOperator(
            FREQUENCIES,
            frequency0=150.0,
            temperature=jnp.array([20.0, 15.0]),
            temperature_patch_indices=patches,
            beta=beta_dust,
            beta_patch_indices=patches,
            in_structure=structure,
        ),
        synchrotron=SynchrotronOperator(
            FREQUENCIES, frequency0=20.0, beta_pl=-3.0, in_structure=structure
        ),
    )
    (frequency_maps,) = jax.tree.leaves(A.out_structure)
    weights = 1.0 + jax.random.uniform(jax.random.key(0), frequency_maps.shape)
    N = DiagonalOperator(weights, in_structure=A.out_structure)
    return (A.T @ N.I @ A).reduce()


def _components(stokes: ValidStokesLiteral):
    landscape = HealpixLandscape(NSIDE, stokes)
    keys = jax.random.split(jax.random.key(1), 3)
    return {name: landscape.normal(key) for name, key in zip(['cmb', 'dust', 'synchrotron'], keys)}


BETA_DUST = jnp.array([1.54, 1.6])


class TestBlockDiagonalBasic:
    def test_solves_tree_system(self):
        A, b, x_true = _tree_system()
        result = BlockDiagonalSolver()(A, b)
        assert isinstance(result, BlockDiagonalResult)
        assert tree_equal(result.solution, x_true, rtol=1e-12)

    def test_ignores_initial_guess_and_preconditioner(self):
        A, b, x_true = _tree_system()
        preconditioner = HomothetyOperator(2.0, in_structure=A.in_structure)
        result = BlockDiagonalSolver()(A, b, x_true, preconditioner=preconditioner)
        assert tree_equal(result.solution, x_true, rtol=1e-12)

    @pytest.mark.parametrize('stokes', ['I', 'QU', 'IQU'])
    def test_component_separation(self, stokes: ValidStokesLiteral):
        M = _normal_matrix(stokes, BETA_DUST)
        b = _components(stokes)
        expected = unravel(M.in_structure, jnp.linalg.solve(M.as_matrix(), ravel(b)))
        actual = BlockDiagonalSolver()(M, b).solution
        assert tree_equal(actual, expected, rtol=1e-10, atol=1e-12)


def _normal_system():
    """Return the QU component-separation normal matrix, a right-hand side and its solution."""
    M = _normal_matrix('QU', BETA_DUST)
    b = _components('QU')
    x_true = unravel(M.in_structure, jnp.linalg.solve(M.as_matrix(), ravel(b)))
    return M, b, x_true


class TestBlockDiagonalInverse:
    def test_inverse_solver_argument(self):
        M, b, x_true = _normal_system()
        actual = M.I(solver=BlockDiagonalSolver())(b)
        assert tree_equal(actual, x_true, rtol=1e-10, atol=1e-12)

    def test_config(self):
        M, b, x_true = _normal_system()
        with Config(solver=BlockDiagonalSolver()):
            actual = M.I(b)
        assert tree_equal(actual, x_true, rtol=1e-10, atol=1e-12)

    def test_throw(self):
        M, b, x_true = _normal_system()
        with Config(solver=BlockDiagonalSolver(), solver_throw=True):
            actual = M.I(b)
        assert tree_equal(actual, x_true, rtol=1e-10, atol=1e-12)

    def test_verbose_callback(self, capsys: pytest.CaptureFixture[str]):
        M, b, _ = _normal_system()
        with Config(solver=BlockDiagonalSolver(), solver_callback=verbose_solver_callback):
            M.I(b)
        jax.effects_barrier()
        assert 'direct inversion' in capsys.readouterr().out


class TestBlockDiagonalJit:
    def test_jit(self):
        M, b, x_true = _normal_system()

        @jax.jit
        def solve(b):
            return M.I(solver=BlockDiagonalSolver())(b)

        assert tree_equal(solve(b), x_true, rtol=1e-10, atol=1e-12)


class TestBlockDiagonalGrad:
    def test_grad_matches_dense_solve(self):
        b = _components('QU')

        def block_quadratic_form(beta_dust: jax.Array) -> jax.Array:
            M = _normal_matrix('QU', beta_dust)
            return dot(b, M.I(solver=BlockDiagonalSolver())(b))

        def dense_quadratic_form(beta_dust: jax.Array) -> jax.Array:
            M = _normal_matrix('QU', beta_dust)
            return ravel(b) @ jnp.linalg.solve(M.as_matrix(), ravel(b))

        actual = jax.grad(block_quadratic_form)(BETA_DUST)
        expected = jax.grad(dense_quadratic_form)(BETA_DUST)
        assert_allclose(actual, expected, rtol=1e-10)
