import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose

from furax.obs.atmosphere import FourierScreenOperator, cell_index, fit_cell
from furax.obs.coords import azel_to_quat, xieta_to_quat

NDET, NSAMP = 3, 16


def _operator(backend: str) -> FourierScreenOperator:
    times = jnp.linspace(0, 10, NSAMP)
    qbore = azel_to_quat(jnp.deg2rad(5 * jnp.sin(times)), jnp.full(NSAMP, jnp.deg2rad(50.0)))
    qdet = xieta_to_quat(jnp.linspace(-0.02, 0.02, NDET), jnp.zeros(NDET), jnp.zeros(NDET))
    # the second detector has no pointing
    qdet = Quaternion.from_array(qdet.wxyz.at[1].set(0))
    return FourierScreenOperator.create(
        cell_index(4 * np.eye(2), q_max=10.0), 4.0, qbore, qdet, times, jnp.array([0.01, -0.02]),
        backend=backend, origin=jnp.array([-2.0, -2.0]),
    )  # fmt: skip


@pytest.mark.parametrize('backend', ['dense', 'nufft'])
def test_matches_matrix(backend: str) -> None:
    op = _operator(backend)
    a = jax.random.normal(jax.random.key(0), op.in_structure.shape)
    tod = op(a)
    assert tod.shape == (NDET, NSAMP)
    assert_allclose(tod.ravel(), op.as_matrix() @ a.ravel(), atol=1e-8)
    assert jnp.all(tod[1] == 0)


def test_transpose() -> None:
    op = _operator('nufft')
    a = jax.random.normal(jax.random.key(0), op.in_structure.shape)
    d = jax.random.normal(jax.random.key(1), (NDET, NSAMP))
    assert_allclose(jnp.vdot(d, op(a)), jnp.vdot(op.T(d), a), rtol=1e-8)


def test_cell_index() -> None:
    index = cell_index(2 * np.pi * np.eye(2), q_max=2.0)
    expected = {(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)}
    assert set(map(tuple, index.tolist())) == expected


def test_fit_cell_contains_samples() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, (NDET, NSAMP, 2)) * [1, 0.1]
    times = np.linspace(0, 10, NSAMP)
    omega = np.array([0.0, 0.05])
    cell, origin = fit_cell(u, times, omega, margin=0.1)
    xi = ((u + times[:, None] * omega) - origin) @ np.linalg.inv(cell).T
    assert np.all((xi > 0) & (xi < 1))
