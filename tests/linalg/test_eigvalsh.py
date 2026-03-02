"""Tests for analytic eigvalsh implementations."""

import jax.numpy as jnp
import numpy as np
import pytest

from furax.linalg import eigvalsh


def assert_eigvalsh_close(A_np, rtol=1e-4, atol=1e-4):
    expected = np.linalg.eigvalsh(A_np)
    actual = np.asarray(eigvalsh(jnp.array(A_np)))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


# --- 1x1 ---


def test_1x1_scalar():
    A = np.array([[[7.0]]])
    assert_eigvalsh_close(A)


# --- 2x2 ---


def test_2x2_identity():
    A = np.eye(2)[None]
    assert_eigvalsh_close(A)


def test_2x2_diagonal():
    A = np.array([[[3.0, 0.0], [0.0, 5.0]]])
    assert_eigvalsh_close(A)


def test_2x2_diagonal_reversed():
    # larger eigenvalue first on diagonal
    A = np.array([[[5.0, 0.0], [0.0, 3.0]]])
    assert_eigvalsh_close(A)


def test_2x2_dense():
    A = np.array([[[3.0, 1.0], [1.0, 3.0]]])
    assert_eigvalsh_close(A)


def test_2x2_batch():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((100, 2, 2)).astype(np.float32)
    A = B @ B.swapaxes(-1, -2) + 2 * np.eye(2)
    assert_eigvalsh_close(A)


# --- 3x3 ---


def test_3x3_identity():
    A = np.eye(3)[None]
    assert_eigvalsh_close(A)


def test_3x3_diagonal():
    A = np.array([[[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]]])
    assert_eigvalsh_close(A)


def test_3x3_block_diagonal():
    # (0,1), (1,0), (0,2), (2,0) are zero — I decoupled from QU
    A = np.array([[[4.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 3.0]]])
    assert_eigvalsh_close(A)


def test_3x3_dense():
    A = np.array([[[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]]])
    assert_eigvalsh_close(A)


def test_3x3_repeated_eigenvalues():
    # scalar multiple of identity — all eigenvalues equal, p -> 0
    A = 3.0 * np.eye(3)[None]
    assert_eigvalsh_close(A)


def test_3x3_two_equal_eigenvalues():
    A = np.array([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]])
    assert_eigvalsh_close(A)


def test_3x3_batch():
    rng = np.random.default_rng(1)
    B = rng.standard_normal((1000, 3, 3)).astype(np.float32)
    A = B @ B.swapaxes(-1, -2) + 3 * np.eye(3)
    assert_eigvalsh_close(A)


def test_3x3_batch_block_diagonal():
    rng = np.random.default_rng(2)
    B = rng.standard_normal((1000, 3, 3)).astype(np.float32)
    A = B @ B.swapaxes(-1, -2) + 3 * np.eye(3)
    A[:, 0, 1] = A[:, 1, 0] = 0.0
    A[:, 0, 2] = A[:, 2, 0] = 0.0
    assert_eigvalsh_close(A)


# --- fallback for n > 3 ---


def test_4x4_fallback():
    A = np.eye(4)[None]
    assert_eigvalsh_close(A)


def test_4x4_batch_fallback():
    rng = np.random.default_rng(3)
    B = rng.standard_normal((50, 4, 4)).astype(np.float32)
    A = B @ B.swapaxes(-1, -2) + 4 * np.eye(4)
    assert_eigvalsh_close(A)


def test_4x4_batch_larger_than_batch_size():
    # batch (20) > batch_size (3): checks lax.map chunking preserves order
    rng = np.random.default_rng(4)
    B = rng.standard_normal((20, 4, 4)).astype(np.float32)
    A = B @ B.swapaxes(-1, -2) + 4 * np.eye(4)
    expected = np.linalg.eigvalsh(A)
    actual = np.asarray(eigvalsh(jnp.array(A), batch_size=3))
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


# --- shape validation ---


def test_invalid_shape_1d():
    with pytest.raises(ValueError, match='eigvalsh requires'):
        eigvalsh(jnp.ones((4,)))


def test_invalid_shape_non_square():
    with pytest.raises(ValueError, match='eigvalsh requires'):
        eigvalsh(jnp.ones((3, 4)))
