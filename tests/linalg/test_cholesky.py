"""Tests for the block-banded Cholesky factorization."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.linalg import BandedCholeskyOperator, banded_cholesky, banded_cholesky_solve


def _banded_spd(n: int, k: int, w: int, seed: int):
    """A random block-banded SPD matrix of bandwidth w, as (dense, bands)."""
    kL, kb = jr.split(jr.key(seed))
    # dense lower block-banded L (well-conditioned diagonal) -> G = L Lᵀ is SPD, block-banded of
    # bandwidth w. w=0 is the block-diagonal degeneration.
    L = jnp.zeros((n * k, n * k))
    for i in range(n):
        for d in range(w + 1):
            j = i - d
            if j >= 0:
                blk = jr.normal(jr.fold_in(kL, i * 10 + d), (k, k))
                if d == 0:
                    blk = jnp.tril(blk) + (k + 1) * jnp.eye(k)
                L = L.at[i * k : (i + 1) * k, j * k : (j + 1) * k].set(blk)
    dense = L @ L.T

    bands = jnp.zeros((n, w + 1, k, k))  # upper band blocks A[j, j+d]
    for j in range(n):
        for d in range(w + 1):
            if j + d < n:
                bands = bands.at[j, d].set(
                    dense[j * k : (j + 1) * k, (j + d) * k : (j + d + 1) * k]
                )
    return dense, bands


@pytest.mark.parametrize('w', [0, 1, 2])
def test_banded_cholesky_solve_matches_dense(w):
    n, k = 6, 2
    dense, bands = _banded_spd(n, k, w, seed=11)
    lb = banded_cholesky(bands)
    b = jr.normal(jr.key(12), (n, k))
    x = banded_cholesky_solve(lb, b)
    expected = jnp.linalg.solve(dense, b.reshape(-1)).reshape(n, k)
    assert_allclose(x, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('w', [0, 1, 2])
def test_banded_cholesky_grad_is_finite(w):
    n, k = 6, 2
    _, bands = _banded_spd(n, k, w, seed=13)
    b = jr.normal(jr.key(14), (n, k))

    def loss(bands, b):
        lb = banded_cholesky(bands)
        x = banded_cholesky_solve(lb, b)
        return jnp.sum(x**2)

    grad_bands, grad_b = jax.grad(loss, argnums=(0, 1))(bands, b)
    assert jnp.all(jnp.isfinite(grad_bands))
    assert jnp.all(jnp.isfinite(grad_b))


def test_banded_cholesky_degenerate_n_blocks_one_matches_dense_cholesky():
    # n_blocks=1 is the fully dense case: must be numerically identical to a direct
    # jnp.linalg.cholesky/cho_solve, not merely close.
    key = jr.key(20)
    k = 5
    a = jr.normal(key, (k, k))
    dense = a @ a.T + k * jnp.eye(k)
    bands = dense[None, None]  # n_blocks=1, w1=1

    lb = banded_cholesky(bands)
    assert_array_equal(lb[0, 0], jnp.linalg.cholesky(dense))

    b = jr.normal(jr.fold_in(key, 1), (1, k))
    x = banded_cholesky_solve(lb, b)
    expected = jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(dense), True), b[0])
    assert_array_equal(x[0], expected)


def test_banded_cholesky_batched_leading_dims():
    # Two independent batch axes, e.g. two separate collections stacked together.
    n, k, w = 4, 2, 1
    dense_bands = [_banded_spd(n, k, w, seed=100 + i) for i in range(6)]
    bands = jnp.stack([b for _, b in dense_bands]).reshape(3, 2, n, w + 1, k, k)
    denses = jnp.stack([d for d, _ in dense_bands]).reshape(3, 2, n * k, n * k)

    lb = banded_cholesky(bands)
    assert lb.shape == bands.shape

    b = jr.normal(jr.key(30), (3, 2, n, k))
    x = banded_cholesky_solve(lb, b)
    expected = jnp.linalg.solve(denses, b.reshape(3, 2, -1, 1))[..., 0].reshape(3, 2, n, k)
    assert_allclose(x, expected, rtol=1e-4, atol=1e-5)


def test_banded_cholesky_jit():
    n, k, w = 4, 2, 1
    _, bands = _banded_spd(n, k, w, seed=50)
    b = jr.normal(jr.key(51), (n, k))

    lb = jax.jit(banded_cholesky)(bands)
    x = jax.jit(banded_cholesky_solve)(lb, b)

    expected_lb = banded_cholesky(bands)
    expected_x = banded_cholesky_solve(expected_lb, b)
    assert_allclose(lb, expected_lb)
    assert_allclose(x, expected_x)


class TestBandedCholeskyOperator:
    """Tests for BandedCholeskyOperator."""

    def test_from_dense_matches_dense_solve(self):
        key = jr.key(60)
        n_dets, k = 5, 3
        a = jr.normal(key, (n_dets, k, k))
        dense = a @ jnp.swapaxes(a, -1, -2) + k * jnp.eye(k)
        struct = jax.ShapeDtypeStruct((n_dets, k), dense.dtype)

        op = BandedCholeskyOperator.from_dense(dense, in_structure=struct)
        b = jr.normal(jr.fold_in(key, 1), (n_dets, k))
        x = op(b)
        expected = jnp.linalg.solve(dense, b[..., None])[..., 0]
        assert_allclose(x, expected, rtol=1e-4, atol=1e-5)

    def test_from_bands_matches_banded_solve(self):
        n, k, w = 5, 2, 1
        dense, bands = _banded_spd(n, k, w, seed=61)
        struct = jax.ShapeDtypeStruct((n, k), bands.dtype)

        op = BandedCholeskyOperator.from_bands(bands, in_structure=struct)
        b = jr.normal(jr.key(62), (n, k))
        x = op(b)
        expected = jnp.linalg.solve(dense, b.reshape(-1)).reshape(n, k)
        assert_allclose(x, expected, rtol=1e-4, atol=1e-5)

    def test_from_dense_infers_in_structure(self):
        key = jr.key(63)
        n_dets, k = 4, 3
        a = jr.normal(key, (n_dets, k, k))
        dense = a @ jnp.swapaxes(a, -1, -2) + k * jnp.eye(k)

        op = BandedCholeskyOperator.from_dense(dense)  # no in_structure given
        assert op.in_structure == jax.ShapeDtypeStruct((n_dets, k), dense.dtype)
        b = jr.normal(jr.fold_in(key, 1), (n_dets, k))
        assert_allclose(
            op(b), BandedCholeskyOperator.from_dense(dense, in_structure=op.in_structure)(b)
        )

    def test_from_bands_infers_in_structure(self):
        n, k, w = 4, 2, 1
        _, bands = _banded_spd(n, k, w, seed=64)

        op = BandedCholeskyOperator.from_bands(bands)  # no in_structure given
        assert op.in_structure == jax.ShapeDtypeStruct((n, k), bands.dtype)
        b = jr.normal(jr.key(65), (n, k))
        assert_allclose(
            op(b), BandedCholeskyOperator.from_bands(bands, in_structure=op.in_structure)(b)
        )

    def test_mv_handles_multi_leaf_pytree(self):
        # A dense matrix acting on a PyTree with several differently-shaped leaves: mv must
        # flatten/concatenate/split consistently regardless of how many leaves there are.
        key = jr.key(70)
        k = 5  # total size, split as leaves of size 2 and 3
        a = jr.normal(key, (k, k))
        dense = a @ a.T + k * jnp.eye(k)
        struct = {
            'a': jax.ShapeDtypeStruct((2,), dense.dtype),
            'b': jax.ShapeDtypeStruct((3,), dense.dtype),
        }

        op = BandedCholeskyOperator.from_dense(dense, in_structure=struct)
        x = {'a': jr.normal(jr.fold_in(key, 1), (2,)), 'b': jr.normal(jr.fold_in(key, 2), (3,))}
        y = op(x)
        flat = jnp.concatenate([x['a'], x['b']])
        expected = jnp.linalg.solve(dense, flat)
        assert_allclose(jnp.concatenate([y['a'], y['b']]), expected, rtol=1e-4, atol=1e-5)

    def test_symmetric(self):
        key = jr.key(80)
        k = 4
        a = jr.normal(key, (k, k))
        dense = a @ a.T + k * jnp.eye(k)
        struct = jax.ShapeDtypeStruct((k,), dense.dtype)
        op = BandedCholeskyOperator.from_dense(dense, in_structure=struct)
        assert op.T is op  # @symmetric

    def test_jit(self):
        key = jr.key(90)
        n_dets, k = 3, 2
        a = jr.normal(key, (n_dets, k, k))
        dense = a @ jnp.swapaxes(a, -1, -2) + k * jnp.eye(k)
        struct = jax.ShapeDtypeStruct((n_dets, k), dense.dtype)
        op = BandedCholeskyOperator.from_dense(dense, in_structure=struct)
        b = jr.normal(jr.fold_in(key, 1), (n_dets, k))
        assert_allclose(jax.jit(lambda x: op(x))(b), op(b))
