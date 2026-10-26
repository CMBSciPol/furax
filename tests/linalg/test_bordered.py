"""Tests for the bordered block-banded matrices, their probing and the arrowhead solve."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.linalg import (
    ArrowheadCholeskyOperator,
    BandedCholeskyOperator,
    BorderedBandedMatrix,
    probe_bordered_banded,
)

BATCH = (2,)

# (n_blocks, k, half width, border): a bordered band, a band alone, a dense block alone
SHAPES = {'bordered': (5, 2, 1, 3), 'banded': (5, 2, 2, 0), 'dense': (0, 0, 0, 3)}


def _random(key, n, k, w, m):
    """A random diagonally dominant matrix of the given structure, and its dense form."""
    size = n * k + m
    block = np.arange(n * k) // k
    mask = np.ones((size, size), bool)
    mask[: n * k, : n * k] = np.abs(block[:, None] - block) <= w
    a = np.asarray(jr.normal(key, (*BATCH, size, size))) * mask
    dense = (a + np.swapaxes(a, -1, -2)) / 2
    dense += np.abs(dense).sum(axis=-1)[..., None] * np.eye(size) + np.eye(size)

    bands = border = corner = None
    if n:
        bands = np.zeros((*BATCH, n, w + 1, k, k))
        for i in range(n):
            for d in range(min(w + 1, n - i)):
                rows, columns = slice(i * k, (i + 1) * k), slice((i + d) * k, (i + d + 1) * k)
                bands[..., i, d, :, :] = dense[..., rows, columns]
    if m:
        corner = dense[..., n * k :, n * k :]
        if n:
            border = dense[..., : n * k, n * k :]
    as_jax = lambda a: None if a is None else jnp.asarray(a)
    return BorderedBandedMatrix(as_jax(bands), as_jax(border), as_jax(corner)), dense


def _pair(key, n, k, m):
    k1, k2 = jr.split(key)
    x_a = jr.normal(k1, (*BATCH, n, k)) if n else None
    x_c = jr.normal(k2, (*BATCH, m)) if m else None
    return x_a, x_c


def _flat(x_a, x_c):
    parts = [] if x_a is None else [x_a.reshape(*BATCH, -1)]
    return jnp.concatenate([*parts, *([] if x_c is None else [x_c])], axis=-1)


@pytest.mark.parametrize('shape', list(SHAPES))
def test_matrix_and_its_inverse_match_the_dense_matrix(shape):
    n, k, w, m = SHAPES[shape]
    matrix, dense = _random(jr.key(0), n, k, w, m)
    x = _pair(jr.key(1), n, k, m)
    expected = np.einsum('...ab,...b->...a', dense, _flat(*x))
    assert_allclose(_flat(*matrix(x)), expected, rtol=1e-10)
    assert_allclose(_flat(*matrix.cholesky()(matrix(x))), _flat(*x), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('shape', list(SHAPES))
def test_probing_recovers_the_matrix(shape):
    n, k, w, m = SHAPES[shape]
    matrix, _ = _random(jr.key(4), n, k, w, m)
    probed, extra = probe_bordered_banded(
        lambda x: (matrix(x), None),
        batch_shape=BATCH,
        n_blocks=n,
        k=k,
        n_border=m,
        half_width=w,
        dtype=jnp.float64,
    )
    assert extra is None
    x = _pair(jr.key(5), n, k, m)
    assert_allclose(_flat(*probed(x)), _flat(*matrix(x)), rtol=1e-10)


def test_probing_collects_the_other_outputs_of_the_border_probes():
    n, k, w, m = SHAPES['bordered']
    matrix, _ = _random(jr.key(6), n, k, w, m)
    _, extra = probe_bordered_banded(
        lambda x: (matrix(x), x[1]),
        batch_shape=BATCH,
        n_blocks=n,
        k=k,
        n_border=m,
        half_width=w,
        dtype=jnp.float64,
    )
    # the border probes, one unit vector per border unknown
    assert_allclose(extra, jnp.broadcast_to(jnp.eye(m)[:, None], (m, *BATCH, m)))


@pytest.mark.parametrize('n_colors', [None, 4])
def test_probing_with_more_colors_recovers_the_matrix_and_returns_every_output(n_colors):
    n, k, w, m = SHAPES['bordered']
    matrix, _ = _random(jr.key(7), n, k, w, m)
    probed, extra = probe_bordered_banded(
        lambda x: (matrix(x), x),
        batch_shape=BATCH,
        n_blocks=n,
        k=k,
        n_border=m,
        half_width=w,
        dtype=jnp.float64,
        n_colors=n_colors,
        all_extras=True,
    )
    x = _pair(jr.key(8), n, k, m)
    assert_allclose(_flat(*probed(x)), _flat(*matrix(x)), rtol=1e-10)
    # every probe: the core ones, color by color then function by function, then the border ones
    colors = 2 * w + 1 if n_colors is None else n_colors
    core, border = extra
    assert core.shape == (colors * k + m, *BATCH, n, k)
    blocks = np.arange(n) % colors
    assert_allclose(core[1, 0], (blocks[:, None] == 0) & (np.arange(k) == 1))
    assert_allclose(border[colors * k :], jnp.broadcast_to(jnp.eye(m)[:, None], (m, *BATCH, m)))


def test_probing_with_a_color_per_block_recovers_a_band_wider_than_the_colors():
    n, k, w, m = 3, 2, 2, 1
    matrix, _ = _random(jr.key(9), n, k, w, m)
    probed, _ = probe_bordered_banded(
        lambda x: (matrix(x), None),
        batch_shape=BATCH,
        n_blocks=n,
        k=k,
        n_border=m,
        half_width=w,
        dtype=jnp.float64,
        n_colors=n,
    )
    x = _pair(jr.key(10), n, k, m)
    assert_allclose(_flat(*probed(x)), _flat(*matrix(x)), rtol=1e-10)


def test_probing_rejects_too_few_colors():
    with pytest.raises(ValueError, match='cannot resolve'):
        probe_bordered_banded(
            lambda x: (x, None),
            batch_shape=BATCH,
            n_blocks=5,
            k=1,
            n_border=0,
            half_width=2,
            dtype=jnp.float64,
            n_colors=4,
        )


def test_arrowhead_solve_matches_the_dense_solve():
    n_blocks, size, n_global = 3, 4, 2
    key = jr.split(jr.key(7), 3)
    a = jr.normal(key[0], (n_blocks, size, size))
    blocks = a @ jnp.swapaxes(a, -1, -2) + size * jnp.eye(size)
    cross = 0.1 * jr.normal(key[1], (n_global, n_blocks, size))
    corner = jnp.eye(n_global) * 10
    inverse = BandedCholeskyOperator.from_dense(blocks)
    op = ArrowheadCholeskyOperator.from_blocks(inverse, cross, corner)

    dense = jax.scipy.linalg.block_diag(*blocks, corner)
    coupling = jnp.moveaxis(cross, 0, -1).reshape(n_blocks * size, n_global)
    dense = dense.at[: n_blocks * size, n_blocks * size :].set(coupling)
    dense = dense.at[n_blocks * size :, : n_blocks * size].set(coupling.T)
    r_l, r_g = jr.normal(key[2], (n_blocks, size)), jnp.ones(n_global)
    x_l, x_g = op((r_l, r_g))
    expected = jnp.linalg.solve(dense, jnp.concatenate([r_l.ravel(), r_g]))
    assert_allclose(jnp.concatenate([x_l.ravel(), x_g]), expected, rtol=1e-8)


def test_matrix_with_more_bands_than_blocks_ignores_the_extra_ones():
    matrix, _ = _random(jr.key(8), 2, 2, 1, 0)
    padding = jnp.zeros((*matrix.bands.shape[:-3], 2, 2, 2))
    padded = BorderedBandedMatrix(jnp.concatenate([matrix.bands, padding], axis=-3))
    x = _pair(jr.key(9), 2, 2, 0)
    assert_allclose(padded(x)[0], matrix(x)[0], rtol=1e-12)
    assert_allclose(padded.cholesky()(x)[0], matrix.cholesky()(x)[0], rtol=1e-10)


def test_matrix_rejects_inconsistent_parts():
    with pytest.raises(ValueError, match='needs a core or a corner'):
        BorderedBandedMatrix()
    with pytest.raises(ValueError, match='border is present exactly when'):
        BorderedBandedMatrix(jnp.ones((2, 1, 1, 1)), jnp.ones((2, 1)))
    with pytest.raises(ValueError, match=r'take a border of shape \(2, 1\), not \(3, 1\)'):
        BorderedBandedMatrix(jnp.ones((2, 1, 1, 1)), jnp.ones((3, 1)), jnp.ones((1, 1)))
