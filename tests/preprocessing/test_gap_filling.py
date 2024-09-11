import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.operators import PackOperator
from furax.operators.toeplitz import SymmetricBandToeplitzOperator
from furax.preprocessing.gap_filling import GapFillingOperator


@pytest.mark.parametrize(
    'n_tt, fft_size, expected_kernel',
    [
        ([1], 1, [1]),
        ([1], 2, [1, 0]),
        ([1], 4, [1, 0, 0, 0]),
        ([1, 2], 4, [1, 2, 0, 2]),
        ([3, 2, 1], 8, [3, 2, 1, 0, 0, 0, 1, 2]),
    ],
)
def test_get_kernel(n_tt: list[int], fft_size: int, expected_kernel: list[int]):
    n_tt = np.array(n_tt)
    expected_kernel = np.array(expected_kernel)
    actual_kernel = GapFillingOperator._get_kernel(n_tt, fft_size)
    assert_allclose(actual_kernel, expected_kernel)


@pytest.mark.parametrize('n_tt, fft_size', [([1, 2], 1), ([1, 2, 3], 4)])
def test_get_kernel_fail_lagmax(n_tt: list[int], fft_size: int):
    # This test should fail because the maximum lag is too large for the required fft_size
    # NB: it is the call to `jnp.pad` that fails with a ValueError
    n_tt = np.array(n_tt)
    with pytest.raises(ValueError):
        _ = GapFillingOperator._get_kernel(n_tt, fft_size)


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize('x_shape', [(1,), (10,), (1, 100), (2, 10), (2, 100), (1, 2, 100)])
def test_generate_realization_shape(x_shape: tuple[int], do_jit: bool):
    x = jnp.zeros(x_shape, dtype=float)
    # dummy toeplitz and pack operators
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), structure)
    pack = PackOperator(jnp.ones_like(x, dtype=bool), structure)
    op = GapFillingOperator(cov, pack)
    func = ft.partial(op._generate_realization_for, seed=1234)
    if do_jit:
        func = jax.jit(func)
    real = func(x)
    assert real.shape == x_shape


@pytest.fixture
def dummy_shape():
    shape = (2, 100)
    return shape


@pytest.fixture
def dummy_mask(dummy_shape):
    mask = jnp.ones(dummy_shape, dtype=bool)
    samples = dummy_shape[-1]
    gap_size = samples // 10
    left, right = (samples - gap_size) // 2, (samples + gap_size) // 2
    mask = mask.at[:, left:right].set(False)
    return mask


@pytest.fixture
def dummy_gap_filling_operator(dummy_shape, dummy_mask):
    x = jnp.ones(dummy_shape, dtype=float)
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), structure)
    pack = PackOperator(dummy_mask, structure)
    return GapFillingOperator(cov, pack)


@pytest.mark.parametrize(
    'n_tt, fft_size', [([1], 1), ([1], 2), ([1], 4), ([1, 2], 4), ([3, 2, 1], 8)]
)
def test_get_psd_non_negative(n_tt, fft_size, dummy_gap_filling_operator):
    n_tt = np.array(n_tt)
    psd = dummy_gap_filling_operator._get_psd(n_tt, fft_size)
    assert np.all(psd >= 0)


@pytest.mark.parametrize(
    'samples, expected_fft_size',
    [(1, 2), (2, 4), (3, 8), (4, 8), (5, 16), (1023, 2048), (1025, 4096), (2049, 8192)],
)
def test_default_size(samples: int, expected_fft_size: int):
    actual_fft_size = GapFillingOperator._get_default_fft_size(samples)
    assert actual_fft_size == expected_fft_size


def test_valid_samples_and_no_nans(dummy_shape, dummy_gap_filling_operator):
    key = jax.random.key(987654321)
    x = jax.random.uniform(key, dummy_shape, dtype=float)
    op = dummy_gap_filling_operator
    func = jax.jit(op)
    y = func(x)
    assert_allclose(op.pack(x), op.pack(y))
    assert not np.any(np.isnan(y))
