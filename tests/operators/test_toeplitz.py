import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jit
from numpy.testing import assert_allclose

from furax.operators.toeplitz import SymmetricBandToeplitzOperator, dense_symmetric_band_toeplitz


@pytest.mark.parametrize(
    'n, band_values, expected_matrix',
    [
        (1, [1], [[1]]),
        (2, [1], [[1, 0], [0, 1]]),
        (2, [1, 2], [[1, 2], [2, 1]]),
        (3, [1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        (3, [1, 2], [[1, 2, 0], [2, 1, 2], [0, 2, 1]]),
        (3, [1, 2, 3], [[1, 2, 3], [2, 1, 2], [3, 2, 1]]),
    ],
)
def test_dense_symmetric_band_toeplitz(
    n: int, band_values: list[int], expected_matrix: list[list[int]]
):
    band_values = np.array(band_values)
    expected_matrix = np.array(expected_matrix)
    actual_matrix = dense_symmetric_band_toeplitz(n, band_values)
    assert_allclose(actual_matrix, expected_matrix)


@pytest.mark.parametrize(
    'n, band_values',
    itertools.chain.from_iterable(
        [[(n, jnp.arange(1, k + 2)) for k in range(n)] for n in range(1, 5)]
    ),
)
def test_fft(n: int, band_values):
    x = jnp.arange(n) + 1
    actual_y = SymmetricBandToeplitzOperator((n, n), band_values, method='fft')(x)
    expected_y = SymmetricBandToeplitzOperator((n, n), band_values, method='dense')(x)
    assert_allclose(actual_y, expected_y)


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize('method', SymmetricBandToeplitzOperator.METHODS)
def test(method: str, do_jit: bool) -> None:
    band_values = jnp.array([4.0, 3, 2, 1])
    shape = (6, 6)
    x = jnp.array([1.0, 2, 3, 4, 5, 6])
    expected_y = jnp.array([20.0, 33, 48, 57, 58, 50])
    func = SymmetricBandToeplitzOperator(shape, band_values, method=method)
    if do_jit:
        func = jit(func)
    y = func(x)
    assert_allclose(y, expected_y, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    'band_number, expected_fft_size',
    [(1, 2), (2, 4), (3, 8), (4, 8), (5, 16), (1023, 2048), (1024, 2048), (1025, 4096)],
)
def test_default_size(band_number: int, expected_fft_size: int):
    actual_fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(band_number)
    assert actual_fft_size == expected_fft_size


def test_overlap_fft_size():
    band_values = jnp.array([1.0, 0.5, 0.25])
    matrix_sizes = [3, 4, 5, 6, 7, 8, 8]
    fft_sizes = [8, 16]
    # TBD
