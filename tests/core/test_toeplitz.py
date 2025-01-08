import itertools

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import jit
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, SymmetricBandToeplitzOperator
from furax.core import dense_symmetric_band_toeplitz


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
    in_structure = jax.ShapeDtypeStruct((n,), jnp.float64)
    actual_y = SymmetricBandToeplitzOperator(band_values, in_structure, method='fft')(x)
    expected_y = SymmetricBandToeplitzOperator(band_values, in_structure, method='dense')(x)
    assert_allclose(actual_y, expected_y)


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize('method', SymmetricBandToeplitzOperator.METHODS)
def test(method: str, do_jit: bool) -> None:
    band_values = jnp.array([4.0, 3, 2, 1])
    in_structure = jax.ShapeDtypeStruct((6,), jnp.float64)
    x = jnp.array([1.0, 2, 3, 4, 5, 6])
    expected_y = jnp.array([20.0, 33, 48, 57, 58, 50])
    op = SymmetricBandToeplitzOperator(band_values, in_structure, method=method)
    if do_jit:
        # to avoid error: TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'
        # we capture op in the lambda closure
        func = jit(lambda x: SymmetricBandToeplitzOperator.mv(op, x))
    else:
        func = op
    y = func(x)
    assert_allclose(y, expected_y, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('method', SymmetricBandToeplitzOperator.METHODS)
@pytest.mark.parametrize(
    'in_shape, band_shape',
    [
        ((2, 6), (4,)),
        ((2, 6), (1, 4)),
        ((2, 6), (2, 4)),
        ((1, 2, 6), (4,)),
        ((1, 2, 6), (1, 4)),
        ((1, 2, 6), (1, 1, 4)),
        ((1, 2, 6), (1, 2, 4)),
        ((2, 1, 6), (1, 1, 4)),
        ((2, 1, 6), (2, 1, 4)),
        ((2, 2, 6), (1, 1, 4)),
        ((2, 2, 6), (1, 2, 4)),
        ((2, 2, 6), (2, 1, 4)),
        ((2, 2, 6), (2, 2, 4)),
        ((1, 2, 2, 6), (2, 1, 4)),
    ],
    ids=lambda x: str(x).replace(' ', ''),
)
def test_multidimensional(
    in_shape: tuple[int, ...], band_shape: tuple[int, ...], method: str
) -> None:
    band_values = jnp.arange(np.prod(band_shape)).reshape(band_shape)
    in_structure = jax.ShapeDtypeStruct(in_shape, jnp.float64)
    op = SymmetricBandToeplitzOperator(band_values, in_structure, method=method)
    broadcast_band_values = jnp.broadcast_to(band_values, in_shape[:-1] + (4,))
    expected_blocks = [
        dense_symmetric_band_toeplitz(6, band_values_)
        for band_values_ in broadcast_band_values.reshape(-1, 4)
    ]
    expected_matrix = jsl.block_diag(*expected_blocks)

    actual_matrix = AbstractLinearOperator.as_matrix(op)
    assert_allclose(actual_matrix, expected_matrix, rtol=1e-7, atol=1e-7)

    actual_matrix = op.as_matrix()
    assert_allclose(actual_matrix, expected_matrix, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    'band_number, expected_fft_size',
    [(1, 2), (2, 4), (3, 8), (4, 8), (5, 16), (1023, 2048), (1024, 2048), (1025, 4096)],
)
def test_default_size(band_number: int, expected_fft_size: int):
    actual_fft_size = SymmetricBandToeplitzOperator._get_default_fft_size(band_number)
    assert actual_fft_size == expected_fft_size
