import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax._base.indices import IndexOperator
from furax.detectors import FakeDetectorArray
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
    n_tt = jnp.array(n_tt)
    expected_kernel = np.array(expected_kernel)
    actual_kernel = GapFillingOperator.get_kernel(n_tt, fft_size)
    assert_allclose(actual_kernel, expected_kernel)


@pytest.mark.parametrize('n_tt, fft_size', [([1, 2], 1), ([1, 2, 3], 4)])
def test_get_kernel_fail_lagmax(n_tt: list[int], fft_size: int):
    # This test should fail because the maximum lag is too large for the required fft_size
    n_tt = jnp.array(n_tt)
    with pytest.raises(ValueError):
        _ = GapFillingOperator.get_kernel(n_tt, fft_size)


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize('x_shape', [(1,), (10,), (1, 100), (2, 10), (2, 100), (1, 2, 100)])
def test_generate_realization_shape(x_shape: tuple[int, ...], do_jit: bool):
    x = jnp.zeros(x_shape, dtype=float)
    key = jax.random.key(31415926539)
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), structure)
    indices = jnp.where(jnp.ones_like(x, dtype=bool))[0]
    pack = IndexOperator(indices, in_structure=structure)
    dets = FakeDetectorArray(x_shape[:-1])
    op = GapFillingOperator(cov, pack, dets)
    if do_jit:
        # avoid error: TypeError: unhashable type: 'jaxlib.xla_extension.ArrayImpl'
        func = jax.jit(lambda x, k: op._generate_realization_for(x, k))
    else:
        func = op._generate_realization_for
    real = func(x, key)
    assert real.shape == x_shape


@pytest.fixture
def dummy_shape():
    shape = (2, 100)
    return shape


@pytest.fixture
def dummy_x(dummy_shape):
    key = jax.random.key(987654321)
    x = jax.random.uniform(key, dummy_shape, dtype=float)
    return x


@pytest.fixture
def dummy_detectors(dummy_shape):
    return FakeDetectorArray(dummy_shape[0])


@pytest.fixture
def dummy_mask(dummy_shape):
    mask = jnp.ones(dummy_shape, dtype=bool)
    samples = dummy_shape[-1]
    gap_size = samples // 10
    left, right = (samples - gap_size) // 2, (samples + gap_size) // 2
    mask = mask.at[:, left:right].set(False)
    return mask


@pytest.fixture
def dummy_pack(dummy_x, dummy_mask):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    indices = jnp.where(dummy_mask)[0]
    pack = IndexOperator(indices, in_structure=structure)
    return pack


@pytest.fixture
def dummy_cov(dummy_x):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), structure)
    return cov


@pytest.fixture
def dummy_gap_filling_operator(dummy_shape, dummy_mask, dummy_detectors):
    x = jnp.ones(dummy_shape, dtype=float)
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), structure)
    indices = jnp.where(dummy_mask)[0]
    pack = IndexOperator(indices, in_structure=structure)
    return GapFillingOperator(cov, pack, dummy_detectors)


@pytest.mark.parametrize(
    'n_tt, fft_size', [([1], 1), ([1], 2), ([1], 4), ([1, 2], 4), ([3, 2, 1], 8)]
)
def test_get_psd_non_negative(n_tt, fft_size):
    n_tt = np.array(n_tt)
    psd = GapFillingOperator.folded_psd(n_tt, fft_size)
    assert np.all(psd >= 0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_valid_samples_and_no_nans(do_jit, dummy_shape, dummy_x, dummy_gap_filling_operator):
    op = dummy_gap_filling_operator
    if do_jit:
        func = jax.jit(lambda k, x: op(k, x))
    else:
        func = op
    y = func(jax.random.key(1234), dummy_x)
    assert_allclose(op.pack(dummy_x), op.pack(y))
    assert not np.any(np.isnan(y))
