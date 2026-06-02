import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax import IndexOperator, SymmetricBandToeplitzOperator
from furax.mapmaking import GapFillingOperator
from furax.mapmaking._observation import HashedObservationMetadata
from furax.mapmaking.gap_filling import _folded_psd, _get_kernel, generate_noise_realization


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
    expected_kernel = np.array(expected_kernel)
    actual_kernel = _get_kernel(jnp.array(n_tt), fft_size)
    assert_allclose(actual_kernel, expected_kernel)


@pytest.mark.parametrize('n_tt, fft_size', [([1, 2], 1), ([1, 2, 3], 4)])
def test_get_kernel_fail_lagmax(n_tt: list[int], fft_size: int):
    # This test should fail because the maximum lag is too large for the required fft_size
    with pytest.raises(ValueError):
        _ = _get_kernel(jnp.array(n_tt), fft_size)


@pytest.mark.parametrize('shape', [(1,), (10,), (1, 100), (2, 10), (2, 100), (1, 2, 100)])
def test_generate_realization_shape(shape: tuple[int, ...]):
    key = jax.random.key(31415926539)
    structure = jax.ShapeDtypeStruct(shape, float)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), in_structure=structure)
    real = generate_noise_realization(key, cov, 1.0)
    assert real.shape == shape


def test_generate_realization_with_metadata():
    """generate_noise_realization must accept a populated HashedObservationMetadata.

    The metadata is a JAX-registered dataclass whose fields are array leaves and are
    folded into the random key (a traced operation). Passing it must not require the
    metadata to be hashable.
    """
    ndet, nsamp = 3, 100
    structure = jax.ShapeDtypeStruct((ndet, nsamp), float)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), in_structure=structure)
    metadata = HashedObservationMetadata(
        uid=jnp.uint32(1),
        telescope_uid=jnp.uint32(2),
        detector_uids=jnp.arange(ndet, dtype=jnp.uint32),
    )
    real = generate_noise_realization(jax.random.key(0), cov, 1.0, metadata=metadata)
    assert real.shape == (ndet, nsamp)


def test_generate_realization_metadata_under_scan():
    """generate_noise_realization must accept traced metadata inside a scan.

    Mirrors ``accumulate_rhs``: the per-observation ``HashedObservationMetadata`` is
    obtained by scanning over a batched metadata pytree, so its array leaves are JAX
    tracers inside the scan body. The metadata must therefore be handled as a traced
    pytree argument rather than a static (hashable) one.
    """
    ndet, nsamp = 3, 100
    structure = jax.ShapeDtypeStruct((ndet, nsamp), float)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), in_structure=structure)
    key = jax.random.key(0)
    nobs = 2
    metadata = HashedObservationMetadata(
        uid=jnp.arange(nobs, dtype=jnp.uint32),
        telescope_uid=jnp.zeros(nobs, dtype=jnp.uint32),
        detector_uids=jnp.broadcast_to(jnp.arange(ndet, dtype=jnp.uint32), (nobs, ndet)),
    )

    def step(carry, meta):
        return carry, generate_noise_realization(key, cov, 1.0, metadata=meta)

    _, reals = jax.lax.scan(step, None, metadata)
    assert reals.shape == (nobs, ndet, nsamp)


@pytest.fixture
def dummy_shape():
    return (2, 2, 100)


@pytest.fixture
def dummy_x(dummy_shape):
    key = jax.random.key(987654321)
    return jax.random.uniform(key, dummy_shape, dtype=float)


@pytest.fixture
def dummy_mask(dummy_shape):
    mask = jnp.ones(dummy_shape, dtype=bool)
    samples = dummy_shape[-1]
    gap_size = samples // 10
    left, right = (samples - gap_size) // 2, (samples + gap_size) // 2
    return mask.at[:, left:right].set(False)


@pytest.fixture
def dummy_mask_op(dummy_x, dummy_mask):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    indices = jnp.where(dummy_mask)
    mask_op = IndexOperator(indices, in_structure=structure)
    return mask_op


@pytest.fixture
def dummy_cov(dummy_x):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    return SymmetricBandToeplitzOperator(jnp.array([1.0]), in_structure=structure)


@pytest.fixture
def dummy_gap_filling_operator(dummy_shape, dummy_mask):
    x = jnp.ones(dummy_shape, dtype=float)
    structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    cov = SymmetricBandToeplitzOperator(jnp.array([1.0]), in_structure=structure)
    indices = jnp.where(dummy_mask)
    mask_op = IndexOperator(indices, in_structure=structure)
    return GapFillingOperator(cov, mask_op)


@pytest.mark.parametrize(
    'n_tt, fft_size', [([1], 1), ([1], 2), ([1], 4), ([1, 2], 4), ([3, 2, 1], 8)]
)
def test_get_psd_non_negative(n_tt, fft_size):
    psd = _folded_psd(n_tt, fft_size)
    assert np.all(psd >= 0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_valid_samples_and_no_nans(do_jit, dummy_x, dummy_gap_filling_operator):
    op = dummy_gap_filling_operator
    if do_jit:
        func = jax.jit(lambda k, x: op(k, x))
    else:
        func = op
    y = func(jax.random.key(1234), dummy_x)
    assert_array_equal(op.pack(dummy_x), op.pack(y))
    assert not np.any(np.isnan(y))
