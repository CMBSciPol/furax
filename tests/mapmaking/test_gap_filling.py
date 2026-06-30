import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax import MaskOperator, SymmetricBandToeplitzOperator
from furax.core import BlockDiagonalOperator
from furax.mapmaking import gap_fill
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
    return mask.at[..., left:right].set(False)


@pytest.fixture
def dummy_ninv(dummy_x):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    # Correlated inverse-noise (banded Toeplitz, diagonally dominant -> SPD).
    return SymmetricBandToeplitzOperator(jnp.array([1.0, -0.2, 0.05]), in_structure=structure)


@pytest.fixture
def dummy_mask_op(dummy_x, dummy_mask):
    structure = jax.ShapeDtypeStruct(dummy_x.shape, dummy_x.dtype)
    return MaskOperator.from_boolean_mask(dummy_mask, in_structure=structure)


@pytest.mark.parametrize(
    'n_tt, fft_size', [([1], 1), ([1], 2), ([1], 4), ([1, 2], 4), ([3, 2, 1], 8)]
)
def test_get_psd_non_negative(n_tt, fft_size):
    psd = _folded_psd(n_tt, fft_size)
    assert np.all(psd >= 0)


@pytest.mark.parametrize('do_jit', [False, True])
def test_valid_samples_and_no_nans(do_jit, dummy_x, dummy_ninv, dummy_mask_op):
    func = jax.jit(gap_fill) if do_jit else gap_fill
    y = func(jax.random.key(1234), dummy_x, dummy_ninv, dummy_mask_op)
    good = dummy_mask_op.to_boolean_mask()
    assert_array_equal(y[good], dummy_x[good])  # good samples untouched
    assert not jnp.any(jnp.isnan(y))


def test_filled_gaps_have_zero_weighted_residual(dummy_x, dummy_ninv, dummy_mask_op):
    """The defining constrained-realization condition: `(N⁻¹ (x_fill − ξ))` vanishes on the gaps.

    Verifying it directly proves the fill is the inverse-noise-only solve -- no covariance `N`.
    """
    rate = 1.0
    y = gap_fill(
        jax.random.key(7),
        dummy_x,
        dummy_ninv,
        dummy_mask_op,
        rate=rate,
        max_cg_steps=200,
        rtol=1e-12,
    )
    xi = generate_noise_realization(jax.random.key(7), dummy_ninv, rate, inverse=True)
    weighted_residual = dummy_ninv(y - xi)
    bad = ~dummy_mask_op.to_boolean_mask()
    assert jnp.any(bad)  # the test is vacuous without gaps
    assert_allclose(weighted_residual[bad], 0.0, atol=1e-6)


def test_gap_fill_pytree_tod():
    """gap_fill handles a pytree TOD with a multi-block inverse-noise operator.

    Each leaf is filled independently: good samples stay exact and the weighted residual
    `(N⁻¹ (x_fill − ξ))` vanishes on that leaf's gaps.
    """
    sa = jax.ShapeDtypeStruct((2, 100), float)
    sb = jax.ShapeDtypeStruct((3, 100), float)
    blocks = {
        'a': SymmetricBandToeplitzOperator(jnp.array([1.0, -0.2, 0.05]), in_structure=sa),
        'b': SymmetricBandToeplitzOperator(jnp.array([1.0, -0.1]), in_structure=sb),
    }
    ninv = BlockDiagonalOperator(blocks)
    x = {
        'a': jax.random.normal(jax.random.key(1), (2, 100)),
        'b': jax.random.normal(jax.random.key(2), (3, 100)),
    }
    good = {
        'a': jnp.ones((2, 100), bool).at[:, 45:55].set(False),
        'b': jnp.ones((3, 100), bool).at[:, 20:30].set(False),
    }
    mask = MaskOperator.from_boolean_mask(good, in_structure=ninv.in_structure)

    rate = 1.0
    y = gap_fill(jax.random.key(7), x, ninv, mask, rate=rate, max_cg_steps=200, rtol=1e-12)
    xi = generate_noise_realization(jax.random.key(7), ninv, rate, inverse=True)
    weighted_residual = ninv(jax.tree.map(jnp.subtract, y, xi))
    boolean = mask.to_boolean_mask()
    for leaf in ('a', 'b'):
        assert_array_equal(y[leaf][boolean[leaf]], x[leaf][boolean[leaf]])  # good untouched
        bad = ~boolean[leaf]
        assert jnp.any(bad)
        assert_allclose(weighted_residual[leaf][bad], 0.0, atol=1e-6)
