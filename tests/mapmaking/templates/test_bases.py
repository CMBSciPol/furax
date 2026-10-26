from math import prod

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose

from furax.mapmaking.templates.bases import (
    Basis,
    DecimatedBasis,
    KroneckerBasis,
    OffsetBasis,
    SegmentedBasis,
    SplineBasis,
    TensorBasis,
    WindowedBasis,
)
from tests.mapmaking.templates.helpers import TOL, adjoint_residual, wrap


def _dense_from_factors(factors: tuple[Array, ...]) -> Array:
    """values[k_0,...,k_{N-1}, s] = prod_i factors[i][k_i, s], built by broadcasting."""
    n = len(factors)
    n_points = factors[0].shape[-1]
    shape = tuple(f.shape[0] for f in factors)
    dense = jnp.ones((*shape, n_points), dtype=factors[0].dtype)
    for i, f in enumerate(factors):
        bshape = (1,) * i + (shape[i],) + (1,) * (n - 1 - i) + (n_points,)
        dense = dense * f.reshape(bshape)
    return dense


# ---------------------------------------------------------------------------
# TensorBasis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
def test_tensor_basis_columns_are_its_values(shape: tuple[int, ...]) -> None:
    n_points = 7
    values = jr.normal(jr.key(0), (*shape, n_points))
    basis = TensorBasis(values)
    assert basis.shape == shape
    assert basis.size == np.prod(shape)
    assert basis.in_structure == jax.ShapeDtypeStruct(shape, values.dtype)
    assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), values.dtype)
    # columns in the row-major order of the amplitude index
    assert_allclose(basis.as_matrix(), values.reshape(-1, n_points).T, rtol=TOL)


# ---------------------------------------------------------------------------
# KroneckerBasis
# ---------------------------------------------------------------------------


def _factors(shape: tuple[int, ...], n_points: int, seed: int) -> tuple[Float[Array, 'd n'], ...]:
    return tuple(jr.normal(jr.key(seed + i), (d, n_points)) for i, d in enumerate(shape))


@pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
def test_kronecker_basis_is_the_tensor_basis_of_the_outer_product(shape: tuple[int, ...]) -> None:
    n_points = 6
    factors = _factors(shape, n_points, 200)
    kron = KroneckerBasis(factors)
    assert kron.shape == shape
    assert kron.n_points == n_points
    dense = TensorBasis(_dense_from_factors(factors))
    coeffs = jr.normal(jr.key(210), shape)
    signal = jr.normal(jr.key(211), (n_points,))
    assert_allclose(kron(coeffs), dense(coeffs), rtol=TOL)
    assert_allclose(kron.T(signal), dense.T(signal), rtol=TOL)


# ---------------------------------------------------------------------------
# DecimatedBasis
# ---------------------------------------------------------------------------


def _decimated(shape: tuple[int, ...], n_dec: int, q: int, n_samples: int, seed: int):
    values = jr.normal(jr.key(seed), (*shape, n_dec))
    return DecimatedBasis(TensorBasis(values), q, n_samples)


class TestDecimatedBasis:
    def test_holds_each_coarse_value_over_its_block(self) -> None:
        # one basis function [1, 2, 3] on a q=2 grid covering 5 samples: each coarse value is
        # held over its 2 samples, and the last block is trimmed to fit.
        basis = DecimatedBasis(TensorBasis(jnp.array([[1.0, 2.0, 3.0]])), 2, 5)
        assert (basis.shape, basis.n_points) == ((1,), 5)
        assert_allclose(basis(jnp.array([1.0])), [1.0, 1.0, 2.0, 2.0, 3.0], rtol=TOL)
        # project is its transpose: [0,1,2,3,4] block-sums to [1, 5, 4], then weights by the
        # basis function -> 1*1 + 2*5 + 3*4.
        assert_allclose(basis.T(jnp.arange(5.0)), [23.0], rtol=TOL)

    def test_gram_matches_the_dense_gram(self) -> None:
        # taken on the coarse grid, from block-summed weights, the Gram is still exact
        basis = _decimated((3,), 5, 4, 18, 512)
        weights = jr.uniform(jr.key(513), (18,), minval=0.5, maxval=2.0)
        matrix = basis.as_matrix()
        expected = matrix.T @ (weights[:, None] * matrix)
        assert_allclose(basis.gram(weights)[0, 0], expected, rtol=TOL, atol=TOL)

    def test_create_rejects_bad_q(self) -> None:
        with pytest.raises(ValueError, match='decimation_factor must be >= 1'):
            DecimatedBasis(TensorBasis(jr.normal(jr.key(514), (4, 5))), 0, 5)

    @pytest.mark.parametrize('n_samples', [16, 21])  # 5*4=20: n_samples must be in (16, 20]
    def test_create_rejects_an_inner_basis_not_covering_the_samples(self, n_samples: int) -> None:
        with pytest.raises(ValueError, match='does not cover'):
            DecimatedBasis(TensorBasis(jr.normal(jr.key(515), (4, 5))), 4, n_samples)


# ---------------------------------------------------------------------------
# SegmentedBasis
# ---------------------------------------------------------------------------


def _segmented(n_segments: int, k: int, n_points: int, seed: int):
    segment = jnp.sort(jr.randint(jr.key(seed), (n_points,), 0, n_segments))
    values = jr.normal(jr.key(seed + 1), (k, n_points))
    return SegmentedBasis(segment.astype(jnp.int32), values, n_segments), segment, values


def test_segmented_basis_is_its_dense_tensor_basis() -> None:
    # dense[j, k, s] = (segment[s] == j) * values[k, s]
    n_seg, k, n_points = 3, 4, 50
    basis, segment, values = _segmented(n_seg, k, n_points, 610)
    assert basis.shape == (n_seg, k)
    assert basis.n_points == n_points
    onehot = (segment[None, :] == jnp.arange(n_seg)[:, None]).astype(values.dtype)
    dense = TensorBasis(onehot[:, None, :] * values[None, :, :])
    coeffs = jr.normal(jr.key(611), (n_seg, k))
    signal = jr.normal(jr.key(612), (n_points,))
    assert_allclose(basis(coeffs), dense(coeffs), rtol=TOL)
    assert_allclose(basis.T(signal), dense.T(signal), rtol=TOL)


# ---------------------------------------------------------------------------
# OffsetBasis
# ---------------------------------------------------------------------------


def _offset_as_segmented(tau: int, n_points: int) -> SegmentedBasis:
    """The `SegmentedBasis` an `OffsetBasis` stands for: unit values, the tail in no segment."""
    n_int = n_points // tau
    samples = jnp.arange(n_points)
    segment = jnp.minimum(samples // tau, n_int - 1).astype(jnp.int32)
    values = (samples < n_int * tau)[None].astype(jnp.float64)
    return SegmentedBasis(segment, values, n_int)


@pytest.mark.parametrize('n_points', [12, 14], ids=['no_tail', 'tail'])
def test_offset_basis_is_its_segmented_basis(n_points: int) -> None:
    # one constant per interval of tau samples, the tail in none: the regular SegmentedBasis
    tau = 4
    basis = OffsetBasis(tau, n_points, jnp.float64)
    segmented = _offset_as_segmented(tau, n_points)
    coeffs = jr.normal(jr.key(670), (n_points // tau,))
    signal = jr.normal(jr.key(671), (n_points,))
    w = jr.uniform(jr.key(672), (n_points,), minval=0.5, maxval=2.0)
    assert basis.shape == (n_points // tau,)
    assert_allclose(basis(coeffs), segmented(coeffs[:, None]), rtol=TOL)
    assert_allclose(basis.T(signal), segmented.T(signal).ravel(), rtol=TOL)
    assert_allclose(basis.gram(w), segmented.gram(w), rtol=TOL)
    assert_allclose(basis.support().as_matrix(), basis.as_matrix(), rtol=TOL, atol=TOL)
    assert_allclose(basis.as_matrix(), segmented.as_matrix(), rtol=TOL, atol=TOL)


# ---------------------------------------------------------------------------
# WindowedBasis
# ---------------------------------------------------------------------------


def _windowed(n_blocks: int, n_window: int, k: int, n_points: int, seed: int):
    # offset kept in [0, n_blocks - n_window] so every window stays in range.
    offset = jnp.sort(jr.randint(jr.key(seed), (n_points,), 0, n_blocks - n_window + 1))
    block_weights = jr.normal(jr.key(seed + 1), (n_window, n_points))
    sub_values = jr.normal(jr.key(seed + 2), (k, n_points))
    basis = WindowedBasis(offset.astype(jnp.int32), block_weights.T, sub_values, n_blocks)
    return basis, offset, block_weights, sub_values


def _windowed_dense_values(offset, block_weights, sub_values, n_blocks):
    # dense[b, j, s] = block_weights[b - offset[s], s] * sub_values[j, s] inside the window.
    n_window, n_points = block_weights.shape
    rel = jnp.arange(n_blocks)[:, None] - offset[None, :]  # (n_blocks, n_points)
    in_window = (rel >= 0) & (rel < n_window)
    taper = jnp.where(
        in_window, block_weights[jnp.clip(rel, 0, n_window - 1), jnp.arange(n_points)], 0.0
    )
    return taper[:, None, :] * sub_values[None, :, :]  # (n_blocks, k, n_points)


@pytest.mark.parametrize(('n_window', 'k'), [(4, 1), (4, 3), (1, 5), (3, 2)])
def test_windowed_basis_is_its_dense_tensor_basis(n_window: int, k: int) -> None:
    n_blocks, n_points = 6, 50
    basis, offset, bw, sv = _windowed(n_blocks, n_window, k, n_points, 810)
    assert basis.shape == (n_blocks, k)
    assert basis.n_points == n_points
    dense = TensorBasis(_windowed_dense_values(offset, bw, sv, n_blocks))
    coeffs = jr.normal(jr.key(811), (n_blocks, k))
    signal = jr.normal(jr.key(812), (n_points,))
    assert_allclose(basis(coeffs), dense(coeffs), rtol=TOL)
    assert_allclose(basis.T(signal), dense.T(signal), rtol=TOL)


# ---------------------------------------------------------------------------
# Batches of bases (batch_shape)
# ---------------------------------------------------------------------------


def _batched_and_singles(
    flavour: str, batch_shape: tuple[int, ...], seed: int
) -> tuple[Basis, list[Basis]]:
    """A batch of bases and the equivalent flat list of single bases, on the same arrays."""
    n_points = 30
    n = prod(batch_shape)
    k = jr.split(jr.key(seed), 4)

    def split(*arrays: Array) -> list[tuple[Array, ...]]:
        """Each single basis's arrays, batch axes flattened."""
        flat = [a.reshape(n, *a.shape[len(batch_shape) :]) for a in arrays]
        return list(zip(*flat, strict=True))

    if flavour == 'tensor':
        values = jr.normal(k[0], (*batch_shape, 2, n_points))
        singles = [TensorBasis(v) for (v,) in split(values)]
        return TensorBasis(values, batch_shape=batch_shape), singles
    if flavour == 'kronecker':
        f0 = jr.normal(k[0], (*batch_shape, 2, n_points))
        f1 = jr.normal(k[1], (*batch_shape, 3, n_points))
        singles = [KroneckerBasis((a, b)) for a, b in split(f0, f1)]
        return KroneckerBasis((f0, f1), batch_shape=batch_shape), singles
    if flavour == 'segmented':
        n_seg = 3
        segment = jnp.sort(jr.randint(k[0], (*batch_shape, n_points), 0, n_seg), axis=-1)
        segment = segment.astype(jnp.int32)
        values = jr.normal(k[1], (*batch_shape, 2, n_points))
        singles = [SegmentedBasis(g, v, n_seg) for g, v in split(segment, values)]
        return SegmentedBasis(segment, values, n_seg, batch_shape=batch_shape), singles
    if flavour == 'windowed':
        n_blocks, n_window = 6, 4
        offset = jr.randint(k[0], (*batch_shape, n_points), 0, n_blocks - n_window + 1)
        offset = jnp.sort(offset, axis=-1).astype(jnp.int32)
        weights = jr.normal(k[1], (*batch_shape, n_points, n_window))
        sub = jr.normal(k[2], (*batch_shape, 2, n_points))
        singles = [WindowedBasis(o, w, v, n_blocks) for o, w, v in split(offset, weights, sub)]
        return WindowedBasis(offset, weights, sub, n_blocks, batch_shape=batch_shape), singles
    if flavour == 'spline':
        n_knots, n_window = 6, 4
        offset = jr.randint(k[0], (*batch_shape, n_points), 0, n_knots - n_window + 1)
        offset = jnp.sort(offset, axis=-1).astype(jnp.int32)
        taps = jr.normal(k[1], (*batch_shape, n_points, n_window))
        singles = [SplineBasis(o, t, n_knots) for o, t in split(offset, taps)]
        return SplineBasis(offset, taps, n_knots, batch_shape=batch_shape), singles
    if flavour == 'decimated':  # a segmented basis on a 4-times coarser grid
        n_seg, q = 3, 4
        n_dec = -(-n_points // q)
        segment = jnp.sort(jr.randint(k[0], (*batch_shape, n_dec), 0, n_seg), axis=-1)
        segment = segment.astype(jnp.int32)
        values = jr.normal(k[1], (*batch_shape, 2, n_dec))
        singles = [
            DecimatedBasis(SegmentedBasis(g, v, n_seg), q, n_points)
            for g, v in split(segment, values)
        ]
        inner = SegmentedBasis(segment, values, n_seg, batch_shape=batch_shape)
        return DecimatedBasis(inner, q, n_points), singles
    raise AssertionError(flavour)


FLAVOURS = ['tensor', 'kronecker', 'segmented', 'windowed', 'spline', 'decimated']
BATCH_SHAPES = [(3,), (2, 3)]


class TestBatchedBasis:
    @pytest.mark.parametrize('batch_shape', BATCH_SHAPES)
    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_expand_and_project_apply_each_basis_to_its_own_batch_element(
        self, flavour: str, batch_shape: tuple[int, ...]
    ) -> None:
        basis, singles = _batched_and_singles(flavour, batch_shape, 910)
        # `shape` stays one basis's amplitude index; the operator structures gain the batch axes
        assert basis.shape == singles[0].shape
        assert basis.n_points == singles[0].n_points
        assert basis.in_structure.shape == (*batch_shape, *singles[0].shape)
        assert basis.out_structure.shape == (*batch_shape, singles[0].n_points)
        amps = jr.normal(jr.key(911), basis.in_structure.shape)
        signal = jr.normal(jr.key(912), basis.out_structure.shape)
        flat_amps = amps.reshape(-1, *basis.shape)
        flat_signal = signal.reshape(-1, basis.n_points)
        expanded = [b(a) for b, a in zip(singles, flat_amps, strict=True)]
        projected = [b.T(s) for b, s in zip(singles, flat_signal, strict=True)]
        assert_allclose(basis(amps), jnp.stack(expanded).reshape(signal.shape), rtol=TOL)
        assert_allclose(basis.T(signal), jnp.stack(projected).reshape(amps.shape), rtol=TOL)
        assert adjoint_residual(basis, amps, signal) < TOL

    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_batch_axes_broadcast(self, flavour: str) -> None:
        # shared amplitudes broadcast over the bases; extra leading axes over everything
        basis, singles = _batched_and_singles(flavour, (3,), 920)
        amps = jr.normal(jr.key(921), (2, 1, *basis.shape))
        expected = jnp.stack([jnp.stack([b(a[0]) for b in singles]) for a in amps])
        assert_allclose(basis(amps), expected, rtol=TOL)

    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_grams_are_each_basis_own(self, flavour: str) -> None:
        basis, singles = _batched_and_singles(flavour, (3,), 930)
        weights = jr.uniform(jr.key(931), (3, basis.n_points), minval=0.5, maxval=2.0)
        expected = jnp.stack([b.gram(w) for b, w in zip(singles, weights, strict=True)])
        assert_allclose(basis.gram(weights), expected, rtol=TOL)

    def test_rejects_arrays_without_the_batch_axes(self) -> None:
        with pytest.raises(ValueError, match=r'batch_shape=\(2,\) must lead every array'):
            SegmentedBasis(jnp.zeros((3, 30), jnp.int32), jnp.ones((2, 2, 30)), 3, batch_shape=(2,))

    def test_rejects_a_batch_of_bases_with_no_array(self) -> None:
        with pytest.raises(ValueError, match='OffsetBasis stores no array to batch'):
            OffsetBasis(2, 5, jnp.float32, batch_shape=(3,))


@pytest.mark.parametrize('flavour', [*FLAVOURS, 'offset'])
def test_template_operator_applies_the_basis_to_every_detector(flavour: str) -> None:
    # the template operator expands and projects all detectors in one pass, which must agree with
    # one basis application per detector
    if flavour == 'offset':
        basis = OffsetBasis(4, 30, jnp.float64)
    else:
        basis = _batched_and_singles(flavour, (1,), 1000)[1][0]
    n_dets = 5
    coeffs = jr.normal(jr.key(1001), (n_dets, *basis.shape))
    signals = jr.normal(jr.key(1002), (n_dets, basis.n_points))
    op = wrap(basis, n_dets)
    assert_allclose(op({'t': coeffs}), jnp.stack([basis(c) for c in coeffs]), rtol=TOL)
    expected = jnp.stack([basis.T(signal) for signal in signals])
    assert_allclose(op.T(signals)['t'], expected, rtol=TOL)


# ---------------------------------------------------------------------------
# Structured Gram views (Basis.gram, Basis.support)
# ---------------------------------------------------------------------------


class TestStructuredViews:
    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_support_encodes_the_same_matrix_as_the_basis(self, flavour: str) -> None:
        # the column-support view and expand/project are two encodings of one basis: they must
        # agree, or every Gram built from the view is silently wrong.
        _, singles = _batched_and_singles(flavour, (1,), 1100)
        basis = singles[0]
        assert_allclose(basis.support().as_matrix(), basis.as_matrix(), rtol=TOL, atol=TOL)

    def test_tensor_views_flatten_a_multidimensional_index(self) -> None:
        basis = TensorBasis(jr.normal(jr.key(1105), (2, 3, 40)))
        weights = jr.uniform(jr.key(1106), (40,), minval=0.5, maxval=2.0)
        matrix = basis.as_matrix()  # (samp, 6), columns in the row-major order of (2, 3)
        assert_allclose(basis.support().as_matrix(), matrix, rtol=TOL, atol=TOL)
        expected = matrix.T @ (weights[:, None] * matrix)
        assert_allclose(basis.gram(weights)[0, 0], expected, rtol=TOL, atol=TOL)

    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_gram_bands_match_the_dense_gram(self, flavour: str) -> None:
        # the banded build must match the dense Bᵀ diag(w) B block for block within the band,
        # whether a flavour takes the default from its support view or overrides it
        _, singles = _batched_and_singles(flavour, (1,), 1110)
        basis = singles[0]
        weights = jr.uniform(jr.key(1111), (basis.n_points,), minval=0.5, maxval=2.0)

        bands = basis.gram(weights)  # (n_blocks, w1, k, k), upper band
        n_blocks, w1, k, _ = bands.shape
        matrix = basis.as_matrix()  # (samp, n_blocks * k), columns ordered (block, sub)
        dense = (matrix.T @ (weights[:, None] * matrix)).reshape(n_blocks, k, n_blocks, k)
        for j in range(n_blocks):
            for d in range(w1):
                if j + d < n_blocks:
                    assert_allclose(bands[j, d], dense[j, :, j + d, :], rtol=TOL, atol=TOL)


# segments of 4, 2 and 4 samples: an interior block of 2, the longest of 4
_SEGMENTS = jnp.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2], jnp.int32)


@pytest.mark.parametrize(
    ('block_lengths', 'segment'),
    [
        ((3, 4), _SEGMENTS),  # an interior block shorter than stated
        ((2, 3), _SEGMENTS),  # a block longer than stated
        ((2, 4), jnp.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], jnp.int32)),  # block 1 skipped
    ],
    ids=['shorter', 'longer', 'skipped'],
)
@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
def test_blocks_breaking_the_stated_lengths_raise(block_lengths, segment, jit):
    def build(segment):
        basis = SegmentedBasis(segment, jnp.ones((1, 10)), 3, block_lengths=block_lengths)
        return basis(jnp.ones((3, 1)))

    with pytest.raises(RuntimeError, match='breaking its block_lengths'):
        jax.block_until_ready((jax.jit(build) if jit else build)(segment))


def test_blocks_within_the_stated_lengths_pass():
    basis = SegmentedBasis(_SEGMENTS, jnp.ones((1, 10)), 3, block_lengths=(2, 4))
    assert_allclose(basis(jnp.ones((3, 1))), jnp.ones(10))


def test_offsets_and_their_decimation_state_their_block_lengths():
    # intervals of 3 samples, the last holding the 2 samples of the tail
    offsets = OffsetBasis(3, 11, jnp.float64)
    assert offsets.block_lengths == (3, 5)
    decimated = DecimatedBasis(offsets, decimation_factor=2, n_samples=22)
    assert decimated.block_lengths == (6, 10)
