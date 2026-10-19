from math import prod

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose

from furax.mapmaking.templates import TemplateOperator
from furax.mapmaking.templates.bases import (
    Basis,
    DecimatedBasis,
    KroneckerBasis,
    OffsetBasis,
    SegmentedBasis,
    TensorBasis,
    WindowedBasis,
)
from furax.tree import as_structure


def _wrap(basis: Basis, n_dets: int) -> TemplateOperator:
    """Wrap a bare basis as a single-template TemplateOperator: amplitudes are {'t': ...},
    output is a bare (n_dets, samp) array."""
    return TemplateOperator({'t': basis}, n_dets)


# float64 everywhere: structural/adjoint residuals run ~1e-14, so one tight tolerance fits all.
TOL = 1e-12


def _adjoint_residual(basis, coeffs: Array, signal: Array) -> float:
    """|<expand(coeffs), signal> - <coeffs, project(signal)>|."""
    lhs = jnp.vdot(basis(coeffs), signal)
    rhs = jnp.vdot(coeffs, basis.T(signal))
    return float(jnp.abs(lhs - rhs))


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


class TestTensorBasis:
    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_create_structure(self, shape: tuple[int, ...]) -> None:
        n_points = 7
        values = jr.normal(jr.key(0), (*shape, n_points))
        basis = TensorBasis(values)
        assert basis.shape == shape
        assert basis.size == np.prod(shape)
        assert basis.n_points == n_points
        assert basis.in_structure == jax.ShapeDtypeStruct(shape, values.dtype)
        assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), values.dtype)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jr.normal(jr.key(5), (*shape, n_points))
        coeffs = jr.normal(jr.key(6), shape)
        signal = jr.normal(jr.key(7), (n_points,))
        basis = TensorBasis(values)
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    @pytest.mark.parametrize('shape', [(4,), (3, 5)])
    def test_transpose_as_matrix(self, shape: tuple[int, ...]) -> None:
        n_points = 5
        values = jr.normal(jr.key(8), (*shape, n_points))
        basis = TensorBasis(values)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=TOL)

    def test_out_structure_roundtrip(self) -> None:
        values = jr.normal(jr.key(9), (3, 5))
        basis = TensorBasis(values)
        coeffs = jr.normal(jr.key(10), (3,))
        assert as_structure(basis(coeffs)) == basis.out_structure


# ---------------------------------------------------------------------------
# KroneckerBasis
# ---------------------------------------------------------------------------


def _factors(shape: tuple[int, ...], n_points: int, seed: int) -> tuple[Float[Array, 'd n'], ...]:
    return tuple(jr.normal(jr.key(seed + i), (d, n_points)) for i, d in enumerate(shape))


class TestKroneckerBasis:
    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_create_structure(self, shape: tuple[int, ...]) -> None:
        n_points = 7
        factors = _factors(shape, n_points, 100)
        basis = KroneckerBasis(factors)
        assert basis.shape == shape
        assert basis.n_points == n_points
        assert basis.in_structure == jax.ShapeDtypeStruct(shape, factors[0].dtype)
        assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), factors[0].dtype)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_equivalent_to_dense_tensor_basis(self, shape: tuple[int, ...]) -> None:
        """KroneckerBasis == TensorBasis whose values are the outer product of factors."""
        n_points = 6
        factors = _factors(shape, n_points, 200)
        kron = KroneckerBasis(factors)
        dense = TensorBasis(_dense_from_factors(factors))

        coeffs = jr.normal(jr.key(210), shape)
        signal = jr.normal(jr.key(211), (n_points,))
        assert_allclose(kron(coeffs), dense(coeffs), rtol=TOL)
        assert_allclose(kron.T(signal), dense.T(signal), rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        factors = _factors(shape, n_points, 300)
        basis = KroneckerBasis(factors)
        coeffs = jr.normal(jr.key(310), shape)
        signal = jr.normal(jr.key(311), (n_points,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL


# ---------------------------------------------------------------------------
# DecimatedBasis
# ---------------------------------------------------------------------------


def _decimated(shape: tuple[int, ...], n_dec: int, q: int, n_samples: int, seed: int):
    values = jr.normal(jr.key(seed), (*shape, n_dec))
    return DecimatedBasis(TensorBasis(values), q, n_samples)


class TestDecimatedBasis:
    @pytest.mark.parametrize('shape', [(4,), (2, 3)])
    def test_structure(self, shape: tuple[int, ...]) -> None:
        n_dec, q, n_samples = 5, 4, 18
        basis = _decimated(shape, n_dec, q, n_samples, 500)
        assert basis.shape == shape
        assert basis.n_points == n_samples
        assert basis.out_structure == jax.ShapeDtypeStruct((n_samples,), basis.dtype)

    def test_holds_each_coarse_value_over_its_block(self) -> None:
        # one basis function [1, 2, 3] on a q=2 grid covering 5 samples: each coarse value is
        # held over its 2 samples, and the last block is trimmed to fit.
        basis = DecimatedBasis(TensorBasis(jnp.array([[1.0, 2.0, 3.0]])), 2, 5)
        assert_allclose(basis(jnp.array([1.0])), [1.0, 1.0, 2.0, 2.0, 3.0], rtol=TOL)
        # project is its transpose: [0,1,2,3,4] block-sums to [1, 5, 4], then weights by the
        # basis function -> 1*1 + 2*5 + 3*4.
        assert_allclose(basis.T(jnp.arange(5.0)), [23.0], rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (2, 3)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_dec, q, n_samples = 5, 4, 18
        basis = _decimated(shape, n_dec, q, n_samples, 505)
        coeffs = jr.normal(jr.key(506), shape)
        signal = jr.normal(jr.key(507), (n_samples,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_q_one_matches_the_inner_basis(self) -> None:
        values = jr.normal(jr.key(509), (4, 18))
        plain = TensorBasis(values)
        decimated = DecimatedBasis(plain, 1, 18)
        coeffs = jr.normal(jr.key(510), (4,))
        signal = jr.normal(jr.key(511), (18,))
        assert_allclose(decimated(coeffs), plain(coeffs), rtol=TOL)
        assert_allclose(decimated.T(signal), plain.T(signal), rtol=TOL)

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


class TestSegmentedBasis:
    def test_structure(self) -> None:
        basis, _, _ = _segmented(3, 4, 50, 600)
        assert basis.shape == (3, 4)
        assert basis.n_points == 50
        assert basis.out_structure == jax.ShapeDtypeStruct((50,), basis.dtype)

    def test_equivalent_to_dense_tensor_basis(self) -> None:
        # dense[j, k, s] = (segment[s] == j) * values[k, s]
        n_seg, k, n_points = 3, 4, 50
        basis, segment, values = _segmented(n_seg, k, n_points, 610)
        onehot = (segment[None, :] == jnp.arange(n_seg)[:, None]).astype(values.dtype)
        dense = TensorBasis(onehot[:, None, :] * values[None, :, :])
        coeffs = jr.normal(jr.key(611), (n_seg, k))
        signal = jr.normal(jr.key(612), (n_points,))
        assert_allclose(basis(coeffs), dense(coeffs), rtol=TOL)
        assert_allclose(basis.T(signal), dense.T(signal), rtol=TOL)

    def test_expand_project_adjoint(self) -> None:
        basis, _, _ = _segmented(3, 4, 50, 620)
        coeffs = jr.normal(jr.key(621), basis.shape)
        signal = jr.normal(jr.key(622), (50,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_each_sample_only_sees_its_own_segment(self) -> None:
        # perturbing one segment's amplitudes only changes samples in that segment.
        n_seg, k, n_points = 3, 4, 50
        basis, segment, _ = _segmented(n_seg, k, n_points, 630)
        coeffs = jr.normal(jr.key(631), (n_seg, k))
        bumped = coeffs.at[0].add(1.0)
        diff = basis(bumped) - basis(coeffs)
        assert_allclose(diff[segment != 0], 0.0, atol=TOL)
        assert jnp.max(jnp.abs(diff[segment == 0])) > 0.0

    def test_projects_every_detector_as_it_projects_each(self) -> None:
        # the template operator projects all detectors in one pass, which must agree with
        # projecting each stream on its own
        n_dets, n_points = 5, 50
        basis, _, _ = _segmented(3, 4, n_points, 640)
        signals = jr.normal(jr.key(642), (n_dets, n_points))
        expected = jnp.stack([basis.T(signal) for signal in signals])
        assert_allclose(_wrap(basis, n_dets).T(signals)['t'], expected, rtol=TOL)

    def test_expands_every_detector_as_it_expands_each(self) -> None:
        n_dets, n_points = 5, 50
        basis, _, _ = _segmented(3, 4, n_points, 650)
        coeffs = jr.normal(jr.key(651), (n_dets, *basis.shape))
        expected = jnp.stack([basis(c) for c in coeffs])
        assert_allclose(_wrap(basis, n_dets)({'t': coeffs}), expected, rtol=TOL)


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


class TestOffsetBasis:
    @pytest.mark.parametrize('n_points', [12, 14], ids=['no_tail', 'tail'])
    def test_equivalent_to_segmented_basis(self, n_points: int) -> None:
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

    def test_expand_project_adjoint(self) -> None:
        basis = OffsetBasis(4, 14, jnp.float64)
        coeffs = jr.normal(jr.key(680), basis.shape)
        signal = jr.normal(jr.key(681), (14,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_expands_and_projects_every_detector_as_each(self) -> None:
        n_dets, n_points = 5, 14
        basis = OffsetBasis(4, n_points, jnp.float64)
        coeffs = jr.normal(jr.key(690), (n_dets, *basis.shape))
        signals = jr.normal(jr.key(691), (n_dets, n_points))
        op = _wrap(basis, n_dets)
        assert_allclose(op({'t': coeffs}), jnp.stack([basis(c) for c in coeffs]), rtol=TOL)
        expected = jnp.stack([basis.T(signal) for signal in signals])
        assert_allclose(op.T(signals)['t'], expected, rtol=TOL)


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


class TestWindowedBasis:
    def test_structure(self) -> None:
        basis, *_ = _windowed(n_blocks=6, n_window=4, k=3, n_points=50, seed=800)
        assert basis.shape == (6, 3)
        assert basis.n_points == 50
        assert basis.out_structure == jax.ShapeDtypeStruct((50,), basis.dtype)

    @pytest.mark.parametrize(('n_window', 'k'), [(4, 1), (4, 3), (1, 5), (3, 2)])
    def test_equivalent_to_dense_tensor_basis(self, n_window: int, k: int) -> None:
        n_blocks, n_points = 6, 50
        basis, offset, bw, sv = _windowed(n_blocks, n_window, k, n_points, 810)
        dense = TensorBasis(_windowed_dense_values(offset, bw, sv, n_blocks))
        coeffs = jr.normal(jr.key(811), (n_blocks, k))
        signal = jr.normal(jr.key(812), (n_points,))
        assert_allclose(basis(coeffs), dense(coeffs), rtol=TOL)
        assert_allclose(basis.T(signal), dense.T(signal), rtol=TOL)

    def test_reduces_to_segmented_when_window_is_one(self) -> None:
        # O=1 with unit weights is exactly SegmentedBasis (offset == segment id).
        n_seg, k, n_points = 4, 3, 50
        segment = jnp.sort(jr.randint(jr.key(820), (n_points,), 0, n_seg))
        values = jr.normal(jr.key(821), (k, n_points))
        segmented = SegmentedBasis(segment.astype(jnp.int32), values, n_seg)
        windowed = WindowedBasis(
            segment.astype(jnp.int32),
            jnp.ones((n_points, 1), values.dtype),
            values,
            n_seg,
        )
        coeffs = jr.normal(jr.key(822), (n_seg, k))
        signal = jr.normal(jr.key(823), (n_points,))
        assert_allclose(windowed(coeffs), segmented(coeffs), rtol=TOL)
        assert_allclose(windowed.T(signal), segmented.T(signal), rtol=TOL)

    @pytest.mark.parametrize(('n_window', 'k'), [(4, 1), (4, 3), (1, 5)])
    def test_expand_project_adjoint(self, n_window: int, k: int) -> None:
        basis, *_ = _windowed(6, n_window, k, 50, 830)
        coeffs = jr.normal(jr.key(831), basis.shape)
        signal = jr.normal(jr.key(832), (50,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_each_sample_only_sees_its_window(self) -> None:
        # perturbing one block's amplitudes only changes samples whose window covers it.
        n_blocks, n_window, k, n_points = 6, 4, 2, 50
        basis, offset, *_ = _windowed(n_blocks, n_window, k, n_points, 850)
        coeffs = jr.normal(jr.key(851), (n_blocks, k))
        target = 2
        bumped = coeffs.at[target].add(1.0)
        diff = basis(bumped) - basis(coeffs)
        covers = (offset <= target) & (target < offset + n_window)
        assert_allclose(diff[~covers], 0.0, atol=TOL)

    def test_projects_every_detector_as_it_projects_each(self) -> None:
        # as for SegmentedBasis: one pass over all detectors agrees with one per detector
        n_dets, n_points = 5, 50
        basis, *_ = _windowed(6, 4, 3, n_points, 860)
        signals = jr.normal(jr.key(861), (n_dets, n_points))
        expected = jnp.stack([basis.T(signal) for signal in signals])
        assert_allclose(_wrap(basis, n_dets).T(signals)['t'], expected, rtol=TOL)


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


FLAVOURS = ['tensor', 'kronecker', 'segmented', 'windowed', 'decimated']
BATCH_SHAPES = [(3,), (2, 3)]


class TestBatchedBasis:
    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_structures_carry_the_batch_axes(self, flavour: str) -> None:
        # `shape` stays one basis's amplitude index; the operator structures gain the batch axes
        basis, singles = _batched_and_singles(flavour, (3,), 900)
        assert basis.shape == singles[0].shape
        assert basis.n_points == singles[0].n_points
        assert basis.in_structure.shape == (3, *singles[0].shape)
        assert basis.out_structure.shape == (3, singles[0].n_points)

    @pytest.mark.parametrize('batch_shape', BATCH_SHAPES)
    @pytest.mark.parametrize('flavour', FLAVOURS)
    def test_expand_and_project_apply_each_basis_to_its_own_batch_element(
        self, flavour: str, batch_shape: tuple[int, ...]
    ) -> None:
        basis, singles = _batched_and_singles(flavour, batch_shape, 910)
        amps = jr.normal(jr.key(911), basis.in_structure.shape)
        signal = jr.normal(jr.key(912), basis.out_structure.shape)
        flat_amps = amps.reshape(-1, *basis.shape)
        flat_signal = signal.reshape(-1, basis.n_points)
        expanded = [b(a) for b, a in zip(singles, flat_amps, strict=True)]
        projected = [b.T(s) for b, s in zip(singles, flat_signal, strict=True)]
        assert_allclose(basis(amps), jnp.stack(expanded).reshape(signal.shape), rtol=TOL)
        assert_allclose(basis.T(signal), jnp.stack(projected).reshape(amps.shape), rtol=TOL)
        assert _adjoint_residual(basis, amps, signal) < TOL

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
