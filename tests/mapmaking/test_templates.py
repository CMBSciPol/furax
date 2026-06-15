import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.config import BinsConfig, PolynomialOrders, SplineHWPSSConfig
from furax.mapmaking.templates import (
    ATOPProjectionOperator,
    KroneckerBasis,
    PerDetectorTemplate,
    SegmentedBasis,
    TensorBasis,
    WindowedBasis,
    _bin_weights,
    _harmonics,
    _legendre,
)
from furax.math import bspline
from furax.tree import as_structure


def spline_4f_hwpss_basis(
    times: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    n_knots: int,
) -> Float[Array, '2k samp']:
    """Dense reference for `PerDetectorTemplate.spline_hwpss`'s `WindowedBasis`.

    Returns:
        B: (2K, N) basis matrix, interleaved rows [phi_j sin(4χ), phi_j cos(4χ)].
    """
    phi = bspline.spline_basis(times, n_knots)  # (K, N)
    sin4 = jnp.sin(4.0 * hwp_angles)
    cos4 = jnp.cos(4.0 * hwp_angles)
    B = jnp.empty((2 * phi.shape[0], phi.shape[1]), dtype=phi.dtype)
    B = B.at[0::2].set(phi * sin4)
    B = B.at[1::2].set(phi * cos4)
    return B


# float64 everywhere: structural/adjoint residuals run ~1e-14, so one tight tolerance fits all.
TOL = 1e-12


def _adjoint_residual(basis, coeffs: Array, signal: Array) -> float:
    """|<expand(coeffs), signal> - <coeffs, project(signal)>|."""
    lhs = jnp.vdot(basis.expand(coeffs), signal)
    rhs = jnp.vdot(coeffs, basis.project(signal))
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
        basis = TensorBasis.create(values)
        assert basis.shape == shape
        assert basis.size == np.prod(shape)
        assert basis.n_points == n_points
        assert basis.in_structure == jax.ShapeDtypeStruct(shape, values.dtype)
        assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), values.dtype)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_matches_einsum(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jr.normal(jr.key(1), (*shape, n_points))
        coeffs = jr.normal(jr.key(2), shape)
        basis = TensorBasis.create(values)

        axes = 'abcdefg'[: len(shape)]
        expected = jnp.einsum(f'{axes},{axes}s->s', coeffs, values)
        assert_allclose(basis.expand(coeffs), expected, rtol=TOL)
        # mv is bound to expand
        assert_allclose(basis(coeffs), expected, rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_project_matches_einsum(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jr.normal(jr.key(3), (*shape, n_points))
        signal = jr.normal(jr.key(4), (n_points,))
        basis = TensorBasis.create(values)

        axes = 'abcdefg'[: len(shape)]
        expected = jnp.einsum(f'{axes}s,s->{axes}', values, signal)
        assert_allclose(basis.project(signal), expected, rtol=TOL)
        # transpose mv is bound to project
        assert_allclose(basis.T(signal), expected, rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jr.normal(jr.key(5), (*shape, n_points))
        coeffs = jr.normal(jr.key(6), shape)
        signal = jr.normal(jr.key(7), (n_points,))
        basis = TensorBasis.create(values)
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    @pytest.mark.parametrize('shape', [(4,), (3, 5)])
    def test_transpose_as_matrix(self, shape: tuple[int, ...]) -> None:
        n_points = 5
        values = jr.normal(jr.key(8), (*shape, n_points))
        basis = TensorBasis.create(values)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=TOL)

    def test_out_structure_roundtrip(self) -> None:
        values = jr.normal(jr.key(9), (3, 5))
        basis = TensorBasis.create(values)
        coeffs = jr.normal(jr.key(10), (3,))
        assert as_structure(basis.expand(coeffs)) == basis.out_structure


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
        basis = KroneckerBasis.create(factors)
        assert basis.shape == shape
        assert basis.n_points == n_points
        assert basis.in_structure == jax.ShapeDtypeStruct(shape, factors[0].dtype)
        assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), factors[0].dtype)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_equivalent_to_dense_tensor_basis(self, shape: tuple[int, ...]) -> None:
        """KroneckerBasis == TensorBasis whose values are the outer product of factors."""
        n_points = 6
        factors = _factors(shape, n_points, 200)
        kron = KroneckerBasis.create(factors)
        dense = TensorBasis.create(_dense_from_factors(factors))

        coeffs = jr.normal(jr.key(210), shape)
        signal = jr.normal(jr.key(211), (n_points,))
        assert_allclose(kron.expand(coeffs), dense.expand(coeffs), rtol=TOL)
        assert_allclose(kron.project(signal), dense.project(signal), rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        factors = _factors(shape, n_points, 300)
        basis = KroneckerBasis.create(factors)
        coeffs = jr.normal(jr.key(310), shape)
        signal = jr.normal(jr.key(311), (n_points,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    @pytest.mark.parametrize('shape', [(4,), (3, 5)])
    def test_transpose_as_matrix(self, shape: tuple[int, ...]) -> None:
        n_points = 5
        factors = _factors(shape, n_points, 400)
        basis = KroneckerBasis.create(factors)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=TOL)


# ---------------------------------------------------------------------------
# TensorBasis (decimated: q > 1)
# ---------------------------------------------------------------------------


def _decimated(shape: tuple[int, ...], n_dec: int, q: int, n_full: int, seed: int):
    values = jr.normal(jr.key(seed), (*shape, n_dec))
    return TensorBasis.create(values, q=q, n_full=n_full)


class TestDecimatedTensorBasis:
    @pytest.mark.parametrize('shape', [(4,), (2, 3)])
    def test_structure(self, shape: tuple[int, ...]) -> None:
        n_dec, q, n_full = 5, 4, 18
        basis = _decimated(shape, n_dec, q, n_full, 500)
        assert basis.shape == shape
        assert basis.n_dec == n_dec
        assert basis.n_points == n_full
        assert basis.out_structure == jax.ShapeDtypeStruct((n_full,), basis.dtype)

    def test_expand_holds_each_coarse_value_over_its_block(self) -> None:
        # synthesis = einsum on coarse grid, then repeat each value q times, trimmed.
        shape, n_dec, q, n_full = (4,), 5, 4, 18  # 5*4 = 20 > 18 -> trailing block trimmed
        basis = _decimated(shape, n_dec, q, n_full, 501)
        coeffs = jr.normal(jr.key(502), shape)
        coarse = jnp.einsum('k,kd->d', coeffs, basis.values)
        expected = jnp.repeat(coarse, q)[:n_full]
        assert_allclose(basis.expand(coeffs), expected, rtol=TOL)

    def test_project_sums_each_block(self) -> None:
        # analysis downsamples by block-summing, then einsum back to coefficients.
        shape, n_dec, q, n_full = (4,), 5, 4, 18
        basis = _decimated(shape, n_dec, q, n_full, 503)
        signal = jr.normal(jr.key(504), (n_full,))
        pad = n_dec * q - n_full
        block_sum = jnp.pad(signal, (0, pad)).reshape(n_dec, q).sum(axis=-1)
        expected = jnp.einsum('kd,d->k', basis.values, block_sum)
        assert_allclose(basis.project(signal), expected, rtol=TOL)

    @pytest.mark.parametrize('shape', [(4,), (2, 3)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_dec, q, n_full = 5, 4, 18
        basis = _decimated(shape, n_dec, q, n_full, 505)
        coeffs = jr.normal(jr.key(506), shape)
        signal = jr.normal(jr.key(507), (n_full,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_transpose_as_matrix(self) -> None:
        basis = _decimated((4,), 5, 4, 18, 508)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=TOL)

    def test_q_one_matches_plain_dense_basis(self) -> None:
        # q=1 stores the full grid: decimation collapses to the plain dense basis.
        values = jr.normal(jr.key(509), (4, 18))
        decimated = TensorBasis.create(values, q=1, n_full=18)
        plain = TensorBasis.create(values)
        coeffs = jr.normal(jr.key(510), (4,))
        signal = jr.normal(jr.key(511), (18,))
        assert_allclose(decimated.expand(coeffs), plain.expand(coeffs), rtol=TOL)
        assert_allclose(decimated.project(signal), plain.project(signal), rtol=TOL)

    def test_create_rejects_bad_q(self) -> None:
        with pytest.raises(ValueError, match='q must be >= 1'):
            TensorBasis.create(jr.normal(jr.key(512), (4, 5)), q=0)

    @pytest.mark.parametrize('n_full', [4, 21])  # 5*4=20: n_full must be in (16, 20]
    def test_create_rejects_inconsistent_n_full(self, n_full: int) -> None:
        with pytest.raises(ValueError, match='inconsistent'):
            TensorBasis.create(jr.normal(jr.key(513), (4, 5)), q=4, n_full=n_full)


# ---------------------------------------------------------------------------
# SegmentedBasis
# ---------------------------------------------------------------------------


def _segmented(n_segments: int, k: int, n_points: int, seed: int):
    segment = jr.randint(jr.key(seed), (n_points,), 0, n_segments)
    values = jr.normal(jr.key(seed + 1), (k, n_points))
    return SegmentedBasis.create(segment.astype(jnp.int32), values, n_segments), segment, values


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
        dense = TensorBasis.create(onehot[:, None, :] * values[None, :, :])
        coeffs = jr.normal(jr.key(611), (n_seg, k))
        signal = jr.normal(jr.key(612), (n_points,))
        assert_allclose(basis.expand(coeffs), dense.expand(coeffs), rtol=TOL)
        assert_allclose(basis.project(signal), dense.project(signal), rtol=TOL)

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
        diff = basis.expand(bumped) - basis.expand(coeffs)
        assert_allclose(diff[segment != 0], 0.0, atol=TOL)
        assert jnp.max(jnp.abs(diff[segment == 0])) > 0.0


# ---------------------------------------------------------------------------
# WindowedBasis
# ---------------------------------------------------------------------------


def _windowed(n_blocks: int, n_window: int, k: int, n_points: int, seed: int):
    # offset kept in [0, n_blocks - n_window] so every window stays in range.
    offset = jr.randint(jr.key(seed), (n_points,), 0, n_blocks - n_window + 1)
    block_weights = jr.normal(jr.key(seed + 1), (n_window, n_points))
    sub_values = jr.normal(jr.key(seed + 2), (k, n_points))
    basis = WindowedBasis.create(offset.astype(jnp.int32), block_weights, sub_values, n_blocks)
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
        dense = TensorBasis.create(_windowed_dense_values(offset, bw, sv, n_blocks))
        coeffs = jr.normal(jr.key(811), (n_blocks, k))
        signal = jr.normal(jr.key(812), (n_points,))
        assert_allclose(basis.expand(coeffs), dense.expand(coeffs), rtol=TOL)
        assert_allclose(basis.project(signal), dense.project(signal), rtol=TOL)

    def test_reduces_to_segmented_when_window_is_one(self) -> None:
        # O=1 with unit weights is exactly SegmentedBasis (offset == segment id).
        n_seg, k, n_points = 4, 3, 50
        segment = jr.randint(jr.key(820), (n_points,), 0, n_seg)
        values = jr.normal(jr.key(821), (k, n_points))
        segmented = SegmentedBasis.create(segment.astype(jnp.int32), values, n_seg)
        windowed = WindowedBasis.create(
            segment.astype(jnp.int32),
            jnp.ones((1, n_points), values.dtype),
            values,
            n_seg,
        )
        coeffs = jr.normal(jr.key(822), (n_seg, k))
        signal = jr.normal(jr.key(823), (n_points,))
        assert_allclose(windowed.expand(coeffs), segmented.expand(coeffs), rtol=TOL)
        assert_allclose(windowed.project(signal), segmented.project(signal), rtol=TOL)

    @pytest.mark.parametrize(('n_window', 'k'), [(4, 1), (4, 3), (1, 5)])
    def test_expand_project_adjoint(self, n_window: int, k: int) -> None:
        basis, *_ = _windowed(6, n_window, k, 50, 830)
        coeffs = jr.normal(jr.key(831), basis.shape)
        signal = jr.normal(jr.key(832), (50,))
        assert _adjoint_residual(basis, coeffs, signal) < TOL

    def test_transpose_as_matrix(self) -> None:
        basis, *_ = _windowed(6, 4, 3, 40, 840)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=TOL)

    def test_each_sample_only_sees_its_window(self) -> None:
        # perturbing one block's amplitudes only changes samples whose window covers it.
        n_blocks, n_window, k, n_points = 6, 4, 2, 50
        basis, offset, *_ = _windowed(n_blocks, n_window, k, n_points, 850)
        coeffs = jr.normal(jr.key(851), (n_blocks, k))
        target = 2
        bumped = coeffs.at[target].add(1.0)
        diff = basis.expand(bumped) - basis.expand(coeffs)
        covers = (offset <= target) & (target < offset + n_window)
        assert_allclose(diff[~covers], 0.0, atol=TOL)


# ---------------------------------------------------------------------------
# Basis-building helpers (_bin_weights, _legendre, _harmonics)
# ---------------------------------------------------------------------------


class TestBinWeights:
    def test_hard_assignment_is_one_hot(self) -> None:
        x = jr.normal(jr.key(700), (200,))
        w = _bin_weights(x, n_bins=4, interpolate=False, smooth=False, dtype=jnp.float64)
        assert w.shape == (4, 200)
        # exactly one bin per sample, weight 1
        assert_allclose(jnp.sum(w, axis=0), 1.0, atol=TOL)
        assert_allclose(jnp.sum(w == 1.0, axis=0), 1.0, atol=TOL)

    @pytest.mark.parametrize('smooth', [False, True])
    def test_interpolated_weights_form_a_partition_of_unity(self, smooth: bool) -> None:
        x = jnp.linspace(-2.0, 3.0, 200)
        w = _bin_weights(x, n_bins=5, interpolate=True, smooth=smooth, dtype=jnp.float64)
        assert_allclose(jnp.sum(w, axis=0), 1.0, atol=TOL)
        assert jnp.all(w >= 0.0)


class TestLegendreAndHarmonics:
    def test_legendre_order_zero_is_constant_one(self) -> None:
        x = jnp.linspace(0.0, 10.0, 64)
        legs = _legendre(x, 0, 3, jnp.float64)
        assert legs.shape == (4, 64)
        assert_allclose(legs[0], 1.0, atol=TOL)  # P_0 == 1
        # P_1(u) == u, the rescaled coordinate spanning [-1, 1]
        u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
        assert_allclose(legs[1], u, atol=TOL)

    @pytest.mark.parametrize('dc', [False, True])
    def test_harmonics_rows_and_dc(self, dc: bool) -> None:
        angles = jnp.linspace(0.0, 4.0, 128)
        n_harm = 3
        h = _harmonics(angles, n_harm, jnp.float64, dc=dc)
        assert h.shape == (2 * n_harm + int(dc), 128)
        if dc:
            assert_allclose(h[0], 1.0, atol=TOL)  # leading constant row
        # first non-DC row is the fundamental sine
        assert_allclose(h[int(dc)], jnp.sin(angles), atol=TOL)


# ---------------------------------------------------------------------------
# PerDetectorTemplate wrapper (shared vs per-detector basis, transpose)
# ---------------------------------------------------------------------------


class TestPerDetectorTemplate:
    def test_shared_basis_applies_same_basis_to_every_detector(self) -> None:
        n_dets, k, n_points = 3, 4, 30
        values = jr.normal(jr.key(800), (k, n_points))
        op = PerDetectorTemplate.from_basis(TensorBasis.create(values), n_dets, shared=True)
        assert op.in_structure.shape == (n_dets, k)
        assert op.out_structure.shape == (n_dets, n_points)
        coeffs = jr.normal(jr.key(801), (n_dets, k))
        assert_allclose(op(coeffs), coeffs @ values, rtol=TOL)

    def test_transpose_is_a_per_detector_template_and_is_adjoint(self) -> None:
        n_dets, k, n_points = 3, 4, 30
        values = jr.normal(jr.key(810), (k, n_points))
        op = PerDetectorTemplate.from_basis(TensorBasis.create(values), n_dets, shared=True)
        assert isinstance(op.T, PerDetectorTemplate)
        coeffs = jr.normal(jr.key(811), op.in_structure.shape)
        signal = jr.normal(jr.key(812), op.out_structure.shape)
        assert_allclose(jnp.vdot(op(coeffs), signal), jnp.vdot(coeffs, op.T(signal)), rtol=TOL)


# ---------------------------------------------------------------------------
# Synchronous templates (factory constructors)
# ---------------------------------------------------------------------------


def _geometry(n_samps: int):
    azimuth = jnp.linspace(0.0, 10.0, n_samps)
    hwp = jnp.linspace(0.0, 40.0, n_samps)
    return azimuth, hwp


class TestSynchronousTemplates:
    def test_scan_synchronous_is_shared_legendre(self) -> None:
        n_dets, n_samps = 3, 200
        azimuth, _ = _geometry(n_samps)
        legendre = PolynomialOrders(0, 3)
        op = PerDetectorTemplate.scan_synchronous(legendre, azimuth, n_dets, jnp.float64)
        assert op.shared_basis
        assert op.in_structure.shape == (n_dets, legendre.n_orders)
        coeffs = jr.normal(jr.key(901), op.in_structure.shape)
        legs = _legendre(azimuth, legendre.min_order, legendre.max_order, jnp.float64)
        assert_allclose(op(coeffs), coeffs @ legs, rtol=TOL)

    def test_binaz_synchronous_one_amplitude_per_bin(self) -> None:
        n_dets, n_samps = 3, 200
        azimuth, _ = _geometry(n_samps)
        bins = BinsConfig(n_bins=6, interpolate=False, smooth=False)
        op = PerDetectorTemplate.binaz_synchronous(bins, azimuth, n_dets, jnp.float64)
        assert op.in_structure.shape == (n_dets, bins.n_bins)
        coeffs = jr.normal(jr.key(911), op.in_structure.shape)
        signal = jr.normal(jr.key(912), op.out_structure.shape)
        assert_allclose(jnp.vdot(op(coeffs), signal), jnp.vdot(coeffs, op.T(signal)), rtol=TOL)

    def test_hwp_synchronous_has_two_rows_per_harmonic(self) -> None:
        n_dets, n_samps, n_harm = 3, 200, 4
        _, hwp = _geometry(n_samps)
        op = PerDetectorTemplate.hwp_synchronous(n_harm, hwp, n_dets, jnp.float64)
        assert op.in_structure.shape == (n_dets, 2 * n_harm)  # no DC row
        coeffs = jr.normal(jr.key(921), op.in_structure.shape)
        matrix = _harmonics(hwp, n_harm, jnp.float64, dc=False)
        assert_allclose(op(coeffs), coeffs @ matrix, rtol=TOL)

    def test_azhwp_synchronous_is_kronecker_shaped(self) -> None:
        n_dets, n_samps, n_harm = 2, 200, 3
        azimuth, hwp = _geometry(n_samps)
        legendre = PolynomialOrders(0, 2)
        op = PerDetectorTemplate.azhwp_synchronous(
            legendre, n_harm, azimuth, hwp, n_dets, jnp.float64
        )
        # azimuth Legendre orders x HWP harmonics with DC (2*n_harm + 1)
        assert op.in_structure.shape == (n_dets, legendre.n_orders, 2 * n_harm + 1)
        coeffs = jr.normal(jr.key(931), op.in_structure.shape)
        signal = jr.normal(jr.key(932), op.out_structure.shape)
        assert_allclose(jnp.vdot(op(coeffs), signal), jnp.vdot(coeffs, op.T(signal)), rtol=TOL)

    def test_azhwp_scan_mask_zeroes_flagged_samples(self) -> None:
        # zeroing the azimuth leg kills every basis function there -> zero synthesis.
        n_dets, n_samps, n_harm = 2, 200, 3
        azimuth, hwp = _geometry(n_samps)
        scan_mask = (jnp.arange(n_samps) % 3 != 0).astype(jnp.float64)
        op = PerDetectorTemplate.azhwp_synchronous(
            PolynomialOrders(0, 2), n_harm, azimuth, hwp, n_dets, jnp.float64, scan_mask=scan_mask
        )
        coeffs = jr.normal(jr.key(941), op.in_structure.shape)
        out = op(coeffs)
        assert_allclose(out[:, scan_mask == 0], 0.0, atol=TOL)

    def test_binazhwp_synchronous_is_kronecker_shaped(self) -> None:
        n_dets, n_samps, n_harm = 2, 200, 3
        azimuth, hwp = _geometry(n_samps)
        bins = BinsConfig(n_bins=5, interpolate=False, smooth=False)
        op = PerDetectorTemplate.binazhwp_synchronous(
            bins, n_harm, azimuth, hwp, n_dets, jnp.float64
        )
        assert op.in_structure.shape == (n_dets, bins.n_bins, 2 * n_harm + 1)
        coeffs = jr.normal(jr.key(951), op.in_structure.shape)
        signal = jr.normal(jr.key(952), op.out_structure.shape)
        assert_allclose(jnp.vdot(op(coeffs), signal), jnp.vdot(coeffs, op.T(signal)), rtol=TOL)


# ---------------------------------------------------------------------------
# PerDetectorTemplate.polynomial (structure, gaps)
# ---------------------------------------------------------------------------


class TestPolynomialStructure:
    def test_amplitude_shape_is_dets_intervals_orders(self) -> None:
        n_samps, n_dets, order = 200, 3, 3
        intervals = jnp.array([[0, 100], [100, 200]])
        times = jnp.arange(n_samps, dtype=jnp.float64)
        op = PerDetectorTemplate.polynomial(order, intervals, times, n_dets, jnp.float64)
        # one polynomial (orders 0..order) per interval per detector
        assert op.in_structure.shape == (n_dets, 2, order + 1)
        assert op.out_structure.shape == (n_dets, n_samps)

    def test_samples_in_gaps_carry_no_signal(self) -> None:
        # a gap between intervals: those samples sit in no segment -> zero column.
        n_samps, n_dets, order = 200, 2, 3
        intervals = jnp.array([[0, 80], [120, 200]])  # 80..120 is a gap
        times = jnp.arange(n_samps, dtype=jnp.float64)
        op = PerDetectorTemplate.polynomial(order, intervals, times, n_dets, jnp.float64)
        coeffs = jr.normal(jr.key(960), op.in_structure.shape)
        out = op(coeffs)
        gap = (jnp.arange(n_samps) >= 80) & (jnp.arange(n_samps) < 120)
        assert_allclose(out[:, gap], 0.0, atol=TOL)


# ---------------------------------------------------------------------------
# PerDetectorTemplate.polynomial (masking)
# ---------------------------------------------------------------------------


class TestPolynomialMask:
    n_samps = 200
    n_dets = 3
    max_poly_order = 3

    def _setup(self, valid_mask):
        half = self.n_samps // 2
        intervals = jnp.array([[0, half], [half, self.n_samps]])
        times = jnp.arange(self.n_samps, dtype=jnp.float64)
        return PerDetectorTemplate.polynomial(
            max_poly_order=self.max_poly_order,
            intervals=intervals,
            times=times,
            n_dets=self.n_dets,
            dtype=jnp.float64,
            valid_mask=valid_mask,
        )

    def _valid_mask(self) -> Float[Array, ' samp']:
        # flag every 5th sample
        return (jnp.arange(self.n_samps) % 5 != 0).astype(jnp.float64)

    def test_mask_zeros_template_at_flagged_samples(self) -> None:
        valid = self._valid_mask()
        op = self._setup(valid)
        coeffs = jr.normal(jr.key(20), op.in_structure.shape)
        out = op(coeffs)
        # flagged samples carry no template signal
        assert_allclose(out[:, valid == 0], 0.0, atol=TOL)

    def test_mask_drops_flagged_from_projection(self) -> None:
        valid = self._valid_mask()
        op = self._setup(valid)
        signal = jr.normal(jr.key(21), op.out_structure.shape)
        # corrupting flagged samples must not change the projected coefficients
        corrupt = signal.at[:, valid == 0].add(1e3)
        assert_allclose(op.T(signal), op.T(corrupt), atol=TOL)

    def test_no_mask_matches_explicit_ones(self) -> None:
        op_none = self._setup(None)
        op_ones = self._setup(jnp.ones(self.n_samps, dtype=jnp.float64))
        coeffs = jr.normal(jr.key(22), op_none.in_structure.shape)
        assert_allclose(op_none(coeffs), op_ones(coeffs), rtol=TOL)


# ---------------------------------------------------------------------------
# PerDetectorTemplate per-detector basis (T2P leakage template)
# ---------------------------------------------------------------------------


class TestTemperatureTemplate:
    def test_synthesis_is_per_detector_scaled_template(self) -> None:
        # each detector's basis is its own temperature stream -> mv(lambda) = lambda * T_d
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(40), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        assert op.in_structure.shape == (n_dets, 1)  # one amplitude per detector
        lam = jr.normal(jr.key(41), op.in_structure.shape)
        assert_allclose(op(lam), lam * T, rtol=TOL)

    def test_projection_is_per_detector_inner_product(self) -> None:
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(42), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        x = jr.normal(jr.key(43), op.out_structure.shape)
        assert_allclose(op.T(x), jnp.sum(T * x, axis=-1)[:, None], rtol=TOL)

    def test_adjoint(self) -> None:
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(44), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        lam = jr.normal(jr.key(45), op.in_structure.shape)
        x = jr.normal(jr.key(46), op.out_structure.shape)
        assert_allclose(jnp.vdot(op(lam), x), jnp.vdot(lam, op.T(x)), rtol=TOL)

    def test_scan_stacking_over_observations(self) -> None:
        # the per-detector basis values gain an obs axis under lax.scan; one program
        n_obs, n_dets, n_samps = 3, 4, 200
        Ts = jr.normal(jr.key(47), (n_obs, n_dets, n_samps))
        _, stack = jax.lax.scan(
            lambda _, Ti: (None, PerDetectorTemplate.temperature(Ti, jnp.float64)), None, Ts
        )
        op0 = jax.tree.map(lambda leaf: leaf[0], stack)
        lam = jr.normal(jr.key(48), op0.in_structure.shape)
        assert_allclose(op0(lam), lam * Ts[0], rtol=TOL)

    def test_fit_band_limits_the_basis_to_the_band(self) -> None:
        # fit_band band-passes the temperature basis: synthesis lives only in (f0, f1)
        n_dets, n_samps, fs = 2, 1024, 10.0
        f0, f1 = 0.5, 2.0
        T = jr.normal(jr.key(49), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64, fit_band=(f0, f1), sample_rate=fs)
        lam = jnp.ones((n_dets, 1))
        out = op(lam)  # = bandpass(T) since lambda = 1
        spec = jnp.abs(jnp.fft.rfft(out, axis=-1))
        freqs = jnp.fft.rfftfreq(n_samps, d=1.0 / fs)
        out_of_band = (freqs <= f0) | (freqs >= f1)
        assert jnp.max(spec[:, out_of_band]) < TOL  # power confined to the band

    def test_none_leg_has_no_amplitudes_and_zero_output(self) -> None:
        n_dets, n_samps = 4, 200
        empty = PerDetectorTemplate.none(n_dets, n_samps, jnp.float64)
        assert empty.in_structure.shape == (n_dets, 0)  # no amplitudes to fit
        out = empty(jnp.zeros(empty.in_structure.shape))
        assert out.shape == (n_dets, n_samps)
        assert_allclose(out, 0.0, atol=0.0)

    def test_decimated_synthesis_is_block_averaged_and_held(self) -> None:
        # decimation_factor=q stores T on a q-coarser grid; lambda = 1 -> synthesis is the
        # block-averaged temperature, held over each block.
        n_dets, n_samps, q = 2, 60, 4
        T = jr.normal(jr.key(970), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64, decimation_factor=q)
        assert op.in_structure.shape == (n_dets, 1)
        out = op(jnp.ones((n_dets, 1)))
        assert out.shape == (n_dets, n_samps)

        n_dec = -(-n_samps // q)
        pad = n_dec * q - n_samps
        avg = jnp.pad(T, [(0, 0), (0, pad)]).reshape(n_dets, n_dec, q).mean(axis=-1)
        held = jnp.repeat(avg, q, axis=-1)[:, :n_samps]
        assert_allclose(out, held, rtol=TOL)


# ---------------------------------------------------------------------------
# ATOPProjectionOperator
# ---------------------------------------------------------------------------


class TestATOPProjectionOperator:
    def make_op(self, n_det: int, n_samp: int, tau: int) -> ATOPProjectionOperator:
        return ATOPProjectionOperator(
            tau, in_structure=jax.ShapeDtypeStruct((n_det, n_samp), jnp.float64)
        )

    @pytest.mark.parametrize(
        'tau,x_vals,expected_vals',
        [
            (2, [1, 3, 5, 7], [-1, 1, -1, 1]),
            (3, [0, 3, 6, 10, 10, 10], [-3, 0, 3, 0, 0, 0]),
            (4, [0, 0, 0, 0, 1, 1, 1, 1, 2, 4, 6, 8], [0, 0, 0, 0, 0, 0, 0, 0, -3, -1, 1, 3]),
        ],
    )
    def test_demeaning_per_interval(self, tau: int, x_vals: list, expected_vals: list):
        """Each interval has its own mean removed, not a global one."""
        x = jnp.array([x_vals], dtype=jnp.float64)
        y = self.make_op(1, len(x_vals), tau)(x)
        assert_allclose(y, [expected_vals])

    def test_tail_passed_through_unchanged(self):
        """Samples in the partial tail interval are returned as-is."""
        n_det, n_samp, tau = 1, 10, 4  # tail at indices 8, 9
        x = jnp.arange(n_det * n_samp, dtype=jnp.float64).reshape(n_det, n_samp)
        y = self.make_op(n_det, n_samp, tau)(x)
        assert_array_equal(y[0, 2 * tau :], x[0, 2 * tau :])

    def test_multiple_detectors_demeaned_independently(self):
        """Each detector row is demeaned using its own interval means."""
        tau = 4
        n_det, n_samp = 3, 8
        x = jnp.stack([jnp.full((n_samp,), float(d)) for d in range(n_det)])
        y = self.make_op(n_det, n_samp, tau)(x)
        assert_array_equal(y, jnp.zeros((n_det, n_samp)))

    @pytest.mark.parametrize('n_samp,tau', [(8, 4), (10, 4)])
    def test_idempotent(self, n_samp: int, tau: int):
        """Test that op(op(x)) == op(x): operator is a projector."""
        n_det = 2
        x = jax.random.normal(jax.random.PRNGKey(0), (n_det, n_samp))
        op = self.make_op(n_det, n_samp, tau)
        assert_allclose(op(op(x)), op(x), atol=1e-6)

    def test_tau_one(self):
        """With tau=1, every sample is its own interval, so output is all zeros."""
        n_det, n_samp = 2, 6
        x = jax.random.normal(jax.random.PRNGKey(1), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=1)(x)
        assert_allclose(y, jnp.zeros((n_det, n_samp)), atol=1e-6)

    def test_tau_equals_n_samp(self):
        """With tau=n_samp, a single interval covers all samples; output sums to zero."""
        n_det, n_samp = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(2), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=n_samp)(x)
        assert_allclose(y.sum(axis=-1), jnp.zeros(n_det), atol=1e-6)


# ---------------------------------------------------------------------------
# SplineHWPSS basis functions and template
# ---------------------------------------------------------------------------


class TestSplineHWPSSConfig:
    @pytest.mark.parametrize(
        ('n_knots', 'samples_per_knot', 'expected'),
        [
            (10, None, 10),
            (None, 20, 5),
            (10, 20, 10),  # n_knots takes precedence
            (1, None, 2),  # at least 2 knots
            (None, 4000, 2),  # at least 2 knots
        ],
    )
    def test_resolve_n_knots(
        self, n_knots: int | None, samples_per_knot: int | None, expected: int
    ) -> None:
        config = SplineHWPSSConfig(n_knots=n_knots, samples_per_knot=samples_per_knot)
        assert config.resolve_n_knots(100) == expected

    def test_requires_one_of_n_knots_or_samples_per_knot(self) -> None:
        with pytest.raises(ValueError, match='one of'):
            SplineHWPSSConfig(n_knots=None, samples_per_knot=None)


class TestSplineHWPSSTemplate:
    def test_4f_basis_structure(self) -> None:
        t = jnp.linspace(0, 10, 100)
        hwp = jnp.linspace(0, 2 * jnp.pi, 100)
        n_knots = 3
        B = spline_4f_hwpss_basis(t, hwp, n_knots=n_knots)
        K = n_knots + 2
        # 2K because cos and sin blocks
        assert B.shape == (2 * K, 100)

    def test_4f_modulation_nonzero(self) -> None:
        t = jnp.linspace(0, 10, 100)
        hwp = jnp.linspace(0, 2 * jnp.pi, 100)
        B = spline_4f_hwpss_basis(t, hwp, n_knots=3)
        sin_part = B[0]
        cos_part = B[1]
        # should not be identical
        assert not jnp.allclose(sin_part, cos_part)

    def test_template_structure(self) -> None:
        n_dets, n_samps, n_knots = 2, 100, 3
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        op = PerDetectorTemplate.bspline_hwpss(t, hwp, n_dets, n_knots=n_knots, dtype=jnp.float64)

        K = n_knots + 2
        # WindowedBasis amplitudes: (K knots, 2 = cos/sin) per detector
        assert op.in_structure.shape == (n_dets, K, 2)
        assert op.out_structure.shape == (n_dets, n_samps)

    def test_equivalent_to_dense_4f_basis(self) -> None:
        # WindowedBasis spline_hwpss reproduces the dense (2K, N) interleaved basis.
        n_dets, n_samps, n_knots = 1, 120, 5
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 6 * jnp.pi, n_samps)
        op = PerDetectorTemplate.bspline_hwpss(t, hwp, n_dets, n_knots=n_knots, dtype=jnp.float64)
        dense = TensorBasis.create(spline_4f_hwpss_basis(t, hwp, n_knots))  # rows 2j=sin, 2j+1=cos

        K = n_knots + 2
        a = jr.normal(jr.key(1010), (K, 2))  # WindowedBasis amplitudes a[j] = (sin amp, cos amp)
        # dense uses the same interleaved order [sin_0, cos_0, sin_1, cos_1, ...]
        assert_allclose(op.operator.expand(a), dense.expand(a.reshape(-1)), rtol=TOL)

    def test_adjoint(self) -> None:
        n_dets, n_samps, n_knots = 2, 100, 3
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        op = PerDetectorTemplate.bspline_hwpss(t, hwp, n_dets, n_knots=n_knots, dtype=jnp.float64)

        coeffs = jr.normal(jr.key(1001), op.in_structure.shape)
        signal = jr.normal(jr.key(1002), op.out_structure.shape)
        assert _adjoint_residual(op.operator, coeffs[0], signal[0]) < TOL
        assert_allclose(jnp.vdot(op(coeffs), signal), jnp.vdot(coeffs, op.T(signal)), rtol=TOL)

    def test_spline_hwpss_multiple_harmonics(self) -> None:
        n_dets, n_samps = 2, 100
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        harmonics = [2, 4]
        op = PerDetectorTemplate.bspline_hwpss(t, hwp, n_dets, n_knots=4, harmonics=harmonics)
        # amplitudes: (det, K, 2 * n_harm)
        # K = n_knots + 2 = 4 + 2 = 6
        # 2 * n_harm = 2 * 2 = 4
        assert op.in_structure.shape == (n_dets, 6, 4)

    def test_spline_hwpss_int_harmonics(self) -> None:
        # an int n is the harmonics 1..n, matching `_harmonics`' convention.
        n_dets, n_samps = 2, 100
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        op = PerDetectorTemplate.bspline_hwpss(t, hwp, n_dets, n_knots=4, harmonics=3)
        # 1..3 -> 3 harmonics -> 2 * 3 = 6 columns; K = 4 + 2 = 6
        assert op.in_structure.shape == (n_dets, 6, 6)
