import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking import ReaderField
from furax.mapmaking.config import (
    BinsConfig,
    HWPSynchronousConfig,
    PolynomialConfig,
    PolynomialOrders,
    SplineDriftConfig,
    SplineHWPSynchronousConfig,
    T2PConfig,
    TemplatesConfig,
)
from furax.mapmaking.templates import (
    Template,
    TemplateOperator,
    build_templates,
    polynomial_templates,
    required_reader_fields,
    t2p_templates,
    zernike_couplings,
)
from furax.mapmaking.templates._builders import (
    _bin_weights,
    _harmonics,
    _legendre,
    azimuth_hwp_synchronous_basis,
    binned_azimuth_hwp_synchronous_basis,
    binned_azimuth_synchronous_basis,
    hwp_synchronous_basis,
    scan_polynomial_basis,
    scan_synchronous_basis,
    spline_drift_basis,
    spline_hwp_synchronous_basis,
    t2p_basis,
)
from furax.mapmaking.templates.bases import Basis, TensorBasis
from furax.mapmaking.templates.pomme import PommeProjectionOperator
from furax.math import bspline
from furax.math.coords import XiEtaAngles
from furax.math.zernike import zernike
from furax.obs import TODFormat


def _wrap(basis: Basis, n_dets: int) -> TemplateOperator:
    """Wrap a bare basis as a single-template TemplateOperator: amplitudes are {'t': ...},
    output is a bare (n_dets, samp) array."""
    tod_format = TODFormat((n_dets, basis.n_points), basis.dtype)
    return TemplateOperator({'t': Template(basis, tod_format)})


def spline_4f_hwp_basis(
    times: Float[Array, ' samp'],
    hwp_angles: Float[Array, ' samp'],
    n_knots: int,
) -> Float[Array, '2k samp']:
    """Dense reference for `spline_hwp_synchronous_basis`'s `WindowedBasis`.

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
    lhs = jnp.vdot(basis(coeffs), signal)
    rhs = jnp.vdot(coeffs, basis.T(signal))
    return float(jnp.abs(lhs - rhs))


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
        values = _legendre(x, 0, 3, jnp.float64)
        assert values.shape == (4, 64)
        assert_allclose(values[0], 1.0, atol=TOL)  # P_0 == 1
        # P_1(u) == u, the rescaled coordinate spanning [-1, 1]
        u = -1.0 + 2.0 * (x - jnp.min(x)) / jnp.ptp(x)
        assert_allclose(values[1], u, atol=TOL)

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
# Synchronous templates (bare basis factories)
# ---------------------------------------------------------------------------
# Per-detector broadcasting and transpose-adjoint mechanics are TemplateOperator's job,
# covered by test_template_operator.py; these test the basis construction itself.


def _geometry(n_samps: int):
    azimuth = jnp.linspace(0.0, 10.0, n_samps)
    hwp = jnp.linspace(0.0, 40.0, n_samps)
    return azimuth, hwp


class TestSynchronousTemplates:
    def test_scan_synchronous_is_shared_legendre(self) -> None:
        n_samps = 200
        azimuth, _ = _geometry(n_samps)
        legendre = PolynomialOrders(0, 3)
        basis = scan_synchronous_basis(legendre, azimuth, jnp.float64)
        assert basis.shape == (legendre.n_orders,)
        coeffs = jr.normal(jr.key(901), basis.shape)
        values = _legendre(azimuth, legendre.min_order, legendre.max_order, jnp.float64)
        assert_allclose(basis(coeffs), coeffs @ values, rtol=TOL)

    def test_binned_azimuth_synchronous_one_amplitude_per_bin(self) -> None:
        n_samps = 200
        azimuth, _ = _geometry(n_samps)
        bins = BinsConfig(n_bins=6, interpolate=False, smooth=False)
        basis = binned_azimuth_synchronous_basis(bins, azimuth, jnp.float64)
        assert basis.shape == (bins.n_bins,)
        coeffs = jr.normal(jr.key(911), basis.shape)
        signal = jr.normal(jr.key(912), basis.out_structure.shape)
        assert_allclose(
            jnp.vdot(basis(coeffs), signal), jnp.vdot(coeffs, basis.T(signal)), rtol=TOL
        )

    def test_hwp_synchronous_has_two_rows_per_harmonic(self) -> None:
        n_samps, n_harm = 200, 4
        _, hwp = _geometry(n_samps)
        basis = hwp_synchronous_basis(n_harm, hwp, jnp.float64)
        assert basis.shape == (2 * n_harm,)  # no DC row
        coeffs = jr.normal(jr.key(921), basis.shape)
        matrix = _harmonics(hwp, n_harm, jnp.float64, dc=False)
        assert_allclose(basis(coeffs), coeffs @ matrix, rtol=TOL)

    def test_azimuth_hwp_synchronous_is_kronecker_shaped(self) -> None:
        n_samps, n_harm = 200, 3
        azimuth, hwp = _geometry(n_samps)
        legendre = PolynomialOrders(0, 2)
        basis = azimuth_hwp_synchronous_basis(legendre, n_harm, azimuth, hwp, jnp.float64)
        # azimuth Legendre orders x HWP harmonics with DC (2*n_harm + 1)
        assert basis.shape == (legendre.n_orders, 2 * n_harm + 1)
        coeffs = jr.normal(jr.key(931), basis.shape)
        signal = jr.normal(jr.key(932), basis.out_structure.shape)
        assert_allclose(
            jnp.vdot(basis(coeffs), signal), jnp.vdot(coeffs, basis.T(signal)), rtol=TOL
        )

    def test_azimuth_hwp_scan_mask_zeroes_flagged_samples(self) -> None:
        # zeroing the azimuth factor kills every basis function there -> zero synthesis.
        n_samps, n_harm = 200, 3
        azimuth, hwp = _geometry(n_samps)
        scan_mask = (jnp.arange(n_samps) % 3 != 0).astype(jnp.float64)
        basis = azimuth_hwp_synchronous_basis(
            PolynomialOrders(0, 2), n_harm, azimuth, hwp, jnp.float64, scan_mask=scan_mask
        )
        coeffs = jr.normal(jr.key(941), basis.shape)
        out = basis(coeffs)
        assert_allclose(out[scan_mask == 0], 0.0, atol=TOL)

    def test_binned_azimuth_hwp_synchronous_is_kronecker_shaped(self) -> None:
        n_samps, n_harm = 200, 3
        azimuth, hwp = _geometry(n_samps)
        bins = BinsConfig(n_bins=5, interpolate=False, smooth=False)
        basis = binned_azimuth_hwp_synchronous_basis(bins, n_harm, azimuth, hwp, jnp.float64)
        assert basis.shape == (bins.n_bins, 2 * n_harm + 1)
        coeffs = jr.normal(jr.key(951), basis.shape)
        signal = jr.normal(jr.key(952), basis.out_structure.shape)
        assert_allclose(
            jnp.vdot(basis(coeffs), signal), jnp.vdot(coeffs, basis.T(signal)), rtol=TOL
        )


# ---------------------------------------------------------------------------
# scan_polynomial_basis (structure, gaps)
# ---------------------------------------------------------------------------


class TestPolynomialStructure:
    def test_amplitude_shape_is_intervals_orders(self) -> None:
        n_samps, order = 200, 3
        intervals = jnp.array([[0, 100], [100, 200]])
        times = jnp.arange(n_samps, dtype=jnp.float64)
        basis = scan_polynomial_basis(order, intervals, times, jnp.float64)
        # one polynomial (orders 0..order) per interval
        assert basis.shape == (2, order + 1)
        assert basis.out_structure.shape == (n_samps,)

    def test_samples_in_gaps_carry_no_signal(self) -> None:
        # a gap between intervals: those samples sit in no segment -> zero column.
        n_samps, order = 200, 3
        intervals = jnp.array([[0, 80], [120, 200]])  # 80..120 is a gap
        times = jnp.arange(n_samps, dtype=jnp.float64)
        basis = scan_polynomial_basis(order, intervals, times, jnp.float64)
        coeffs = jr.normal(jr.key(960), basis.shape)
        out = basis(coeffs)
        gap = (jnp.arange(n_samps) >= 80) & (jnp.arange(n_samps) < 120)
        assert_allclose(out[gap], 0.0, atol=TOL)

    def test_segments_are_sorted(self) -> None:
        # `SegmentedBasis` requires non-decreasing segment ids, gaps included
        intervals = jnp.array([[10, 80], [120, 160], [160, 190]])
        times = jnp.arange(200, dtype=jnp.float64)
        basis = scan_polynomial_basis(3, intervals, times, jnp.float64)
        assert jnp.all(jnp.diff(basis.segment) >= 0)


# ---------------------------------------------------------------------------
# scan_polynomial_basis (masking)
# ---------------------------------------------------------------------------


class TestPolynomialMask:
    n_samps = 200
    max_poly_order = 3

    def _setup(self, valid_mask):
        half = self.n_samps // 2
        intervals = jnp.array([[0, half], [half, self.n_samps]])
        times = jnp.arange(self.n_samps, dtype=jnp.float64)
        return scan_polynomial_basis(self.max_poly_order, intervals, times, jnp.float64, valid_mask)

    def _valid_mask(self) -> Float[Array, ' samp']:
        # flag every 5th sample
        return (jnp.arange(self.n_samps) % 5 != 0).astype(jnp.float64)

    def test_mask_zeros_template_at_flagged_samples(self) -> None:
        valid = self._valid_mask()
        basis = self._setup(valid)
        coeffs = jr.normal(jr.key(20), basis.shape)
        out = basis(coeffs)
        # flagged samples carry no template signal
        assert_allclose(out[valid == 0], 0.0, atol=TOL)

    def test_mask_drops_flagged_from_projection(self) -> None:
        valid = self._valid_mask()
        basis = self._setup(valid)
        signal = jr.normal(jr.key(21), basis.out_structure.shape)
        # corrupting flagged samples must not change the projected coefficients
        corrupt = signal.at[valid == 0].add(1e3)
        assert_allclose(basis.T(signal), basis.T(corrupt), atol=TOL)

    def test_no_mask_matches_explicit_ones(self) -> None:
        basis_none = self._setup(None)
        basis_ones = self._setup(jnp.ones(self.n_samps, dtype=jnp.float64))
        coeffs = jr.normal(jr.key(22), basis_none.shape)
        assert_allclose(basis_none(coeffs), basis_ones(coeffs), rtol=TOL)


# ---------------------------------------------------------------------------
# t2p_basis (T2P leakage template's per-detector basis)
# ---------------------------------------------------------------------------


class TestTemperatureTemplate:
    def test_synthesis_is_per_detector_scaled_template(self) -> None:
        # each detector's basis is its own temperature stream -> mv(lambda) = lambda * T_d
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(40), (n_dets, n_samps))
        op = _wrap(t2p_basis(T, jnp.float64), n_dets)
        assert op.in_structure['t'].shape == (n_dets, 1)  # one amplitude per detector
        lam = jr.normal(jr.key(41), (n_dets, 1))
        assert_allclose(op({'t': lam}), lam * T, rtol=TOL)

    def test_adjoint(self) -> None:
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(44), (n_dets, n_samps))
        op = _wrap(t2p_basis(T, jnp.float64), n_dets)
        lam = jr.normal(jr.key(45), (n_dets, 1))
        x = jr.normal(jr.key(46), (n_dets, n_samps))
        assert_allclose(jnp.vdot(op({'t': lam}), x), jnp.vdot(lam, op.T(x)['t']), rtol=TOL)

    def test_scan_stacking_over_observations(self) -> None:
        # the per-detector basis values gain an obs axis under lax.scan; one program
        n_obs, n_dets, n_samps = 3, 4, 200
        Ts = jr.normal(jr.key(47), (n_obs, n_dets, n_samps))
        _, stack = jax.lax.scan(
            lambda _, Ti: (None, _wrap(t2p_basis(Ti, jnp.float64), n_dets)),
            None,
            Ts,
        )
        op0 = jax.tree.map(lambda leaf: leaf[0], stack)
        lam = jr.normal(jr.key(48), (n_dets, 1))
        assert_allclose(op0({'t': lam}), lam * Ts[0], rtol=TOL)

    def test_fit_band_limits_the_basis_to_the_band(self) -> None:
        # fit_band band-passes the temperature basis: synthesis lives only in (f0, f1)
        n_dets, n_samps, fs = 2, 1024, 10.0
        f0, f1 = 0.5, 2.0
        T = jr.normal(jr.key(49), (n_dets, n_samps))
        basis = t2p_basis(T, jnp.float64, fit_band=(f0, f1), sample_rate=fs)
        op = _wrap(basis, n_dets)
        out = op({'t': jnp.ones((n_dets, 1))})  # = bandpass(T) since lambda = 1
        spec = jnp.abs(jnp.fft.rfft(out, axis=-1))
        freqs = jnp.fft.rfftfreq(n_samps, d=1.0 / fs)
        out_of_band = (freqs <= f0) | (freqs >= f1)
        assert jnp.max(spec[:, out_of_band]) < TOL  # power confined to the band

    def test_decimated_synthesis_is_block_averaged_and_held(self) -> None:
        # decimation_factor=q stores T on a q-coarser grid; lambda = 1 -> synthesis is the
        # block-averaged temperature, held over each block.
        n_dets, n_samps, q = 2, 60, 4
        T = jr.normal(jr.key(970), (n_dets, n_samps))
        op = _wrap(t2p_basis(T, jnp.float64, decimation_factor=q), n_dets)
        assert op.in_structure['t'].shape == (n_dets, 1)
        out = op({'t': jnp.ones((n_dets, 1))})
        assert out.shape == (n_dets, n_samps)

        n_dec = -(-n_samps // q)
        pad = n_dec * q - n_samps
        avg = jnp.pad(T, [(0, 0), (0, pad)]).reshape(n_dets, n_dec, q).mean(axis=-1)
        held = jnp.repeat(avg, q, axis=-1)[:, :n_samps]
        assert_allclose(out, held, rtol=TOL)


# ---------------------------------------------------------------------------
# PommeProjectionOperator
# ---------------------------------------------------------------------------


class TestPommeProjectionOperator:
    def make_op(self, n_det: int, n_samp: int, tau: int) -> PommeProjectionOperator:
        return PommeProjectionOperator(
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
        x = jax.random.normal(jax.random.key(0), (n_det, n_samp))
        op = self.make_op(n_det, n_samp, tau)
        assert_allclose(op(op(x)), op(x), atol=1e-6)

    def test_tau_one(self):
        """With tau=1, every sample is its own interval, so output is all zeros."""
        n_det, n_samp = 2, 6
        x = jax.random.normal(jax.random.key(1), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=1)(x)
        assert_allclose(y, jnp.zeros((n_det, n_samp)), atol=1e-6)

    def test_tau_equals_n_samp(self):
        """With tau=n_samp, a single interval covers all samples; output sums to zero."""
        n_det, n_samp = 2, 8
        x = jax.random.normal(jax.random.key(2), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=n_samp)(x)
        assert_allclose(y.sum(axis=-1), jnp.zeros(n_det), atol=1e-6)


# ---------------------------------------------------------------------------
# Spline HWP-synchronous basis functions and template
# ---------------------------------------------------------------------------


class TestSplineHWPSynchronousConfig:
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
        config = SplineHWPSynchronousConfig(n_knots=n_knots, samples_per_knot=samples_per_knot)
        assert config.resolve_n_knots(100) == expected

    def test_requires_one_of_n_knots_or_samples_per_knot(self) -> None:
        with pytest.raises(ValueError, match='one of'):
            SplineHWPSynchronousConfig(n_knots=None, samples_per_knot=None)


class TestSplineHWPSynchronousTemplate:
    def test_template_structure(self) -> None:
        n_samps, n_knots = 100, 3
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        basis = spline_hwp_synchronous_basis(t, hwp, n_knots, (4,), jnp.float64)

        K = n_knots + 2
        # WindowedBasis amplitudes: (K knots, 2 = cos/sin)
        assert basis.shape == (K, 2)
        assert basis.out_structure.shape == (n_samps,)

    def test_offsets_are_sorted(self) -> None:
        # `WindowedBasis` requires non-decreasing window offsets
        t = jnp.linspace(0, 10, 100)
        basis = spline_hwp_synchronous_basis(t, jnp.zeros(100), 6, (4,), jnp.float64)
        assert jnp.all(jnp.diff(basis.offset) >= 0)

    def test_equivalent_to_dense_4f_basis(self) -> None:
        # WindowedBasis spline_hwp_synchronous reproduces the dense (2K, N) interleaved basis.
        n_samps, n_knots = 120, 5
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 6 * jnp.pi, n_samps)
        basis = spline_hwp_synchronous_basis(t, hwp, n_knots, (4,), jnp.float64)
        dense = TensorBasis(spline_4f_hwp_basis(t, hwp, n_knots))  # rows 2j=sin, 2j+1=cos

        K = n_knots + 2
        a = jr.normal(jr.key(1010), (K, 2))  # WindowedBasis amplitudes a[j] = (sin amp, cos amp)
        # dense uses the same interleaved order [sin_0, cos_0, sin_1, cos_1, ...]
        assert_allclose(basis(a), dense(a.reshape(-1)), rtol=TOL)

    def test_adjoint(self) -> None:
        n_samps, n_knots = 100, 3
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        basis = spline_hwp_synchronous_basis(t, hwp, n_knots, (4,), jnp.float64)

        coeffs = jr.normal(jr.key(1001), basis.shape)
        signal = jr.normal(jr.key(1002), basis.out_structure.shape)
        assert _adjoint_residual(basis, coeffs, signal) < TOL
        assert_allclose(
            jnp.vdot(basis(coeffs), signal), jnp.vdot(coeffs, basis.T(signal)), rtol=TOL
        )

    def test_spline_hwp_synchronous_multiple_harmonics(self) -> None:
        n_samps = 100
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        harmonics = [2, 4]
        basis = spline_hwp_synchronous_basis(t, hwp, 4, harmonics, jnp.float64)
        # amplitudes: (K, 2 * n_harm)
        # K = n_knots + 2 = 4 + 2 = 6
        # 2 * n_harm = 2 * 2 = 4
        assert basis.shape == (6, 4)

    def test_spline_hwp_synchronous_int_harmonics(self) -> None:
        # an int n is the harmonics 1..n, matching `_harmonics`' convention.
        n_samps = 100
        t = jnp.linspace(0, 10, n_samps)
        hwp = jnp.linspace(0, 2 * jnp.pi, n_samps)
        basis = spline_hwp_synchronous_basis(t, hwp, 4, 3, jnp.float64)
        # 1..3 -> 3 harmonics -> 2 * 3 = 6 columns; K = 4 + 2 = 6
        assert basis.shape == (6, 6)


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

N_DETS = 3
N_SAMPS = 120


def _data(stokes: str | None) -> dict:
    """An observation's fields, as the reader returns them, for the template builders."""
    times = jnp.linspace(0.0, 1.0, N_SAMPS)
    data = {
        ReaderField.TIMESTAMPS: times,
        ReaderField.SCANNING_INTERVALS: jnp.array([[0, 60], [60, N_SAMPS]]),
        ReaderField.VALID_SCANNING_MASKS: jnp.ones(N_SAMPS, bool),
        ReaderField.HWP_ANGLES: 2 * jnp.pi * 4 * times,
        ReaderField.AZIMUTH: jnp.sin(2 * jnp.pi * times),
    }
    if stokes is not None:
        tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes=stokes)
        data[ReaderField.SAMPLE_DATA] = tod_format.normal(jr.key(0))
    return data


def test_build_templates_splits_explicit_and_implicit():
    config = TemplatesConfig(
        polynomial=PolynomialConfig(explicit=True),
        hwp_synchronous=HWPSynchronousConfig(2, explicit=False),
    )
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64)
    explicit, implicit = build_templates(config, _data(None), tod_format)
    assert set(explicit) == {'polynomial'} and set(implicit) == {'hwp_synchronous'}
    assert all(
        t.tod_format == tod_format and t.streams is None
        for t in [*explicit.values(), *implicit.values()]
    )
    assert required_reader_fields(config) <= set(_data(None))


@pytest.mark.parametrize(
    ('legendre', 'streams'),
    [
        (PolynomialOrders(0, 3), {'polynomial': 'iqu'}),
        (
            {'i': PolynomialOrders(0, 3), 'qu': PolynomialOrders(1, 3)},
            {'polynomial_i': 'i', 'polynomial_qu': 'qu'},
        ),
        ({'u': PolynomialOrders(0, 1)}, {'polynomial_u': 'u'}),
    ],
)
def test_streams_fitted_alike_form_one_template(legendre, streams):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='IQU')
    templates = polynomial_templates(PolynomialConfig(legendre=legendre), _data('IQU'), tod_format)
    assert {name: t.streams for name, t in templates.items()} == streams


@pytest.mark.parametrize(('stokes', 'streams'), [('IQU', 'qu'), ('I', None)])
def test_t2p_covers_the_q_and_u_streams(stokes, streams):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes=stokes)
    templates = t2p_templates(T2PConfig(), _data('IQU'), tod_format)
    assert (templates['t2p'].streams if templates else None) == streams


def test_build_templates_needs_an_active_template():
    with pytest.raises(ValueError, match='no template is active'):
        build_templates(TemplatesConfig(), _data(None), TODFormat((N_DETS, N_SAMPS), jnp.float64))


def _detector_angles():
    r = jr.split(jr.key(5), 3)
    xi, eta = (jr.uniform(k, (N_DETS,), minval=-0.2, maxval=0.2) for k in r[:2])
    return xi, eta, jr.uniform(r[2], (N_DETS,), maxval=jnp.pi)


@pytest.mark.parametrize('order', [0, 2])
def test_zernike_couplings_of_one_stream_are_a_cos_sin_pair_per_polynomial(order):
    xi, eta, gamma = _detector_angles()
    coupling = zernike_couplings(xi, eta, gamma, order, 0.31)
    modes = [(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)]
    assert coupling.shape == (N_DETS, 2 * len(modes))
    x, y = xi / 0.31, eta / 0.31
    first = zernike(*modes[-1], x, y)
    assert_allclose(coupling[:, -2], first * jnp.cos(2 * gamma))
    assert_allclose(coupling[:, -1], first * jnp.sin(2 * gamma))


def test_zernike_couplings_of_q_and_u_are_the_real_and_imaginary_parts_of_one_field():
    xi, eta, gamma = _detector_angles()
    coupling = zernike_couplings(xi, eta, gamma, 2, 0.31, streams='qu')
    assert coupling.shape == (2, N_DETS, 12)
    c = jr.normal(jr.key(6), (6, 2))  # (Re, Im) of each polynomial
    field = sum(
        complex(*c[k]) * zernike(n, m, xi / 0.31, eta / 0.31)
        for k, (n, m) in enumerate((n, m) for n in range(3) for m in range(-n, n + 1, 2))
    )
    leakage = field * jnp.exp(2j * gamma)
    assert_allclose(coupling @ c.ravel(), jnp.stack([leakage.real, leakage.imag]), atol=1e-12)


def test_zernike_couplings_reject_other_streams_than_q_and_u():
    xi, eta, gamma = _detector_angles()
    with pytest.raises(ValueError, match='Q and U streams only'):
        zernike_couplings(xi, eta, gamma, 1, 0.31, streams='qi')


@pytest.mark.parametrize(
    ('joint_qu', 'per', 'shape'), [(True, (), (1, 12)), (False, ('stream',), (2, 1, 12))]
)
def test_t2p_field_shares_its_amplitudes_between_the_detectors(joint_qu, per, shape):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='IQU')
    angles = XiEtaAngles(*_detector_angles())
    data = _data('IQU') | {
        ReaderField.DETECTOR_QUATERNIONS: jnp.stack(angles.to_quaternion().to_components(), -1)
    }
    config = T2PConfig(zernike_order=2, joint_qu=joint_qu)
    template = t2p_templates(config, data, tod_format)['t2p']
    assert template.per == per and template.in_structure.shape == shape
    assert not template.per_detector
    assert ReaderField.DETECTOR_QUATERNIONS in required_reader_fields(TemplatesConfig(t2p=config))


def test_spline_drift_needs_no_scanning_intervals():
    config = TemplatesConfig(spline_drift=SplineDriftConfig())
    assert ReaderField.SCANNING_INTERVALS not in required_reader_fields(config)
    assert ReaderField.SCANNING_INTERVALS in required_reader_fields(
        TemplatesConfig(polynomial=PolynomialConfig())
    )


def test_spline_drift_is_a_partition_of_unity_on_valid_samples():
    n_samps, n_knots = 300, 10
    t = jnp.linspace(0, 30, n_samps)
    valid = jnp.ones(n_samps).at[100:200].set(0.0)  # a gap wider than the knot spacing
    basis = spline_drift_basis(n_knots, t, jnp.float64, valid)
    assert basis.shape == (n_knots + 2,)

    assert_allclose(basis(jnp.ones(basis.shape)), valid, atol=TOL)
    matrix = basis.as_matrix()
    assert jnp.sum(jnp.diagonal(matrix.T @ matrix) == 0) > 0  # knots in the gap see no sample
