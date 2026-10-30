import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose

from furax.mapmaking import ReaderField
from furax.mapmaking.config import (
    BinsConfig,
    CommonModeConfig,
    HWPSynchronousConfig,
    PolynomialOrders,
    SplineDriftConfig,
    SplineHWPSynchronousConfig,
    T2PConfig,
    TemplatesConfig,
)
from furax.mapmaking.templates import (
    build_templates,
    common_mode_templates,
    polarization_field_couplings,
    required_reader_fields,
    spline_drift_templates,
    t2p_templates,
    winding_couplings,
    zernike_couplings,
)
from furax.mapmaking.templates._builders import (
    _bin_weights,
    _harmonics,
    _legendre,
    azimuth_bins_basis,
    azimuth_hwp_synchronous_basis,
    azimuth_spline_basis,
    binned_azimuth_hwp_synchronous_basis,
    hwp_synchronous_basis,
    spline_drift_basis,
    spline_hwp_synchronous_basis,
    t2p_basis,
)
from furax.mapmaking.templates.bases import TensorBasis
from furax.mapmaking.templates.pomme import PommeProjectionOperator
from furax.math import bspline
from furax.math.coords import XiEtaAngles
from furax.math.zernike import zernike
from furax.obs import TODFormat
from tests.mapmaking.templates.helpers import TOL, wrap


def spline_4f_hwp_basis(
    times: Float[Array, ' samp'], hwp_angles: Float[Array, ' samp'], n_knots: int
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
# Per-detector application and the adjoint are covered with the bases, in test_bases.py; these
# test the basis construction itself.


def _geometry(n_samps: int):
    azimuth = jnp.linspace(0.0, 10.0, n_samps)
    hwp = jnp.linspace(0.0, 40.0, n_samps)
    return azimuth, hwp


@pytest.mark.parametrize(
    ('build', 'shape'),
    [
        (
            lambda az, hwp: azimuth_bins_basis(
                BinsConfig(n_bins=6, interpolate=False, smooth=False), az, jnp.float64
            ),
            (6,),
        ),
        # a set of functions per scan mask
        (
            lambda az, hwp: azimuth_bins_basis(
                BinsConfig(n_bins=6), az, jnp.float64, scan_masks=(az < 5, az >= 5)
            ),
            (12,),
        ),
        # two rows per harmonic, no DC row
        (lambda az, hwp: hwp_synchronous_basis(4, hwp, jnp.float64), (8,)),
        # azimuth functions x HWP harmonics with their DC row
        (
            lambda az, hwp: azimuth_hwp_synchronous_basis(
                PolynomialOrders(0, 2), 3, az, hwp, jnp.float64
            ),
            (3, 7),
        ),
        (
            lambda az, hwp: binned_azimuth_hwp_synchronous_basis(
                BinsConfig(n_bins=5, interpolate=False, smooth=False), 3, az, hwp, jnp.float64
            ),
            (5, 7),
        ),
    ],
    ids=['binned_scan', 'binned_scan_split', 'hwp', 'azimuth_hwp', 'binned_azimuth_hwp'],
)
def test_synchronous_basis_amplitudes(build, shape) -> None:
    azimuth, hwp = _geometry(200)
    basis = build(azimuth, hwp)
    assert basis.shape == shape
    assert basis.n_points == 200


def _scan(n_samps: int):
    """A triangle wave of azimuth, its left- and right-going samples, and a few in neither."""
    phase = jnp.arange(n_samps) % 50 / 25
    azimuth = jnp.where(phase < 1, phase, 2 - phase)
    right = phase < 1
    turnaround = (phase < 0.08) | (jnp.abs(phase - 1) < 0.04)
    return azimuth, ~right & ~turnaround, right & ~turnaround


@pytest.mark.parametrize('split', [False, True])
def test_azimuth_spline_basis_is_the_splines_without_the_constant(split) -> None:
    n_knots = 6
    azimuth, left, right = _scan(400)
    masks = (left, right) if split else None
    basis = azimuth_spline_basis(n_knots, azimuth, jnp.float64, scan_masks=masks)
    values = basis.as_matrix()  # (samp, function)
    n_sets = 2 if split else 1
    assert values.shape == (400, n_sets * n_knots - 1)
    assert jnp.linalg.matrix_rank(values) == n_sets * n_knots - 1

    def residual(signal):
        fit = values @ jnp.linalg.lstsq(values, signal)[0]
        return jnp.max(jnp.abs(signal - fit))

    # the splines sum to one; every one whose coefficients sum to zero is in the span, the
    # constant is not
    splines = bspline.spline_basis(azimuth, n_knots - 1)[:-1].T  # (samp, knot)
    assert_allclose(splines.sum(axis=1), 1.0, atol=TOL)
    coefficients = jr.normal(jr.key(0), (n_knots,))
    signal = splines @ (coefficients - coefficients.mean())
    in_scans = (left | right) if split else jnp.ones(400, bool)
    assert residual(jnp.where(in_scans, signal, 0)) < 1e-10
    assert residual(in_scans.astype(jnp.float64)) > 0.1
    if split:
        # the offset of one scan direction to the other is kept, and the turnarounds are not seen
        assert residual(left.astype(jnp.float64) - right) < 1e-10
        assert_allclose(values[~in_scans], 0.0, atol=TOL)


def test_azimuth_hwp_scan_mask_zeroes_flagged_samples() -> None:
    # zeroing the azimuth factor kills every basis function there -> zero synthesis.
    n_samps, n_harm = 200, 3
    azimuth, hwp = _geometry(n_samps)
    scan_mask = (jnp.arange(n_samps) % 3 != 0).astype(jnp.float64)
    basis = azimuth_hwp_synchronous_basis(
        PolynomialOrders(0, 2), n_harm, azimuth, hwp, jnp.float64, scan_mask=scan_mask
    )
    out = basis(jr.normal(jr.key(941), basis.shape))
    assert_allclose(out[scan_mask == 0], 0.0, atol=TOL)
    assert jnp.all(out[scan_mask == 1] != 0)


# ---------------------------------------------------------------------------
# t2p_basis (T2P leakage template's per-detector basis)
# ---------------------------------------------------------------------------


class TestTemperatureTemplate:
    def test_synthesis_is_per_detector_scaled_template(self) -> None:
        # each detector's basis is its own temperature stream -> mv(lambda) = lambda * T_d
        n_dets, n_samps = 4, 200
        T = jr.normal(jr.key(40), (n_dets, n_samps))
        op = wrap(t2p_basis(T, jnp.float64), n_dets)
        assert op.in_structure['t'].shape == (n_dets, 1)  # one amplitude per detector
        lam = jr.normal(jr.key(41), (n_dets, 1))
        assert_allclose(op({'t': lam}), lam * T, rtol=TOL)

    def test_fit_band_limits_the_basis_to_the_band(self) -> None:
        # fit_band band-passes the temperature basis: synthesis lives only in (f0, f1)
        n_dets, n_samps, fs = 2, 1024, 10.0
        f0, f1 = 0.5, 2.0
        T = jr.normal(jr.key(49), (n_dets, n_samps))
        basis = t2p_basis(T, jnp.float64, fit_band=(f0, f1), sample_rate=fs)
        op = wrap(basis, n_dets)
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
        op = wrap(t2p_basis(T, jnp.float64, decimation_factor=q), n_dets)
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


def _pomme(n_det: int, n_samp: int, tau: int) -> PommeProjectionOperator:
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
def test_pomme_projection_removes_the_mean_of_each_interval(tau, x_vals, expected_vals):
    """Each interval has its own mean removed, not a global one."""
    x = jnp.array([x_vals], dtype=jnp.float64)
    assert_allclose(_pomme(1, len(x_vals), tau)(x), [expected_vals])


@pytest.mark.parametrize(
    'n_samp,tau',
    [(8, 4), (10, 4), (6, 1), (8, 8)],
    ids=['whole', 'tail', 'tau_one', 'one_interval'],
)
def test_pomme_projection_is_the_interval_demeaning_projector(n_samp: int, tau: int):
    # each detector demeaned on its own intervals; the partial tail interval is left as is
    x = jax.random.normal(jax.random.key(0), (3, n_samp))
    op = _pomme(3, n_samp, tau)
    n_whole = n_samp // tau * tau
    blocks = x[:, :n_whole].reshape(3, -1, tau)
    expected = jnp.concatenate(
        [(blocks - blocks.mean(-1, keepdims=True)).reshape(3, -1), x[:, n_whole:]], axis=-1
    )
    assert_allclose(op(x), expected, atol=1e-12)
    assert_allclose(op(op(x)), op(x), atol=1e-12)


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


@pytest.mark.parametrize(
    ('harmonics', 'shape'),
    # n_knots + 2 = 6 knots, each with a (sin, cos) pair per harmonic; an int n is the harmonics
    # 1..n, matching `_harmonics`' convention
    [((4,), (6, 2)), ([2, 4], (6, 4)), (3, (6, 6))],
)
def test_spline_hwp_synchronous_amplitudes(harmonics, shape) -> None:
    t = jnp.linspace(0, 10, 100)
    hwp = jnp.linspace(0, 2 * jnp.pi, 100)
    basis = spline_hwp_synchronous_basis(t, hwp, 4, harmonics, jnp.float64)
    assert basis.shape == shape
    assert basis.out_structure.shape == (100,)


def test_spline_hwp_synchronous_is_the_dense_4f_basis() -> None:
    n_samps, n_knots = 120, 5
    t = jnp.linspace(0, 10, n_samps)
    hwp = jnp.linspace(0, 6 * jnp.pi, n_samps)
    basis = spline_hwp_synchronous_basis(t, hwp, n_knots, (4,), jnp.float64)
    dense = TensorBasis(spline_4f_hwp_basis(t, hwp, n_knots))  # rows 2j=sin, 2j+1=cos
    a = jr.normal(jr.key(1010), (n_knots + 2, 2))  # amplitudes a[j] = (sin amp, cos amp)
    # dense uses the same interleaved order [sin_0, cos_0, sin_1, cos_1, ...]
    assert_allclose(basis(a), dense(a.reshape(-1)), rtol=TOL)


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
        spline_drift=SplineDriftConfig(explicit=True),
        hwp_synchronous=HWPSynchronousConfig(2, explicit=False),
    )
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64)
    explicit, implicit = build_templates(config, _data(None), tod_format)
    assert set(explicit) == {'spline_drift'} and set(implicit) == {'hwp_synchronous'}
    assert all(
        t.tod_format == tod_format and t.streams is None
        for t in [*explicit.values(), *implicit.values()]
    )
    assert required_reader_fields(config) <= set(_data(None))


@pytest.mark.parametrize(
    ('samples_per_knot', 'streams'),
    [
        (50, {'spline_drift': 'iqu'}),
        ({'i': 50, 'qu': 25}, {'spline_drift_i': 'i', 'spline_drift_qu': 'qu'}),
        ({'u': 50}, {'spline_drift_u': 'u'}),
    ],
)
def test_streams_fitted_alike_form_one_template(samples_per_knot, streams):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='IQU')
    config = SplineDriftConfig(samples_per_knot=samples_per_knot)
    templates = spline_drift_templates(config, _data('IQU'), tod_format)
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
    'field',
    [{'zernike_order': 2}, {'quadrupole_order': 2}],  # 6 complex functions each
    ids=['zernike', 'quadrupole'],
)
@pytest.mark.parametrize(
    ('joint_qu', 'per', 'shape'), [(True, (), (1, 12)), (False, ('stream',), (2, 1, 12))]
)
def test_t2p_field_shares_its_amplitudes_between_the_detectors(field, joint_qu, per, shape):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='IQU')
    config = T2PConfig(**field, joint_qu=joint_qu)
    template = t2p_templates(config, _field_data('IQU'), tod_format)['t2p']
    assert template.per == per and template.in_structure.shape == shape
    assert not template.per_detector
    assert ReaderField.DETECTOR_QUATERNIONS in required_reader_fields(TemplatesConfig(t2p=config))


def test_polarization_field_couplings_of_q_and_u_are_the_leakage_of_a_complex_field():
    _, _, gamma = _detector_angles()
    re, im = jr.normal(jr.key(7), (2, N_DETS, 3))
    fields = re + 1j * im
    c = jr.normal(jr.key(8), (3, 2))  # (Re, Im) of each function
    coupling = polarization_field_couplings(fields, gamma, streams='qu')
    assert coupling.shape == (2, N_DETS, 6)
    leakage = (fields @ (c[:, 0] + 1j * c[:, 1])) * jnp.exp(2j * gamma)
    assert_allclose(coupling @ c.ravel(), jnp.stack([leakage.real, leakage.imag]), atol=1e-12)


def test_polarization_field_couplings_of_one_stream_are_the_real_and_imaginary_parts():
    _, _, gamma = _detector_angles()
    re, im = jr.normal(jr.key(9), (2, N_DETS, 2))
    fields = re + 1j * im
    coupling = polarization_field_couplings(fields, gamma)
    assert coupling.shape == (N_DETS, 4)
    w = fields * jnp.exp(2j * gamma)[:, None]
    assert_allclose(coupling[:, 0::2], w.real, atol=1e-12)
    assert_allclose(coupling[:, 1::2], w.imag, atol=1e-12)


def test_polarization_field_couplings_reject_other_streams_than_q_and_u():
    with pytest.raises(ValueError, match='Q and U streams only'):
        polarization_field_couplings(jnp.ones((N_DETS, 1)), jnp.zeros(N_DETS), streams='qi')


@pytest.mark.parametrize(
    ('decentred', 'uniform', 'exponents'),
    [
        (True, True, [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]),
        (False, False, [(2, 0), (3, 1)]),
    ],
    ids=['decentred', 'centred'],
)
def test_winding_couplings_are_those_of_the_monomials_by_degree_and_turns(
    decentred, uniform, exponents
):
    # the modes are the monomials ζ^p ζ̄^q of ζ = (-η - iξ) / R, by degree p + q, then by turns
    # p - q: two turns about the axis for a centred one
    xi, eta, gamma = _detector_angles()
    zeta = (-eta - 1j * xi) / 0.31
    fields = jnp.stack([zeta**p * jnp.conj(zeta) ** q for p, q in exponents], axis=-1)
    coupling = winding_couplings(
        xi, eta, gamma, 2, 0.31, decentred=decentred, uniform=uniform, streams='qu'
    )
    assert_allclose(coupling, polarization_field_couplings(fields, gamma, streams='qu'), atol=TOL)


def test_winding_couplings_reject_a_radial_order_below_one():
    xi, eta, gamma = _detector_angles()
    with pytest.raises(ValueError, match="'radial_order' must be >= 1"):
        winding_couplings(xi, eta, gamma, 0, 0.31)


def _field_data(stokes: str) -> dict:
    """The fields the polarization templates are built from: those of `_data` and the detector
    quaternions."""
    angles = XiEtaAngles(*_detector_angles())
    quaternions = jnp.stack(angles.to_quaternion().to_components(), -1)
    return _data(stokes) | {ReaderField.DETECTOR_QUATERNIONS: quaternions}


@pytest.mark.parametrize(
    ('config', 'n_modes'),
    [
        (CommonModeConfig(samples_per_knot=30), 4),  # the quadrupole and the shifted axis
        (CommonModeConfig(samples_per_knot=30, uniform=True), 6),
        (CommonModeConfig(samples_per_knot=30, quadrupole_order=2, decentred=False), 4),
    ],
)
def test_common_mode_has_one_spline_per_mode_shared_by_the_detectors(config, n_modes):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='IQU')
    template = common_mode_templates(config, _field_data('IQU'), tod_format)['common_mode']
    assert template.streams == 'qu' and template.per == ()
    assert template.in_structure.shape == (N_SAMPS // 30 + 2, n_modes)  # (spline, mode)
    assert not template.per_detector


def test_common_mode_needs_the_q_and_u_streams():
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64, stokes='I')
    assert common_mode_templates(CommonModeConfig(), _field_data('I'), tod_format) == {}


def test_spline_drift_basis_is_the_masked_dense_spline():
    n_samps, n_knots = 300, 10
    t = jnp.linspace(0, 30, n_samps)
    valid = jnp.ones(n_samps).at[100:200].set(0.0)
    basis = spline_drift_basis(n_knots, t, jnp.float64, valid)
    dense = bspline.spline_basis(t, n_knots) * valid  # (K, N)
    assert_allclose(basis.as_matrix(), dense.T, atol=TOL)
    assert basis._n_blocks == n_knots + 2  # time-local: its Gram is banded


def test_spline_drift_is_a_partition_of_unity_on_valid_samples():
    n_samps, n_knots = 300, 10
    t = jnp.linspace(0, 30, n_samps)
    valid = jnp.ones(n_samps).at[100:200].set(0.0)  # a gap wider than the knot spacing
    basis = spline_drift_basis(n_knots, t, jnp.float64, valid)
    assert basis.shape == (n_knots + 2,)

    assert_allclose(basis(jnp.ones(basis.shape)), valid, atol=TOL)
    matrix = basis.as_matrix()
    assert jnp.sum(jnp.diagonal(matrix.T @ matrix) == 0) > 0  # knots in the gap see no sample
