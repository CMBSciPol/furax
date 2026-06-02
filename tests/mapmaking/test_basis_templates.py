import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_allclose

from furax.mapmaking.basis_templates import (
    KroneckerBasis,
    PerDetectorTemplate,
    TensorBasis,
)
from furax.tree import as_structure


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


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
        values = jax.random.normal(_key(0), (*shape, n_points))
        basis = TensorBasis.create(values)
        assert basis.shape == shape
        assert basis.size == int(jnp.prod(jnp.array(shape)))
        assert basis.n_points == n_points
        assert basis.in_structure == jax.ShapeDtypeStruct(shape, values.dtype)
        assert basis.out_structure == jax.ShapeDtypeStruct((n_points,), values.dtype)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_matches_einsum(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jax.random.normal(_key(1), (*shape, n_points))
        coeffs = jax.random.normal(_key(2), shape)
        basis = TensorBasis.create(values)

        axes = 'abcdefg'[: len(shape)]
        expected = jnp.einsum(f'{axes},{axes}s->s', coeffs, values)
        assert_allclose(basis.expand(coeffs), expected, rtol=1e-12)
        # mv is bound to expand
        assert_allclose(basis.mv(coeffs), expected, rtol=1e-12)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_project_matches_einsum(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jax.random.normal(_key(3), (*shape, n_points))
        signal = jax.random.normal(_key(4), (n_points,))
        basis = TensorBasis.create(values)

        axes = 'abcdefg'[: len(shape)]
        expected = jnp.einsum(f'{axes}s,s->{axes}', values, signal)
        assert_allclose(basis.project(signal), expected, rtol=1e-12)
        # transpose mv is bound to project
        assert_allclose(basis.T.mv(signal), expected, rtol=1e-12)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        values = jax.random.normal(_key(5), (*shape, n_points))
        coeffs = jax.random.normal(_key(6), shape)
        signal = jax.random.normal(_key(7), (n_points,))
        basis = TensorBasis.create(values)
        assert _adjoint_residual(basis, coeffs, signal) < 1e-10

    @pytest.mark.parametrize('shape', [(4,), (3, 5)])
    def test_transpose_as_matrix(self, shape: tuple[int, ...]) -> None:
        n_points = 5
        values = jax.random.normal(_key(8), (*shape, n_points))
        basis = TensorBasis.create(values)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=1e-12)

    def test_out_structure_roundtrip(self) -> None:
        values = jax.random.normal(_key(9), (3, 5))
        basis = TensorBasis.create(values)
        coeffs = jax.random.normal(_key(10), (3,))
        assert as_structure(basis(coeffs)) == basis.out_structure


# ---------------------------------------------------------------------------
# KroneckerBasis
# ---------------------------------------------------------------------------


def _factors(shape: tuple[int, ...], n_points: int, seed: int) -> tuple[Float[Array, 'd n'], ...]:
    return tuple(jax.random.normal(_key(seed + i), (d, n_points)) for i, d in enumerate(shape))


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

        coeffs = jax.random.normal(_key(210), shape)
        signal = jax.random.normal(_key(211), (n_points,))
        assert_allclose(kron.expand(coeffs), dense.expand(coeffs), rtol=1e-11)
        assert_allclose(kron.project(signal), dense.project(signal), rtol=1e-11)

    @pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
    def test_expand_project_adjoint(self, shape: tuple[int, ...]) -> None:
        n_points = 6
        factors = _factors(shape, n_points, 300)
        basis = KroneckerBasis.create(factors)
        coeffs = jax.random.normal(_key(310), shape)
        signal = jax.random.normal(_key(311), (n_points,))
        assert _adjoint_residual(basis, coeffs, signal) < 1e-10

    @pytest.mark.parametrize('shape', [(4,), (3, 5)])
    def test_transpose_as_matrix(self, shape: tuple[int, ...]) -> None:
        n_points = 5
        factors = _factors(shape, n_points, 400)
        basis = KroneckerBasis.create(factors)
        assert_allclose(basis.T.as_matrix().T, basis.as_matrix(), rtol=1e-11)


# ---------------------------------------------------------------------------
# PerDetectorTemplate.polynomial (masking)
# ---------------------------------------------------------------------------


class TestPolynomialMask:
    def _setup(self, valid_mask):
        n_samps, n_dets = 200, 3
        intervals = jnp.array([[0, 100], [100, 200]])
        times = jnp.arange(n_samps, dtype=jnp.float64)
        op = PerDetectorTemplate.polynomial(
            max_poly_order=3,
            intervals=intervals,
            times=times,
            n_dets=n_dets,
            dtype=jnp.float64,
            valid_mask=valid_mask,
        )
        return op, n_dets, n_samps

    def test_mask_zeros_template_at_flagged_samples(self) -> None:
        n_samps = 200
        valid = (jnp.arange(n_samps) % 5 != 0).astype(jnp.float64)
        op, n_dets, _ = self._setup(valid)
        coeffs = jax.random.normal(_key(20), op.in_structure.shape)
        out = op.mv(coeffs)
        # flagged samples carry no template signal
        flagged = valid == 0
        assert_allclose(out[:, flagged], 0.0, atol=1e-12)

    def test_mask_drops_flagged_from_projection(self) -> None:
        n_samps = 200
        valid = (jnp.arange(n_samps) % 5 != 0).astype(jnp.float64)
        op, n_dets, _ = self._setup(valid)
        signal = jax.random.normal(_key(21), op.out_structure.shape)
        # corrupting flagged samples must not change the projected coefficients
        corrupt = signal.at[:, valid == 0].add(1e3)
        assert_allclose(op.T.mv(signal), op.T.mv(corrupt), atol=1e-9)

    def test_no_mask_matches_explicit_ones(self) -> None:
        op_none, _, n_samps = self._setup(None)
        op_ones, _, _ = self._setup(jnp.ones(200, dtype=jnp.float64))
        coeffs = jax.random.normal(_key(22), op_none.in_structure.shape)
        assert_allclose(op_none.mv(coeffs), op_ones.mv(coeffs), rtol=1e-12)


# ---------------------------------------------------------------------------
# PerDetectorTemplate per-detector basis (T2P leakage template)
# ---------------------------------------------------------------------------


class TestTemperatureTemplate:
    def test_synthesis_is_per_detector_scaled_template(self) -> None:
        # each detector's basis is its own temperature stream -> mv(lambda) = lambda * T_d
        n_dets, n_samps = 4, 200
        T = jax.random.normal(_key(40), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        assert op.in_structure.shape == (n_dets, 1)  # one amplitude per detector
        lam = jax.random.normal(_key(41), op.in_structure.shape)
        assert_allclose(op.mv(lam), lam * T, rtol=1e-12)

    def test_projection_is_per_detector_inner_product(self) -> None:
        n_dets, n_samps = 4, 200
        T = jax.random.normal(_key(42), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        x = jax.random.normal(_key(43), op.out_structure.shape)
        assert_allclose(op.T.mv(x), jnp.sum(T * x, axis=-1)[:, None], rtol=1e-11)

    def test_adjoint(self) -> None:
        n_dets, n_samps = 4, 200
        T = jax.random.normal(_key(44), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64)
        lam = jax.random.normal(_key(45), op.in_structure.shape)
        x = jax.random.normal(_key(46), op.out_structure.shape)
        assert float(jnp.abs(jnp.vdot(op.mv(lam), x) - jnp.vdot(lam, op.T.mv(x)))) < 1e-9

    def test_scan_stacking_over_observations(self) -> None:
        # the per-detector basis values gain an obs axis under lax.scan; one program
        n_obs, n_dets, n_samps = 3, 4, 200
        Ts = jax.random.normal(_key(47), (n_obs, n_dets, n_samps))
        _, stack = jax.lax.scan(
            lambda _, Ti: (None, PerDetectorTemplate.temperature(Ti, jnp.float64)), None, Ts
        )
        op0 = jax.tree.map(lambda leaf: leaf[0], stack)
        lam = jax.random.normal(_key(48), op0.in_structure.shape)
        assert_allclose(op0.mv(lam), lam * Ts[0], rtol=1e-12)

    def test_fit_band_limits_the_basis_to_the_band(self) -> None:
        # fit_band band-passes the temperature basis: synthesis lives only in (f0, f1)
        n_dets, n_samps, fs = 2, 1024, 10.0
        f0, f1 = 0.5, 2.0
        T = jax.random.normal(_key(49), (n_dets, n_samps))
        op = PerDetectorTemplate.temperature(T, jnp.float64, fit_band=(f0, f1), sample_rate=fs)
        lam = jnp.ones((n_dets, 1))
        out = op.mv(lam)  # = bandpass(T) since lambda = 1
        spec = jnp.abs(jnp.fft.rfft(out, axis=-1))
        freqs = jnp.fft.rfftfreq(n_samps, d=1.0 / fs)
        out_of_band = (freqs <= f0) | (freqs >= f1)
        assert float(jnp.max(spec[:, out_of_band])) < 1e-9  # power confined to the band

    def test_none_leg_has_no_amplitudes_and_zero_output(self) -> None:
        n_dets, n_samps = 4, 200
        empty = PerDetectorTemplate.none(n_dets, n_samps, jnp.float64)
        assert empty.in_structure.shape == (n_dets, 0)  # no amplitudes to fit
        out = empty.mv(jnp.zeros(empty.in_structure.shape))
        assert out.shape == (n_dets, n_samps)
        assert_allclose(out, 0.0, atol=0.0)
