import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax import MaskOperator, SymmetricBandToeplitzOperator
from furax.mapmaking.config import NestedConfig
from furax.mapmaking.weight import NestedWeightOperator, WeightOperator

# A banded SPD inverse-noise; the covariance N is its dense inverse.
NINV_BAND = np.array([2.5, -1.0, -0.3])


def _band_toeplitz(band: np.ndarray, n: int) -> np.ndarray:
    """Dense symmetric banded Toeplitz matrix from its first column band."""
    idx = np.arange(n)
    lag = np.abs(idx[:, None] - idx[None, :])
    m = np.zeros((n, n))
    for k, b in enumerate(band):
        m[lag == k] = b
    return m


def _w_exact(cov: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Dense reference W_exact = Pᵀ (P N Pᵀ)⁻¹ P for a 1-D boolean good-sample mask."""
    p = np.eye(good.size)[good]
    return p.T @ np.linalg.inv(p @ cov @ p.T) @ p


def _gappy_mask(ndet: int, n: int, *, burst: int = 6, n_bursts: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    good = np.ones((ndet, n), bool)
    for d in range(ndet):
        for s in rng.integers(0, n - burst, size=n_bursts):
            good[d, s : s + burst] = False
    return good


def _inv_noise_op(ndet: int, n: int) -> SymmetricBandToeplitzOperator:
    struct = jax.ShapeDtypeStruct((ndet, n), jnp.float64)
    band = jnp.broadcast_to(jnp.asarray(NINV_BAND), (ndet, NINV_BAND.size))
    return SymmetricBandToeplitzOperator(band, in_structure=struct)


@pytest.mark.parametrize('ndet', [1, 3])
def test_nested_weight_matches_w_exact(ndet: int) -> None:
    n = 120
    ninv_mat = _band_toeplitz(NINV_BAND, n)
    cov = np.linalg.inv(ninv_mat)  # N = (N⁻¹)⁻¹
    good = _gappy_mask(ndet, n)
    struct = jax.ShapeDtypeStruct((ndet, n), jnp.float64)

    inv_noise = _inv_noise_op(ndet, n)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    # Budget comfortably above the flag count -> Woodbury path; fixed iterations (rtol=atol=0).
    config = NestedConfig(max_flag_fraction=0.5, inner_steps=200, rtol=0.0, atol=0.0)
    weight = NestedWeightOperator.create(inv_noise, mask, config)

    x = np.asarray(jax.random.normal(jax.random.key(0), (ndet, n), dtype=jnp.float64))
    out = np.asarray(weight(jnp.asarray(x)))

    ref = np.stack([_w_exact(cov, good[d]) @ x[d] for d in range(ndet)])
    assert_allclose(out, ref, rtol=1e-6, atol=1e-8)

    # Discriminating: the cheap inner-mask weight Z N⁻¹ Z is *not* W_exact.
    for d in range(ndet):
        z = np.diag(good[d].astype(float))
        assert not np.allclose(z @ ninv_mat @ z, _w_exact(cov, good[d]), atol=1e-3)


@pytest.mark.parametrize('ndet', [1, 3])
def test_over_budget_falls_back_to_inner_mask(ndet: int) -> None:
    """When the flag count exceeds n_flag_max, mv returns Z N⁻¹ Z (unbiased), not W_exact."""
    n = 120
    ninv_mat = _band_toeplitz(NINV_BAND, n)
    cov = np.linalg.inv(ninv_mat)
    good = _gappy_mask(ndet, n)
    struct = jax.ShapeDtypeStruct((ndet, n), jnp.float64)

    inv_noise = _inv_noise_op(ndet, n)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    # Budget below the flag count -> fallback branch.
    config = NestedConfig(max_flag_fraction=0.01, inner_steps=50)
    weight = NestedWeightOperator.create(inv_noise, mask, config)

    x = np.asarray(jax.random.normal(jax.random.key(1), (ndet, n), dtype=jnp.float64))
    out = np.asarray(weight(jnp.asarray(x)))

    inner_mask_ref = np.stack(
        [np.diag(good[d]) @ ninv_mat @ np.diag(good[d]) @ x[d] for d in range(ndet)]
    )
    assert_allclose(out, inner_mask_ref, rtol=1e-10, atol=1e-12)
    w_exact_ref = np.stack([_w_exact(cov, good[d]) @ x[d] for d in range(ndet)])
    assert not np.allclose(out, w_exact_ref, atol=1e-3)


def test_fully_masked_contributes_zero() -> None:
    """A fully-flagged (gated) observation exceeds any budget -> Z N⁻¹ Z = 0."""
    n = 64
    struct = jax.ShapeDtypeStruct((1, n), jnp.float64)
    inv_noise = _inv_noise_op(1, n)
    mask = MaskOperator.from_boolean_mask(jnp.zeros((1, n), bool), in_structure=struct)
    weight = NestedWeightOperator.create(
        inv_noise, mask, NestedConfig(max_flag_fraction=0.2, inner_steps=10)
    )
    x = jax.random.normal(jax.random.key(2), (1, n), dtype=jnp.float64)
    assert_allclose(np.asarray(weight(x)), 0.0, atol=1e-12)


def test_nested_weight_is_symmetric() -> None:
    n = 64
    struct = jax.ShapeDtypeStruct((1, n), jnp.float64)
    inv_noise = _inv_noise_op(1, n)
    good = np.ones((1, n), bool)
    good[0, 20:26] = good[0, 40:44] = False
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    weight = NestedWeightOperator.create(
        inv_noise, mask, NestedConfig(max_flag_fraction=0.5, inner_steps=200)
    )

    assert weight.is_symmetric
    assert weight.T is weight
    matrix = np.asarray(weight.as_matrix())
    assert_allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8)
    # Flagged rows/cols are zeroed automatically (Woodbury touches data only through good samples).
    assert_allclose(matrix[~good.ravel()], 0.0, atol=1e-8)


def test_with_mask_rebuilds_weight_only() -> None:
    n = 32
    struct = jax.ShapeDtypeStruct((1, n), jnp.float64)
    inv_noise = _inv_noise_op(1, n)
    good = np.ones((1, n), bool)
    good[0, 10:14] = False
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    weight = NestedWeightOperator.create(
        inv_noise, mask, NestedConfig(max_flag_fraction=0.5, inner_steps=50)
    )

    new_good = np.ones((1, n), bool)
    new_good[0, 5:9] = False
    new_mask = MaskOperator.from_boolean_mask(jnp.asarray(new_good), in_structure=struct)
    rebuilt = weight.with_mask(new_mask)

    assert isinstance(rebuilt, NestedWeightOperator)
    assert rebuilt.n_flag_max == weight.n_flag_max
    assert rebuilt.max_flag_fraction == weight.max_flag_fraction
    assert rebuilt.inner_steps == weight.inner_steps
    assert rebuilt.ninv is weight.ninv
    assert_allclose(np.asarray(rebuilt.mask.to_boolean_mask()), new_good)


def test_with_mask_reresolves_n_flag_max() -> None:
    """with_mask resolves n_flag_max from the *new* mask's size, not by copying the old budget."""
    struct32 = jax.ShapeDtypeStruct((1, 32), jnp.float64)
    mask32 = MaskOperator.from_boolean_mask(jnp.ones((1, 32), bool), in_structure=struct32)
    weight = NestedWeightOperator.create(
        _inv_noise_op(1, 32), mask32, NestedConfig(max_flag_fraction=0.5, inner_steps=10)
    )
    assert weight.n_flag_max == 16  # ceil(0.5 * 32)

    # A new mask over a larger TOD must grow the budget (only the budget field is inspected here).
    struct64 = jax.ShapeDtypeStruct((1, 64), jnp.float64)
    mask64 = MaskOperator.from_boolean_mask(jnp.ones((1, 64), bool), in_structure=struct64)
    rebuilt = weight.with_mask(mask64)
    assert rebuilt.max_flag_fraction == weight.max_flag_fraction
    assert rebuilt.n_flag_max == 32  # ceil(0.5 * 64), re-resolved from mask64


def test_weight_operator_with_mask() -> None:
    """The plain inner-mask weight gained a matching ``with_mask`` for the polymorphic Z setter."""
    n = 16
    struct = jax.ShapeDtypeStruct((n,), jnp.float64)
    inv_noise = SymmetricBandToeplitzOperator(jnp.array([2.0, -0.9]), in_structure=struct)
    good = np.ones(n, bool)
    good[4:6] = False
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    weight = WeightOperator.create(inv_noise, mask)

    new_good = np.ones(n, bool)
    new_good[8:10] = False
    new_mask = MaskOperator.from_boolean_mask(jnp.asarray(new_good), in_structure=struct)
    rebuilt = weight.with_mask(new_mask)

    assert isinstance(rebuilt, WeightOperator)
    assert rebuilt.weight is weight.weight
    assert_allclose(np.asarray(rebuilt.mask.to_boolean_mask()), new_good)
