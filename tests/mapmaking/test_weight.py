import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator, HomothetyOperator, MaskOperator, SymmetricBandToeplitzOperator
from furax.core import DenseBlockDiagonalOperator
from furax.mapmaking.config import NestedConfig
from furax.mapmaking.templates import Template, TemplateOperator
from furax.mapmaking.templates.bases import TensorBasis
from furax.mapmaking.templates.gram import factor, template_gram
from furax.mapmaking.weight import MarginalizedWeightOperator, NestedWeightOperator, _chebyshev
from furax.obs import TODFormat

# A banded SPD inverse-noise; the covariance N is its dense inverse.
NINV_BAND = np.array([2.5, -1.0, 0.2])


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


def _nested_weight(good: np.ndarray, config: NestedConfig) -> NestedWeightOperator:
    """The nested weight of the banded noise, over the good samples `good`, `(det, samp)`."""
    struct = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    white = HomothetyOperator(NINV_BAND[0], in_structure=struct)  # the zero lag of N⁻¹
    return NestedWeightOperator.create(_inv_noise_op(*good.shape), mask, config, white)


def _gaps_of_every_length(ndet: int, n: int) -> np.ndarray:
    """Gaps from one sample to much longer than the correlation, in each channel."""
    good = np.ones((ndet, n), bool)
    for start, length in [(5, 1), (20, 5), (40, 9), (70, 30)]:
        good[:, start : start + length] = False
    return good


@pytest.mark.parametrize('ndet', [1, 3])
def test_nested_weight_matches_w_exact(ndet: int) -> None:
    n = 120
    ninv_mat = _band_toeplitz(NINV_BAND, n)
    cov = np.linalg.inv(ninv_mat)  # N = (N⁻¹)⁻¹
    good = _gappy_mask(ndet, n) & _gaps_of_every_length(ndet, n)
    weight = _nested_weight(good, NestedConfig(inner_steps=60))

    x = np.asarray(jax.random.normal(jax.random.key(0), (ndet, n), dtype=jnp.float64))
    out = np.asarray(weight(jnp.asarray(x)))

    ref = np.stack([_w_exact(cov, good[d]) @ x[d] for d in range(ndet)])
    assert_allclose(out, ref, rtol=1e-6, atol=1e-8)

    # Discriminating: the cheap inner-mask weight Z N⁻¹ Z is *not* W_exact.
    for d in range(ndet):
        z = np.diag(good[d].astype(float))
        assert not np.allclose(z @ ninv_mat @ z, _w_exact(cov, good[d]), atol=1e-3)


def test_fully_masked_contributes_zero() -> None:
    """A fully-flagged (gated) observation contributes nothing."""
    n = 64
    weight = _nested_weight(np.zeros((1, n), bool), NestedConfig(inner_steps=10))
    x = jax.random.normal(jax.random.key(2), (1, n), dtype=jnp.float64)
    assert_allclose(np.asarray(weight(x)), 0.0, atol=1e-12)


@pytest.mark.parametrize('inner_steps', [0, 1, 4])
def test_weight_lies_between_exact_and_inner_mask(inner_steps: int) -> None:
    """At any number of steps, M N⁻¹ M ⪰ W ⪰ W_exact, and W is symmetric."""
    n = 120
    ninv_mat = _band_toeplitz(NINV_BAND, n)
    good = _gaps_of_every_length(1, n)
    weight = _nested_weight(good, NestedConfig(inner_steps=inner_steps))

    matrix = np.asarray(weight.as_matrix())
    assert_allclose(matrix, matrix.T, atol=1e-12)
    z = np.diag(good[0].astype(float))
    assert np.linalg.eigvalsh(z @ ninv_mat @ z - matrix).min() > -1e-10
    assert np.linalg.eigvalsh(matrix - _w_exact(np.linalg.inv(ninv_mat), good[0])).min() > -1e-10


@pytest.mark.parametrize('steps', [0, 1, 6])
def test_residual_polynomial(steps: int) -> None:
    """On eigenvalues in, below and at the top of the interval, `1 − λ p(λ)` is the shifted
    Chebyshev residual, between 0 and 1."""
    lower, upper = 0.5, 4.0
    eigenvalues = jnp.array([1e-3, 0.1, 0.5, 1.3, 2.7, 4.0])
    p = _chebyshev(DiagonalOperator(eigenvalues), jnp.ones(6), jnp.array([lower, upper]), steps)
    residual = np.asarray(1 - eigenvalues * p)

    k = steps + 1
    z = (upper + lower - 2 * np.asarray(eigenvalues)) / (upper - lower)
    inside = np.cos(k * np.arccos(np.clip(z, -1, 1)))
    t = np.where(np.abs(z) <= 1, inside, np.cosh(k * np.arccosh(np.maximum(np.abs(z), 1))))
    t0 = np.cosh(k * np.arccosh((upper + lower) / (upper - lower)))
    assert_allclose(residual, (1 + t) / (1 + t0), atol=1e-12)
    assert np.all(residual >= 0) and np.all(residual < 1)


def test_preconditioned_solve_converges() -> None:
    """With the preconditioned spectrum inside the interval, it approaches the inverse."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    scale = rng.uniform(1, 10, 8)
    # A = S^½ Q diag(λ) Qᵀ S^½, preconditioned by P = S⁻¹: P A has the eigenvalues λ
    root = np.sqrt(scale)
    matrix = root[:, None] * (q * np.linspace(1, 3, 8)) @ q.T * root[None, :]
    A = DenseBlockDiagonalOperator(
        jnp.asarray(matrix), in_structure=jax.ShapeDtypeStruct((8,), jnp.float64)
    )
    b = jnp.asarray(rng.normal(size=8))
    x = _chebyshev(A, b, jnp.array([1.0, 3.0]), 30, DiagonalOperator(jnp.asarray(1 / scale)))
    assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-10)


@pytest.mark.parametrize('inner_steps', [2, 20])
def test_spectrum_bounds_that_of_the_preconditioned_inner_block(inner_steps: int) -> None:
    """The upper end holds every eigenvalue of `P A` over the live channels, and is nearly reached;
    the lower end is the smallest response, raised to what the steps resolve, at most half the upper.

    The channels have white levels of their own, and the last one, of level zero, is dead.
    """
    n = 120
    good = _gaps_of_every_length(3, n)
    levels = np.array([1.0, 2.0, 0.0]) * NINV_BAND[0]
    struct = jax.ShapeDtypeStruct(good.shape, jnp.float64)
    mask = MaskOperator.from_boolean_mask(jnp.asarray(good), in_structure=struct)
    white = DiagonalOperator(
        jnp.broadcast_to(jnp.asarray(levels)[:, None], good.shape), in_structure=struct
    )
    config = NestedConfig(inner_steps=inner_steps)
    weight = NestedWeightOperator.create(_inv_noise_op(3, n), mask, config, white)
    lower, upper = np.asarray(weight.spectrum)

    flagged = ~good[0]
    inner = np.linalg.eigvalsh(_band_toeplitz(NINV_BAND, n)[np.ix_(flagged, flagged)])
    largest = inner.max() / levels[0]  # the live channel of the lowest level
    assert largest <= upper < 1.1 * largest  # a 30-sample gap reaches it
    smallest_response = NINV_BAND[0] + 2 * NINV_BAND[1:].sum()  # at zero frequency, for this band
    raised = max(smallest_response / levels[1], upper * (3 / inner_steps) ** 2)
    assert_allclose(lower, min(raised, upper / 2), rtol=1e-12)


def test_marginalized_weight_is_blind_to_its_templates():
    tod_format = TODFormat((2, 16), jnp.float64)
    keys = jax.random.split(jax.random.key(0), 2)
    basis = TensorBasis(jax.random.normal(keys[0], (3, 16)))
    templates = TemplateOperator({'drift': Template(basis, tod_format)})
    W = tod_format.diagonal(jax.random.uniform(keys[1], (1, 2, 16), minval=0.5, maxval=2.0))
    inverse, _ = factor(template_gram(templates, W))
    W_marginal = np.asarray(MarginalizedWeightOperator.create(W, templates, inverse).as_matrix())
    assert_allclose(W_marginal, W_marginal.T, atol=1e-12)
    assert_allclose(W_marginal @ np.asarray(templates.as_matrix()), 0, atol=1e-10)
