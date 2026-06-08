import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_allclose

from furax.core import DiagonalOperator
from furax.mapmaking._marginalize import marginal_weight
from furax.mapmaking.basis_templates import PerDetectorTemplate, TensorBasis
from furax.tree import as_structure


def _key(seed: int) -> Array:
    return jax.random.PRNGKey(seed)


def _weight(n_dets: int, n_samps: int, seed: int) -> DiagonalOperator:
    # strictly positive per-(det, sample) white-noise weight
    w = jax.random.uniform(_key(seed), (n_dets, n_samps), minval=0.5, maxval=2.0)
    return DiagonalOperator(w, in_structure=as_structure(w))


def _tensor_template(n_dets: int, n_samps: int, k: int, seed: int) -> PerDetectorTemplate:
    # shared dense basis (e.g. scan-synchronous Legendre): coupled block = whole k, no
    # independent basis axis -> lead = (det,).
    values = jax.random.normal(_key(seed), (k, n_samps))
    return PerDetectorTemplate.from_basis(TensorBasis.create(values), n_dets=n_dets)


def _segmented_template(
    n_dets: int, n_samps: int, n_intervals: int, k: int, seed: int
) -> PerDetectorTemplate:
    # per-interval polynomial: one-hot segment partition -> block-diagonal over intervals,
    # lead = (det, interval), coupled block = k.
    times = jnp.linspace(0.0, 1.0, n_samps)
    edges = jnp.linspace(0, n_samps, n_intervals + 1).astype(int)
    intervals = jnp.stack([edges[:-1], edges[1:]], axis=1)
    return PerDetectorTemplate.polynomial(
        max_poly_order=k - 1,
        intervals=intervals,
        times=times,
        n_dets=n_dets,
        dtype=jnp.float64,
    )


def _dense_marginal_weight(W, T_m, x):
    """Reference W_m(x) using a *full* (non-block) Gram solve, validating that the
    block-diagonal inverse equals the dense inverse (i.e. the Gram really is block-diagonal)."""
    amp_struct = T_m.in_structure
    amp0 = jnp.zeros(amp_struct.shape, amp_struct.dtype)
    n = amp0.size

    # dense Gram G = T_mᵀ W T_m by probing every amplitude unit vector
    def g_col(j):
        e = amp0.reshape(-1).at[j].set(1.0).reshape(amp_struct.shape)
        return T_m.T(W(T_m(e))).reshape(-1)

    G = jnp.stack([g_col(j) for j in range(n)], axis=1)  # (n, n), column j = G e_j
    y = T_m.T(W(x)).reshape(-1)
    a = jnp.linalg.solve(G, y).reshape(amp_struct.shape)
    return W(x) - W(T_m(a))


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize(
    'make_template',
    [
        lambda: _tensor_template(4, 60, 3, 10),
        lambda: _segmented_template(4, 60, 5, 3, 11),
        lambda: _segmented_template(3, 48, 4, 1, 12),  # k=1 (T2P-like): 1x1 blocks
    ],
)
def test_projector_kills_template(make_template) -> None:
    """W_m T_m = 0: the marginalised weight annihilates the template subspace."""
    n_dets, n_samps = 4, 60
    T_m = make_template()
    n_dets = T_m.in_structure.shape[0]
    n_samps = T_m.out_structure.shape[-1]
    W = _weight(n_dets, n_samps, seed=20)
    W_m = marginal_weight(W, T_m)

    a = jax.random.normal(_key(30), T_m.in_structure.shape)
    out = W_m(T_m(a))
    assert_allclose(out, jnp.zeros_like(out), atol=1e-8)


@pytest.mark.parametrize(
    'make_template',
    [
        lambda: _tensor_template(4, 60, 3, 10),
        lambda: _segmented_template(4, 60, 5, 3, 11),
    ],
)
def test_matches_dense_reference(make_template) -> None:
    """Block-diagonal Gram inverse reproduces the full dense marginalised weight."""
    T_m = make_template()
    n_dets = T_m.in_structure.shape[0]
    n_samps = T_m.out_structure.shape[-1]
    W = _weight(n_dets, n_samps, seed=21)
    W_m = marginal_weight(W, T_m)

    x = jax.random.normal(_key(31), (n_dets, n_samps))
    assert_allclose(W_m(x), _dense_marginal_weight(W, T_m, x), atol=1e-8, rtol=1e-6)


def test_symmetric() -> None:
    """W_m is symmetric: <u, W_m v> = <v, W_m u>."""
    T_m = _segmented_template(4, 60, 5, 3, 11)
    n_dets, n_samps = 4, 60
    W = _weight(n_dets, n_samps, seed=22)
    W_m = marginal_weight(W, T_m)

    u = jax.random.normal(_key(40), (n_dets, n_samps))
    v = jax.random.normal(_key(41), (n_dets, n_samps))
    assert_allclose(jnp.vdot(u, W_m(v)), jnp.vdot(v, W_m(u)), atol=1e-8, rtol=1e-6)
