import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.core import BlockRowOperator
from furax.mapmaking._deprojection import build_gram_inverse, deprojector, marginal_weight
from furax.mapmaking.templates import PerDetectorTemplate, TensorBasis

N_DETS = 3
N_SAMPS = 64


def _template(key, k):
    values = jr.normal(key, (k, N_SAMPS))
    return PerDetectorTemplate.from_basis(TensorBasis.create(values), n_dets=N_DETS)


def _weight(key):
    w = jr.uniform(key, (N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
    return DiagonalOperator(w, in_structure=jax.ShapeDtypeStruct((N_DETS, N_SAMPS), w.dtype))


@pytest.mark.parametrize('k', [1, 4])
def test_marginal_weight_annihilates_template(k):
    kt, kw, ka = jr.split(jr.key(0), 3)
    T = _template(kt, k)
    W = _weight(kw)
    Wm = marginal_weight(W, T)

    amps = jr.normal(ka, T.in_structure.shape)
    out = Wm(T(amps))
    assert_allclose(out, jnp.zeros_like(out), atol=1e-9)


def test_deprojection_is_idempotent_projector():
    kt, kw, kx = jr.split(jr.key(1), 3)
    T = _template(kt, 4)
    W = _weight(kw)
    P = deprojector(W, T)

    x = jr.normal(kx, (N_DETS, N_SAMPS))
    px = P(x)
    assert_allclose(P(px), px, rtol=1e-6, atol=1e-9)


def test_marginal_weight_symmetric():
    kt, kw, kx, ky = jr.split(jr.key(2), 4)
    T = _template(kt, 4)
    W = _weight(kw)
    Wm = marginal_weight(W, T)

    x = jr.normal(kx, (N_DETS, N_SAMPS))
    y = jr.normal(ky, (N_DETS, N_SAMPS))
    assert_allclose(jnp.vdot(x, Wm(y)), jnp.vdot(y, Wm(x)), rtol=1e-6)


def test_marginal_weight_equals_explicit_fit_residual():
    # W_m d = W (d - T a*) with a* = argmin ||d - T a||^2_W = G^-1 T^T W d (per detector).
    kt, kw, kd = jr.split(jr.key(3), 3)
    T = _template(kt, 4)
    W = _weight(kw)
    Wm = marginal_weight(W, T)

    d = jr.normal(kd, (N_DETS, N_SAMPS))
    # explicit GLS amplitude fit, per detector: a* = (TᵀWT)⁻¹ TᵀW d
    rhs = T.T(W(d))  # (n_dets, k)
    gram_inverse = build_gram_inverse((T.T @ W @ T).reduce(), T.in_structure)
    a_star = gram_inverse(rhs)
    expected = W(d - T(a_star))
    assert_allclose(Wm(d), expected, rtol=1e-6, atol=1e-9)


def test_combined_families_capture_cross_coupling():
    # Two non-orthogonal families combined into one T_m; W_m must annihilate the joint range.
    k1, k2, kw, ka, kb = jr.split(jr.key(4), 5)
    T1 = _template(k1, 3)
    T2 = _template(k2, 2)
    T = BlockRowOperator([T1, T2])
    W = _weight(kw)
    Wm = marginal_weight(W, T)

    a = jr.normal(ka, T1.in_structure.shape)
    b = jr.normal(kb, T2.in_structure.shape)
    tod = T([a, b])
    out = Wm(tod)
    assert_allclose(out, jnp.zeros_like(out), atol=1e-9)
