import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax.mapmaking.templates import (
    SegmentedBasis,
    StokesTemplateOperator,
    TemplateOperator,
    TensorBasis,
)

N_DETS = 3
N_SAMPS = 64


def _seg(n):
    return jnp.repeat(jnp.arange(n), N_SAMPS // n).astype(jnp.int32)


def _expand(basis, a):
    """Per-detector reference: broadcast a shared basis's expand over the detector axis."""
    return jax.vmap(basis.expand)(a)


def _project(basis, s):
    """Per-detector reference: broadcast a shared basis's project over the detector axis."""
    return jax.vmap(basis.project)(s)


def test_template_operator_forward():
    # mv/transpose == the sum / per-template split of the equivalent per-detector
    # expand/project.
    k = jr.split(jr.key(0), 4)
    b1 = TensorBasis(jr.normal(k[0], (3, N_SAMPS)))
    b2 = SegmentedBasis(_seg(4), jr.normal(k[1], (2, N_SAMPS)), 4)
    T = TemplateOperator({'scan': b1, 'poly': b2}, n_dets=N_DETS)

    amps = {'scan': jr.normal(k[2], (N_DETS, 3)), 'poly': jr.normal(k[3], (N_DETS, 4, 2))}
    ref = _expand(b1, amps['scan']) + _expand(b2, amps['poly'])
    assert_allclose(T(amps), ref, rtol=1e-5, atol=1e-6)

    tod = jr.normal(jr.key(1), (N_DETS, N_SAMPS))
    got = T.T(tod)
    assert_allclose(got['scan'], _project(b1, tod), rtol=1e-5, atol=1e-6)
    assert_allclose(got['poly'], _project(b2, tod), rtol=1e-5, atol=1e-6)


def test_stokes_template_operator_forward():
    # Per-Stokes-leg templates (polynomial on i/q/u, T2P on q/u only): output is a Stokes
    # pytree, each leg the sum of the templates enabled on it.
    k = jr.split(jr.key(2), 8)
    poly = {
        leg: SegmentedBasis(_seg(4), jr.normal(k[i], (2, N_SAMPS)), 4)
        for i, leg in enumerate('iqu')
    }
    t2p = {
        'q': TensorBasis(jr.normal(k[3], (1, N_SAMPS))),
        'u': TensorBasis(jr.normal(k[4], (1, N_SAMPS))),
    }
    T = StokesTemplateOperator({'poly': poly, 't2p': t2p}, n_dets=N_DETS, stokes='IQU')

    amps = {
        'poly': {
            leg: jr.normal(jr.fold_in(k[5], i), (N_DETS, 4, 2)) for i, leg in enumerate('iqu')
        },
        't2p': {leg: jr.normal(jr.fold_in(k[6], i), (N_DETS, 1)) for i, leg in enumerate('qu')},
    }
    out = T(amps)
    for leg in 'iqu':
        ref = _expand(poly[leg], amps['poly'][leg])
        if leg in ('q', 'u'):
            ref = ref + _expand(t2p[leg], amps['t2p'][leg])
        assert_allclose(getattr(out, leg), ref, rtol=1e-5, atol=1e-6)


def test_template_operator_transpose_is_adjoint():
    # <T(amps), tod> == <amps, T.T(tod)> for several templates, shared + per-detector mix.
    k = jr.split(jr.key(4), 6)
    shared = TensorBasis(jr.normal(k[0], (3, N_SAMPS)))
    per_det = TensorBasis.per_detector_stack(values=jr.normal(k[1], (N_DETS, 2, N_SAMPS)))
    T = TemplateOperator({'shared': shared, 'per_det': per_det}, n_dets=N_DETS)
    amps = {
        'shared': jr.normal(k[2], (N_DETS, 3)),
        'per_det': jr.normal(k[3], (N_DETS, 2)),
    }
    tod = jr.normal(k[4], (N_DETS, N_SAMPS))
    lhs = jnp.vdot(T(amps), tod)
    back = T.T(tod)
    rhs = jnp.vdot(amps['shared'], back['shared']) + jnp.vdot(amps['per_det'], back['per_det'])
    assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-6)


def test_template_operator_stacks_under_vmap():
    # Only the bases are dynamic, so obs-stacking gains a leading axis and vmaps cleanly — this is
    # exactly how the multi-observation jax.lax.scan applies it.
    k = jr.split(jr.key(3), 3)

    def make(kk):
        b = SegmentedBasis(_seg(4), jr.normal(kk, (2, N_SAMPS)), 4)
        return TemplateOperator({'poly': b}, n_dets=N_DETS)

    t0, t1 = make(k[0]), make(k[1])
    stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), t0, t1)  # leading obs axis on the bases
    amps = {'poly': jr.normal(k[2], (2, N_DETS, 4, 2))}
    out = jax.vmap(lambda op, x: op(x))(stacked, amps)
    assert out.shape == (2, N_DETS, N_SAMPS)
    assert_allclose(out[0], t0({'poly': amps['poly'][0]}), rtol=1e-5, atol=1e-6)
    assert_allclose(out[1], t1({'poly': amps['poly'][1]}), rtol=1e-5, atol=1e-6)


def test_stokes_template_operator_rejects_legs_outside_stokes():
    # `stokes` declares the leg axis once: a template keyed by anything else is caught at
    # construction, rather than as a bare KeyError from inside mv.
    b = TensorBasis(jnp.ones((2, N_SAMPS)))
    with pytest.raises(ValueError, match=r"template 'p' has legs \['i'\] outside stokes='QU'"):
        StokesTemplateOperator({'p': {'i': b}}, n_dets=N_DETS, stokes='QU')


def test_stokes_template_operator_rejects_a_basis_that_is_not_keyed_by_leg():
    # a Stokes-valued operator needs one basis per leg, never a single shared one
    b = TensorBasis(jnp.ones((2, N_SAMPS)))
    with pytest.raises(TypeError, match="template 'poly' needs one basis per Stokes leg"):
        StokesTemplateOperator({'poly': b}, n_dets=N_DETS, stokes='QU')
