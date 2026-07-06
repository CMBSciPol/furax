import jax
import jax.numpy as jnp
import jax.random as jr
from numpy.testing import assert_allclose

from furax.mapmaking.templates import SegmentedBasis, TemplateFamily, TemplateOperator, TensorBasis

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


def test_template_operator_forward_modulated():
    # mv/transpose == the sum / per-family split of the equivalent per-detector expand/project.
    k = jr.split(jr.key(0), 4)
    b1 = TensorBasis.create(jr.normal(k[0], (3, N_SAMPS)))
    b2 = SegmentedBasis.create(_seg(4), jr.normal(k[1], (2, N_SAMPS)), 4)
    T = TemplateOperator.create(
        [TemplateFamily('scan', b1), TemplateFamily('poly', b2)], n_dets=N_DETS
    )

    amps = {'scan': jr.normal(k[2], (N_DETS, 3)), 'poly': jr.normal(k[3], (N_DETS, 4, 2))}
    ref = _expand(b1, amps['scan']) + _expand(b2, amps['poly'])
    assert_allclose(T(amps), ref, rtol=1e-5, atol=1e-6)

    tod = jr.normal(jr.key(1), (N_DETS, N_SAMPS))
    got = T.T(tod)
    assert_allclose(got['scan'], _project(b1, tod), rtol=1e-5, atol=1e-6)
    assert_allclose(got['poly'], _project(b2, tod), rtol=1e-5, atol=1e-6)


def test_template_operator_forward_demodulated():
    # Per-Stokes-leg families (polynomial on i/q/u, T2P on q/u only): output is a Stokes pytree,
    # each leg the sum of the families active on it.
    k = jr.split(jr.key(2), 8)
    poly = TemplateFamily(
        'poly',
        {
            leg: SegmentedBasis.create(_seg(4), jr.normal(k[i], (2, N_SAMPS)), 4)
            for i, leg in enumerate('iqu')
        },
    )
    t2p = TemplateFamily(
        't2p',
        {
            'q': TensorBasis.create(jr.normal(k[3], (1, N_SAMPS))),
            'u': TensorBasis.create(jr.normal(k[4], (1, N_SAMPS))),
        },
        explicit=True,
    )
    T = TemplateOperator.create([poly, t2p], n_dets=N_DETS, stokes='IQU')

    amps = {
        'poly': {
            leg: jr.normal(jr.fold_in(k[5], i), (N_DETS, 4, 2)) for i, leg in enumerate('iqu')
        },
        't2p': {leg: jr.normal(jr.fold_in(k[6], i), (N_DETS, 1)) for i, leg in enumerate('qu')},
    }
    out = T(amps)
    for leg in 'iqu':
        ref = _expand(poly.bases[leg], amps['poly'][leg])
        if leg in ('q', 'u'):
            ref = ref + _expand(t2p.bases[leg], amps['t2p'][leg])
        assert_allclose(getattr(out, leg), ref, rtol=1e-5, atol=1e-6)


def test_template_operator_transpose_is_adjoint():
    # <T(amps), tod> == <amps, T.T(tod)> for a multi-family, shared + per-detector-basis mix.
    k = jr.split(jr.key(4), 6)
    shared = TensorBasis.create(jr.normal(k[0], (3, N_SAMPS)))
    per_det = TensorBasis(
        values=jr.normal(k[1], (N_DETS, 2, N_SAMPS)),
        n_full=N_SAMPS,
        in_structure=jax.ShapeDtypeStruct((2,), jnp.float64),
    )
    T = TemplateOperator.create(
        [TemplateFamily('shared', shared), TemplateFamily('per_det', per_det, shared=False)],
        n_dets=N_DETS,
    )
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
        b = SegmentedBasis.create(_seg(4), jr.normal(kk, (2, N_SAMPS)), 4)
        return TemplateOperator.create([TemplateFamily('poly', b)], n_dets=N_DETS)

    t0, t1 = make(k[0]), make(k[1])
    stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), t0, t1)  # leading obs axis on the bases
    amps = {'poly': jr.normal(k[2], (2, N_DETS, 4, 2))}
    out = jax.vmap(lambda op, x: op(x))(stacked, amps)
    assert out.shape == (2, N_DETS, N_SAMPS)
    assert_allclose(out[0], t0({'poly': amps['poly'][0]}), rtol=1e-5, atol=1e-6)
    assert_allclose(out[1], t1({'poly': amps['poly'][1]}), rtol=1e-5, atol=1e-6)
