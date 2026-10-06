import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import IdentityOperator
from furax.mapmaking.gram import gram_inverse
from furax.mapmaking.templates import (
    KroneckerBasis,
    SegmentedBasis,
    SharedBasis,
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


def test_template_operator_expands_dense_templates_together():
    # dense templates are expanded in one matrix product and the others one by one; the sum must
    # not depend on which is which: shared dense (tensor, kronecker), decimated and per-detector
    # tensors (kept apart), and a segmented basis
    k = jr.split(jr.key(6), 10)
    bases = {
        'tensor': TensorBasis(jr.normal(k[0], (3, N_SAMPS))),
        'kron': KroneckerBasis((jr.normal(k[1], (2, N_SAMPS)), jr.normal(k[2], (3, N_SAMPS)))),
        'coarse': TensorBasis(jr.normal(k[3], (2, N_SAMPS // 4)), q=4, n_full=N_SAMPS),
        'per_det': TensorBasis.per_detector_stack(values=jr.normal(k[4], (N_DETS, 1, N_SAMPS))),
        'poly': SegmentedBasis(_seg(4), jr.normal(k[5], (2, N_SAMPS)), 4),
    }
    T = TemplateOperator(bases, n_dets=N_DETS)
    amps = {
        name: jr.normal(jr.fold_in(k[6], i), s.shape)
        for i, (name, s) in enumerate(T.in_structure.items())
    }
    ref = sum(
        jax.vmap(lambda b, a: b.expand(a), in_axes=(0 if b.per_detector else None, 0))(b, amps[n])
        for n, b in bases.items()
    )
    assert_allclose(T(amps), ref, rtol=1e-12, atol=1e-12)


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
    # a bare basis does not say which legs it covers
    b = TensorBasis(jnp.ones((2, N_SAMPS)))
    with pytest.raises(TypeError, match="template 'poly' needs its bases keyed by Stokes leg"):
        StokesTemplateOperator({'poly': b}, n_dets=N_DETS, stokes='QU')


def test_stokes_template_operator_rejects_a_leg_in_two_groups():
    b = TensorBasis(jnp.ones((2, N_SAMPS)))
    with pytest.raises(ValueError, match=r"template 'p' has legs \['q'\] in several groups"):
        StokesTemplateOperator({'p': {'q': b, 'qu': b}}, n_dets=N_DETS, stokes='QU')


def test_stokes_template_operator_leg_group_acts_as_one_basis_per_leg():
    # a leg group stores one basis for several legs, each keeping its own amplitudes: the operator,
    # its transpose and its Gram inverse must be those of the basis repeated on every leg. Two
    # templates, so the Gram takes the coupled path.
    k = jr.split(jr.key(5), 4)
    poly = SegmentedBasis(_seg(4), jr.normal(k[0], (2, N_SAMPS)), 4)
    leak = TensorBasis(jr.normal(k[1], (1, N_SAMPS)))
    grouped = StokesTemplateOperator(
        {'poly': {'iqu': poly}, 't2p': {'qu': leak}}, n_dets=N_DETS, stokes='IQU'
    )
    per_leg = StokesTemplateOperator(
        {'poly': dict.fromkeys('iqu', poly), 't2p': dict.fromkeys('qu', leak)},
        n_dets=N_DETS,
        stokes='IQU',
    )
    assert grouped.in_structure == per_leg.in_structure

    amps = jax.tree.map(lambda s: jr.normal(k[2], s.shape), per_leg.in_structure)
    assert_allclose(grouped(amps).data, per_leg(amps).data, rtol=1e-12)
    tod = per_leg.out_structure.from_array(jr.normal(k[3], (3, N_DETS, N_SAMPS)))
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-12), grouped.T(tod), per_leg.T(tod))

    weight = IdentityOperator(in_structure=per_leg.out_structure)
    expected = gram_inverse(per_leg, weight)(amps)
    jax.tree.map(
        lambda a, b: assert_allclose(a, b, rtol=1e-10),
        gram_inverse(grouped, weight)(amps),
        expected,
    )


def _shared(key, n_modes=2):
    k = jr.split(key, 2)
    return SharedBasis(
        jr.normal(k[0], (N_DETS, n_modes)), TensorBasis(jr.normal(k[1], (3, N_SAMPS)))
    )


def test_template_operator_with_a_shared_template():
    # a shared template adds its expansion to the per-detector ones, and has one set of amplitudes
    k = jr.split(jr.key(20), 5)
    poly = SegmentedBasis(_seg(4), jr.normal(k[0], (2, N_SAMPS)), 4)
    shared = _shared(k[1])
    T = TemplateOperator({'poly': poly, 'common': shared}, n_dets=N_DETS)
    assert T.in_structure['common'].shape == (2, 3)
    amps = {'poly': jr.normal(k[2], (N_DETS, 4, 2)), 'common': jr.normal(k[3], (2, 3))}
    assert_allclose(T(amps), _expand(poly, amps['poly']) + shared(amps['common']), rtol=1e-12)

    tod = jr.normal(k[4], (N_DETS, N_SAMPS))
    back = T.T(tod)
    rhs = jnp.vdot(amps['poly'], back['poly']) + jnp.vdot(amps['common'], back['common'])
    assert_allclose(jnp.vdot(T(amps), tod), rhs, rtol=1e-12)


def test_stokes_template_operator_with_a_shared_template_on_some_legs():
    k = jr.split(jr.key(21), 4)
    shared = _shared(k[0])
    T = StokesTemplateOperator({'common': {'qu': shared}}, n_dets=N_DETS, stokes='IQU')
    amps = {'common': {'q': jr.normal(k[1], (2, 3)), 'u': jr.normal(k[2], (2, 3))}}
    out = T(amps)
    assert_allclose(out.i, jnp.zeros((N_DETS, N_SAMPS)))
    assert_allclose(out.q, shared(amps['common']['q']), rtol=1e-12)
    assert_allclose(out.u, shared(amps['common']['u']), rtol=1e-12)


def test_template_operator_rejects_a_shared_template_of_other_detectors():
    with pytest.raises(ValueError, match='couples 3 detectors, not 4'):
        TemplateOperator({'common': _shared(jr.key(22))}, n_dets=N_DETS + 1)


def test_template_operator_with_a_shared_template_stacks_under_vmap():
    # stacking gives the couplings a leading observation axis, which must leave the amplitude
    # structure alone
    t0, t1 = (TemplateOperator({'common': _shared(k)}, n_dets=N_DETS) for k in jr.split(jr.key(23)))
    stacked = jax.tree.map(lambda a, b: jnp.stack([a, b]), t0, t1)
    assert stacked.in_structure == t0.in_structure
    amps = {'common': jr.normal(jr.key(24), (2, 2, 3))}
    out = jax.vmap(lambda op, x: op(x))(stacked, amps)
    assert_allclose(out[1], t1({'common': amps['common'][1]}), rtol=1e-12)


def test_gram_inverse_rejects_a_shared_template():
    T = TemplateOperator({'common': _shared(jr.key(25))}, n_dets=N_DETS)
    with pytest.raises(NotImplementedError, match='shared template'):
        gram_inverse(T, IdentityOperator(in_structure=T.out_structure))
