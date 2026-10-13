import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator, tree
from furax.linalg import BandedCholeskyOperator
from furax.mapmaking.bases import (
    DecimatedBasis,
    KroneckerBasis,
    SegmentedBasis,
    TensorBasis,
    WindowedBasis,
)
from furax.mapmaking.gram import (
    _BorderedGramInverse,
    _gram_inverse_and_ridge,
    cross_gram,
    gram_inverse,
)
from furax.mapmaking.pomme import PommeProjectionOperator
from furax.mapmaking.templates import StokesTemplateOperator, TemplateOperator

N_DETS = 3
N_SAMPS = 64


def _weight(key):
    w = jr.uniform(key, (N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
    return DiagonalOperator(w, in_structure=jax.ShapeDtypeStruct((N_DETS, N_SAMPS), w.dtype))


def _windowed_basis(key, n_blocks, k, O):
    offset = jnp.sort(jr.uniform(key, (N_SAMPS,)) * (n_blocks - O + 1)).astype(jnp.int32)
    ko, ks = jr.split(jr.fold_in(key, 1))
    return WindowedBasis(offset, jr.normal(ko, (O, N_SAMPS)), jr.normal(ks, (k, N_SAMPS)), n_blocks)


@pytest.mark.parametrize(
    'make_a, make_b',
    [
        # segmented (local) x tensor (global): the cross block is dense in the segment axis
        (
            lambda k: SegmentedBasis(_seg(4), jr.normal(k, (2, N_SAMPS)), 4),
            lambda k: TensorBasis(jr.normal(k, (3, N_SAMPS))),
        ),
        # two segmented (local x local): cross block sparse where segments coincide
        (
            lambda k: SegmentedBasis(_seg(4), jr.normal(k, (2, N_SAMPS)), 4),
            lambda k: SegmentedBasis(_seg(4), jr.normal(k, (2, N_SAMPS)), 4),
        ),
        # windowed (overlapping) x tensor
        (
            lambda k: _windowed_basis(k, n_blocks=6, k=2, O=3),
            lambda k: TensorBasis(jr.normal(k, (3, N_SAMPS))),
        ),
        # two windowed, each sample under several blocks of both
        (
            lambda k: _windowed_basis(k, n_blocks=6, k=2, O=3),
            lambda k: _windowed_basis(k, n_blocks=5, k=3, O=2),
        ),
        # two global bases: a plain matrix product
        (
            lambda k: KroneckerBasis((jr.normal(k, (2, N_SAMPS)), jr.normal(k, (3, N_SAMPS)))),
            lambda k: TensorBasis(jr.normal(k, (4, N_SAMPS))),
        ),
    ],
    ids=[
        'segmented_x_tensor',
        'segmented_x_segmented',
        'windowed_x_tensor',
        'windowed_x_windowed',
        'kronecker_x_tensor',
    ],
)
def test_cross_gram_matches_dense_cross_block(make_a, make_b):
    # cross_gram(A, B, w) == the (A, B) cross block of the dense Gram of [A | B], built from tags.
    ka, kb, kw = jr.split(jr.key(20), 3)
    A, B = make_a(ka), make_b(kb)
    w = jr.uniform(kw, (N_SAMPS,), minval=0.5, maxval=2.0)
    b_a, b_b = A.as_matrix(), B.as_matrix()  # (samp, n_a*k_a), (samp, n_b*k_b)
    assert_allclose(cross_gram(A, B, w), b_a.T @ (w[:, None] * b_b), rtol=1e-5, atol=1e-6)
    # self-Gram consistency: pairwise(A, A) is the dense self-Gram
    assert_allclose(cross_gram(A, A, w), b_a.T @ (w[:, None] * b_a), rtol=1e-5, atol=1e-6)


def _seg(n_seg):
    return jnp.repeat(jnp.arange(n_seg), N_SAMPS // n_seg).astype(jnp.int32)


def _unstructured_basis(key, k, n_samps=N_SAMPS):
    # a two-dimensional TensorBasis index has no structured view: its Gram is probed
    return TensorBasis(jr.normal(key, (k, 1, n_samps)))


def _per_det_basis(key, k, n_samps=N_SAMPS):
    # one basis per detector, as for T2P
    return TensorBasis(jr.normal(key, (N_DETS, k, n_samps)), batch_shape=(N_DETS,))


def _decimated_per_det_basis(key, k, n_samps=N_SAMPS, q=4):
    # one basis per detector on a coarser grid, as for decimated T2P
    values = jr.normal(key, (N_DETS, k, -(-n_samps // q)))
    return DecimatedBasis(TensorBasis(values, batch_shape=(N_DETS,)), q, n_samps)


def _detector_matrix(basis, d):
    """Detector `d`'s basis matrix, `(samp, size)`."""
    if not basis.batch_shape:
        return basis.as_matrix()
    return basis.as_matrix().reshape(N_DETS, basis.n_points, N_DETS, basis.size)[d, :, d]


PER_DET_OR_PROBED = pytest.mark.parametrize(
    'make',
    [_per_det_basis, _decimated_per_det_basis, _unstructured_basis],
    ids=['per_det', 'decimated_per_det', 'probed'],
)


@pytest.mark.parametrize('stokes', [None, 'IQU'])
@PER_DET_OR_PROBED
def test_gram_inverse_inverts_the_gram(make, stokes):
    # per-detector bases take the structured path, a basis with no structured view the probe;
    # with a Stokes axis the basis sits on q and u only, each leg weighted differently
    kt, kw, ka = jr.split(jr.key(7), 3)
    basis = make(kt, 2)
    if stokes is None:
        T = TemplateOperator({'t': basis}, n_dets=N_DETS)
        W = _weight(kw)
    else:
        T = StokesTemplateOperator({'t': {'qu': basis}}, N_DETS, stokes)
        w = jr.uniform(kw, (3, N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
        W = DiagonalOperator(w, in_structure=T.out_structure)

    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    gram = T.T @ W @ T
    recovered = gram_inverse(T, W, allow_probe=make is _unstructured_basis)(gram(amps))
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-8, atol=1e-10), recovered, amps)


def test_gram_inverse_probes_only_when_allowed():
    kt, kw, ka = jr.split(jr.key(6), 3)
    W = _weight(kw)

    # A basis with no structured Gram raises by default (the O(K) probe is never used silently)
    # but works when allow_probe=True is passed explicitly.
    unstructured = TemplateOperator({'t': _unstructured_basis(kt, 2)}, n_dets=N_DETS)
    with pytest.raises(NotImplementedError, match='structured Gram construction not possible'):
        gram_inverse(unstructured, W)
    probed = gram_inverse(unstructured, W, allow_probe=True)
    amps_pd = {'t': jr.normal(ka, unstructured.in_structure['t'].shape)}
    assert jnp.all(jnp.isfinite(probed(amps_pd)['t']))


def _local_and_global_bases(key):
    kw, ks, kt, kf0, kf1 = jr.split(key, 5)
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    return {
        'spline': _windowed_basis(kw, n_blocks=6, k=2, O=3),
        'poly': SegmentedBasis(segment, jr.normal(ks, (2, N_SAMPS)), 4),
        'hwp': TensorBasis(jr.normal(kt, (3, N_SAMPS))),
        'az_hwp': KroneckerBasis((jr.normal(kf0, (2, N_SAMPS)), jr.normal(kf1, (3, N_SAMPS)))),
    }


@pytest.mark.parametrize(
    ('names', 'inverse_class'),
    [
        (('spline', 'hwp', 'az_hwp'), _BorderedGramInverse),
        (('poly', 'hwp'), _BorderedGramInverse),
        (('spline', 'poly', 'hwp'), _BorderedGramInverse),
        (('hwp', 'az_hwp'), BandedCholeskyOperator),
    ],
    ids=['banded-core', 'block-diagonal-core', 'two-local', 'no-local'],
)
def test_stream_gram_inverse_borders_the_time_local_core(names, inverse_class):
    # the time-local template with the most amplitudes keeps its band structure, the others
    # bordering it; with no time-local template the few amplitudes share one dense block
    kb, kw = jr.split(jr.key(15))
    bases = _local_and_global_bases(kb)
    T = TemplateOperator({name: bases[name] for name in names}, n_dets=N_DETS)
    assert type(gram_inverse(T, _weight(kw))) is inverse_class


@pytest.mark.parametrize('names', [('poly', 'hwp'), ('spline', 'poly', 'hwp')])
def test_stream_gram_inverse_survives_unobserved_amplitudes(names):
    # masking the first quarter of the samples leaves a polynomial interval, and a spline knot,
    # seen by no weighted sample: their Gram rows are zero, which must not turn the solve into NaNs
    kb, kw, ka = jr.split(jr.key(16), 3)
    bases = _local_and_global_bases(kb)
    T = TemplateOperator({name: bases[name] for name in names}, n_dets=N_DETS)
    w = jr.uniform(kw, (N_DETS, N_SAMPS), minval=0.5, maxval=2.0).at[:, : N_SAMPS // 4].set(0.0)
    W = DiagonalOperator(w, in_structure=jax.ShapeDtypeStruct((N_DETS, N_SAMPS), w.dtype))
    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gram_inverse(T, W)(amps)))


# ---------------------------------------------------------------------------
# Pomme-filtered Gram: (F B)ᵀ W (F B), assembled by Schur-eliminating the intervals
# ---------------------------------------------------------------------------

TAU = 5


def _pomme_weight(key, n_samps):
    # per-detector constant weight, whole intervals masked (2..4) plus the tail: W and F commute
    n_int = n_samps // TAU
    w = jr.uniform(key, (N_DETS, 1), minval=0.5, maxval=2.0)
    m = jnp.ones((N_DETS, n_samps)).at[:, 2 * TAU : 4 * TAU].set(0.0).at[:, n_int * TAU :].set(0.0)
    diag = w * m
    return diag, DiagonalOperator(diag, in_structure=jax.ShapeDtypeStruct(diag.shape, diag.dtype))


def _pomme_reference(basis, diag, amps):
    """`(FB)ᵀ diag(w) (FB)` inverted densely, per detector, for `basis.as_matrix()`."""
    n_samps = basis.n_points
    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    fb = f @ basis.as_matrix()
    out = []
    for d in range(N_DETS):
        gram = fb.T @ (diag[d][:, None] * fb)
        out.append(jnp.linalg.solve(gram, amps[d].reshape(-1)).reshape(amps[d].shape))
    return jnp.stack(out)


def _aligned_segmented(key, n_samps):
    # segments of 3*TAU samples: every interval sits inside one segment, so the Gram keeps its
    # block-diagonal form and only the diagonal blocks get a correction
    span = 3 * TAU
    n_seg = -(-n_samps // span)  # ceil
    segment = (jnp.arange(n_samps) // span).astype(jnp.int32)
    return SegmentedBasis(segment, jr.normal(key, (2, n_samps)), n_seg)


def _straddling_segmented(key, n_samps):
    # segments of 12 samples do not align with the TAU=5 grid: intervals straddle segments
    segment = jnp.minimum(jnp.arange(n_samps) // 12, 5).astype(jnp.int32)
    return SegmentedBasis(segment, jr.normal(key, (2, n_samps)), 6)


def _straddling_windowed(key, n_samps):
    n_blocks, O = 6, 3
    offset = jnp.minimum(jnp.arange(n_samps) // 9, n_blocks - O).astype(jnp.int32)
    ko, ks = jr.split(key)
    return WindowedBasis(offset, jr.normal(ko, (O, n_samps)), jr.normal(ks, (2, n_samps)), n_blocks)


@pytest.mark.parametrize('n_samps', [70, 73], ids=['no_tail', 'tail'])
@pytest.mark.parametrize(
    'make',
    [
        lambda k, n: TensorBasis(jr.normal(k, (4, n))),
        lambda k, n: KroneckerBasis((jr.normal(k, (2, n)), jr.normal(jr.fold_in(k, 1), (3, n)))),
        lambda k, n: _aligned_segmented(k, n),
        _straddling_segmented,
        _straddling_windowed,
    ],
    ids=['tensor', 'kronecker', 'segmented_aligned', 'segmented_straddling', 'windowed'],
)
def test_pomme_gram_inverse_matches_filtered_dense(make, n_samps):
    kb, kw, ka = jr.split(jr.key(30), 3)
    basis = make(kb, n_samps)
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    diag, W = _pomme_weight(kw, n_samps)
    amps = jr.normal(ka, T.in_structure['t'].shape)

    actual = gram_inverse(T, W, pomme_tau=TAU)({'t': amps})['t']
    assert_allclose(actual, _pomme_reference(basis, diag, amps), rtol=1e-8, atol=1e-10)


def _tensor(key, n_samps):
    # each basis needs a key of its own: draws from one key share their prefix, which would make
    # the joint basis rank-deficient
    return TensorBasis(jr.normal(key, (3, n_samps)))


@pytest.mark.parametrize(
    'make',
    [
        lambda k, n: {'a': _straddling_segmented(k, n), 't': _tensor(jr.fold_in(k, 1), n)},
        lambda k, n: {'a': _straddling_windowed(k, n), 't': _tensor(jr.fold_in(k, 1), n)},
        lambda k, n: {
            'a': _straddling_segmented(k, n),
            'w': _straddling_windowed(jr.fold_in(k, 1), n),
            't': _tensor(jr.fold_in(k, 2), n),
        },
    ],
    ids=['segmented_core', 'windowed_core', 'time_local_in_border'],
)
def test_pomme_gram_inverse_coupled_matches_filtered_dense(make):
    # several templates: a time-local core bordered by the others, every block carrying the Schur
    # correction, the core's from its band layout
    kb, kw, ka = jr.split(jr.key(31), 3)
    n_samps = 73
    bases = make(kb, n_samps)
    T = TemplateOperator(bases, n_dets=N_DETS)
    diag, W = _pomme_weight(kw, n_samps)
    amps = {
        name: jr.normal(jr.fold_in(ka, i), T.in_structure[name].shape)
        for i, name in enumerate(bases)
    }

    actual = gram_inverse(T, W, pomme_tau=TAU)(amps)
    # reference: the joint basis of every template as one dense matrix
    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    fb = f @ jnp.concatenate([basis.as_matrix() for basis in bases.values()], axis=1)
    for d in range(N_DETS):
        gram = fb.T @ (diag[d][:, None] * fb)
        rhs = jnp.concatenate([amps[name][d].reshape(-1) for name in bases])
        got = jnp.concatenate([actual[name][d].reshape(-1) for name in bases])
        assert_allclose(got, jnp.linalg.solve(gram, rhs), rtol=1e-8, atol=1e-10)


@PER_DET_OR_PROBED
def test_pomme_gram_inverse_per_det_and_probed_match_filtered_dense(make):
    # the structured path for per-detector bases, and the column probe, recover Tᵀ W F T
    kt, kw, ka = jr.split(jr.key(32), 3)
    n_samps = 73
    basis = make(kt, 2, n_samps)
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    diag, W = _pomme_weight(kw, n_samps)
    amps = jr.normal(ka, T.in_structure['t'].shape)
    probe = make is _unstructured_basis
    actual = gram_inverse(T, W, pomme_tau=TAU, allow_probe=probe)({'t': amps})['t']
    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    for d in range(N_DETS):
        fb = f @ _detector_matrix(basis, d)
        gram = fb.T @ (diag[d][:, None] * fb)
        assert_allclose(actual[d], jnp.linalg.solve(gram, amps[d]), rtol=1e-8, atol=1e-10)


def test_pomme_marginal_weight_deprojects_templates_and_intervals():
    # W' = WF − WF T G̃⁻¹ Tᵀ F W annihilates both the templates and the interval offsets, even
    # when a template column is constant (killed by F, so the Gram needs the ridge).
    kt, kw, ka, kz = jr.split(jr.key(33), 4)
    n_samps = 73
    n_int = n_samps // TAU
    values = jr.normal(kt, (3, n_samps)).at[0].set(1.0)
    T = TemplateOperator({'t': TensorBasis(values)}, n_dets=N_DETS)
    _, W = _pomme_weight(kw, n_samps)
    F = PommeProjectionOperator(TAU, in_structure=W.in_structure)
    G = gram_inverse(T, W, regularization=1e-10, pomme_tau=TAU)
    WF = (W @ F).reduce()
    W_prime = (WF - WF @ T @ G @ T.T @ WF).reduce()

    template_signal = T({'t': jr.normal(ka, T.in_structure['t'].shape)})
    offsets = jnp.repeat(jr.normal(kz, (N_DETS, n_int)), TAU, axis=1)
    offsets = jnp.pad(offsets, [(0, 0), (0, n_samps - n_int * TAU)])
    scale = jnp.abs(WF(template_signal)).max()
    assert_allclose(W_prime(template_signal) / scale, 0.0, atol=1e-8)
    assert_allclose(W_prime(offsets), 0.0, atol=1e-12)


def test_pomme_gram_inverse_handles_a_fully_masked_block():
    # A segment covered entirely by masked intervals has a zero Gram block, singular on its own.
    # It is replaced by the identity, so its (unconstrained) amplitudes come back unchanged and
    # the other segments are unaffected.
    n_samps = 70
    segment = (jnp.arange(n_samps) // (2 * TAU)).astype(jnp.int32)  # segment 1 = the masked range
    basis = SegmentedBasis(segment, jr.normal(jr.key(35), (2, n_samps)), 7)
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    diag, W = _pomme_weight(jr.key(36), n_samps)
    assert jnp.all(diag[:, 2 * TAU : 4 * TAU] == 0)  # the segment really is fully masked
    amps = jr.normal(jr.key(37), T.in_structure['t'].shape)

    actual = gram_inverse(T, W, pomme_tau=TAU)({'t': amps})['t']
    assert jnp.all(jnp.isfinite(actual))
    assert_allclose(actual[:, 1], amps[:, 1], rtol=1e-12)


def test_pomme_gram_rejects_blocks_shorter_than_an_interval():
    # a 3-sample segment inside a 5-sample interval: the interval touches 3 blocks, more than the
    # widened band (2) can hold
    n_samps = 30
    segment = (jnp.arange(n_samps) // 3).astype(jnp.int32)
    T = TemplateOperator({'t': SegmentedBasis(segment, jnp.ones((1, n_samps)), 10)}, n_dets=N_DETS)
    _, W = _pomme_weight(jr.key(34), n_samps)
    with pytest.raises(Exception, match='spans more template blocks'):
        gram_inverse(T, W, pomme_tau=TAU)


def test_pomme_gram_accepts_a_short_block_reached_only_by_masked_samples():
    # The interval [10, 15) reads segments 0, 1 and 2, as segment 1 is only 3 samples long: one
    # block more than the widened band holds. The interval is masked, so it couples nothing:
    # segment 1 is unconstrained and the other two are solved as usual.
    n_samps = 30
    segment = jnp.searchsorted(jnp.array([11, 14]), jnp.arange(n_samps), side='right')
    basis = SegmentedBasis(segment.astype(jnp.int32), jr.normal(jr.key(38), (1, n_samps)), 3)
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    diag, W = _pomme_weight(jr.key(39), n_samps)  # masks samples 10 to 19
    amps = jr.normal(jr.key(40), T.in_structure['t'].shape)

    actual = gram_inverse(T, W, pomme_tau=TAU)({'t': amps})['t']

    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    fb = (f @ basis.as_matrix())[:, jnp.array([0, 2])]
    for d in range(N_DETS):
        observed = jnp.linalg.solve(fb.T @ (diag[d][:, None] * fb), amps[d, jnp.array([0, 2]), 0])
        assert_allclose(actual[d, jnp.array([0, 2]), 0], observed, rtol=1e-10)
    assert_allclose(actual[:, 1], amps[:, 1], rtol=1e-12)


# ---------------------------------------------------------------------------
# Ridge regularization
# ---------------------------------------------------------------------------

REGULARIZATION = 1e-3


def _ridge_bases(kind, key):
    """One set of bases per structural path of the Gram assembly."""
    k1, k2 = jr.split(key)
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    segmented = SegmentedBasis(segment, jr.normal(k1, (2, N_SAMPS)), 4)
    if kind == 'banded':
        return {'p': segmented}
    if kind == 'dense':
        return {'t': TensorBasis(jr.normal(k1, (3, N_SAMPS)))}
    if kind == 'coupled':  # two global bases share one dense block
        return {
            't': TensorBasis(jr.normal(k1, (3, N_SAMPS))),
            'h': TensorBasis(jr.normal(k2, (2, N_SAMPS))),
        }
    if kind == 'bordered':  # a time-local core bordered by a global template
        return {'p': segmented, 't': TensorBasis(jr.normal(k2, (3, N_SAMPS)))}
    if kind == 'per_det_bordered':  # the same, the border one basis per detector
        return {'p': segmented, 't': _per_det_basis(k2, 2)}
    assert kind == 'probed'
    return {'t': _unstructured_basis(k1, 2)}


@pytest.mark.parametrize(
    'kind', ['banded', 'dense', 'coupled', 'bordered', 'per_det_bordered', 'probed']
)
def test_gram_ridge_is_the_one_added_before_factoring(kind):
    # The explicit normal system adds the returned ridge to the Gram it assembles itself, so the
    # ridge must be exactly the one the inverse was factored from, on every assembly path:
    # applying the inverse to the ridged Gram must give back the amplitudes.
    kb, kw, ka = jr.split(jr.key(40), 3)
    bases = _ridge_bases(kind, kb)
    T = TemplateOperator(bases, n_dets=N_DETS)
    W = _weight(kw)
    inverse, ridge = _gram_inverse_and_ridge(T, W, REGULARIZATION, allow_probe=(kind == 'probed'))
    assert ridge is not None

    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    gram = T.T(W(T(amps)))  # G a, without any ridge
    ridged = tree.add(gram, ridge(amps))
    assert_allclose_tree(inverse(ridged), amps, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('kind', ['banded', 'dense'])
def test_gram_ridge_scales_with_the_data_alone(kind):
    # An amplitude no weighted sample sees gets a unit placeholder on the Gram's diagonal. The
    # ridge must be scaled from the Gram before the placeholder goes in, otherwise it depends on
    # that arbitrary 1: scaling the weights must scale the ridge by the same factor.
    kb, kw = jr.split(jr.key(42))
    if kind == 'banded':  # one sub-basis function vanishes on segment 2
        segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
        values = jr.normal(kb, (2, N_SAMPS)).at[1, segment == 2].set(0.0)
        basis = SegmentedBasis(segment, values, 4)
    else:  # one function vanishes everywhere
        basis = TensorBasis(jr.normal(kb, (3, N_SAMPS)).at[2].set(0.0))
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    w = jr.uniform(kw, (N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
    structure = jax.ShapeDtypeStruct(w.shape, w.dtype)
    ones = tree.ones_like(T.in_structure)

    def ridge_of(weights):
        W = DiagonalOperator(weights, in_structure=structure)
        _, ridge = _gram_inverse_and_ridge(T, W, REGULARIZATION)
        return ridge(ones)['t']

    assert_allclose(ridge_of(1e-3 * w), 1e-3 * ridge_of(w), rtol=1e-12)


@pytest.mark.parametrize('kind', ['banded', 'coupled', 'bordered'])
def test_gram_ridge_is_none_without_regularization(kind):
    kb, kw = jr.split(jr.key(41))
    T = TemplateOperator(_ridge_bases(kind, kb), n_dets=N_DETS)
    inverse, ridge = _gram_inverse_and_ridge(T, _weight(kw), 0.0)
    assert ridge is None
    assert inverse is not None


def assert_allclose_tree(actual, desired, **kwargs):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(desired), strict=True):
        assert_allclose(a, b, **kwargs)
