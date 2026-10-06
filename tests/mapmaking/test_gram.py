import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.linalg import BandedCholeskyOperator
from furax.mapmaking.gram import _BorderedGramInverse, cross_gram, gram_inverse
from furax.mapmaking.templates import (
    KroneckerBasis,
    SegmentedBasis,
    StokesTemplateOperator,
    TemplateOperator,
    TensorBasis,
    WindowedBasis,
)

N_DETS = 3
N_SAMPS = 64


def _template(key, k):
    values = jr.normal(key, (k, N_SAMPS))
    return TemplateOperator({'t': TensorBasis(values)}, n_dets=N_DETS)


def _weight(key):
    w = jr.uniform(key, (N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
    return DiagonalOperator(w, in_structure=jax.ShapeDtypeStruct((N_DETS, N_SAMPS), w.dtype))


def _segmented_template(key, n_seg, k):
    # contiguous partition of the samples into n_seg equal segments
    segment = jnp.repeat(jnp.arange(n_seg), N_SAMPS // n_seg).astype(jnp.int32)
    values = jr.normal(key, (k, N_SAMPS))
    basis = SegmentedBasis(segment, values, n_seg)
    return TemplateOperator({'t': basis}, n_dets=N_DETS)


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


def _per_det_template(key, k):
    # per-detector basis (as in T2P) -> no shared structured Gram.
    basis = TensorBasis.per_detector_stack(values=jr.normal(key, (N_DETS, k, N_SAMPS)))
    return TemplateOperator({'t': basis}, n_dets=N_DETS)


@pytest.mark.parametrize('stokes', [None, 'IQU'])
def test_probed_gram_inverse_inverts_the_gram(stokes):
    # a per-detector basis takes the probe path; with a Stokes axis it sits on q and u only, each
    # leg weighted differently
    kt, kw, ka = jr.split(jr.key(7), 3)
    basis = TensorBasis.per_detector_stack(values=jr.normal(kt, (N_DETS, 2, N_SAMPS)))
    if stokes is None:
        T = TemplateOperator({'t': basis}, n_dets=N_DETS)
        W = _weight(kw)
    else:
        T = StokesTemplateOperator({'t': {'qu': basis}}, N_DETS, stokes)
        w = jr.uniform(kw, (3, N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
        W = DiagonalOperator(w, in_structure=T.out_structure)

    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    gram = T.T @ W @ T
    recovered = gram_inverse(T, W, allow_probe=True)(gram(amps))
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-8, atol=1e-10), recovered, amps)


@pytest.mark.parametrize('n_seg,k', [(4, 3), (8, 1)])
def test_gram_inverse_matches_dense_probe_segmented(n_seg, k):
    # The structured (block-per-segment) inverse Gram must act identically to the dense
    # column-probe fallback, which lumps the segment axis into the coupled index.
    kt, kw, ka = jr.split(jr.key(5), 3)
    T = _segmented_template(kt, n_seg, k)
    W = _weight(kw)

    dense = gram_inverse(T, W, allow_probe=True)  # K = n_seg * k probes
    structured = gram_inverse(T, W)  # segmented + diagonal weight -> fast path taken

    amps = {'t': jr.normal(ka, T.in_structure['t'].shape)}  # (N_DETS, n_seg, k)
    assert_allclose(structured(amps)['t'], dense(amps)['t'], rtol=1e-5, atol=1e-6)


def test_gram_inverse_dense_tensorbasis_matches_and_raises_without_dense_probe():
    kt, kw, ka = jr.split(jr.key(6), 3)
    W = _weight(kw)

    # Dense TensorBasis (block_ndim=0, one k×k block per detector) is supported and matches.
    T = _template(kt, 4)
    structured = gram_inverse(T, W)
    amps = {'t': jr.normal(ka, T.in_structure['t'].shape)}
    assert_allclose(
        structured(amps)['t'],
        gram_inverse(T, W, allow_probe=True)(amps)['t'],
        rtol=1e-5,
        atol=1e-6,
    )

    # A per-detector basis has no shared structured Gram: raises by
    # default (the O(K) probe is never used silently on the implicit path) but works when
    # allow_probe=True is passed explicitly (the explicit/small-K path).
    per_det = _per_det_template(kt, 2)
    with pytest.raises(NotImplementedError, match='structured Gram construction not possible'):
        gram_inverse(per_det, W)
    probed = gram_inverse(per_det, W, allow_probe=True)
    amps_pd = {'t': jr.normal(ka, per_det.in_structure['t'].shape)}
    assert jnp.all(jnp.isfinite(probed(amps_pd)['t']))


def test_gram_inverse_kronecker_matches_dense_probe():
    # KroneckerBasis (the azimuth_hwp_synchronous / binned_azimuth_hwp_synchronous templates):
    # dense Gram over the flattened product index.
    kf0, kf1, kw, ka = jr.split(jr.key(13), 4)
    d0, d1 = 3, 4
    basis = KroneckerBasis((jr.normal(kf0, (d0, N_SAMPS)), jr.normal(kf1, (d1, N_SAMPS))))
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    W = _weight(kw)

    structured = gram_inverse(T, W)
    amps = {'t': jr.normal(ka, T.in_structure['t'].shape)}  # (N_DETS, d0, d1)
    dense = gram_inverse(T, W, allow_probe=True)
    assert_allclose(structured(amps)['t'], dense(amps)['t'], rtol=1e-4, atol=1e-5)


def test_gram_inverse_windowed_matches_dense_probe():
    # Full banded path as an operator (WindowedBasis.gram -> block-banded Cholesky ->
    # block-triangular solve) must act like the dense column-probe inverse.
    ko, kb, ks, kw, ka = jr.split(jr.key(12), 5)
    n_blocks, k, O = 6, 2, 3
    offset = jnp.sort(jr.uniform(ko, (N_SAMPS,)) * (n_blocks - O + 1)).astype(jnp.int32)
    basis = WindowedBasis(
        offset, jr.normal(kb, (O, N_SAMPS)), jr.normal(ks, (k, N_SAMPS)), n_blocks
    )
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    W = _weight(kw)

    structured = gram_inverse(T, W)
    amps = {'t': jr.normal(ka, T.in_structure['t'].shape)}  # (N_DETS, n_blocks, k)
    dense = gram_inverse(T, W, allow_probe=True)
    assert_allclose(structured(amps)['t'], dense(amps)['t'], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('stokes', [None, 'IQU'])
def test_coupled_gram_inverse_matches_dense_probe(stokes):
    # several templates on one stream couple into a joint Gram block; with a Stokes axis each leg
    # gets its own block over the templates it carries, here all three legs for 'poly' and only q
    # and u for 'hwp', each leg weighted differently
    kp, kh, kw, ka = jr.split(jr.key(14), 4)
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    poly = SegmentedBasis(segment, jr.normal(kp, (2, N_SAMPS)), 4)
    hwp = TensorBasis(jr.normal(kh, (3, N_SAMPS)))
    if stokes is None:
        T = TemplateOperator({'poly': poly, 'hwp': hwp}, n_dets=N_DETS)
        W = _weight(kw)
    else:
        T = StokesTemplateOperator({'poly': {'iqu': poly}, 'hwp': {'qu': hwp}}, N_DETS, stokes)
        w = jr.uniform(kw, (3, N_DETS, N_SAMPS), minval=0.5, maxval=2.0)
        W = DiagonalOperator(w, in_structure=T.out_structure)

    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    structured = gram_inverse(T, W)(amps)
    dense = gram_inverse(T, W, allow_probe=True)(amps)
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-4, atol=1e-5), structured, dense)


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
def test_stream_gram_inverse_matches_dense_probe(names, inverse_class):
    # the time-local template with the most amplitudes keeps its band structure, the others
    # bordering it; with no time-local template the few amplitudes share one dense block. Either
    # way the inverse must act as the dense column-probe inverse.
    kb, kw, ka = jr.split(jr.key(15), 3)
    bases = _local_and_global_bases(kb)
    T = TemplateOperator({name: bases[name] for name in names}, n_dets=N_DETS)
    W = _weight(kw)

    structured = gram_inverse(T, W)
    assert type(structured) is inverse_class
    amps = jax.tree.map(lambda s: jr.normal(ka, s.shape), T.in_structure)
    dense = gram_inverse(T, W, allow_probe=True)(amps)
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-8, atol=1e-10), structured(amps), dense)


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
