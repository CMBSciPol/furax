import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.mapmaking.gram import cross_gram, gram_inverse
from furax.mapmaking.pomme import PommeProjectionOperator
from furax.mapmaking.templates import (
    KroneckerBasis,
    SegmentedBasis,
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
    offset = (jr.uniform(key, (N_SAMPS,)) * (n_blocks - O + 1)).astype(jnp.int32)
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
    ],
    ids=['segmented_x_tensor', 'segmented_x_segmented', 'windowed_x_tensor'],
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
    offset = (jr.uniform(ko, (N_SAMPS,)) * (n_blocks - O + 1)).astype(jnp.int32)
    basis = WindowedBasis(
        offset, jr.normal(kb, (O, N_SAMPS)), jr.normal(ks, (k, N_SAMPS)), n_blocks
    )
    T = TemplateOperator({'t': basis}, n_dets=N_DETS)
    W = _weight(kw)

    structured = gram_inverse(T, W)
    amps = {'t': jr.normal(ka, T.in_structure['t'].shape)}  # (N_DETS, n_blocks, k)
    dense = gram_inverse(T, W, allow_probe=True)
    assert_allclose(structured(amps)['t'], dense(amps)['t'], rtol=1e-4, atol=1e-5)


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


def test_pomme_gram_inverse_coupled_matches_filtered_dense():
    # two bases: the dense per-detector joint block gets the Schur correction on every pair
    ka_, kb_, kw, ka = jr.split(jr.key(31), 4)
    n_samps = 73
    a, b = _straddling_segmented(ka_, n_samps), TensorBasis(jr.normal(kb_, (3, n_samps)))
    T = TemplateOperator({'a': a, 'b': b}, n_dets=N_DETS)
    diag, W = _pomme_weight(kw, n_samps)
    amps = {
        'a': jr.normal(ka, T.in_structure['a'].shape),
        'b': jr.normal(ka, T.in_structure['b'].shape),
    }

    actual = gram_inverse(T, W, pomme_tau=TAU)(amps)
    # reference: the joint basis [A | B] as one dense TensorBasis-like matrix
    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    fb = f @ jnp.concatenate([a.as_matrix(), b.as_matrix()], axis=1)
    for d in range(N_DETS):
        gram = fb.T @ (diag[d][:, None] * fb)
        rhs = jnp.concatenate([amps['a'][d].reshape(-1), amps['b'][d].reshape(-1)])
        sol = jnp.linalg.solve(gram, rhs)
        got = jnp.concatenate([actual['a'][d].reshape(-1), actual['b'][d].reshape(-1)])
        assert_allclose(got, sol, rtol=1e-8, atol=1e-10)


def test_pomme_gram_inverse_probe_matches_filtered_dense():
    # per-detector basis: the column probe recovers Tᵀ W F T
    kt, kw, ka = jr.split(jr.key(32), 3)
    n_samps = 73
    T = TemplateOperator(
        {'t': TensorBasis.per_detector_stack(values=jr.normal(kt, (N_DETS, 2, n_samps)))},
        n_dets=N_DETS,
    )
    diag, W = _pomme_weight(kw, n_samps)
    amps = jr.normal(ka, T.in_structure['t'].shape)
    actual = gram_inverse(T, W, pomme_tau=TAU, allow_probe=True)({'t': amps})['t']
    f = PommeProjectionOperator(
        TAU, in_structure=jax.ShapeDtypeStruct((1, n_samps), diag.dtype)
    ).as_matrix()
    for d in range(N_DETS):
        fb = f @ T.bases['t'].values[d].T
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
