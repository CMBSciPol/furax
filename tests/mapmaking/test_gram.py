import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.mapmaking.gram import cross_gram, gram_inverse
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
