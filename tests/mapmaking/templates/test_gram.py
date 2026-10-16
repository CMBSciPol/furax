import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, symmetric, tree
from furax.mapmaking.pomme import PommeProjectionOperator
from furax.mapmaking.templates import Template, TemplateOperator
from furax.mapmaking.templates._gram import BlockGram, factor, template_gram
from furax.mapmaking.templates.bases import OffsetBasis, SegmentedBasis, TensorBasis, _einsum
from furax.obs import TODFormat

N_DETS = 4
N_SAMPS = 64
STOKES = 'IQU'
WAFERS = np.array([0, 0, 1, 1])  # the group of each detector
ALL = np.arange(N_DETS)
TAU = 5  # Pomme intervals, straddling the blocks of 8 samples of the offsets


def _tod_format(dets=ALL):
    return TODFormat((len(dets), N_SAMPS), jnp.float64, stokes=STOKES)


def _local(key, dets=ALL):
    """Templates with a set per detector: a polynomial per interval (time-local), harmonics, and
    a leakage of each detector's own stream."""
    k = jr.split(key, 3)
    tod_format = _tod_format(dets)
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    stream = jr.normal(k[2], (N_DETS, 1, N_SAMPS))[dets]
    return {
        'poly': Template(
            SegmentedBasis(segment, jr.normal(k[0], (2, N_SAMPS)), 4), tod_format, streams='iqu'
        ),
        'hwp': Template(TensorBasis(jr.normal(k[1], (3, N_SAMPS))), tod_format, streams='qu'),
        't2p': Template(TensorBasis(stream, batch_shape=(len(dets),)), tod_format, streams='qu'),
    }


def _global(key, dets=ALL):
    """Templates shared by detectors: a common mode seen with gains (time-local), a field of a
    polarization common to Q and U, and offsets per wafer."""
    k = jr.split(key, 3)
    tod_format = _tod_format(dets)
    offsets = OffsetBasis(8, N_SAMPS, jnp.float64)
    stream = jr.normal(k[1], (N_DETS, 1, N_SAMPS))[dets]
    wafers = jnp.asarray(WAFERS[dets][:, None] == np.arange(2))
    return {
        'common': Template(
            offsets,
            tod_format,
            streams='qu',
            per=('stream',),
            coupling=jr.normal(k[0], (1, N_DETS, 1))[:, dets],
        ),
        'field': Template(
            TensorBasis(stream, batch_shape=(len(dets),)),
            tod_format,
            streams='qu',
            per=(),
            coupling=jr.normal(k[2], (2, N_DETS, 2))[:, dets],
        ),
        'wafer': Template(offsets, tod_format, streams='i', per=('det',), det_groups=wafers),
    }


def _mixed(key, dets=ALL):
    return _local(key, dets) | _global(jr.fold_in(key, 1), dets)


TEMPLATES = {'local': _local, 'global': _global, 'mixed': _mixed}


def _operator(templates, key, dets=ALL):
    return TemplateOperator(TEMPLATES[templates](key, dets))


@symmetric
class _DenseWeight(AbstractLinearOperator):
    """A weight given as a dense matrix over the flattened TOD."""

    matrix: jax.Array

    def mv(self, x):
        flat, unravel = ravel_pytree(x)
        return unravel(self.matrix @ flat)


def _weight(op, key, bandwidth=0, streams=False, dets=ALL):
    """A positive definite weight that couples no two detectors, nor samples more than
    `bandwidth` apart, and couples the streams of a detector if `streams`. Each detector's own block
    depends on that detector only."""
    n_streams = len(op.tod_format.stokes or 'x')  # a single stream unless demodulated
    stream, det, samp = (a.ravel() for a in np.indices((n_streams, N_DETS, N_SAMPS)))
    reach = (det[:, None] == det) & (np.abs(samp[:, None] - samp) <= bandwidth)
    if not streams:
        reach &= stream[:, None] == stream
    a = np.asarray(jr.normal(key, reach.shape)) * reach
    matrix = (a + a.T) / 2
    matrix += np.diag(np.abs(matrix).sum(axis=1) + 1)  # diagonally dominant
    kept = np.isin(det, dets)
    return _DenseWeight(jnp.asarray(matrix[kept][:, kept]), in_structure=op.out_structure)


def _dense_gram(op, weight):
    a = np.asarray(op.as_matrix())
    return a.T @ np.asarray(weight.as_matrix()) @ a


def _dense(gram, op, cross=True):
    """The assembled Gram as a dense matrix over the flattened amplitudes."""
    zeros = tree.zeros_like(op.in_structure)
    flat, unravel = ravel_pytree(zeros)

    def apply(v):
        x = unravel(v)
        out = zeros
        for part in (gram.local, gram.global_):
            if part is not None:
                filled = part.layout.scatter(*part.matrix(part.layout.gather(x)))
                out = tree.add(out, zeros | filled)
        if cross and gram.cross is not None:
            local, global_ = gram.local.layout, gram.global_.layout
            r_local = local.flat(*local.gather(x))
            r_global = global_.flat(*global_.gather(x))[0]
            to_local = _einsum('gdl,g->dl', gram.cross, r_global)
            to_global = _einsum('gdl,dl->g', gram.cross, r_local)
            filled = local.scatter(*local.unflat(to_local))
            filled |= global_.scatter(*global_.unflat(to_global[None]))
            out = tree.add(out, zeros | filled)
        return ravel_pytree(out)[0]

    return np.asarray(jax.vmap(apply, out_axes=1)(jnp.eye(flat.size, dtype=flat.dtype)))


WEIGHTS = {
    'diagonal': {'bandwidth': 0},
    'banded': {'bandwidth': 3},
    'streams_coupled': {'bandwidth': 2, 'streams': True},
}


# The assembly from the structure of the bases is for a diagonal weight only.
CASES = [
    pytest.param(templates, weight, method, id=f'{templates}-{weight}-{method}')
    for templates in TEMPLATES
    for weight in WEIGHTS
    for method in ('probe', 'structure')
    if method == 'probe' or weight == 'diagonal'
]


@pytest.mark.parametrize(('templates', 'weight', 'method'), CASES)
def test_gram_is_the_dense_gram(templates, weight, method):
    op = _operator(templates, jr.key(0))
    W = _weight(op, jr.key(1), **WEIGHTS[weight])
    _check_gram(op, W, WEIGHTS[weight]['bandwidth'], weight == 'streams_coupled', method)


def _check_gram(op, W, bandwidth, streams_coupled, method, batch_size=1, effective=None, **kw):
    """The Gram assembled with and without the local-global coupling, against the dense one, under
    the weight `effective` if given."""
    reach = {'bandwidth': bandwidth, 'streams_coupled': streams_coupled, 'method': method}
    reach |= {'batch_size': batch_size} | kw
    coupled = template_gram(op, W, coupled=True, **reach)
    expected = _dense_gram(op, W if effective is None else effective)
    assert_allclose(_dense(coupled, op), expected, rtol=1e-8, atol=1e-8)

    # the blocks alone, without their coupling
    blocks = template_gram(op, W, **reach)
    assert blocks.cross is None
    assert_allclose(_dense(blocks, op), _dense(coupled, op, cross=False), rtol=1e-8, atol=1e-8)


def _templates_on(tod_format, key):
    """A template of each kind on any TOD: per detector (time-local, and with a basis per
    detector), and shared (a field, a common mode per stream, offsets per wafer)."""
    k = jr.split(key, 5)
    det = tod_format.det_shape
    streams = 'iqu' if tod_format.stokes else None
    stream_axis = (len(streams),) if streams else ()
    same_on_every_stream = (1,) if streams else ()
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    offsets = OffsetBasis(8, N_SAMPS, jnp.float64)
    stream = TensorBasis(jr.normal(k[1], (*det, 1, N_SAMPS)), batch_shape=det)
    wafers = jnp.asarray(WAFERS.reshape(det)[..., None] == np.arange(2))
    return {
        'poly': Template(
            SegmentedBasis(segment, jr.normal(k[0], (2, N_SAMPS)), 4), tod_format, streams
        ),
        't2p': Template(stream, tod_format, streams),
        'field': Template(
            stream,
            tod_format,
            streams,
            per=(),
            coupling=jr.normal(k[2], (*stream_axis, *det, 2)),
        ),
        'common': Template(
            offsets,
            tod_format,
            streams,
            per=('stream',),
            coupling=jr.normal(k[3], (*same_on_every_stream, *det, 1)),
        ),
        'wafer': Template(offsets, tod_format, streams, per=('det',), det_groups=wafers),
    }


@pytest.mark.parametrize('method', ['probe', 'structure'])
@pytest.mark.parametrize(
    ('stokes', 'det_shape'), [('IQU', (2, 2)), (None, (N_DETS,))], ids=['det_axes', 'no_streams']
)
def test_gram_on_other_tods_is_the_dense_gram(stokes, det_shape, method):
    tod_format = TODFormat((*det_shape, N_SAMPS), jnp.float64, stokes=stokes)
    op = TemplateOperator(_templates_on(tod_format, jr.key(14)))
    W = _weight(op, jr.key(15))
    # batches of 3 of the 4 detectors: the last one overlaps the first
    _check_gram(op, W, 0, False, method, batch_size=3)


def _pomme_weight(op, key):
    """A diagonal weight constant on each Pomme interval: one value per detector, two intervals
    and the tail masked."""
    w = np.repeat(np.asarray(jr.uniform(key, (N_DETS, 1), minval=0.5, maxval=2.0)), N_SAMPS, 1)
    w[:, 2 * TAU : 4 * TAU] = 0
    w[:, N_SAMPS // TAU * TAU :] = 0
    return _DenseWeight(jnp.diag(jnp.asarray(w.ravel())), in_structure=op.out_structure)


@pytest.mark.parametrize('method', ['probe', 'structure'])
def test_gram_under_pomme_is_the_dense_gram(method):
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64)
    op = TemplateOperator(_templates_on(tod_format, jr.key(18)))
    W = _pomme_weight(op, jr.key(19))
    F = PommeProjectionOperator(TAU, in_structure=tod_format.structure).as_matrix()
    effective = _DenseWeight(W.matrix @ F, in_structure=op.out_structure)
    _check_gram(op, W, 0, False, method, batch_size=3, effective=effective, pomme_tau=TAU)


def test_pomme_is_for_a_tod_that_is_not_demodulated():
    op = _operator('local', jr.key(20))
    with pytest.raises(ValueError, match='not demodulated'):
        template_gram(op, _weight(op, jr.key(21)), pomme_tau=TAU)


def test_blocks_shorter_than_a_pomme_interval_give_a_nan_gram():
    # blocks of 2 samples: an interval of 5 spans 3 of them, beyond the band
    segment = jnp.arange(N_SAMPS, dtype=jnp.int32) // 2
    basis = SegmentedBasis(segment, jnp.ones((1, N_SAMPS)), N_SAMPS // 2)
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64)
    op = TemplateOperator({'poly': Template(basis, tod_format)})
    gram = template_gram(op, _pomme_weight(op, jr.key(22)), pomme_tau=TAU)
    assert jnp.all(jnp.isnan(gram.local.matrix.bands))


def test_structure_is_for_a_diagonal_weight_only():
    op = _operator('local', jr.key(16))
    W = _weight(op, jr.key(17), bandwidth=1)
    with pytest.raises(ValueError, match='diagonal weight only'):
        template_gram(op, W, bandwidth=1, method='structure')


@pytest.mark.parametrize('method', ['probe', 'structure'])
@pytest.mark.parametrize('coupled', [True, False], ids=['coupled', 'blocks'])
def test_grams_of_detector_batches_combine(coupled, method):
    op = _operator('mixed', jr.key(2))
    W = _weight(op, jr.key(3))
    grams = []
    for dets in (ALL[:3], ALL[3:]):
        batch = _operator('mixed', jr.key(2), dets)
        W_batch = _weight(batch, jr.key(3), dets=dets)
        grams.append(template_gram(batch, W_batch, coupled=coupled, method=method))
    combined = BlockGram.combine(grams)
    expected = template_gram(op, W, coupled=coupled, method=method)
    assert_allclose(_dense(combined, op), _dense(expected, op), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('regularization', [0.0, 0.1], ids=['plain', 'ridge'])
@pytest.mark.parametrize('coupled', [True, False], ids=['exact', 'blocks'])
@pytest.mark.parametrize('templates', list(TEMPLATES))
def test_factor_inverts_the_gram(templates, coupled, regularization):
    op = _operator(templates, jr.key(4))
    W = _weight(op, jr.key(5), bandwidth=2)
    gram = template_gram(op, W, bandwidth=2, coupled=coupled)
    inverse, ridge = factor(gram, regularization=regularization)
    assert (ridge is None) == (regularization == 0)

    x = tree.normal_like(op.in_structure, jr.key(6))
    flat, unravel = ravel_pytree(x)
    gram_x = jnp.asarray(_dense(gram, op)) @ flat
    if ridge is not None:
        gram_x = gram_x + ravel_pytree(ridge(x))[0]
    assert_allclose(ravel_pytree(inverse(unravel(gram_x)))[0], flat, rtol=1e-7, atol=1e-8)


def test_unseen_amplitudes_come_back_unchanged():
    # a wafer whose detectors are all masked out: its offsets no weighted sample sees
    op = _operator('global', jr.key(7))
    W = _weight(op, jr.key(8))
    mask = np.ones((len(STOKES), N_DETS, N_SAMPS))
    mask[:, WAFERS == 1] = 0
    mask = jnp.asarray(mask.ravel())
    W = _DenseWeight(mask[:, None] * W.matrix * mask, in_structure=op.out_structure)
    inverse, _ = factor(template_gram(op, W))

    x = tree.normal_like(op.in_structure, jr.key(9))
    assert_allclose(inverse(x)['wafer'][1], x['wafer'][1])


@pytest.mark.parametrize(
    ('streams_coupled', 'method'),
    [(False, 'probe'), (False, 'structure'), (True, 'probe')],
    ids=['streams_apart-probe', 'streams_apart-structure', 'streams_coupled-probe'],
)
def test_units_split_the_streams_the_weight_does_not_couple(streams_coupled, method):
    # every template here has its own amplitudes on each stream
    templates = _local(jr.key(10)) | {'common': _global(jr.key(11))['common']}
    op = TemplateOperator(templates)
    W = _weight(op, jr.key(12), streams=streams_coupled)
    gram = template_gram(op, W, streams_coupled=streams_coupled, method=method)

    n_streams = 1 if streams_coupled else len(STOKES)
    assert gram.local.matrix.bands.shape[0] == N_DETS * n_streams
    assert gram.global_.matrix.bands.shape[0] == n_streams
    expected = template_gram(op, W, streams_coupled=streams_coupled, coupled=True, method=method)
    assert_allclose(_dense(gram, op), _dense(expected, op, cross=False), rtol=1e-8, atol=1e-8)


def test_blocks_too_short_for_the_probes_are_rejected():
    # three 1-sample intervals: samples 3 apart couple blocks 3 apart, beyond the 2 probed
    segment = jnp.repeat(
        jnp.arange(4), jnp.array([1, 1, 1, N_SAMPS - 3]), total_repeat_length=N_SAMPS
    )
    basis = SegmentedBasis(segment.astype(jnp.int32), jnp.ones((1, N_SAMPS)), 4)
    op = TemplateOperator({'poly': Template(basis, _tod_format(), streams='iqu')})
    W = _weight(op, jr.key(13), bandwidth=3)
    with pytest.raises(ValueError, match='beyond the 2 probed'):
        template_gram(op, W, bandwidth=3)
    traced = jax.jit(lambda op: template_gram(op, W, bandwidth=3))(op)
    assert jnp.all(jnp.isnan(traced.local.matrix.bands))
