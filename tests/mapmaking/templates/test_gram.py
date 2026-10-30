import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

import furax.mapmaking.templates.gram as gram_module
from furax import AbstractLinearOperator, symmetric, tree
from furax.linalg import BorderedBandedMatrix
from furax.mapmaking.templates import Template, TemplateOperator
from furax.mapmaking.templates._builders import spline_drift_basis
from furax.mapmaking.templates.bases import OffsetBasis, SegmentedBasis, TensorBasis
from furax.mapmaking.templates.gram import (
    _regularized,
    factor,
    template_gram,
)
from furax.mapmaking.templates.pomme import PommeProjectionOperator
from furax.mapmaking.weight import MarginalizedWeightOperator
from furax.obs import TODFormat

N_DETS = 4
N_SAMPS = 64
BLOCKS = (N_SAMPS // 4, N_SAMPS // 4)  # the block lengths of the polynomial per interval
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
            SegmentedBasis(segment, jr.normal(k[0], (2, N_SAMPS)), 4, block_lengths=BLOCKS),
            tod_format,
            streams='iqu',
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


def _dense(gram, op):
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
        return ravel_pytree(out)[0]

    return np.asarray(jax.vmap(apply, out_axes=1)(jnp.eye(flat.size, dtype=flat.dtype)))


def _without_coupling(matrix, op):
    """`matrix`, over the flattened amplitudes, without the coupling of the local and global ones."""
    local = {
        name: jnp.full(s.shape, op.templates[name].per_detector)
        for name, s in op.in_structure.items()
    }
    local = np.asarray(ravel_pytree(local)[0], dtype=bool)
    return np.where(local[:, None] == local[None, :], matrix, 0)


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
    """The Gram assembled, against the dense one without the local-global coupling, under the
    weight `effective` if given."""
    reach = {'bandwidth': bandwidth, 'streams_coupled': streams_coupled, 'method': method}
    reach |= {'batch_size': batch_size} | kw
    gram = template_gram(op, W, **reach)
    expected = _without_coupling(_dense_gram(op, W if effective is None else effective), op)
    assert_allclose(_dense(gram, op), expected, rtol=1e-8, atol=1e-8)


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
            SegmentedBasis(segment, jr.normal(k[0], (2, N_SAMPS)), 4, block_lengths=BLOCKS),
            tod_format,
            streams,
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


@pytest.mark.parametrize('block_lengths', [(2, 2), None], ids=['short', 'unknown'])
def test_blocks_not_stated_longer_than_a_pomme_interval_are_rejected(block_lengths):
    # blocks of 2 samples: an interval of 5 spans 3 of them, beyond the band
    segment = jnp.arange(N_SAMPS, dtype=jnp.int32) // 2
    basis = SegmentedBasis(
        segment, jnp.ones((1, N_SAMPS)), N_SAMPS // 2, block_lengths=block_lengths
    )
    tod_format = TODFormat((N_DETS, N_SAMPS), jnp.float64)
    op = TemplateOperator({'poly': Template(basis, tod_format)})
    with pytest.raises(ValueError, match='at least 4 samples'):
        template_gram(op, _pomme_weight(op, jr.key(22)), pomme_tau=TAU)


def test_structure_is_for_a_diagonal_weight_only():
    op = _operator('local', jr.key(16))
    W = _weight(op, jr.key(17), bandwidth=1)
    with pytest.raises(ValueError, match='diagonal weight only'):
        template_gram(op, W, bandwidth=1, method='structure')


@pytest.mark.parametrize('regularization', [0.0, 0.1], ids=['plain', 'ridge'])
@pytest.mark.parametrize('templates', list(TEMPLATES))
def test_factor_inverts_the_gram(templates, regularization):
    op = _operator(templates, jr.key(4))
    W = _weight(op, jr.key(5), bandwidth=2)
    gram = template_gram(op, W, bandwidth=2)
    inverse, ridge = factor(gram, regularization=regularization)
    assert (ridge is None) == (regularization == 0)

    x = tree.normal_like(op.in_structure, jr.key(6))
    flat, unravel = ravel_pytree(x)
    gram_x = jnp.asarray(_dense(gram, op)) @ flat
    if ridge is not None:
        gram_x = gram_x + ravel_pytree(ridge(x))[0]
    assert_allclose(ravel_pytree(inverse(unravel(gram_x)))[0], flat, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('regularization', [0.0, 0.1], ids=['plain', 'ridge'])
@pytest.mark.parametrize('method', ['probe', 'structure'])
def test_gram_under_a_marginalized_weight(method, regularization):
    local = TemplateOperator(_local(jr.key(9)))
    shared = TemplateOperator(_global(jr.key(10)))
    W = _weight(TemplateOperator(local.templates | shared.templates), jr.key(11))
    local_inverse, _ = factor(template_gram(local, W, method=method), regularization=regularization)
    W_marginal = MarginalizedWeightOperator.create(W, local, local_inverse)
    gram = template_gram(shared, W_marginal, method=method)
    inverse, ridge = factor(gram, regularization=regularization)

    x = tree.normal_like(shared.in_structure, jr.key(12))
    flat, unravel = ravel_pytree(x)
    gram_x = jnp.asarray(_dense_gram(shared, W_marginal)) @ flat
    if ridge is not None:
        # that of the Gram under the plain weight as one dense block: each amplitude's own
        plain_ridge = regularization * np.diag(_dense_gram(shared, W)) * flat
        assert_allclose(ravel_pytree(ridge(x))[0], plain_ridge, rtol=1e-12)
        gram_x = gram_x + ravel_pytree(ridge(x))[0]
    assert_allclose(ravel_pytree(inverse(unravel(gram_x)))[0], flat, rtol=1e-7, atol=1e-8)


LONG = 512  # samples: more blocks of the common mode than colors of its probes


def _time_local(key, block_lengths=(64, 64), field=False):
    """A common mode of 64 blocks, and a piecewise polynomial of each detector over 8 blocks.

    With `field`, the common mode is bordered by a field of a stream of each detector, which the
    polynomials partly absorb.
    """
    k = jr.split(key, 4)
    tod_format = TODFormat((N_DETS, LONG), jnp.float64)
    segment = jnp.repeat(jnp.arange(8), LONG // 8).astype(jnp.int32)
    basis = SegmentedBasis(segment, jr.normal(k[0], (2, LONG)), 8, block_lengths=block_lengths)
    local = {'drift': Template(basis, tod_format)}
    common = Template(
        OffsetBasis(8, LONG, jnp.float64), tod_format, per=(), coupling=jr.normal(k[1], (N_DETS, 1))
    )
    shared = {'common': common}
    if field:
        # slow, as an atmospheric signal, so that the polynomials take most of it
        stream = jnp.cumsum(jr.normal(k[2], (N_DETS, 1, LONG)), axis=-1)
        shared['field'] = Template(
            TensorBasis(stream, batch_shape=(N_DETS,)),
            tod_format,
            per=(),
            coupling=jr.normal(k[3], (N_DETS, 2)),
        )
    return TemplateOperator(local), TemplateOperator(shared)


def _banded_weight(op, key, bandwidth):
    """A positive definite weight coupling no two detectors, nor samples more than `bandwidth`
    apart."""
    det, samp = (a.ravel() for a in np.indices((N_DETS, LONG)))
    reach = (det[:, None] == det) & (np.abs(samp[:, None] - samp) <= bandwidth)
    a = np.asarray(jr.normal(key, reach.shape)) * reach
    matrix = (a + a.T) / 2
    matrix += np.diag(np.abs(matrix).sum(axis=1) + 1)
    return _DenseWeight(jnp.asarray(matrix), in_structure=op.out_structure)


@pytest.mark.parametrize('field', [False, True], ids=['core', 'bordered'])
@pytest.mark.parametrize('bandwidth', [1, 9])
def test_gram_of_time_local_templates_under_a_marginalized_weight_is_probed_by_color(
    bandwidth, field
):
    # The coupling of a time-local common mode to time-local templates of each detector is read
    # off the colored probes of its band, fewer than its amplitudes. The block keeps its band, the
    # Gram under the plain weight, and the term the marginalized weight takes from it is that on
    # the coarse amplitudes and the border, when the block is too large to be dense.
    local, shared = _time_local(jr.key(13), field=field)
    W = _banded_weight(shared, jr.key(14), bandwidth)
    local_inverse, _ = factor(template_gram(local, W, bandwidth=bandwidth), regularization=0.1)
    W_marginal = MarginalizedWeightOperator.create(W, local, local_inverse)
    gram = template_gram(shared, W_marginal, bandwidth=bandwidth, max_dense=0)
    assert gram.coarse is not None
    plain = _dense_gram(shared, W)
    assert_allclose(gram.global_.matrix.as_matrix(), plain, rtol=1e-7, atol=1e-8)
    projection = _coarse_matrix(
        gram.coarse, n_border=plain.shape[0] - gram.coarse.n_blocks * gram.coarse.k
    )
    marginal = projection.T @ (plain - _dense_gram(shared, W_marginal)) @ projection
    scale = np.abs(marginal).max()
    assert_allclose(gram.marginal[0], marginal, rtol=1e-7, atol=1e-9 * scale)

    inverse, _ = factor(gram, regularization=0.1)
    dense = np.asarray(inverse.as_matrix())
    assert_allclose(dense, dense.T, atol=1e-10 * np.abs(dense).max())
    assert np.linalg.eigvalsh((dense + dense.T) / 2).min() > 0


@pytest.mark.parametrize('bandwidth', [1, 9])
@pytest.mark.parametrize('block_lengths', [(64, 64), None], ids=['small', 'unbounded'])
def test_marginalized_gram_of_time_local_templates_is_inverted_exactly_when_dense(
    bandwidth, block_lengths
):
    # A block of at most `max_dense` amplitudes is assembled dense and inverted exactly, as is any
    # block when the local templates state no block lengths: the coarse amplitudes then have no
    # bounded band.
    local, shared = _time_local(jr.key(13), block_lengths=block_lengths)
    W = _banded_weight(shared, jr.key(14), bandwidth)
    local_inverse, _ = factor(template_gram(local, W, bandwidth=bandwidth), regularization=0.1)
    W_marginal = MarginalizedWeightOperator.create(W, local, local_inverse)
    gram = template_gram(shared, W_marginal, bandwidth=bandwidth)
    assert gram.coarse is None
    inverse, ridge = factor(gram, regularization=0.1)
    assert ridge is not None

    x = tree.normal_like(shared.in_structure, jr.key(15))
    flat, unravel = ravel_pytree(x)
    gram_x = jnp.asarray(_dense_gram(shared, W_marginal)) @ flat + ravel_pytree(ridge(x))[0]
    assert_allclose(ravel_pytree(inverse(unravel(gram_x)))[0], flat, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize(
    ('bandwidth', 'regularization'),
    [(0, 0.0), (9, 0.0), (0, 1e-6)],
    ids=['structure', 'probe', 'ridge'],
)
def test_absorbed_directions_are_inverted_at_the_ridge_or_the_rounding_cutoff(
    bandwidth, regularization
):
    # The local templates absorb some directions of the common mode, which only the ridge holds:
    # `r / (1 + r)` normalized by the diagonal of the regularized Gram. The inverse solves the
    # Gram on the other directions, and inverts those at the ridge, or at the rounding cutoff if
    # larger, so that it stays positive definite.
    shared, W, W_marginal = _absorbed_common_mode(bandwidth)
    gram = template_gram(shared, W_marginal, bandwidth=bandwidth)
    inverse, _ = factor(gram, regularization=regularization)

    G = np.asarray(_dense_gram(shared, W_marginal))
    assert np.linalg.matrix_rank(G, tol=1e-8 * np.abs(G).max()) < G.shape[0]
    plain = np.diag(_dense_gram(shared, W))
    G_ridge = G + np.diag(regularization * plain)
    dense = np.asarray(inverse.as_matrix())
    x = np.asarray(jr.normal(jr.key(18), (G.shape[0],)))
    assert_allclose(
        G_ridge @ dense @ G_ridge @ x, G_ridge @ x, atol=1e-6 * np.abs(G_ridge @ x).max()
    )

    root = np.sqrt(plain * (1 + regularization))
    normalized = root[:, None] * dense * root[None, :]
    eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) / 2)
    floor = max(np.finfo(np.float64).eps ** 0.5, regularization / (1 + regularization))
    assert eigenvalues.min() > 0
    assert_allclose(eigenvalues.max(), 1 / floor, rtol=1e-3)


def test_coarse_expand_and_reduce_are_transposes():
    k = jr.split(jr.key(30), 4)
    coarse = gram_module._Coarse(jnp.array([0, 2, 3, 7]), jr.normal(k[0], (4, 3)), n_blocks=10, k=2)
    y = jr.normal(k[1], (coarse.size + 3,))  # the coarse amplitudes, then a border of 3
    core, border = jr.normal(k[2], (10, 2)), jr.normal(k[3], (3,))
    expanded_core, expanded_border = coarse.expand(y)
    assert_allclose(
        jnp.vdot(expanded_core, core) + jnp.vdot(expanded_border, border),
        jnp.vdot(y, coarse.reduce(core, border)),
    )


def test_banded_coupling_apply_is_the_transpose_of_contract():
    k = jr.split(jr.key(31), 6)
    n_colors, n_k, n_units, n_local, n_blocks, n_border = 3, 2, 2, 5, 6, 2
    coupling = gram_module._BandedCoupling(
        jr.normal(k[0], (n_colors, n_k, n_units, n_local, 1)),
        jr.randint(k[1], (n_colors, n_local), 0, n_blocks + 1),  # `n_blocks` reaches none
        jr.normal(k[2], (n_border, n_units, n_local)),
        jr.permutation(k[3], n_blocks * n_k + n_border),
    )
    v = jr.normal(k[4], (coupling.n_rows,))
    w = jr.normal(k[5], (n_units, n_local))
    assert_allclose(jnp.vdot(coupling.apply(v), w), jnp.vdot(v, coupling.contract(w)))


def test_coarse_amplitudes_synthesize_the_local_functions():
    # Constants over 64 samples, onto offsets over 8: the offsets synthesize each constant exactly.
    tod_format = TODFormat((N_DETS, LONG), jnp.float64)
    segment = jnp.repeat(jnp.arange(8), LONG // 8).astype(jnp.int32)
    local = SegmentedBasis(segment, jnp.ones((1, LONG)), 8, block_lengths=(64, 64))
    global_ = OffsetBasis(8, LONG, jnp.float64)
    coupling = jr.normal(jr.key(33), (N_DETS, 1))
    operator = TemplateOperator(
        {
            'drift': Template(local, tod_format),
            'common': Template(global_, tod_format, per=(), coupling=coupling),
        }
    )
    layout = gram_module._Layout.of(
        operator, ('common',), local=False, dense=False, streams_coupled=True
    )
    reading = gram_module._Layout.of(
        operator, ('drift',), local=True, dense=False, streams_coupled=False
    )
    coarse = gram_module._coarse_projection(operator, layout, reading, batch_size=4)
    assert coarse is not None and coarse.size == local.size
    for j in range(local.size):
        unit = jnp.zeros(local.size).at[j].set(1)
        synthesized = global_(coarse.lift(unit.reshape(-1, 1))[:, 0])
        assert_allclose(synthesized, local(unit.reshape(local.shape)), atol=1e-12)


@pytest.mark.parametrize('n_border', [0, 2], ids=['core', 'bordered'])
def test_two_level_inverse_is_that_of_the_band_plus_the_coarse_correction(n_border):
    # The band inverts the core alone; the border is inverted by the coarse level, with the coarse
    # amplitudes.
    basis = spline_drift_basis(10, jnp.arange(200.0), jnp.float64)
    bands = basis.gram(jnp.ones(200)).at[:, 0].add(jnp.eye(1))  # positive definite
    k = jr.split(jr.key(34), 3)
    border = corner = None
    if n_border:
        border = 0.1 * jr.normal(k[1], (1, 12, n_border))
        corner = 10 * jnp.eye(n_border)[None]  # diagonally dominant, so positive definite
    matrix = BorderedBandedMatrix(bands[None], border, corner)
    coarse = gram_module._Coarse(jnp.array([0, 3, 6, 9]), jr.normal(k[0], (4, 3)), 12, 1)
    projection = _coarse_matrix(coarse, n_border)
    plain = np.asarray(matrix.as_matrix())
    marginal = 0.5 * projection.T @ plain @ projection
    inverse = gram_module._two_level(matrix, jnp.asarray(marginal)[None], coarse, 0.0, 2)
    coarse_block = projection.T @ plain @ projection - marginal
    band = np.zeros_like(plain)
    band[:12, :12] = np.linalg.inv(plain[:12, :12])
    expected = band + projection @ np.linalg.inv(coarse_block) @ projection.T
    assert_allclose(inverse.as_matrix(), expected, rtol=1e-8, atol=1e-10)


def _coarse_matrix(coarse, n_border=0):
    r"""The coarse amplitudes and the border as a dense matrix $Z = \Pi \oplus I$, from the core
    amplitudes then the border to the coarse ones then the border."""
    units = jnp.eye(coarse.size).reshape(coarse.size, -1, coarse.k)
    projection = np.asarray(jax.vmap(lambda y: coarse.lift(y).ravel())(units)).T
    return block_diag(projection, np.eye(n_border))


def _absorbed_common_mode(bandwidth):
    """A common mode, the weight, and that weight with a template absorbing some of it marginalized.

    Each detector's constant over 8 blocks absorbs those of the common mode: the Gram of the common
    mode under the marginalized weight is singular. The constants state no block lengths, so that
    Gram is assembled and inverted dense.
    """
    k = jr.split(jr.key(17), 2)
    tod_format = TODFormat((N_DETS, LONG), jnp.float64)
    segment = jnp.repeat(jnp.arange(8), LONG // 8).astype(jnp.int32)
    basis = SegmentedBasis(segment, jnp.ones((1, LONG)), 8)
    local = TemplateOperator({'drift': Template(basis, tod_format)})
    common = Template(
        OffsetBasis(8, LONG, jnp.float64), tod_format, per=(), coupling=jr.normal(k[1], (N_DETS, 1))
    )
    shared = TemplateOperator({'common': common})
    W = _banded_weight(shared, k[0], bandwidth)
    local_inverse, _ = factor(template_gram(local, W, bandwidth=bandwidth))
    return shared, W, MarginalizedWeightOperator.create(W, local, local_inverse)


@pytest.mark.parametrize('bandwidth', [0, 3, 20])
def test_reach_of_time_blocks_matches_the_bases(bandwidth):
    local, shared = _time_local(jr.key(16))
    local_basis = local.templates['drift'].basis
    global_basis = shared.templates['common'].basis
    first, last = gram_module._reach(local_basis, global_basis, bandwidth)
    # brute force: the global blocks read by samples within `bandwidth` of each local block's
    local_blocks = np.asarray(local_basis.support().blocks)
    global_blocks = np.asarray(global_basis.support().blocks)
    for block in range(local_basis._n_blocks):
        samples = np.flatnonzero((local_blocks == block).any(axis=1))
        near = np.unique(
            np.clip(
                np.arange(samples.min() - bandwidth, samples.max() + bandwidth + 1), 0, LONG - 1
            )
        )
        reached = global_blocks[near].ravel()
        assert (int(first[block]), int(last[block])) == (reached.min(), reached.max())


def test_gram_under_a_marginalized_weight_is_of_global_templates_only():
    local = TemplateOperator(_local(jr.key(9)))
    shared = TemplateOperator(_global(jr.key(10)))
    joint = TemplateOperator(local.templates | shared.templates)
    W = _weight(joint, jr.key(11))
    inverse, _ = factor(template_gram(local, W))
    W_marginal = MarginalizedWeightOperator.create(W, local, inverse)
    with pytest.raises(ValueError, match='global templates'):
        template_gram(joint, W_marginal)


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
    expected = _without_coupling(_dense_gram(op, W), op)
    assert_allclose(_dense(gram, op), expected, rtol=1e-8, atol=1e-8)


def test_blocks_of_unknown_length_are_probed_whole():
    # three 1-sample intervals: samples 3 apart couple blocks 3 apart, which a band estimated
    # from the mean block length would miss
    segment = jnp.repeat(
        jnp.arange(4), jnp.array([1, 1, 1, N_SAMPS - 3]), total_repeat_length=N_SAMPS
    )
    basis = SegmentedBasis(segment.astype(jnp.int32), jnp.ones((1, N_SAMPS)), 4)
    op = TemplateOperator({'poly': Template(basis, _tod_format(), streams='iqu')})
    W = _weight(op, jr.key(13), bandwidth=3)
    gram = template_gram(op, W, bandwidth=3)
    assert_allclose(_dense(gram, op), _dense_gram(op, W), rtol=1e-10, atol=1e-10)


def test_regularized_adds_the_ridge_and_a_unit_diagonal_on_unseen_rows():
    units, n, k, m = (2,), 5, 2, 3
    keys = jr.split(jr.key(30), 3)
    a = jr.normal(keys[0], (*units, n * k + m, n * k + m))
    dense = a @ jnp.swapaxes(a, -1, -2) + jnp.eye(n * k + m)
    # the first core function and the last border amplitude are seen by nothing
    unseen = jnp.zeros(n * k + m, bool).at[: n * k : k].set(True).at[-1].set(True)
    dense = jnp.where(unseen[:, None] | unseen, 0, dense)
    blocks = dense[..., : n * k, : n * k].reshape(*units, n, k, n, k)
    bands = jnp.stack(  # every band, the core being dense: (*units, n, n, k, k)
        [
            jnp.stack([blocks[..., i, :, min(i + d, n - 1), :] * (i + d < n) for d in range(n)], -3)
            for i in range(n)
        ],
        axis=-4,
    )
    matrix = BorderedBandedMatrix(bands, dense[..., : n * k, n * k :], dense[..., n * k :, n * k :])

    regularized, (ridge_a, ridge_c) = _regularized(matrix, 0.1)

    diagonal = jnp.diagonal(bands[..., 0, :, :], axis1=-2, axis2=-1)
    # the seen functions of a core block share its mean diagonal, the unseen ones take no ridge
    assert_allclose(ridge_a[..., 1:], 0.1 * diagonal.mean(axis=-1, keepdims=True))
    assert_allclose(ridge_a[..., 0], 0)
    # each border amplitude in its own units
    assert_allclose(ridge_c, 0.1 * jnp.diagonal(matrix.corner, axis1=-2, axis2=-1))
    assert_allclose(regularized.bands[..., 0, 0, 0], 1)
    assert_allclose(regularized.corner[..., -1, -1], 1)
    x = (jr.normal(keys[1], (*units, n, k)), jr.normal(keys[2], (*units, m)))
    x_a, x_c = regularized.cholesky()(x)
    assert jnp.all(jnp.isfinite(x_a)) and jnp.all(jnp.isfinite(x_c))
    # the unseen amplitudes come back as they came in
    assert_allclose(x_a[..., 0], x[0][..., 0], rtol=1e-12)
    assert_allclose(x_c[..., -1], x[1][..., -1], rtol=1e-12)
