import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax import tree
from furax.mapmaking.templates._operator import Template, TemplateOperator
from furax.mapmaking.templates.bases import OffsetBasis, SegmentedBasis, TensorBasis
from furax.obs import TODFormat

N_DETS = 6
N_SAMPS = 32
DET_GROUPS = np.eye(2, dtype=bool)[[0, 1, 0, 1, 1, 0]]
N_MODES = 2


def _tod_format(stokes, det_shape=(N_DETS,)):
    return TODFormat((*det_shape, N_SAMPS), jnp.float64, stokes=stokes)


def _bases(key):
    """One basis per storage: dense, segmented, batched over detectors, regular offsets."""
    k = jr.split(key, 3)
    segment = jnp.repeat(jnp.arange(4), N_SAMPS // 4).astype(jnp.int32)
    return {
        'tensor': TensorBasis(jr.normal(k[0], (3, N_SAMPS))),
        'segmented': SegmentedBasis(segment, jr.normal(k[1], (2, N_SAMPS)), 4),
        'per_det': TensorBasis(jr.normal(k[2], (N_DETS, 2, N_SAMPS)), batch_shape=(N_DETS,)),
        'offset': OffsetBasis(8, N_SAMPS, jnp.float64),
    }


LEGS = {'tensor': 'iqu', 'segmented': 'qu', 'per_det': 'qu', 'offset': 'u'}

COUPLINGS = {
    'none': None,
    'full': (3, N_DETS),
    'same_on_every_stream': (1, N_DETS),
    'same_on_every_detector': (3, 1),
}

# The amplitude sets of a case: `per`, and whether the detectors are grouped along 'det'.
SETS = {
    'per_stream_and_det': (('stream', 'det'), False),
    'per_det': (('det',), False),
    'per_stream': (('stream',), False),
    'shared': ((), False),
    'per_group': (('det',), True),
    'per_stream_and_group': (('stream', 'det'), True),
}


def _templates(sets, coupling, stokes, key):
    """One template per basis, each with the amplitude sets and the coupling of the case; on a
    demodulated TOD, the templates cover different streams."""
    per, grouped = SETS[sets]
    tod_format = _tod_format(stokes)
    templates = {}
    for i, (name, basis) in enumerate(_bases(key).items()):
        on = LEGS[name] if stokes else None
        c = None
        if (shape := COUPLINGS[coupling]) is not None:
            streams_axis = () if on is None else (min(shape[0], len(on)),)  # only if demodulated
            c = jr.normal(jr.fold_in(key, 10 + i), (*streams_axis, shape[1], N_MODES))
        groups = jnp.asarray(DET_GROUPS) if grouped else None
        templates[name] = Template(
            basis, tod_format, streams=on, per=per, coupling=c, det_groups=groups
        )
    return templates


def _basis_matrices(basis):
    """Each detector's basis matrix, `(det, samp, size)`."""
    matrix = np.asarray(basis.as_matrix())
    if not basis.batch_shape:
        return np.broadcast_to(matrix, (N_DETS, basis.n_points, basis.size))
    matrix = matrix.reshape(N_DETS, basis.n_points, N_DETS, basis.size)
    return np.stack([matrix[d, :, d] for d in range(N_DETS)])


def _reference_signal(template, matrices, a, stream, d):
    """Stream number `stream` of detector `d`, from the definition, one detector at a time."""
    a = np.asarray(a)
    if 'stream' in template.per:
        a = a[stream]
    if 'det' in template.per:
        groups = template.det_groups
        a = a[d] if groups is None else a[int(np.argmax(groups[d]))]
    if template.coupling is not None:
        c = np.asarray(template.coupling)
        if template.streams is not None:
            c = c[min(stream, c.shape[0] - 1)]
        a = a @ c[min(d, c.shape[0] - 1)]
    return matrices[d] @ a.ravel()


def _reference_tod(templates, x, stokes):
    streams = list(stokes.lower()) if stokes else [None]  # a single stream unless demodulated
    tod = np.zeros((len(streams), N_DETS, N_SAMPS))
    for name, template in templates.items():
        matrices = _basis_matrices(template.basis)
        covered = [None] if template.streams is None else list(template.streams)
        for j, stream in enumerate(streams):
            if stream not in covered:
                continue
            index = covered.index(stream)
            for d in range(N_DETS):
                tod[j, d] += _reference_signal(template, matrices, x[name], index, d)
    return tod if stokes else tod[0]


# On a TOD that is not demodulated, amplitudes per stream and a coupling the same on every stream
# change nothing: those cases repeat others, and are left out.
CASES = [
    pytest.param(stokes, sets, coupling, id=f'{stokes or "no_streams"}-{sets}-{coupling}')
    for stokes in ('IQU', None)
    for sets in SETS
    for coupling in COUPLINGS
    if stokes or ('stream' not in SETS[sets][0] and coupling != 'same_on_every_stream')
]


@pytest.mark.parametrize(('stokes', 'sets', 'coupling'), CASES)
def test_operator_matches_the_definition(stokes, sets, coupling):
    templates = _templates(sets, coupling, stokes, jr.key(0))
    op = TemplateOperator(templates)
    x = tree.normal_like(op.in_structure, jr.key(1))

    tod = op(x)
    got = tod.data if stokes else tod
    assert_allclose(got, _reference_tod(templates, x, stokes), rtol=1e-10, atol=1e-10)

    # the transpose is the adjoint
    y = tree.normal_like(op.out_structure, jr.key(2))
    assert_allclose(tree.dot(tod, y), tree.dot(x, op.T(y)), rtol=1e-10)


def test_template_alone_is_zero_off_its_streams():
    template = _templates('per_stream', 'full', 'IQU', jr.key(3))['segmented']  # on Q and U
    a = jr.normal(jr.key(4), template.in_structure.shape)
    tod = template(a)
    assert_allclose(tod.i, 0)
    assert_allclose(tod.data, TemplateOperator({'t': template})({'t': a}).data)
    y = tree.normal_like(template.out_structure, jr.key(5))
    assert_allclose(tree.dot(tod, y), tree.dot(a, template.T(y)), rtol=1e-10)


def _on_detector_axes(template, tod_format):
    """The same template on a TOD whose detectors span several axes."""
    det = tod_format.det_shape
    basis, coupling, groups = template.basis, template.coupling, template.det_groups
    if basis.batch_shape:
        basis = TensorBasis(basis.values.reshape(*det, *basis.shape, N_SAMPS), batch_shape=det)
    if coupling is not None:
        n = coupling.shape[-2]
        coupling = coupling.reshape(*coupling.shape[:-2], *(det if n > 1 else (1,) * len(det)), -1)
    if groups is not None:
        groups = groups.reshape(*det, -1)
    return Template(basis, tod_format, template.streams, template.per, coupling, groups)


@pytest.mark.parametrize('sets', ['per_stream_and_det', 'shared', 'per_stream_and_group'])
def test_detector_axes_are_those_of_a_flat_detector_axis(sets):
    flat = _templates(sets, 'full', 'IQU', jr.key(6))
    tod_format = _tod_format('IQU', (2, 3))
    shaped = {name: _on_detector_axes(t, tod_format) for name, t in flat.items()}
    op_flat, op = TemplateOperator(flat), TemplateOperator(shaped)
    x = tree.normal_like(op_flat.in_structure, jr.key(7))
    x_shaped = {name: a.reshape(op.in_structure[name].shape) for name, a in x.items()}

    got = op(x_shaped).data
    assert got.shape == (3, 2, 3, N_SAMPS)
    assert_allclose(got.reshape(3, N_DETS, N_SAMPS), op_flat(x).data, rtol=1e-12)
    y = tree.normal_like(op.out_structure, jr.key(8))
    assert_allclose(tree.dot(op(x_shaped), y), tree.dot(x_shaped, op.T(y)), rtol=1e-10)


def test_single_timestream_has_no_detector_axis():
    tod_format = TODFormat((N_SAMPS,), jnp.float64, stokes='QU')
    template = Template(TensorBasis(jnp.ones((2, N_SAMPS))), tod_format, streams='qu')
    assert template.per == ('stream',)
    assert template.in_structure.shape == (2, 2)
    assert template(jnp.ones((2, 2))).q.shape == (N_SAMPS,)


@pytest.mark.parametrize(
    ('kwargs', 'per', 'shape'),
    [
        ({'streams': 'qu', 'per': ('det', 'stream')}, ('stream', 'det'), (2, N_DETS, 4, 2)),
        (
            {'streams': 'qu', 'per': ('stream',), 'coupling': np.ones((2, N_DETS, N_MODES))},
            ('stream',),
            (2, 4, 2, N_MODES),
        ),
        ({'streams': 'qu', 'per': ('det',), 'det_groups': DET_GROUPS}, ('det',), (2, 4, 2)),
        ({}, ('det',), (N_DETS, 4, 2)),  # only a demodulated TOD has a stream axis
    ],
    ids=['order', 'modes', 'groups', 'no_streams'],
)
def test_amplitudes_put_sets_then_basis_then_modes(kwargs, per, shape):
    basis = SegmentedBasis(jnp.zeros(N_SAMPS, jnp.int32), jnp.ones((2, N_SAMPS)), 4)
    template = Template(basis, _tod_format('QU' if 'streams' in kwargs else None), **kwargs)
    assert template.per == per
    assert template.in_structure.shape == shape


@pytest.mark.parametrize('coupled', [False, True], ids=['uncoupled', 'coupled'])
@pytest.mark.parametrize('sets', list(SETS))
def test_canonical_amplitudes_have_fixed_axes(sets, coupled):
    # (stream set, detector set, *basis.shape, mode), of size 1 where shared, whatever `per`
    per, grouped = SETS[sets]
    basis = SegmentedBasis(jnp.zeros(N_SAMPS, jnp.int32), jnp.ones((2, N_SAMPS)), 4)
    coupling = jnp.ones((2, N_DETS, N_MODES)) if coupled else None
    det_groups = jnp.asarray(DET_GROUPS) if grouped else None
    template = Template(
        basis, _tod_format('QU'), streams='qu', per=per, coupling=coupling, det_groups=det_groups
    )
    a = jr.normal(jr.key(0), template.in_structure.shape)

    canonical = template._to_canonical(a)
    n_det_sets = N_DETS if template.n_det_groups is None else template.n_det_groups
    assert canonical.shape == (
        2 if 'stream' in per else 1,
        n_det_sets if 'det' in per else 1,
        4,
        2,
        N_MODES if coupled else 1,
    )
    assert_allclose(template._from_canonical(canonical), a)


def test_operator_stacks_under_vmap():
    # only the templates are dynamic, so stacking observations gains a leading axis on their
    # arrays, as the multi-observation scan applies it
    ops = [
        TemplateOperator(_templates('per_stream_and_group', 'full', 'IQU', jr.key(i)))
        for i in (0, 1)
    ]
    x = tree.stack([tree.normal_like(op.in_structure, jr.key(2 + i)) for i, op in enumerate(ops)])
    out = jax.vmap(lambda op, x: op(x).data)(tree.stack(ops), x)
    for i, op in enumerate(ops):
        assert_allclose(out[i], op(tree.stacked_get(x, i)).data, rtol=1e-12)


def _template(stokes='QU', n_samps=N_SAMPS, dtype=jnp.float64, **kwargs):
    basis = TensorBasis(jnp.ones((1, n_samps), dtype))
    return Template(basis, _tod_format(stokes), **kwargs)


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'per': ('wafer',)}, 'distinct axes among'),
        ({'per': ('det', 'det')}, 'distinct axes among'),
        ({'streams': 'QU'}, 'distinct and lower case'),
        ({'streams': ''}, 'distinct and lower case'),
        ({'streams': 'i'}, r"streams \['i'\]"),
        ({}, 'covers no stream'),
        ({'stokes': None, 'streams': 'q'}, r"streams \['q'\]"),
        ({'streams': 'q', 'per': ('stream',), 'det_groups': DET_GROUPS}, 'groups the amplitudes'),
        ({'streams': 'q', 'det_groups': DET_GROUPS[:2]}, 'groups the amplitudes'),
        ({'streams': 'q', 'coupling': np.ones((N_DETS, 1))}, r'shape \(1 or 1, 6, mode\)'),
        ({'streams': 'iqu', 'stokes': 'IQU', 'coupling': np.ones((2, N_DETS, 1))}, r'1 or 3'),
        ({'streams': 'q', 'coupling': np.ones((1, 2, 1))}, r'shape \(1 or 1, 6, mode\)'),
        ({'stokes': None, 'coupling': np.ones((1, N_DETS, 1))}, r'shape \(6, mode\)'),
        ({'streams': 'q', 'n_samps': N_SAMPS + 1}, f'{N_SAMPS + 1} samples, not {N_SAMPS}'),
        ({'streams': 'q', 'dtype': jnp.float32}, 'dtype float32, not float64'),
    ],
)
def test_template_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _template(**kwargs)


def test_template_rejects_a_basis_batch_other_than_the_detectors():
    basis = TensorBasis(jnp.ones((2, 1, N_SAMPS)), batch_shape=(2,))
    with pytest.raises(ValueError, match=r'batch \(2,\), not the detectors \(6,\)'):
        Template(basis, _tod_format('QU'), streams='q')


def test_operator_rejects_no_template_and_different_tods():
    with pytest.raises(ValueError, match='at least one template'):
        TemplateOperator({})
    with pytest.raises(ValueError, match='different TODs'):
        TemplateOperator({'a': _template(streams='q'), 'b': _template('IQU', streams='q')})
