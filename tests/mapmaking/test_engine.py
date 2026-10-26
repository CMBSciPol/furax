import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

from furax.mapmaking import ReaderField
from furax.mapmaking._engine import (
    DetectorFamily,
    Policy,
    ReplicatedFamily,
    StackedFamily,
    ZippedFamily,
    combine,
    family_apply,
    partition,
)

N_MEMBERS = 6
N_SAMPS = 5


def _members():
    return jnp.arange(N_MEMBERS * N_SAMPS, dtype=jnp.float64).reshape(N_MEMBERS, N_SAMPS)


@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
def test_outputs_combine_over_the_members(jit):
    members = _members()
    shared = jnp.linspace(0, 1, N_SAMPS)

    def apply(x):
        return {'sum': x, 'stack': 2 * x, 'shared': shared + 1}

    def run(members):
        spec = {'sum': 'sum', 'stack': 'stack', 'shared': 'shared'}
        out, kinds = family_apply(StackedFamily(members), apply, spec)
        assert kinds == spec
        return out

    out = (jax.jit(run) if jit else run)(members)
    assert_allclose(out['sum'], members.sum(axis=0))
    assert_allclose(out['stack'], 2 * members)
    assert_allclose(out['shared'], shared + 1)


def test_auto_stacks_what_depends_on_the_member_only():
    members = _members()
    shared = jnp.linspace(0, 1, N_SAMPS)
    out, kinds = family_apply(StackedFamily(members), lambda x: (x, shared**2, x.sum()), 'auto')
    assert kinds == ('stack', 'shared', 'stack')
    assert_allclose(out[0], members)
    assert_allclose(out[1], shared**2)
    assert_allclose(out[2], members.sum(axis=1))


def test_a_single_kind_applies_to_every_output_leaf():
    members = _members()
    out, _ = family_apply(StackedFamily(members), lambda x: (x.sum(), x.max()), 'sum')
    assert_allclose(out[0], members.sum())
    assert_allclose(out[1], members.max(axis=1).sum())


def test_the_computation_is_traced_independently_of_the_number_of_members():
    def traces(n_members):
        count = 0

        def apply(x):
            nonlocal count
            count += 1
            return x.sum()

        jax.make_jaxpr(lambda m: family_apply(StackedFamily(m), apply, 'sum')[0])(
            jnp.zeros((n_members, N_SAMPS))
        )
        return count

    assert traces(1) == traces(N_MEMBERS)


def test_detector_batches_slice_the_per_detector_fields_only():
    n_dets = 4
    tod = jnp.arange(2 * n_dets * N_SAMPS, dtype=jnp.float64).reshape(2, n_dets, N_SAMPS)
    timestamps = jnp.linspace(0, 1, N_SAMPS)
    data = {ReaderField.SAMPLE_DATA: tod, ReaderField.TIMESTAMPS: timestamps}
    family = DetectorFamily(data, batch_size=2)
    assert family.size == 2

    def apply(data):
        return data[ReaderField.SAMPLE_DATA], data[ReaderField.TIMESTAMPS]

    (batches, times), kinds = family_apply(family, apply, 'auto')
    assert kinds == ('stack', 'shared')
    assert batches.shape == (2, 2, 2, N_SAMPS)  # (batch, stokes, det, samp)
    assert_allclose(jnp.concatenate(list(batches), axis=1), tod)
    assert_allclose(times, timestamps)


@pytest.mark.parametrize('width', [None, 1, 5], ids=['index', 'one', 'all'])
def test_stacked_members_may_lie_along_any_axis(width):
    tod = _members().T[None]  # (stokes, det, samp): the entries on axis -2
    family = StackedFamily(tod, axis=-2, width=width)
    assert family.size == N_SAMPS // (width or 1)
    out, _ = family_apply(family, lambda x: x, 'stack')
    if width is None:
        assert_allclose(jnp.moveaxis(out, 0, -2), tod)
    else:
        assert_allclose(jnp.concatenate(list(out), axis=-2), tod)


def test_zipped_families_hand_out_their_members_together():
    ids = jnp.arange(N_MEMBERS)
    pairs = jnp.stack([ids, 10 * ids])  # (2, member)
    family = ZippedFamily((StackedFamily(ids), StackedFamily(pairs, axis=-1, width=1)))
    out, _ = family_apply(family, lambda members: members, 'stack')
    assert_allclose(out[0], ids)
    assert_allclose(jnp.concatenate(list(out[1]), axis=-1), pairs)


def test_stacked_members_hold_the_shared_leaves_whole():
    # the 'stack' and 'shared' outputs of a family make a family again
    members = _members()
    shared = jnp.linspace(0, 1, N_SAMPS)
    outputs, kinds = family_apply(
        StackedFamily(members), lambda x: (x, shared), ('stack', 'shared')
    )
    stacked, others = partition(outputs, kinds)
    assert stacked[1] is None and others[0] is None
    out, _ = family_apply(
        StackedFamily(stacked, shared=others), lambda pair: pair[0] + pair[1], 'stack'
    )
    assert_allclose(out, members + shared)
    assert combine(stacked, others) == outputs


@pytest.mark.distributed
def test_a_single_round_keeps_the_members_where_they_are_sharded():
    # members split over an `Auto` mesh axis, outside `Policy.devices`: vectorized over the
    # stacks, the round neither gathers nor reduces them across devices
    mesh = jax.make_mesh((jax.device_count(),), ('obs',), axis_types=(AxisType.Auto,))
    members = _members().repeat(jax.device_count(), axis=0)
    sharded = jax.device_put(members, NamedSharding(mesh, P('obs')))

    def run(x):
        family = ZippedFamily((StackedFamily(x), ReplicatedFamily(2.0, x.shape[0])))
        out, _ = family_apply(family, lambda m: m[0] * m[1], 'stack', Policy(batch=x.shape[0]))
        return out

    hlo = jax.jit(run).lower(sharded).compile().as_text()
    assert not any(c in hlo for c in ('all-gather', 'all-reduce', 'all-to-all'))
    out = jax.jit(run)(sharded)
    assert out.sharding.spec == P('obs')
    assert_allclose(out, 2 * members)


def test_sharded_members_take_replicated_inputs_as_arguments():
    # the map of a system: held whole by every device, handed to every member
    mesh = jax.make_mesh((jax.device_count(),), ('obs',))
    members = jnp.arange(2 * jax.device_count() * N_SAMPS, dtype=jnp.float64)
    members = members.reshape(2 * jax.device_count(), N_SAMPS)
    offset = jnp.linspace(0, 1, N_SAMPS)

    def apply(member):
        x, common = member
        return x * common, x + common

    family = ZippedFamily((StackedFamily(members), ReplicatedFamily(offset, len(members))))
    expected, _ = family_apply(family, apply, ('sum', 'stack'))
    with jax.set_mesh(mesh):
        sharded = jax.device_put(members, NamedSharding(mesh, P('obs')))
        family = ZippedFamily((StackedFamily(sharded), ReplicatedFamily(offset, len(members))))
        (total, stacked), _ = family_apply(family, apply, ('sum', 'stack'), Policy(devices='obs'))
    assert_allclose(total, expected[0])
    assert_allclose(stacked, expected[1])


@pytest.mark.parametrize('batch', [1, 2, N_MEMBERS])
def test_rounds_of_vectorized_members_give_the_same_outputs(batch):
    members = _members()
    scale = jnp.linspace(1, 2, N_SAMPS)
    family = ZippedFamily(
        (StackedFamily((members, None), shared=(None, scale)), StackedFamily(2 * members))
    )

    def apply(member):
        (x, s), y = member
        return x * s + y, x.sum(), scale**2

    spec = ('stack', 'sum', 'auto')
    expected, _ = family_apply(family, apply, spec)
    out, kinds = family_apply(family, apply, spec, Policy(batch=batch))
    assert kinds == ('stack', 'sum', 'shared')
    for got, want in zip(out, expected, strict=True):
        assert_allclose(got, want)


def test_family_apply_runs_inside_shard_map():
    # The members of each shard vary over the mesh axis; the sum and the shared value come out
    # varying too, the carry being typed alike.
    mesh = jax.make_mesh((jax.device_count(),), ('obs',))
    stacked = jnp.arange(jax.device_count() * N_MEMBERS * N_SAMPS, dtype=jnp.float64)
    stacked = stacked.reshape(jax.device_count(), N_MEMBERS, N_SAMPS)
    stacked = jax.device_put(stacked, NamedSharding(mesh, P('obs')))

    @jax.shard_map(mesh=mesh, in_specs=P('obs'), out_specs=P('obs'))
    def per_shard(members):
        def apply(x):
            return x, x[0] * 0 + members[0, 0, 0]

        (total, first), _ = family_apply(StackedFamily(members[0]), apply, ('sum', 'shared'))
        return total[None], first[None]

    total, first = per_shard(stacked)
    assert_allclose(total, stacked.sum(axis=1))
    assert_allclose(first, stacked[:, 0, 0])


@pytest.mark.parametrize(
    'make, match',
    [
        (lambda: DetectorFamily({ReaderField.TIMESTAMPS: jnp.zeros(3)}, 1), 'per-detector'),
        (lambda: DetectorFamily({ReaderField.SAMPLE_DATA: jnp.zeros((4, 3))}, 3), 'do not divide'),
        (lambda: StackedFamily(jnp.zeros((4, 3)), width=3), 'do not divide'),
        (lambda: ZippedFamily((StackedFamily(jnp.zeros(2)), StackedFamily(jnp.zeros(3)))), 'sizes'),
    ],
    ids=['no-detectors', 'indivisible-detectors', 'indivisible-width', 'zipped-sizes'],
)
def test_families_reject(make, match):
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize(
    ('apply', 'match'),
    [
        (lambda: family_apply(StackedFamily(_members()), lambda x: x, 'mean'), 'unknown output'),
        (
            lambda: family_apply(StackedFamily(_members()), lambda x: x, 'stack', Policy(batch=4)),
            'do not divide',
        ),
        (lambda: Policy(batch=0), 'batch'),
        (
            lambda: family_apply(
                StackedFamily(jnp.zeros((jax.device_count(), 2))),
                lambda x: x,
                'auto',
                Policy(devices='obs'),
            ),
            'declared',
        ),
    ],
    ids=['unknown-kind', 'indivisible-rounds', 'empty-round', 'undeclared-sharded-kinds'],
)
def test_family_apply_rejects(apply, match):
    mesh = jax.make_mesh((jax.device_count(),), ('obs',))
    with jax.set_mesh(mesh), pytest.raises(ValueError, match=match):
        apply()
