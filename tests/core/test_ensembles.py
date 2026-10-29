"""The ensemble engine: representations, policies, output strides and their inference."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax import DiagonalOperator
from furax.core import (
    CompositionOperator,
    HostEnsemble,
    ListEnsemble,
    Policy,
    StackedEnsemble,
    lift,
    map_reduce,
)

N = 8
N_SAMPS = 5


def _xs() -> jax.Array:
    return jnp.arange(N * N_SAMPS, dtype=jnp.float64).reshape(N, N_SAMPS)


def _shared() -> jax.Array:
    return jnp.linspace(0.0, 1.0, N_SAMPS)


def _apply(member, scale):
    return {
        'sum': scale * member['x'],
        'stack': member['x'] + member['t'],
        'shared': 2 * member['t'],
    }


def _expected(scale):
    xs, t = _xs(), _shared()
    return {'sum': scale * xs.sum(axis=0), 'stack': xs + t, 'shared': 2 * t}


def _stacked(**keywords) -> StackedEnsemble:
    return StackedEnsemble(N, {'x': _xs(), 't': _shared()}, {'x': 1, 't': None}, **keywords)


def _list() -> ListEnsemble:
    return ListEnsemble([{'x': x, 't': _shared()} for x in _xs()])


def _host(**keywords) -> HostEnsemble:
    xs = np.asarray(_xs())
    return HostEnsemble(
        N,
        lambda i: (xs[i],),
        {'x': jax.ShapeDtypeStruct((N_SAMPS,), jnp.float64), 't': _shared()},
        **keywords,
    )


@pytest.mark.parametrize('batch', [1, 2, 8])
@pytest.mark.parametrize('representation', ['list', 'stacked', 'host'])
def test_outputs_combine_over_the_members(representation, batch):
    if representation == 'list' and batch > 1:
        pytest.skip('a list ensemble is evaluated one member per round')
    policy = Policy(batch=batch)
    ensemble = {
        'list': _list,
        'stacked': lambda: _stacked(policy=policy),
        'host': lambda: _host(policy=policy),
    }[representation]()
    spec = {'sum': 'sum', 'stack': 1, 'shared': None}
    out, kinds = map_reduce(_apply, ensemble, spec, 3.0)
    assert kinds == spec
    expected = _expected(3.0)
    assert_allclose(out['sum'], expected['sum'])
    assert_allclose(out['stack'], expected['stack'])
    assert_allclose(out['shared'], expected['shared'])


@pytest.mark.parametrize('jit', [False, True], ids=['eager', 'jit'])
def test_under_jit(jit):
    def run(scale):
        return map_reduce(_apply, _stacked(), {'sum': 'sum', 'stack': 1, 'shared': None}, scale)[0]

    out = (jax.jit(run) if jit else run)(3.0)
    assert_allclose(out['sum'], _expected(3.0)['sum'])


@pytest.mark.parametrize('batch', [1, 2, 4])
def test_strided_leaves_are_shared_by_runs(batch):
    xs = jnp.arange(float(N))
    by_2 = jnp.array([1.0, 2.0, 3.0, 4.0])  # one entry per run of 2 members
    ensemble = StackedEnsemble(
        N, {'x': xs, 'b': by_2}, {'x': 1, 'b': 2}, policy=Policy(batch=batch)
    )
    out, kinds = map_reduce(
        lambda m: {'y': m['x'] * m['b'], 'b': m['b']}, ensemble, {'y': 1, 'b': 2}
    )
    assert_allclose(out['y'], xs * jnp.repeat(by_2, 2))
    assert_allclose(out['b'], by_2)  # one entry per run, not one per member
    assert kinds == {'y': 1, 'b': 2}


def test_padded_members_do_not_contribute():
    valid = jnp.asarray([True] * 6 + [False] * 2)
    out, _ = map_reduce(_apply, _stacked(valid=valid), 'sum', 1.0)
    assert_allclose(out['sum'], _xs()[:6].sum(axis=0))


def test_auto_resolves_from_the_dataflow():
    t = _shared()

    def apply(member):
        return {
            'per_member': member['x'] * t,
            'constant': jnp.zeros_like(t),  # the shape is static: no dataflow from the member
            'closure': t**2,
            'conservative': member['x'] * 0,  # zero, but the dataflow does not know
        }

    out, kinds = map_reduce(apply, _stacked(), 'auto')
    assert kinds == {'per_member': 1, 'constant': None, 'closure': None, 'conservative': 1}
    assert out['per_member'].shape == (N, N_SAMPS)
    assert_allclose(out['closure'], t**2)
    assert_allclose(out['conservative'], jnp.zeros((N, N_SAMPS)))


def test_auto_never_resolves_to_sum():
    out, kinds = map_reduce(lambda m: m['x'].sum(), _stacked(), 'auto')
    assert kinds == 1
    assert out.shape == (N,)


@pytest.mark.parametrize(
    'spec, match',
    [
        ('reduce', 'unknown output kind'),
        (3, 'does not divide'),
        (5, 'does not divide'),
    ],
)
def test_invalid_specs_are_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        map_reduce(lambda m: m['x'], _stacked(), spec)


def test_stride_must_align_with_the_rounds():
    ensemble = _stacked(policy=Policy(batch=4))
    # 8 % 2 == 0 but neither 2 % 4 nor 4 % 2 != 0: stride 2 is fine with rounds of 4
    map_reduce(lambda m: m['x'], ensemble, 2)
    ensemble3 = StackedEnsemble(12, {'x': jnp.arange(12.0)}, 1, policy=Policy(batch=4))
    with pytest.raises(ValueError, match='not aligned'):
        map_reduce(lambda m: m['x'], ensemble3, 3)


def test_out_spec_must_be_a_prefix():
    with pytest.raises(ValueError, match='prefix'):
        map_reduce(lambda m: m['x'], _stacked(), {'wrong': 'sum'})


def test_rounds_must_divide_the_members():
    with pytest.raises(ValueError, match='do not divide'):
        map_reduce(lambda m: m['x'], _stacked(policy=Policy(batch=3)), 'sum')


def test_list_ensemble_is_not_batched():
    members = [{'x': x} for x in _xs()]
    with pytest.raises(ValueError, match='batched'):
        map_reduce(lambda m: m['x'], ListEnsemble(members, policy=Policy(batch=2)), 'sum')


def test_members_share_their_treedef():
    with pytest.raises(ValueError, match='treedef'):
        ListEnsemble([{'x': jnp.zeros(2)}, {'y': jnp.zeros(2)}])


def test_stacked_leaves_match_their_strides():
    with pytest.raises(ValueError, match='entries'):
        StackedEnsemble(N, {'x': jnp.zeros((3, 2))}, 1)
    with pytest.raises(ValueError, match='does not divide'):
        StackedEnsemble(N, {'x': jnp.zeros((2, 2))}, 3)


def test_stacked_outputs_come_back_in_position_order():
    out, _ = map_reduce(lambda m: m['x'], _stacked(policy=Policy(batch=2)), 1)
    assert_allclose(out, _xs())


def test_the_body_is_traced_once_whatever_the_ensemble_size():
    def hlo(n: int) -> str:
        xs = jnp.arange(float(n * 16)).reshape(n, 16)

        def run(xs):
            ensemble = StackedEnsemble(n, {'x': xs}, 1)
            return map_reduce(lambda m: jnp.fft.fft(m['x']).real.sum(), ensemble, 'sum')[0]

        return jax.jit(run).lower(xs).compile().as_text()

    counts = [len(re.findall(r'"fft"', hlo(n))) for n in (2, 16)]
    assert counts[0] == counts[1]


def test_temporaries_do_not_grow_with_the_members():
    def temp_bytes(n: int, batch: int) -> int:
        xs = jnp.arange(float(n * 1024)).reshape(n, 1024)

        def run(xs):
            ensemble = StackedEnsemble(n, {'x': xs}, 1, policy=Policy(batch=batch))
            return map_reduce(
                lambda m: jnp.fft.ifft(jnp.fft.fft(m['x'])).real.sum(), ensemble, 'sum'
            )[0]

        return jax.jit(run).lower(xs).compile().memory_analysis().temp_size_in_bytes

    assert temp_bytes(16, 1) < 2 * temp_bytes(4, 1)  # flat in N
    assert temp_bytes(16, 8) > 4 * temp_bytes(16, 1)  # linear in K


def test_host_stacked_outputs_feed_a_resident_ensemble():
    out, _ = map_reduce(_apply, _host(), {'sum': 'sum', 'stack': 1, 'shared': None}, 1.0)
    resident = StackedEnsemble(N, {'y': out['stack']}, 1)
    total, _ = map_reduce(lambda m: m['y'], resident, 'sum')
    assert_allclose(total, (_xs() + _shared()).sum(axis=0))


def test_from_budget_fits_the_batch_to_the_temporaries():
    ensemble = _stacked()

    def apply(member):
        return jnp.fft.ifft(jnp.fft.fft(member['x'])).real.sum()

    member = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), ensemble.fetch(0))
    temps = jax.jit(apply).lower(member).compile().memory_analysis().temp_size_in_bytes
    resident = sum(x.nbytes for x in jax.tree.leaves(ensemble.stacked))

    policy = Policy.from_budget(ensemble, apply, budget=resident + int(2.5 * temps), margin=1.0)
    assert policy.batch == 2  # room for 2.5 members, clipped to the divisors of 8
    with pytest.raises(ValueError, match='does not fit'):
        Policy.from_budget(ensemble, apply, budget=resident, margin=1.0)


def test_lift_builds_operators_member_wise():
    diagonals = jnp.arange(float(N * N_SAMPS)).reshape(N, N_SAMPS) + 1
    ensemble = StackedEnsemble(N, {'d': diagonals}, 1)
    built = lift(lambda m: DiagonalOperator(2 * m['d']), ensemble)
    assert isinstance(built, StackedEnsemble)
    assert built.size == N
    assert_allclose(built.fetch(3).diagonal, 2 * diagonals[3])


def test_lift_keeps_the_leaves_and_strides_of_a_structural_function():
    xs = jnp.arange(float(N * N_SAMPS)).reshape(N, N_SAMPS)
    by_2 = jnp.ones((N // 2, N_SAMPS))
    ensemble = StackedEnsemble(N, {'x': xs, 'b': by_2}, {'x': 1, 'b': 2})

    rearranged = lift(lambda m: {'pair': (m['b'], m['x'])}, ensemble)
    assert isinstance(rearranged, StackedEnsemble)
    assert rearranged.stacked['pair'][0] is by_2  # the arrays are reused, not copied
    assert rearranged.stacked['pair'][1] is xs
    assert rearranged.strides == {'pair': (2, 1)}


def test_lift_composes_operators_structurally():
    diagonals = jnp.arange(float(N * N_SAMPS)).reshape(N, N_SAMPS) + 1
    left = lift(lambda m: DiagonalOperator(m['d']), StackedEnsemble(N, {'d': diagonals}, 1))
    right = lift(lambda m: DiagonalOperator(2 * m['d']), StackedEnsemble(N, {'d': diagonals}, 1))
    assert isinstance(left, StackedEnsemble) and isinstance(right, StackedEnsemble)

    composed = lift(lambda a, b: a @ b, left, right)
    assert isinstance(composed, StackedEnsemble)
    assert isinstance(composed.fetch(1), CompositionOperator)
    input_leaves = {id(leaf) for leaf in jax.tree.leaves((left.stacked, right.stacked))}
    assert all(id(leaf) in input_leaves for leaf in jax.tree.leaves(composed.stacked))
    assert_allclose(composed.fetch(1)(jnp.ones(N_SAMPS)), 2 * diagonals[1] ** 2)


def test_lift_over_list_ensembles():
    lifted = lift(lambda m: {'y': 2 * m['x']}, _list())
    assert isinstance(lifted, ListEnsemble)
    assert_allclose(lifted.members[3]['y'], 2 * _xs()[3])


def test_lift_checks_the_sizes():
    with pytest.raises(ValueError, match='different sizes'):
        lift(lambda a, b: a, _list(), ListEnsemble([{'x': jnp.zeros(N_SAMPS)}]))


@pytest.mark.distributed
class TestSharded:
    def _mesh(self):
        return jax.make_mesh((jax.device_count(),), ('obs',))

    def test_sharded_matches_single_device(self):
        reference, _ = map_reduce(
            _apply, _stacked(), {'sum': 'sum', 'stack': 1, 'shared': None}, 3.0
        )
        with jax.set_mesh(self._mesh()):
            ensemble = _stacked().shard('obs')
            assert ensemble.policy.axis == 'obs'
            out, _ = map_reduce(_apply, ensemble, {'sum': 'sum', 'stack': 1, 'shared': None}, 3.0)
            from jax.sharding import PartitionSpec as P

            assert out['sum'].sharding.spec == P()  # summed over the devices, held by all
            assert out['stack'].sharding.spec == P('obs')  # stays with the devices
            assert_allclose(out['sum'], reference['sum'])
            assert_allclose(out['stack'], reference['stack'])
            assert_allclose(out['shared'], reference['shared'])

    def test_auto_is_resolved_under_sharding(self):
        with jax.set_mesh(self._mesh()):
            ensemble = _stacked().shard('obs')
            out, kinds = map_reduce(
                _apply, ensemble, {'sum': 'sum', 'stack': 'auto', 'shared': 'auto'}, 1.0
            )
            assert kinds == {'sum': 'sum', 'stack': 1, 'shared': None}
            assert_allclose(out['stack'], _expected(1.0)['stack'])

    def test_sharded_masks_padded_members(self):
        valid = jnp.asarray([True] * 6 + [False] * 2)
        with jax.set_mesh(self._mesh()):
            ensemble = _stacked(valid=valid).shard('obs')
            out, _ = map_reduce(_apply, ensemble, 'sum', 1.0)
            assert_allclose(out['sum'], _xs()[:6].sum(axis=0))

    def test_host_loop_matches_resident(self):
        reference, _ = map_reduce(
            _apply, _stacked(), {'sum': 'sum', 'stack': 1, 'shared': None}, 1.0
        )
        with jax.set_mesh(self._mesh()):
            ensemble = _host(policy=Policy(axis='obs'))
            out, _ = map_reduce(_apply, ensemble, {'sum': 'sum', 'stack': 1, 'shared': None}, 1.0)
            assert_allclose(out['sum'], reference['sum'])
            assert_allclose(out['stack'], reference['stack'])

    def test_only_a_stacked_ensemble_is_sharded(self):
        with jax.set_mesh(self._mesh()):
            members = [{'x': x} for x in _xs()]
            ensemble = ListEnsemble(members, policy=Policy(axis='obs'))
            with pytest.raises(ValueError, match='mesh axis'):
                map_reduce(lambda m: m['x'], ensemble, 'sum')
