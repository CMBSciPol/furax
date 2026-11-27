import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

import furax
from furax import AbstractLinearOperator, DiagonalOperator, HomothetyOperator
from furax.core import (
    AdditionOperator,
    AssemblyOperator,
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    CompositionOperator,
    DenseBlockDiagonalOperator,
    HostEnsemble,
    Policy,
    RaggedAssemblyOperator,
    StackedEnsemble,
    lift,
    map_reduce,
)

M = 4

# An assembly whose members each read the whole map, their own amplitudes and those of their
# segment
OWNER = np.array([0, 0, 1, 2, 2, 2])
STRIDES = {'map': None, 'own': 1, 'seg': 'segment'}


def _joint(n: int = len(OWNER), owner: np.ndarray = OWNER) -> StackedEnsemble:
    """The joint operators $J_i$, from `{'map', 'own', 'seg'}` to the TOD of member i."""
    k = jax.random.split(jax.random.key(0), 3)
    segments = int(owner.max()) + 1
    diagonals = {
        'map': jax.random.normal(k[0], (n, M)),
        'own': jax.random.normal(k[1], (n, M)),
        'seg': jax.random.normal(k[2], (segments, M)),
    }
    tiles = StackedEnsemble(n, diagonals, STRIDES | {'map': 1}, owner=jnp.asarray(owner))
    joint = lift(
        lambda d: BlockRowOperator({key: DiagonalOperator(v) for key, v in d.items()}), tiles
    )
    assert isinstance(joint, StackedEnsemble)
    return joint


def _unknowns(n: int = len(OWNER), segments: int = int(OWNER.max()) + 1) -> dict[str, jax.Array]:
    k = jax.random.split(jax.random.key(1), 3)
    return {
        'map': jax.random.normal(k[0], (M,)),
        'own': jax.random.normal(k[1], (n, M)),
        'seg': jax.random.normal(k[2], (segments, M)),
    }


def _diagonal_members() -> StackedEnsemble:
    diagonals = jnp.arange(1.0, 1 + len(OWNER) * M).reshape(len(OWNER), M)
    members = lift(DiagonalOperator, StackedEnsemble(len(OWNER), diagonals))
    assert isinstance(members, StackedEnsemble)
    return members


def test_assembly_applies_each_member_to_its_parts():
    joint = _joint()
    normal = AssemblyOperator(lift(lambda J: J.T @ J, joint), in_strides=STRIDES)
    x = _unknowns()
    expected = furax.tree.zeros_like(x)
    for i, s in enumerate(OWNER):
        J = joint.fetch(i)
        y = (J.T @ J)({'map': x['map'], 'own': x['own'][i], 'seg': x['seg'][s]})
        expected['map'] += y['map']
        expected['own'] = expected['own'].at[i].set(y['own'])
        expected['seg'] = expected['seg'].at[s].add(y['seg'])
    assert normal.in_structure == normal.out_structure == furax.tree.as_structure(x)
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-12), normal(x), expected)


def test_assembly_transposes_with_its_strides():
    # non-square: the unknowns to the TOD of each member, stacked
    op = AssemblyOperator(_joint(), in_strides=STRIDES, out_strides=1)
    assert op.out_structure.shape == (len(OWNER), M)
    transposed = op.T
    assert isinstance(transposed, AssemblyOperator)
    assert transposed.in_strides == op.out_strides and transposed.out_strides == op.in_strides
    assert_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-12)
    assert_allclose((op.T @ op).as_matrix(), op.as_matrix().T @ op.as_matrix(), rtol=1e-12)


def test_assembly_tags_need_the_same_strides_on_both_sides():
    members = _diagonal_members()
    assert AssemblyOperator(members, in_strides=1).is_symmetric
    assert not AssemblyOperator(members, in_strides=1, out_strides=None).is_symmetric


def test_assembly_is_an_operand_of_a_sum():
    op = AssemblyOperator(_diagonal_members(), in_strides=1)
    x = jnp.ones((len(OWNER), M))
    assert_allclose((op + op)(x), 2 * op(x))


@pytest.mark.parametrize(
    ('cls', 'operands', 'strides', 'error', 'match'),
    [
        (AssemblyOperator, _diagonal_members, 2, ValueError, 'None, 1 or "segment"'),
        (AssemblyOperator, _diagonal_members, 'segment', ValueError, 'owner index'),
        (AssemblyOperator, lambda: _list_joint(), 1, TypeError, 'RaggedAssemblyOperator'),
        (RaggedAssemblyOperator, _diagonal_members, 1, TypeError, 'a list of operators'),
    ],
    ids=['stride', 'segment', 'list-to-ensemble-assembly', 'ensemble-to-ragged-assembly'],
)
def test_assembly_rejects_invalid_operands_and_strides(cls, operands, strides, error, match):
    with pytest.raises(error, match=match):
        cls(operands(), in_strides=strides)


@pytest.mark.distributed
def test_assembly_over_a_sharded_ensemble():
    n_devices = jax.device_count()
    owner = np.arange(2 * n_devices) // 2  # a segment of two members per device
    joint = _joint(len(owner), owner)
    normal = AssemblyOperator(lift(lambda J: J.T @ J, joint), in_strides=STRIDES)
    x = _unknowns(len(owner), n_devices)
    expected = normal(x)
    mesh = jax.make_mesh((n_devices,), ('obs',))
    with jax.set_mesh(mesh):
        sharded = AssemblyOperator(lift(lambda J: J.T @ J, joint.shard('obs')), in_strides=STRIDES)
        # the strided amplitudes are laid out with their members, the map held by every device
        laid_out = {'own', 'seg'}
        x = {k: jax.device_put(v, P('obs') if k in laid_out else P()) for k, v in x.items()}
        got = sharded(x)
    assert got['own'].sharding.spec[0] == got['seg'].sharding.spec[0] == 'obs'
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-12), got, expected)


# An assembly of a list of operands whose own amplitudes differ in size, which share the map
_SIZES = (2, 3)
_LIST_STRIDES = {'map': None, 'own': 1}


def _struct(n: int) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct((n,), jnp.float64)


def _list_joint() -> list[AbstractLinearOperator]:
    """The joint operators, from `{'map', 'own'}` to a TOD of the size of each one's own part."""
    keys = jax.random.split(jax.random.key(0), 2 * len(_SIZES))
    return [
        BlockRowOperator(
            {
                'map': DenseBlockDiagonalOperator(
                    jax.random.normal(keys[2 * g], (n, M)), in_structure=_struct(M)
                ),
                'own': DiagonalOperator(jax.random.normal(keys[2 * g + 1], (n,))),
            }
        )
        for g, n in enumerate(_SIZES)
    ]


def _list_unknowns() -> dict[str, jax.Array | list[jax.Array]]:
    keys = jax.random.split(jax.random.key(1), 1 + len(_SIZES))
    own = [jax.random.normal(k, (n,)) for k, n in zip(keys[1:], _SIZES)]
    return {'map': jax.random.normal(keys[0], (M,)), 'own': own}


@pytest.mark.parametrize('sequential', [False, True])
def test_list_assembly_applies_each_operand_to_its_parts(sequential: bool) -> None:
    joint = _list_joint()
    normal = RaggedAssemblyOperator(
        [J.T @ J for J in joint], in_strides=_LIST_STRIDES, sequential=sequential
    )
    x = _list_unknowns()
    parts = [(J.T @ J)({'map': x['map'], 'own': x['own'][g]}) for g, J in enumerate(joint)]
    expected = {'map': sum(p['map'] for p in parts), 'own': [p['own'] for p in parts]}
    assert normal.in_structure == normal.out_structure == furax.tree.as_structure(x)
    jax.tree.map(lambda a, b: assert_allclose(a, b, rtol=1e-12), normal(x), expected)


def test_list_assembly_transposes_with_its_strides() -> None:
    # non-square: the unknowns to the TOD of each operand, one entry each
    op = RaggedAssemblyOperator(_list_joint(), in_strides=_LIST_STRIDES, out_strides=1)
    assert [s.shape for s in op.out_structure] == [(n,) for n in _SIZES]
    assert_allclose(op.T.as_matrix(), op.as_matrix().T, rtol=1e-12)
    assert_allclose((op.T @ op).as_matrix(), op.as_matrix().T @ op.as_matrix(), rtol=1e-12)


def test_list_assembly_tags_need_the_same_strides_on_both_sides() -> None:
    structures = [{'map': _struct(M), 'own': _struct(n)} for n in _SIZES]
    scalings = [HomothetyOperator(2.0, in_structure=structure) for structure in structures]
    scaled = RaggedAssemblyOperator(scalings, in_strides=_LIST_STRIDES)
    assert scaled.is_symmetric
    total = scaled + scaled
    assert isinstance(total, AdditionOperator) and total.operand_leaves == [scaled, scaled]
    joint = RaggedAssemblyOperator(_list_joint(), in_strides=_LIST_STRIDES, out_strides=1)
    assert not joint.is_square and not joint.is_symmetric
    # the same strides, but only the first operand is square
    operands = [HomothetyOperator(2.0, in_structure=_struct(3)), _list_joint()[0]]
    assert not RaggedAssemblyOperator(operands, in_strides=1).is_square


@pytest.mark.parametrize(
    ('strides', 'match'),
    [
        ('segment', 'None or 1'),
        (None, 'differ on a part of stride None'),
        ({'map': None, 'other': 1}, 'do not fit the operands'),
    ],
    ids=['segment', 'shared-part-differs', 'not-a-prefix'],
)
def test_list_assembly_rejects_invalid_strides(strides, match) -> None:
    operands = [J.T @ J for J in _list_joint()]
    with pytest.raises(ValueError, match=match):
        RaggedAssemblyOperator(operands, in_strides=strides, out_strides=_LIST_STRIDES)


# The factors of a marginalized normal operator $J^\top (I - T_w K T^\top) D J$, each an assembly
# over the same tiles: the joint operator $J$ to the TOD of each tile, its weight $D$, the modes $T$
# and $T_w = D T$ of each segment, and their core $K$, a block diagonal over the segments
_TILE_STRIDES = STRIDES | {'map': 1, 'd': 1, 't': 1, 'k': 'segment'}


def _tiles(policy: Policy | None = None, owner: np.ndarray = OWNER) -> StackedEnsemble:
    n, segments = len(owner), int(owner.max()) + 1
    k = jax.random.split(jax.random.key(2), 6)
    leaves = {
        'map': jax.random.normal(k[0], (n, M)),
        'own': jax.random.normal(k[1], (n, M)),
        'seg': jax.random.normal(k[2], (segments, M)),
        'd': jax.random.uniform(k[3], (n, M)) + 1,
        't': jax.random.normal(k[4], (n, M)),
        'k': jax.random.normal(k[5], (segments, M)),
    }
    policy = policy or Policy()
    return StackedEnsemble(n, leaves, _TILE_STRIDES, policy=policy, owner=jnp.asarray(owner))


def _factors(tiles: StackedEnsemble) -> dict[str, AbstractLinearOperator]:
    def assembly(build, in_strides, out_strides=1):
        return AssemblyOperator(lift(build, tiles), in_strides, out_strides)

    T = assembly(lambda m: DiagonalOperator(m['t']), 'segment')
    return {
        'J': assembly(
            lambda m: BlockRowOperator({key: DiagonalOperator(m[key]) for key in STRIDES}), STRIDES
        ),
        'D': assembly(lambda m: DiagonalOperator(m['d']), 1),
        'T': T,
        'Tt': T.T,
        'Tw': assembly(lambda m: DiagonalOperator(m['d'] * m['t']), 'segment'),
        'K': BlockDiagonalOperator(lift(lambda m: DiagonalOperator(m['k']), tiles).by_segment()),
    }


def _matrices_agree(fused: AbstractLinearOperator, unfused: AbstractLinearOperator) -> None:
    assert_allclose(fused.as_matrix(), unfused.as_matrix(), rtol=1e-12, atol=1e-12)
    assert_allclose(fused.T.as_matrix(), unfused.as_matrix().T, rtol=1e-12, atol=1e-12)


def test_reduce_keeps_the_arrays_at_their_strides():
    # each member's scale times its segment's, which reducing the member multiplies: lifted as
    # is, the product would take one entry per member
    n, segments = len(OWNER), int(OWNER.max()) + 1
    scales = StackedEnsemble(
        n,
        {'segment': jnp.arange(1.0, segments + 1), 'member': jnp.arange(1.0, n + 1)},
        {'segment': 'segment', 'member': 1},
        owner=jnp.asarray(OWNER),
    )

    def member(m):
        return CompositionOperator(
            [HomothetyOperator(m[k], in_structure=_struct(M)) for k in ('segment', 'member')]
        )

    op = AssemblyOperator(lift(member, scales), 1)
    reduced = op.reduce()
    assert reduced.operands.strides == op.operands.strides == ('segment', 1)
    assert reduced.reduce() is reduced
    _matrices_agree(reduced, op)


def test_composition_fuses_through_one_entry_per_member():
    f = _factors(_tiles())
    unfused = f['Tt'] @ f['D'] @ f['J']
    fused = unfused.reduce()
    assert isinstance(fused, AssemblyOperator)
    assert fused.in_strides == f['J'].in_strides and fused.out_strides == f['Tt'].out_strides
    _matrices_agree(fused, unfused)


def _other_policy() -> AbstractLinearOperator:
    return _factors(_tiles(Policy(batch=2)))['D']


def _other_owner() -> AbstractLinearOperator:
    return _factors(_tiles(owner=np.array([0, 0, 0, 1, 1, 2])))['D']


def _reads_whole() -> AbstractLinearOperator:
    # reads the TOD of every tile whole, rather than its own entry
    diagonals = StackedEnsemble(len(OWNER), jnp.ones((len(OWNER), len(OWNER), M)))
    return AssemblyOperator(lift(DiagonalOperator, diagonals), None, 1)


@pytest.mark.parametrize(
    ('left', 'right'),
    [
        (lambda f: f['T'], lambda f: f['Tt']),
        (lambda f: f['J'], lambda f: f['J'].T),
        (lambda f: _other_policy(), lambda f: f['J']),
        (lambda f: _other_owner(), lambda f: f['J']),
        (lambda f: _reads_whole(), lambda f: f['J']),
    ],
    ids=['through-segment', 'through-none', 'other-policy', 'other-owner', 'mismatched-strides'],
)
def test_composition_does_not_fuse(left, right):
    f = _factors(_tiles())
    unfused = left(f) @ right(f)
    reduced = unfused.reduce()
    assert isinstance(reduced, CompositionOperator) and len(reduced.operands) == 2
    _matrices_agree(reduced, unfused)


def test_sum_fuses_with_the_same_strides():
    f = _factors(_tiles())
    unfused = AdditionOperator([f['Tt'] @ f['Tw'], f['Tt'] @ f['T']])
    fused = unfused.reduce()
    assert isinstance(fused, AssemblyOperator)
    _matrices_agree(fused, unfused)
    # the same strides, over another ensemble
    unfused = AdditionOperator([f['D'], _other_owner()])
    assert isinstance(unfused.reduce(), AdditionOperator)


@pytest.mark.parametrize('transposed', [False, True], ids=['shared-output', 'shared-input'])
def test_sum_fuses_assemblies_sharing_a_vector(transposed):
    # the terms fuse on the side where their assemblies are over the same tiles, and only there
    f = _factors(_tiles())
    unfused = f['J'].T @ _other_owner() + f['J'].T @ f['D']
    if transposed:
        unfused = unfused.T
    fused = unfused.reduce()
    assert isinstance(fused, CompositionOperator)
    sweep, other = fused.operands[::-1] if transposed else fused.operands
    assert isinstance(sweep, AssemblyOperator)
    assert isinstance(other, BlockRowOperator if transposed else BlockColumnOperator)
    _matrices_agree(fused, unfused)


def test_low_rank_normal_operator_takes_two_sweeps():
    # $A - B^\top K B$: the normal operator of each tile, less its modes, coupled by segment
    f = _factors(_tiles())
    modes = f['Tt'] @ f['D'] @ f['J']
    unfused = f['J'].T @ f['D'] @ f['J'] - modes.T @ f['K'] @ modes
    fused = unfused.reduce()
    assert isinstance(fused, CompositionOperator)
    row, sweep = fused.operands
    # one sweep reads the unknowns for both terms, which only the modes leave per segment
    assert isinstance(sweep, AssemblyOperator)
    assert sweep.out_strides == (*f['J'].in_strides, 'segment')
    # the other applies the transpose of the modes, after their core
    assert isinstance(row, BlockRowOperator)
    _, correction = row.blocks
    assert sum(isinstance(op, AssemblyOperator) for op in correction.operands) == 1
    _matrices_agree(fused, unfused)


def test_fused_members_evaluate_over_a_host_ensemble():
    # the TOD of each tile read from the host, the fused operators zipped with it
    f = _factors(_tiles())
    fused = (f['D'] @ f['J']).reduce()
    assert isinstance(fused, AssemblyOperator) and isinstance(fused.operands, StackedEnsemble)
    x = _unknowns()
    ops = fused.operands
    leaves, treedef = jax.tree.flatten(ops.stacked)
    reads = HostEnsemble(
        ops.size,
        lambda i: np.asarray(x['own'][i]),
        jax.ShapeDtypeStruct((M,), x['own'].dtype),
        owner=OWNER,
        segments=int(OWNER.max()) + 1,
    ).zip(
        (leaves, x['map'], x['seg']),
        axes=(list(ops.axes), None, 0),
        strides=(list(ops.strides), None, 'segment'),
    )

    def apply(member):
        own, (op_leaves, map_, seg) = member
        op = jax.tree.unflatten(treedef, op_leaves)
        return op({'map': map_, 'own': own, 'seg': seg})

    got, _ = map_reduce(apply, reads, 1)
    assert_allclose(got, fused(x), rtol=1e-12)
    assert_allclose(got, (f['D'] @ f['J'])(x), rtol=1e-12)
