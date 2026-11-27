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
    BlockRowOperator,
    DenseBlockDiagonalOperator,
    RaggedAssemblyOperator,
    StackedEnsemble,
    lift,
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
