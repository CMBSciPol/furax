import equinox
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    BroadcastDiagonalOperator,
    DiagonalOperator,
)
from furax.core._diagonal import DiagonalInverseOperator
from furax.tree import as_structure
from tests.helpers import arange

# U+A4FB: Lisu Letter Tone Mya Jeu
ꓽ = slice(None)


def id_func(param):
    if isinstance(param, list):
        return ''
    if isinstance(param, int):
        if param < 0:
            return f'({param})'
        return str(param)
    return str(param).replace(' ', '')


@pytest.mark.parametrize(
    'values_shape, input_shape, axes, values_index, input_index',
    [
        ((2,), (), 0, [ꓽ], [None]),
        ((2,), (), 1, [None, ꓽ], [None, None]),
        ((2,), (), -1, [ꓽ], [None]),
        ((2,), (), -2, [ꓽ, None], [None, None]),
        ((2,), (2,), 0, [ꓽ], [ꓽ]),
        ((2,), (3,), 1, [None, ꓽ], [ꓽ, None]),
        ((2,), (3,), 2, [None, None, ꓽ], [ꓽ, None, None]),
        ((2,), (2,), -1, [ꓽ], [ꓽ]),
        ((2,), (3,), -2, [ꓽ, None], [None, ꓽ]),
        ((2,), (3,), -3, [ꓽ, None, None], [None, None, ꓽ]),
        ((3,), (2, 3), -1, [None, ꓽ], [ꓽ, ꓽ]),
        ((2,), (2, 3), -2, [ꓽ, None], [ꓽ, ꓽ]),
        ((4,), (2, 3), -3, [ꓽ, None, None], [None, ꓽ, ꓽ]),
        ((4,), (2, 3), -4, [ꓽ, None, None, None], [None, None, ꓽ, ꓽ]),
        ((2,), (2, 3), 0, [ꓽ, None], [ꓽ, ꓽ]),
        ((3,), (2, 3), 1, [None, ꓽ], [ꓽ, ꓽ]),
        ((4,), (2, 3), 2, [None, None, ꓽ], [ꓽ, ꓽ, None]),
        ((4,), (2, 3), 3, [None, None, None, ꓽ], [ꓽ, ꓽ, None, None]),
        ((2,), (2, 3, 4), 0, [ꓽ, None, None], [ꓽ, ꓽ, ꓽ]),
        ((3,), (2, 3, 4), 1, [None, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((4,), (2, 3, 4), 2, [None, None, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((5,), (2, 3, 4), 3, [None, None, None, ꓽ], [ꓽ, ꓽ, ꓽ, None]),
        ((5,), (2, 3, 4), -4, [ꓽ, None, None, None], [None, ꓽ, ꓽ, ꓽ]),
        ((2,), (2, 3, 4), -3, [ꓽ, None, None], [ꓽ, ꓽ, ꓽ]),
        ((3,), (2, 3, 4), -2, [None, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((4,), (2, 3, 4), -1, [None, None, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((2, 3), (), 0, [ꓽ, ꓽ], [None, None]),
        ((2, 3), (), 1, [None, ꓽ, ꓽ], [None, None, None]),
        ((2, 3), (), -1, [ꓽ, ꓽ], [None, None]),
        ((2, 3), (), -2, [ꓽ, ꓽ, None], [None, None, None]),
        ((2, 3), (2,), (0, 1), [ꓽ, ꓽ], [ꓽ, None]),
        ((2, 3), (2,), 0, [ꓽ, ꓽ], [ꓽ, None]),
        ((2, 4), (2,), (0, 2), [ꓽ, None, ꓽ], [ꓽ, None, None]),
        ((3, 4), (2,), (1, 2), [None, ꓽ, ꓽ], [ꓽ, None, None]),
        ((3, 4), (2,), 1, [None, ꓽ, ꓽ], [ꓽ, None, None]),
        ((4, 2), (2,), (-3, -2), [ꓽ, ꓽ, None], [None, None, ꓽ]),
        ((4, 2), (2,), -2, [ꓽ, ꓽ, None], [None, None, ꓽ]),
        ((4, 2), (2,), (-3, -1), [ꓽ, None, ꓽ], [None, None, ꓽ]),
        ((4, 2), (2,), (-2, -1), [ꓽ, ꓽ], [None, ꓽ]),
        ((4, 2), (2,), -1, [ꓽ, ꓽ], [None, ꓽ]),
        ((2, 3), (2, 3), (0, 1), [ꓽ, ꓽ], [ꓽ, ꓽ]),
        ((2, 3), (2, 3), 0, [ꓽ, ꓽ], [ꓽ, ꓽ]),
        ((2, 4), (2, 3), (0, 2), [ꓽ, None, ꓽ], [ꓽ, ꓽ, None]),
        ((3, 4), (2, 3), (1, 2), [None, ꓽ, ꓽ], [ꓽ, ꓽ, None]),
        ((3, 4), (2, 3), 1, [None, ꓽ, ꓽ], [ꓽ, ꓽ, None]),
        ((4, 2), (2, 3), (-3, -2), [ꓽ, ꓽ, None], [None, ꓽ, ꓽ]),
        ((4, 2), (2, 3), -2, [ꓽ, ꓽ, None], [None, ꓽ, ꓽ]),
        ((4, 3), (2, 3), (-3, -1), [ꓽ, None, ꓽ], [None, ꓽ, ꓽ]),
        ((2, 3), (2, 3), (-2, -1), [ꓽ, ꓽ], [ꓽ, ꓽ]),
        ((2, 3), (2, 3), -1, [ꓽ, ꓽ], [ꓽ, ꓽ]),
        ((2, 3), (2, 3, 4), (0, 1), [ꓽ, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((2, 3), (2, 3, 4), 0, [ꓽ, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((2, 4), (2, 3, 4), (0, 2), [ꓽ, None, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((3, 4), (2, 3, 4), (1, 2), [None, ꓽ, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((3, 4), (2, 3, 4), 1, [None, ꓽ, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((2, 3), (2, 3, 4), (-3, -2), [ꓽ, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((2, 3), (2, 3, 4), -2, [ꓽ, ꓽ, None], [ꓽ, ꓽ, ꓽ]),
        ((2, 4), (2, 3, 4), (-3, -1), [ꓽ, None, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((3, 4), (2, 3, 4), (-2, -1), [None, ꓽ, ꓽ], [ꓽ, ꓽ, ꓽ]),
        ((3, 4), (2, 3, 4), -1, [None, ꓽ, ꓽ], [ꓽ, ꓽ, ꓽ]),
        # add more tests here`
    ],
    ids=id_func,
)
def test_broadcast_diagonal(
    values_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    values_index: tuple[slice | None, ...],
    input_index: tuple[slice | None, ...],
) -> None:
    input = arange(*input_shape)
    values = arange(*values_shape)
    op = BroadcastDiagonalOperator(values, axis_destination=axes, in_structure=as_structure(input))
    assert_allclose(op(input), values[tuple(values_index)] * input[tuple(input_index)])


@pytest.mark.parametrize(
    'values_shape, input_shape, axes',
    [
        ((2,), (2,), 0),
        ((2,), (2,), -1),
        ((2,), (2, 3), 0),
        ((2,), (2, 3), -2),
        ((3,), (2, 3), 1),
        ((3,), (2, 3), -1),
    ],
)
def test_diagonal_as_matrix(
    values_shape: tuple[int, ...],
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
) -> None:
    input = arange(*input_shape)
    values = arange(*values_shape)
    op = DiagonalOperator(values, axis_destination=axes, in_structure=as_structure(input))
    assert_array_equal(op.as_matrix(), AbstractLinearOperator.as_matrix(op))


def test_diagonal_moveaxis() -> None:
    input = arange(3, 4)
    values = arange(4, 3)
    op = DiagonalOperator(values, in_structure=as_structure(input), axis_destination=(1, 0))
    assert_allclose(op(input), values.T * input)
    assert_allclose(op.as_matrix(), jnp.diag(values.T.ravel()))


def test_inverse_diagonal_with_zeros() -> None:
    x = (jnp.ones(4), jnp.array([1.0]))
    op = BlockDiagonalOperator(
        (
            DiagonalOperator(jnp.arange(4), in_structure=as_structure(x[0])),
            DiagonalOperator(jnp.array([0.0]), in_structure=as_structure(x[1])),
        )
    )
    inv_op = op.I
    assert type(inv_op) is BlockDiagonalOperator
    assert type(inv_op.blocks[0]) is DiagonalInverseOperator
    assert type(inv_op.blocks[1]) is DiagonalInverseOperator
    expected_y = jnp.array([0, 1, 1, 1]), jnp.array([0])
    equinox.tree_equal(op.I(op(x)), expected_y)
    assert_allclose(op.I.as_matrix(), jnp.diag(jnp.array([0.0, 1.0, 1 / 2, 1 / 3, 0.0])))


def test_broadcast_diagonal_values_scalar() -> None:
    x = jnp.arange(3)
    values = jnp.array(2.0)
    with pytest.raises(ValueError, match='the diagonal values are scalar'):
        _ = BroadcastDiagonalOperator(values, in_structure=as_structure(x))


def test_broadcast_diagonal_values_pytree() -> None:
    x = jnp.arange(3)
    values = {'a': jnp.array(2.0)}
    with pytest.raises(ValueError, match='the diagonal values cannot be a pytree'):
        _ = BroadcastDiagonalOperator(values, in_structure=as_structure(x))


def test_broadcast_diagonal_wrong_shape() -> None:
    x = jnp.arange(3)
    values = jnp.ones(2)
    with pytest.raises(ValueError):
        _ = BroadcastDiagonalOperator(values, in_structure=as_structure(x))


def test_broadcast_diagonal_dup_axes() -> None:
    x = {'a': arange(3, 3), 'b': jnp.arange(3)}
    values = arange(3, 3)
    with pytest.raises(
        ValueError, match=r'duplicated axis destination \[0, -1\] for leaf of shape \(3,\)'
    ):
        _ = BroadcastDiagonalOperator(
            values, in_structure=as_structure(x), axis_destination=(0, -1)
        )


def test_diagonal_invalid_broadcast1() -> None:
    x = jnp.arange(3)
    values = arange(2, 3)
    with pytest.raises(ValueError, match=r'the input shape \(3,\) cannot be changed to \(2, 3\)'):
        _ = DiagonalOperator(values, in_structure=as_structure(x))


def test_diagonal_invalid_broadcast2() -> None:
    x = jnp.arange(4)
    values = arange(2, 3)
    with pytest.raises(ValueError, match=r'input shape \(4,\) cannot be changed to \(2, 3, 4\)'):
        _ = DiagonalOperator(values, in_structure=as_structure(x), axis_destination=(-3, -2))
    with pytest.raises(ValueError, match=r'input shape \(4,\) cannot be changed to \(4, 2, 3\)'):
        _ = DiagonalOperator(values, in_structure=as_structure(x), axis_destination=1)
