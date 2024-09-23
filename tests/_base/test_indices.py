import jax
import jax.numpy as jnp
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from furax._base.diagonal import DiagonalOperator
from furax._base.indices import IndexOperator
from tests.helpers import arange

# U+A4FB: Lisu Letter Tone Mya Jeu
ꓽ = slice(None)


def id_func_indices(param):
    if isinstance(param, list):
        return ''
    if isinstance(param, int):
        if param < 0:
            return f'({param})'
        return str(param)
    return str(param).replace(' ', '').replace('Ellipsis', '...')


@pytest.mark.parametrize('do_jit', [False, True])
@pytest.mark.parametrize(
    'indices, expected_matrix',
    [
        (
            0,
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
        ),
        (
            (1,),
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ),
        (
            (..., slice(0, 2)),
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
        ),
        (
            (..., jnp.array([True, False, True])),
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ),
        (
            jnp.array([1, 1, 0, 0]),
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
        ),
    ],
    ids=id_func_indices,
)
def test_indices(indices, expected_matrix, do_jit: bool) -> None:
    in_structure = jax.ShapeDtypeStruct((2, 1, 3), jnp.float32)
    if isinstance(indices, tuple) and isinstance(indices[-1], Array) and indices[-1].dtype == bool:
        keywords = {'out_structure': jax.ShapeDtypeStruct((2, 1, 2), jnp.float32)}
    else:
        keywords = {}
    op = IndexOperator(indices, in_structure=in_structure, **keywords)

    if do_jit:
        func = jax.jit(lambda x: op(x))  # https://github.com/jax-ml/jax/issues/13554
    else:
        func = op

    x = arange(2, 1, 3, dtype=jnp.float32)
    assert_array_equal(func(x), x[indices])

    assert_array_equal(op.as_matrix(), expected_matrix)
    assert_array_equal(op.T.as_matrix().T, expected_matrix)


@pytest.mark.parametrize(
    'indices, expected_axes',
    [
        ((0,), [0]),
        ((0, jnp.array([1, 2])), [0, 1]),
        ((0, ꓽ, slice(None, None, 2)), [0, 2]),
        ((0, ꓽ, jnp.array([1, 2])), [0, 2]),
        ((ꓽ, ꓽ, jnp.array([1, 2])), [2]),
        ((ꓽ, ꓽ, jnp.array([True, False, True])), [2]),
        ((ꓽ, ꓽ, ꓽ), []),
        ((0, ...), [0]),
        ((0, jnp.array([1, 2]), ...), [0, 1]),
        ((0, ꓽ, slice(None, None, 2), ...), [0, 2]),
        ((0, ꓽ, jnp.array([1, 2]), ...), [0, 2]),
        ((ꓽ, ꓽ, jnp.array([1, 2]), ...), [2]),
        ((ꓽ, ꓽ, jnp.array([True, False, True]), ...), [2]),
        ((...,), []),
        (
            (
                ꓽ,
                ...,
            ),
            [],
        ),
        (
            (
                ꓽ,
                ꓽ,
                ...,
            ),
            [],
        ),
        (
            (
                ꓽ,
                ꓽ,
                ꓽ,
                ...,
            ),
            [],
        ),
        ((..., ꓽ), []),
        ((..., ꓽ, ꓽ), []),
        ((..., ꓽ, ꓽ, ꓽ), []),
        ((..., 0), [-1]),
        ((..., jnp.array([1, 2])), [-1]),
        ((..., slice(None, None, 2)), [-1]),
        ((..., jnp.array([True, False, True])), [-1]),
        ((..., 0, ꓽ), [-2]),
        ((..., 0, ꓽ, ꓽ), [-3]),
        ((0, ..., jnp.array([1, 2])), [0, -1]),
        ((..., 0, jnp.array([1, 2])), [-2, -1]),
        ((0, ..., 0, jnp.array([1, 2])), [0, -2, -1]),
    ],
    ids=id_func_indices,
)
def test_indexed_axes(indices, expected_axes: list[int]) -> None:
    in_structure = jax.ShapeDtypeStruct((2, 1, 3), jnp.float32)
    if any(isinstance(index, Array) and index.dtype == bool for index in indices):
        keywords = {'out_structure': jax.ShapeDtypeStruct((2, 1, 2), jnp.float32)}
    else:
        keywords = {}
    op = IndexOperator(indices, in_structure=in_structure, **keywords)
    assert op.indexed_axes == expected_axes


@pytest.mark.parametrize(
    'indices',
    [
        0,
        ꓽ,
        slice(1, 2),
        jnp.array([1]),
        (0,),
        (ꓽ,),
        (slice(1, 2),),
        (jnp.array([1]),),
    ],
)
def test_direct_transpose(indices) -> None:
    index_op = IndexOperator(indices, in_structure=jax.ShapeDtypeStruct((2, 3, 2), jnp.float32))
    op = index_op @ index_op.T
    reduced_op = op.reduce()
    assert reduced_op.in_structure() == op.in_structure()
    assert reduced_op.out_structure() == op.out_structure()
    assert_array_equal(reduced_op.as_matrix(), op.as_matrix())


@pytest.mark.parametrize(
    'indices',
    [
        jnp.array([0, 4, 0, 2]),
        (jnp.array([0, 4, 0, 2]),),
    ],
)
def test_rule_direct_transpose_non_unique_indices(indices) -> None:
    index_op = IndexOperator(indices, in_structure=jax.ShapeDtypeStruct((10,), jnp.float32))
    op = index_op @ index_op.T
    reduced_op = op.reduce()
    assert reduced_op.in_structure() == op.in_structure()
    assert reduced_op.out_structure() == op.out_structure()
    assert_array_equal(reduced_op.as_matrix(), op.as_matrix())


@pytest.mark.parametrize(
    'indices',
    [
        (jnp.array([0, 4, 0, 2]), ...),
        (..., jnp.array([0, 4, 0, 2])),
    ],
)
def test_rule_transpose_direct(indices) -> None:
    index_op = IndexOperator(indices, in_structure=jax.ShapeDtypeStruct((5, 6), jnp.float32))
    op = index_op.T @ index_op
    reduced_op = op.reduce()
    assert isinstance(reduced_op, DiagonalOperator)
    assert reduced_op.in_structure() == op.in_structure()
    assert reduced_op.out_structure() == op.out_structure()
    assert_array_equal(reduced_op.as_matrix(), op.as_matrix())
