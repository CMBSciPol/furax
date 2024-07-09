import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from jax import Array
from jaxtyping import Float
from numpy.testing import assert_array_equal

from astrosim._base.dense import DenseBlockDiagonalOperator


@pytest.mark.parametrize(
    'subscripts, expected_subscripts',
    [
        ('ij...,j...->i...', 'ji...,j...->i...'),
        ('hij...,hj...->hi...', 'hji...,hj...->hi...'),
        ('ikj,kj->ki', 'jki,kj->ki'),
    ],
)
def test_get_transposed_subscripts(subscripts: str, expected_subscripts: str) -> None:
    actual_subscripts = DenseBlockDiagonalOperator._get_transposed_subscripts(subscripts)
    assert actual_subscripts == expected_subscripts


@pytest.mark.parametrize(
    'subscripts, x',
    [
        ('ij,j->i', jnp.array([1.0, 2, 3])),
        ('ji,j->i', jnp.array([1.0, 2])),
        ('ij...,j...->i...', jnp.array([1.0, 2, 3])),
        ('ji...,j...->i...', jnp.array([1.0, 2])),
        ('...ij,...j->...i', jnp.array([1.0, 2, 3])),
        ('...ji,...j->...i', jnp.array([1.0, 2])),
    ],
)
def test_blocks2d(subscripts: str, x: Float[Array, '...']) -> None:
    blocks = jnp.array([[1.0, 2, 3], [2, 3, 4]])
    in_structure = jax.ShapeDtypeStruct(x.shape, x.dtype)
    op = DenseBlockDiagonalOperator(blocks, in_structure=in_structure, subscripts=subscripts)

    y = op(x)
    expected_y = jnp.einsum(subscripts, blocks, x)
    assert_array_equal(y, expected_y)

    assert_array_equal(op.T.as_matrix().T, op.as_matrix())


@pytest.mark.parametrize(
    'subscripts, in_shape, x',
    [
        ('kij,kj->ki', None, jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])),
        ('ikj,kj->ki', None, jnp.array([[1, 2], [3, 4], [5, 6]])),
        ('ijk,kj->ki', None, jnp.array([[1, 2, 3], [4, 5, 6]])),
        ('kij,kj->ki', (4, 2), jnp.array([[1, 2]])),
        ('ikj,kj->ki', (3, 2), jnp.array([[1, 2]])),
        ('ijk,kj->ki', (2, 3), jnp.array([[1, 2, 3]])),
        ('kij...,kj...->ki...', None, jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])),
        ('ikj...,kj...->ki...', None, jnp.array([[1, 2], [3, 4], [5, 6]])),
        ('ijk...,kj...->ki...', None, jnp.array([[1, 2, 3], [4, 5, 6]])),
        ('...kij,...kj->...ki', None, jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])),
        ('...ikj,...kj->...ki', None, jnp.array([[1, 2], [3, 4], [5, 6]])),
        ('...ijk,...kj->...ki', None, jnp.array([[1, 2, 3], [4, 5, 6]])),
        ('kij...,kj...->ki...', (4, 2), jnp.array([[1, 2]])),
        ('ikj...,kj...->ki...', (3, 2), jnp.array([[1, 2]])),
        ('ijk...,kj...->ki...', (2, 3), jnp.array([[1, 2, 3]])),
        ('...kij,...kj->...ki', (4, 2), jnp.array([[1, 2]])),
        ('...ikj,...kj->...ki', (3, 2), jnp.array([[1, 2]])),
        ('...ijk,...kj->...ki', (2, 3), jnp.array([[1, 2, 3]])),
        ('ij...,j...->i...', None, jnp.array([[1, -1], [2, -2], [3, -3]])),
        ('...ij,...j->...i', None, jnp.array([[1, -1], [2, -2], [3, -3], [4, -4]])),
        ('ij...,j...->i...', (3, 2), jnp.array([[1, -1]])),
        ('...ij,...j->...i', (4, 2), jnp.array([[1, -1]])),
    ],
)
def test_blocks3d(subscripts: str, in_shape: tuple[int, ...], x: Float[Array, '...']) -> None:
    blocks = jnp.arange(2 * 3 * 4).reshape(4, 3, 2)
    in_structure = jax.ShapeDtypeStruct(in_shape or x.shape, x.dtype)
    op = DenseBlockDiagonalOperator(blocks, in_structure=in_structure, subscripts=subscripts)

    y = op(x)
    expected_y = jnp.einsum(subscripts, blocks, x)
    assert_array_equal(y, expected_y)
    assert_array_equal(op.T.as_matrix().T, op.as_matrix())


@pytest.mark.parametrize(
    'x',
    [
        [jnp.array([1, 2])],
        [jnp.array([1, 2]), jnp.array([3, 4])],
        {'a': jnp.array([1, 2]), 'b': jnp.array([3, 4])},
    ],
)
def test_vector_pytree(x: Float[Array, '...']) -> None:
    blocks = jnp.arange(3 * 2).reshape(3, 2)
    leaves, treedef = jax.tree.flatten(x)
    in_structure = jax.tree.unflatten(
        treedef, [jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves]
    )
    op = DenseBlockDiagonalOperator(blocks, in_structure)
    y = op(x)
    expected_y = jax.tree.unflatten(
        treedef, [jnp.einsum(op.subscripts, blocks, leaf) for leaf in leaves]
    )
    assert tree_equal(y, expected_y)
    assert_array_equal(op.T.as_matrix().T, op.as_matrix())


@pytest.mark.parametrize(
    'x',
    [
        [jnp.array([1, 2])],
        [jnp.array([1, 2]), jnp.array([3, 4])],
        {'a': jnp.array([1, 2]), 'b': jnp.array([3, 4])},
    ],
)
def test_blocks_pytree(x: Float[Array, '...']) -> None:
    leaves, treedef = jax.tree.flatten(x)
    in_structure = jax.tree.unflatten(
        treedef, [jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves]
    )
    block_leaves = [
        jnp.array([[1, 2], [3, 4], [5, 6]]),
        jnp.array([[7, 8], [9, 10], [11, 12]]),
    ][: len(leaves)]

    blocks = jax.tree.unflatten(treedef, block_leaves)
    op = DenseBlockDiagonalOperator(blocks, in_structure)
    y = op(x)
    expected_y = jax.tree.unflatten(
        treedef,
        [jnp.einsum(op.subscripts, block, leaf) for block, leaf in zip(block_leaves, leaves)],
    )
    assert tree_equal(y, expected_y)
    assert_array_equal(op.T.as_matrix().T, op.as_matrix())
