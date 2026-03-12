import itertools

import jax
import pytest
from equinox import tree_equal
from jax import Array
from jax import numpy as jnp
from jax.tree_util import PyTreeDef
from jaxtyping import PyTree
from numpy.testing import assert_array_equal

import furax as fx
from furax.obs.stokes import StokesIQU, StokesIQUV
from furax.tree import _dense_to_tree, _get_outer_treedef, _tree_to_dense


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jnp.ones(2, dtype=jnp.float32), jnp.ones((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((2,), dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [
                jax.ShapeDtypeStruct((2,), dtype=jnp.float32),
                jax.ShapeDtypeStruct((), dtype=jnp.float32),
            ],
        ),
    ],
)
def test_as_promoted_dtype(x, expected_y) -> None:
    y = fx.tree.as_promoted_dtype(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jax.ShapeDtypeStruct((2,), dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((2,), jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
        ),
    ],
)
def test_as_structure(x, expected_y) -> None:
    y = fx.tree.as_structure(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.ones(2, dtype=jnp.float16), jnp.zeros(2, dtype=jnp.float16)),
        (
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.zeros(2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
        ),
    ],
)
def test_zeros_like(x, expected_y) -> None:
    y = fx.tree.zeros_like(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.ones(2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.ones(2, dtype=jnp.float16), jnp.ones((), dtype=jnp.float32)],
        ),
    ],
)
def test_ones_like(x, expected_y) -> None:
    y = fx.tree.ones_like(x)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jnp.full(2, 3, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [jnp.full(2, 3, dtype=jnp.float16), jnp.full((), 3, dtype=jnp.float32)],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jnp.full(2, 3, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [jnp.full(2, 3, dtype=jnp.float16), jnp.full((), 3, dtype=jnp.float32)],
        ),
    ],
)
def test_full_like(x, expected_y) -> None:
    y = fx.tree.full_like(x, 3)
    assert tree_equal(y, expected_y)


key_from_seed = jax.random.PRNGKey(0)
(key0,) = jax.random.split(key_from_seed, 1)
key1, key2 = jax.random.split(key_from_seed)


@pytest.mark.parametrize(
    'x, expected_y',
    [
        (jnp.zeros(2, dtype=jnp.float16), jax.random.normal(key0, 2, dtype=jnp.float16)),
        (
            [jnp.zeros(2, dtype=jnp.float16), jnp.zeros((), dtype=jnp.float32)],
            [
                jax.random.normal(key1, 2, dtype=jnp.float16),
                jax.random.normal(key2, (), dtype=jnp.float32),
            ],
        ),
        (jax.ShapeDtypeStruct((2,), jnp.float16), jax.random.normal(key0, 2, dtype=jnp.float16)),
        (
            [jax.ShapeDtypeStruct((2,), jnp.float16), jax.ShapeDtypeStruct((), jnp.float32)],
            [
                jax.random.normal(key1, 2, jnp.float16),
                jax.random.normal(key2, (), dtype=jnp.float32),
            ],
        ),
    ],
)
def test_normal_like(x, expected_y) -> None:
    y = fx.tree.normal_like(x, key_from_seed)
    assert tree_equal(y, expected_y)


@pytest.mark.parametrize(
    'x, y, expected_xy',
    [
        (jnp.ones((2,)), jnp.full((2,), 3), 6),
        ({'a': -1}, {'a': 2}, -2),
        (
            {'a': jnp.ones((2,)), 'b': jnp.array([1, 0, 1])},
            {'a': jnp.full((2,), 3), 'b': jnp.array([1, 0, -1])},
            6,
        ),
    ],
)
def test_dot(x, y, expected_xy) -> None:
    assert fx.tree.dot(x, y) == expected_xy


def test_dot_invalid_pytrees() -> None:
    with pytest.raises(ValueError, match='Dict key mismatch'):
        _ = fx.tree.dot({'a': 1}, {'b': 2})


@pytest.mark.parametrize(
    'x, expected_norm',
    [
        (jnp.array([3.0, 4.0]), 5.0),
        ({'a': jnp.array([3.0, 0.0]), 'b': jnp.array([0.0, 4.0])}, 5.0),
        ({'a': jnp.array([1.0, 2.0, 2.0])}, 3.0),
    ],
)
def test_norm(x, expected_norm) -> None:
    assert jnp.allclose(fx.tree.norm(x), expected_norm)


@pytest.mark.parametrize(
    'structure, a, x, expected_y',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 2 3 ]
            #     [ 4 5 ]
            {'r1': {'c1': 2, 'c2': 3}, 'r2': {'c1': 4, 'c2': 5}},
            {'c1': jnp.arange(3), 'c2': -1},
            {'r1': jnp.array([-3, -1, 1]), 'r2': jnp.array([-5, -1, 3])},
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 1]
            {
                'i': {'i': 1, 'q': -1, 'u': 0},
                'q': {'i': 1, 'q': 1, 'u': 0},
                'u': {'i': 0, 'q': 0, 'u': 1},
            },
            {'i': 1, 'q': -1, 'u': 3},
            {'i': 2, 'q': 0, 'u': 3},
        ),
        (
            StokesIQU.structure_for((), jnp.int32),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 0]
            StokesIQU(StokesIQU(1, -1, 0), StokesIQU(1, 1, 0), 0),
            StokesIQU(1, -1, 3),
            StokesIQU(2, 0, 0),
        ),
    ],
)
def test_matvec(structure, a, x, expected_y) -> None:
    actual_y = fx.tree.matvec(structure, a, x)
    assert tree_equal(actual_y, expected_y)


@pytest.mark.parametrize(
    'structure, a, x, expected_y',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 2 3 ]
            #     [ 4 5 ]
            {'r1': {'c1': 2, 'c2': 3}, 'r2': {'c1': 4, 'c2': 5}},
            {'r1': jnp.arange(3), 'r2': -1},
            {'c1': jnp.array([-4, -2, 0]), 'c2': jnp.array([-5, -2, 1])},
        ),
        (
            jax.tree.structure({'i': 0, 'q': 0, 'u': 0}),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 1]
            {
                'i': {'i': 1, 'q': -1, 'u': 0},
                'q': {'i': 1, 'q': 1, 'u': 0},
                'u': {'i': 0, 'q': 0, 'u': 1},
            },
            {'i': 1, 'q': -1, 'u': 3},
            {'i': 0, 'q': -2, 'u': 3},
        ),
        (
            StokesIQU.structure_for((), jnp.int32),
            #     [ 1 -1 0]
            # a = [ 1  1 0]
            #     [ 0  0 0]
            StokesIQU(StokesIQU(1, -1, 0), StokesIQU(1, 1, 0), StokesIQU(0, 0, 0)),
            StokesIQU(1, -1, 3),
            StokesIQU(0, -2, 0),
        ),
    ],
)
def test_vecmat(structure, a, x, expected_y) -> None:
    actual_y = fx.tree.vecmat(x, structure, a)
    assert tree_equal(actual_y, expected_y)


@pytest.mark.parametrize(
    'a_structure, a, b_structure, b, expected_mat',
    [
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            # a = [ 1 -1  2 ]
            #     [ 0  2 -1 ]
            {'r1': StokesIQU(1, -1, 2), 'r2': StokesIQU(0, 2, -1)},
            StokesIQU.structure_for((), jnp.int32),
            #     [  1 -1 ]
            # b = [  3  0 ]
            #     [ -1  1 ]
            StokesIQU({'c1': 1, 'c2': -1}, {'c1': 3, 'c2': 0}, {'c1': -1, 'c2': 1}),
            {'r1': {'c1': -4, 'c2': 1}, 'r2': {'c1': 7, 'c2': -1}},
        ),
    ],
)
def test_matmat(a_structure, a, b_structure, b, expected_mat) -> None:
    actual_mat = fx.tree.matmat(a_structure, a, b_structure, b)
    assert tree_equal(actual_mat, expected_mat)


@pytest.mark.parametrize(
    'outer_structure, inner_structure',
    [
        (jax.tree.structure({'r1': 0, 'r2': 0}), jax.tree.structure({'c1': 0, 'c2': 0})),
        (jax.tree.structure([(0,)]), jax.tree.structure(([0],))),
        (StokesIQU(0, 0, 0), StokesIQU(0, 0, 0)),
        (StokesIQUV(0, 0, 0, 0), StokesIQU(0, 0, 0)),
    ],
)
def test_get_outer_treedef(outer_structure: PyTreeDef, inner_structure: PyTreeDef) -> None:
    if not isinstance(outer_structure, PyTreeDef):
        outer_structure = jax.tree.structure(outer_structure)
    if not isinstance(inner_structure, PyTreeDef):
        inner_structure_ = jax.tree.structure(inner_structure)
    else:
        inner_structure_ = inner_structure
    counter = itertools.count()
    num_outer_leaves = outer_structure.num_leaves
    outer_leaves = [
        jax.tree.unflatten(inner_structure_, inner_structure_.num_leaves * [next(counter)])
        for _ in range(num_outer_leaves)
    ]
    tree = jax.tree.unflatten(outer_structure, outer_leaves)
    assert _get_outer_treedef(inner_structure, tree) == outer_structure


@pytest.mark.parametrize(
    'outer_structure, inner_structure, tree, expected_dense',
    [
        (jax.tree.structure(0), jax.tree.structure(0), jnp.ones(10), jnp.ones((10, 1, 1))),
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            jax.tree.structure({'c1': 0, 'c2': 0}),
            {'r1': {'c1': 1, 'c2': 2}, 'r2': {'c1': 3, 'c2': 0}},
            jnp.array([[1, 2], [3, 0]]),
        ),
        (
            StokesIQU(0, 0, 0),
            StokesIQU(0, 0, 0),
            StokesIQU(StokesIQU(1, 0, 0), StokesIQU(0, 2, 0), StokesIQU(0, 0, jnp.array([-1, 1]))),
            jnp.array([[[1, 0, 0], [0, 2, 0], [0, 0, -1]], [[1, 0, 0], [0, 2, 0], [0, 0, 1]]]),
        ),
    ],
)
def test_tree_to_dense(
    outer_structure: PyTreeDef,
    inner_structure: PyTreeDef,
    tree: PyTree[Array],
    expected_dense: Array,
):
    actual_dense = _tree_to_dense(outer_structure, inner_structure, tree)
    assert_array_equal(actual_dense, expected_dense)


@pytest.mark.parametrize(
    'outer_structure, inner_structure, dense, expected_tree',
    [
        (jax.tree.structure(0), jax.tree.structure(0), jnp.ones((10, 1, 1)), jnp.ones(10)),
        (
            jax.tree.structure({'r1': 0, 'r2': 0}),
            jax.tree.structure({'c1': 0, 'c2': 0}),
            jnp.array([[1, 2], [3, 0]]),
            {
                'r1': {'c1': jnp.array(1), 'c2': jnp.array(2)},
                'r2': {'c1': jnp.array(3), 'c2': jnp.array(0)},
            },
        ),
        (
            StokesIQU(0, 0, 0),
            StokesIQU(0, 0, 0),
            jnp.array([[[1, 0, 0], [0, 2, 0], [0, 0, -1]], [[1, 0, 0], [0, 2, 0], [0, 0, 1]]]),
            StokesIQU(
                StokesIQU(jnp.array([1, 1]), jnp.array([0, 0]), jnp.array([0, 0])),
                StokesIQU(jnp.array([0, 0]), jnp.array([2, 2]), jnp.array([0, 0])),
                StokesIQU(jnp.array([0, 0]), jnp.array([0, 0]), jnp.array([-1, 1])),
            ),
        ),
    ],
)
def test_dense_to_tree(
    outer_structure: PyTreeDef,
    inner_structure: PyTreeDef,
    dense: Array,
    expected_tree: PyTree[Array],
):
    actual_tree = _dense_to_tree(outer_structure, inner_structure, dense)
    assert tree_equal(actual_tree, expected_tree)
