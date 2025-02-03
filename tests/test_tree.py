import jax
import pytest
from equinox import tree_equal
from jax import numpy as jnp

import furax as fx
from furax.obs.stokes import StokesIQU


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
