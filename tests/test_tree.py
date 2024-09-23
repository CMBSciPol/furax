import jax
import pytest
from equinox import tree_equal
from jax import numpy as jnp

import furax as fx


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
