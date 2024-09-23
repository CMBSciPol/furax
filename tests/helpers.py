from math import prod
from pathlib import Path

import jax
import jax.numpy as jnp

TEST_DATA = Path(__file__).parent / 'data'
TEST_DATA_PLANCK = TEST_DATA / 'planck'
TEST_DATA_SAT = TEST_DATA / 'sat'


def arange(*shape: int, dtype=jnp.float32) -> jax.Array:
    return jnp.arange(prod(shape), dtype=dtype).reshape(shape)
