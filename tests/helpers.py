from math import prod
from pathlib import Path

import jax
import jax.numpy as jnp

TEST_DATA = Path(__file__).parent / 'data'
TEST_DATA_PLANCK = TEST_DATA / 'planck'
TEST_DATA_SAT = TEST_DATA / 'sat'
TEST_DATA_SEDS = Path(__file__).parent / 'seds/data'


def arange(*shape: int, dtype=jnp.float32, start=1) -> jax.Array:
    return jnp.arange(start, prod(shape) + start, dtype=dtype).reshape(shape)
