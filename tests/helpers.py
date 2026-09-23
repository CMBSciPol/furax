from math import prod

import jax
import jax.numpy as jnp


def arange(*shape: int, dtype=jnp.float32, start=1) -> jax.Array:
    """arange(2, 3) -> jnp.arange(6, dtype=jnp.float32).reshape(2, 3)"""
    return jnp.arange(start, prod(shape) + start, dtype=dtype).reshape(shape)
