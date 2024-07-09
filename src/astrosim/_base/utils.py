from typing import Any

import jax


def is_leaf(x: Any) -> bool:
    """Returns true if the input is a Pytree leaf."""
    leaves = jax.tree.leaves(x)
    return len(leaves) == 1 and x is leaves[0]
