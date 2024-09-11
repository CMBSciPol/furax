from typing import Any, TypeVar, overload

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def is_leaf(x: Any) -> bool:
    """Returns true if the input is a Pytree leaf."""
    leaves = jax.tree.leaves(x)
    return len(leaves) == 1 and x is leaves[0]


T = TypeVar('T')


class DefaultIdentityDict(dict[T, T]):
    """A dict whose default factory is the identity.

    Example:
        >>> d = DefaultIdentityDict({'a': 'b'})
        >>> d['c']
        'c'
    """

    def __getitem__(self, key: T) -> T:
        try:
            return super().__getitem__(key)
        except KeyError:
            return key


@overload
def promote_types_for(*args: ArrayLike) -> tuple[jax.Array, ...]: ...


@overload
def promote_types_for(*args: jax.ShapeDtypeStruct) -> tuple[jax.ShapeDtypeStruct, ...]: ...


def promote_types_for(
    *args: ArrayLike | jax.ShapeDtypeStruct,
) -> tuple[jax.Array | jax.ShapeDtypeStruct, ...]:
    """Promotes the data types of the specified arrays to a common dtype."""
    dtype = jnp.result_type(*args)
    return tuple(
        (
            jax.ShapeDtypeStruct(arg.shape, dtype)
            if isinstance(arg, jax.ShapeDtypeStruct)
            else jnp.astype(arg, dtype)
        )
        for arg in args
    )
