from typing import Any, TypeVar

import jax


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
