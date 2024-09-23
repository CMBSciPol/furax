from typing import TypeVar

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
