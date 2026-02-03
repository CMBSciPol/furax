import dataclasses
from collections.abc import Iterable
from typing import Any, TypeVar

from jax._src.tree_util import GetAttrKey, register_pytree_with_keys

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


_T = TypeVar('_T')


def register_dataclass_with_keys(cls: type[_T]) -> type[_T]:
    """Register a dataclass as a pytree node, bypassing __init__ during unflatten.

    The motivation to reimplement jax.tree_util.register_dataclass comes from the fact that it does not handle
    dataclasses with an __init__ constructor that does not match the fields of the dataclass.
    """
    fields = dataclasses.fields(cls)  # type: ignore[arg-type]
    data_fields = tuple(f.name for f in fields if not f.metadata.get('static', False))
    meta_fields = tuple(f.name for f in fields if f.metadata.get('static', False))

    def flatten_with_keys(obj: _T) -> tuple[list[tuple[GetAttrKey, Any]], tuple[Any, ...]]:
        data = [(GetAttrKey(name), getattr(obj, name)) for name in data_fields]
        meta = tuple(getattr(obj, name) for name in meta_fields)
        return data, meta

    def unflatten(meta: tuple[Any, ...], data: Iterable[Any]) -> _T:
        obj = object.__new__(cls)
        for name, value in zip(data_fields, data):
            object.__setattr__(obj, name, value)
        for name, value in zip(meta_fields, meta):
            object.__setattr__(obj, name, value)
        return obj

    def flatten(obj: _T) -> tuple[list[Any], tuple[Any, ...]]:
        data = [getattr(obj, name) for name in data_fields]
        meta = tuple(getattr(obj, name) for name in meta_fields)
        return data, meta

    register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls
