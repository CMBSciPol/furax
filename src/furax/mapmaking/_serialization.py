"""Conversion between configuration dataclasses and plain YAML-ready data."""

from typing import Any

from typedload.datadumper import Dumper
from typedload.dataloader import Loader
from typedload.exceptions import TypedloadValueError

__all__ = ['deserialize', 'serialize']


def deserialize[T](cls: type[T], data: Any) -> T:
    """Build an instance of `cls` from plain data, such as a mapping read from YAML.

    Raises:
        TypedloadException: If `data` has unknown keys or values of the wrong type.
    """
    return _LOADER.load(data, cls)


def serialize(obj: Any) -> Any:
    """Convert a dataclass instance to plain data, including fields left at their default."""
    return _DUMPER.dump(obj)


def _load_float(value: Any, type_: Any) -> float:
    # YAML reads `0` or `1` as an int; accept it for a float field, but not a bool or a string.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypedloadValueError(f'Got {value!r}, expected float', value=value, type_=type_)
    return float(value)


# Unknown keys and implicit casts (e.g. '12' or 1.5 for an int field) are errors, so that a
# mistyped config fails at load time instead of running with another setting.
_LOADER = Loader(failonextra=True, basiccast=False)
# The adapter lambda reuses typedload's handler parameter names, since type checkers match it
# against the signatures of its built-in handlers.
_LOADER.handlers.insert(
    0, (lambda type_: type_ is float, lambda l, value, type_: _load_float(value, type_))
)
# Write every field, including those left at their default, so a dumped config is self-contained.
_DUMPER = Dumper(hidedefault=False)
