import operator
from abc import ABC
from collections.abc import Callable
from typing import Any, ClassVar, Literal, Self, cast, get_args, overload

import jax
import jax.numpy as jnp
import numpy as np
import wadler_lindig as wl  # type: ignore[import-untyped]
from equinox.internal._omega import _Metaω
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from jaxtyping import DTypeLike, Float, Integer, Key, PyTree, ScalarLike

from furax.tree import (
    as_promoted_dtype,
    dot,
    full_like,
    normal_like,
    ones_like,
    uniform_like,
    zeros_like,
)

__all__ = ['Stokes', 'StokesI', 'StokesQU', 'StokesIQU', 'StokesIQUV', 'ValidStokesType']

ValidStokesType = Literal['I', 'QU', 'IQU', 'IQUV']


class Stokes(ABC):
    """Stokes container backed by a single dense array with the components on the leading axis."""

    stokes: ClassVar[ValidStokesType]
    array: Array

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        letters = self.stokes
        if args and kwargs:
            raise TypeError('Stokes components must be given positionally or by keyword, not both.')
        if kwargs:
            expected = {letter.lower() for letter in letters}
            if set(kwargs) != expected:
                raise TypeError(
                    f'Expected Stokes components {sorted(expected)}, got {sorted(kwargs)}.'
                )
            components: tuple[Any, ...] = tuple(kwargs[letter.lower()] for letter in letters)
        else:
            components = args
        if len(components) != len(letters):
            raise TypeError(
                f'{type(self).__name__} expects {len(letters)} Stokes component(s), '
                f'got {len(components)}.'
            )
        self.array = self._stack(components)

    @staticmethod
    def _stack(components: tuple[Any, ...]) -> Array:
        if isinstance(components[0], jax.ShapeDtypeStruct):
            shape = components[0].shape
            dtype = jnp.result_type(*[c.dtype for c in components])
            return cast(Array, jax.ShapeDtypeStruct((len(components), *shape), dtype))
        return jnp.stack(components, axis=0)

    # ---- pytree (single backing array is the only child) ----------------------------------------
    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]:
        return (self.array,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:  # type: ignore[no-untyped-def]
        # Create an instance directly without going through from_array to avoid tracer issues
        instance = object.__new__(cls)
        instance.array = children[0]
        return instance

    @classmethod
    def from_array(cls, array: ArrayLike | jax.ShapeDtypeStruct) -> Self:
        """Wrap a backing array of shape ``(len(stokes), *shape)``."""
        arr = array if isinstance(array, jax.ShapeDtypeStruct) else jnp.asarray(array)
        if arr.shape[0] != len(cls.stokes):
            raise ValueError(
                f'{cls.__name__} expects a leading axis of {len(cls.stokes)}, got shape {arr.shape}.'
            )
        instance = object.__new__(cls)
        instance.array = cast(Array, arr)
        return instance

    def __pdoc__(self, **kwargs: Any) -> wl.AbstractDoc:
        inner = wl.pdoc(self.array, **kwargs)
        return wl.ConcatDoc(wl.TextDoc(f'{type(self).__name__}('), inner, wl.TextDoc(')'))

    def __repr__(self) -> str:
        return cast(str, wl.pformat(self, width=80))

    # ---- component access -----------------------------------------------------------------------
    def _component(self, letter: str) -> Array:
        idx = self.stokes.find(letter)
        if idx < 0:
            raise AttributeError(f'{type(self).__name__} has no Stokes component {letter!r}.')
        return self.array[idx]

    @property
    def i(self) -> Array:
        return self._component('I')

    @property
    def q(self) -> Array:
        return self._component('Q')

    @property
    def u(self) -> Array:
        return self._component('U')

    @property
    def v(self) -> Array:
        return self._component('V')

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape[1:]

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.structure_for(self.shape, self.dtype)

    def __getitem__(self, index: Integer[Array, '...']) -> Self:
        # ``index`` addresses the data axes (from the first); the leading Stokes axis is preserved.
        idx = index if isinstance(index, tuple) else (index,)
        return self.from_array(self.array[(slice(None), *idx)])

    def __eq__(self, other: object) -> Any:
        # Same-type comparison of the backing array. For ``ShapeDtypeStruct`` backings (structures)
        # this returns a bool (shape/dtype match); for concrete arrays it returns the elementwise
        # array, matching the former dataclass semantics.
        if not isinstance(other, Stokes) or self.stokes != other.stokes:
            return NotImplemented
        return self.array == other.array

    # Defining __eq__ without __hash__ makes instances unhashable (as the former dataclass was).

    def __matmul__(self, other: Any) -> Any:
        """Scalar product between Stokes pytrees."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return dot(self, other)

    def __abs__(self) -> Self:
        return self.from_array(jnp.abs(self.array))

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return self.from_array(-self.array)

    def _operand(self, other: Any) -> Any:
        if isinstance(other, Stokes):
            return other.array if type(other) is type(self) else NotImplemented

        try:
            return jnp.asarray(other)
        except TypeError:
            return NotImplemented

    def _binop(self, other: Any, fn: Callable[[Any, Any], Any], reflected: bool = False) -> Self:
        rhs = self._operand(other)
        if rhs is NotImplemented:
            # mypy's exemption for returning NotImplemented only applies inside a method
            # literally named as a dunder (e.g. __add__), not this shared helper.
            return NotImplemented  # type: ignore[no-any-return]
        return self.from_array(fn(rhs, self.array) if reflected else fn(self.array, rhs))

    def __add__(self, other: Any) -> Self:
        return self._binop(other, operator.add)

    def __sub__(self, other: Any) -> Self:
        return self._binop(other, operator.sub)

    def __mul__(self, other: Any) -> Self:
        return self._binop(other, operator.mul)

    def __truediv__(self, other: Any) -> Self:
        return self._binop(other, operator.truediv)

    def __pow__(self, other: Any) -> Self:
        if isinstance(other, _Metaω):
            return NotImplemented
        return self._binop(other, operator.pow)

    def __radd__(self, other: Any) -> Self:
        return self._binop(other, operator.add, reflected=True)

    def __rsub__(self, other: Any) -> Self:
        return self._binop(other, operator.sub, reflected=True)

    def __rmul__(self, other: Any) -> Self:
        return self._binop(other, operator.mul, reflected=True)

    def __rtruediv__(self, other: Any) -> Self:
        return self._binop(other, operator.truediv, reflected=True)

    def __rpow__(self, other: Any) -> Self:
        return self._binop(other, operator.pow, reflected=True)

    def ravel(self) -> Self:
        """Ravels the batch axes of each Stokes component."""
        return self.from_array(self.array.reshape(self.array.shape[0], -1))

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Reshape the batch axes of each Stokes component."""
        return self.from_array(self.array.reshape(self.array.shape[0], *shape))

    def rotate_qu(self, cos_2angles: Float[Array, '...'], sin_2angles: Float[Array, '...']) -> Self:
        """Rotate the Q, U components by an angle whose double-angle cos/sin are given.

        ``Q' = Q cos2a + U sin2a``, ``U' = -Q sin2a + U cos2a``; I and V are unchanged. Q and U are
        adjacent rows on the leading Stokes axis, so this updates just those two rows in place.
        A type without Q (``StokesI``) is returned unchanged.
        """
        qi = self.stokes.find('Q')
        if qi < 0:
            return self
        q, u = self.array[qi], self.array[qi + 1]
        rotated = jnp.stack([q * cos_2angles + u * sin_2angles, -q * sin_2angles + u * cos_2angles])
        return self.from_array(self.array.at[qi : qi + 2].set(rotated))

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['I']) -> type['StokesI']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['QU']) -> type['StokesQU']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['IQU']) -> type['StokesIQU']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['IQUV']) -> type['StokesIQUV']: ...

    @classmethod
    def class_for(cls, stokes: str) -> type['StokesType']:
        """Returns the StokesPyTree subclass associated to the specified Stokes types."""
        if stokes not in get_args(ValidStokesType):
            raise ValueError(f'Invalid Stokes parameters: {stokes!r}')
        requested_cls = {
            'I': StokesI,
            'QU': StokesQU,
            'IQU': StokesIQU,
            'IQUV': StokesIQUV,
        }[stokes]
        return cast(type[StokesType], requested_cls)

    @classmethod
    def structure_for(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
    ) -> Self:
        return cls.from_array(jax.ShapeDtypeStruct((len(cls.stokes), *shape), dtype))

    @classmethod
    @overload
    def from_stokes(cls, i: ArrayLike) -> 'StokesI': ...

    @classmethod
    @overload
    def from_stokes(cls, i: jax.ShapeDtypeStruct) -> 'StokesI': ...

    @classmethod
    @overload
    def from_stokes(cls, q: ArrayLike, u: ArrayLike) -> 'StokesQU': ...

    @classmethod
    @overload
    def from_stokes(cls, q: jax.ShapeDtypeStruct, u: jax.ShapeDtypeStruct) -> 'StokesQU': ...

    @classmethod
    @overload
    def from_stokes(cls, i: ArrayLike, q: ArrayLike, u: ArrayLike) -> 'StokesIQU': ...

    @classmethod
    @overload
    def from_stokes(
        cls, i: jax.ShapeDtypeStruct, q: jax.ShapeDtypeStruct, u: jax.ShapeDtypeStruct
    ) -> 'StokesIQU': ...

    @classmethod
    @overload
    def from_stokes(
        cls, i: ArrayLike, q: ArrayLike, u: ArrayLike, v: ArrayLike
    ) -> 'StokesIQUV': ...

    @classmethod
    @overload
    def from_stokes(
        cls,
        i: jax.ShapeDtypeStruct,
        q: jax.ShapeDtypeStruct,
        u: jax.ShapeDtypeStruct,
        v: jax.ShapeDtypeStruct,
    ) -> 'StokesIQUV': ...

    @classmethod
    def from_stokes(
        cls,
        *args: Any,
        **keywords: Any,
    ) -> 'Stokes':
        """Returns a StokesPyTree according to the specified Stokes vectors.

        Examples:
            >>> tod_i = Stokes.from_stokes(i)
            >>> tod_qu = Stokes.from_stokes(q, u)
            >>> tod_iqu = Stokes.from_stokes(i, q, u)
            >>> tod_iquv = Stokes.from_stokes(i, q, u, v)
        """
        if args and keywords:
            raise TypeError(
                'The Stokes parameters should be specified either through positional or keyword '
                'arguments.'
            )
        if keywords:
            stokes = ''.join(sorted(keywords))
            if stokes not in get_args(ValidStokesType):
                raise TypeError(
                    f"Invalid Stokes vectors: {stokes!r}. Use 'I', 'QU', 'IQU' or 'IQUV'."
                )
            args = tuple(keywords[stoke] for stoke in stokes)

        args = as_promoted_dtype(args)
        if len(args) == 1:
            return StokesI(*args)
        if len(args) == 2:
            return StokesQU(*args)
        if len(args) == 3:
            return StokesIQU(*args)
        if len(args) == 4:
            return StokesIQUV(*args)
        raise TypeError(f'Unexpected number of Stokes parameters: {len(args)}.')

    @classmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        """Build this Stokes type from the full I, Q, U, V set, keeping only its own components."""
        available = {'I': i, 'Q': q, 'U': u, 'V': v}
        return cls(*(available[letter] for letter in cls.stokes))

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: DTypeLike = float) -> Self:
        return zeros_like(cls.structure_for(shape, dtype))

    @classmethod
    def ones(cls, shape: tuple[int, ...], dtype: DTypeLike = float) -> Self:
        return ones_like(cls.structure_for(shape, dtype))

    @classmethod
    def full(cls, shape: tuple[int, ...], fill_value: ScalarLike, dtype: DTypeLike = float) -> Self:
        return full_like(cls.structure_for(shape, dtype), fill_value)

    @classmethod
    def normal(cls, key: Key[Array, ''], shape: tuple[int, ...], dtype: DTypeLike = float) -> Self:
        return normal_like(cls.structure_for(shape, dtype), key)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, ...],
        key: Key[Array, ''],
        dtype: DTypeLike = float,
        low: float = 0.0,
        high: float = 1.0,
    ) -> Self:
        return uniform_like(cls.structure_for(shape, dtype), key, low, high)


@register_pytree_node_class
class StokesI(Stokes):
    stokes: ClassVar[ValidStokesType] = 'I'


@register_pytree_node_class
class StokesQU(Stokes):
    stokes: ClassVar[ValidStokesType] = 'QU'


@register_pytree_node_class
class StokesIQU(Stokes):
    stokes: ClassVar[ValidStokesType] = 'IQU'


@register_pytree_node_class
class StokesIQUV(Stokes):
    stokes: ClassVar[ValidStokesType] = 'IQUV'


StokesType = StokesI | StokesQU | StokesIQU | StokesIQUV
