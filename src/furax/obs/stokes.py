import operator
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, cast, get_args, overload

import jax
import jax_dataclasses as jdc
import numpy as np
from equinox.internal._omega import _Metaω
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import DTypeLike, Float, Integer, Key, PyTree, ScalarLike

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from furax.exceptions import StructureError
from furax.tree import (
    add,
    as_promoted_dtype,
    dot,
    full_like,
    mul,
    normal_like,
    ones_like,
    power,
    sub,
    truediv,
    uniform_like,
    zeros_like,
)

__all__ = ['Stokes', 'StokesI', 'StokesQU', 'StokesIQU', 'StokesIQUV', 'ValidStokesType']

ValidStokesType = Literal['I', 'QU', 'IQU', 'IQUV']


@jdc.pytree_dataclass
class Stokes(ABC):
    stokes: ClassVar[ValidStokesType]

    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], getattr(self, self.stokes[0].lower()).shape)

    @property
    def dtype(self) -> DTypeLike:
        return cast(DTypeLike, getattr(self, self.stokes[0].lower()).dtype)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.structure_for(self.shape, self.dtype)

    def __getitem__(self, index: Integer[Array, '...']) -> Self:
        arrays = [getattr(self, stoke.lower())[index] for stoke in self.stokes]
        return type(self)(*arrays)

    def __matmul__(self, other: Any) -> Any:
        """Scalar product between Stokes pytrees."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return dot(self, other)

    def __abs__(self) -> Self:
        result: Self = jax.tree.map(operator.abs, self)
        return result

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        result: Self = jax.tree.map(operator.neg, self)
        return result

    def __add__(self, other: Any) -> Self:
        try:
            result: Self = add(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __sub__(self, other: Any) -> Self:
        try:
            result: Self = sub(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __mul__(self, other: Any) -> Self:
        try:
            result: Self = mul(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __truediv__(self, other: Any) -> Self:
        try:
            result: Self = truediv(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __pow__(self, other: Any) -> Self:
        if isinstance(other, _Metaω):
            return NotImplemented
        try:
            result: Self = power(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __radd__(self, other: Any) -> Self:
        try:
            result: Self = add(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __rsub__(self, other: Any) -> Self:
        try:
            result: Self = sub(other, self)
        except StructureError:
            return NotImplemented
        return result

    def __rmul__(self, other: Any) -> Self:
        try:
            result: Self = mul(self, other)
        except StructureError:
            return NotImplemented
        return result

    def __rtruediv__(self, other: Any) -> Self:
        try:
            result: Self = truediv(other, self)
        except StructureError:
            return NotImplemented
        return result

    def __rpow__(self, other: Any) -> Self:
        try:
            result: Self = power(other, self)
        except StructureError:
            return NotImplemented
        return result

    def ravel(self) -> Self:
        """Ravels each Stokes component."""
        return jax.tree.map(lambda x: x.ravel(), self)  # type: ignore[no-any-return]

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Reshape each Stokes component."""
        return jax.tree.map(lambda x: x.reshape(shape), self)  # type: ignore[no-any-return]

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
    def class_for(cls, stokes: str) -> type['StokesPyTreeType']:
        """Returns the StokesPyTree subclass associated to the specified Stokes types."""
        if stokes not in get_args(ValidStokesType):
            raise ValueError(f'Invalid Stokes parameters: {stokes!r}')
        requested_cls = {
            'I': StokesI,
            'QU': StokesQU,
            'IQU': StokesIQU,
            'IQUV': StokesIQUV,
        }[stokes]
        return cast(type[StokesPyTreeType], requested_cls)

    @classmethod
    def structure_for(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
    ) -> Self:
        stokes_arrays = len(cls.stokes) * [jax.ShapeDtypeStruct(shape, dtype)]
        return cls(*stokes_arrays)

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
    @abstractmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        """Returns a StokesPyTree ignoring the Stokes components not in the type."""

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


@jdc.pytree_dataclass
class StokesI(Stokes):
    stokes: ClassVar[ValidStokesType] = 'I'
    i: Array

    @classmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        return cls(i)


@jdc.pytree_dataclass
class StokesQU(Stokes):
    stokes: ClassVar[ValidStokesType] = 'QU'
    q: Array
    u: Array

    @classmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        q, u = as_promoted_dtype((q, u))
        return cls(q, u)


@jdc.pytree_dataclass
class StokesIQU(Stokes):
    stokes: ClassVar[ValidStokesType] = 'IQU'
    i: Array
    q: Array
    u: Array

    @classmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        i, q, u = as_promoted_dtype((i, q, u))
        return cls(i, q, u)


@jdc.pytree_dataclass
class StokesIQUV(Stokes):
    stokes: ClassVar[ValidStokesType] = 'IQUV'
    i: Array
    q: Array
    u: Array
    v: Array

    @classmethod
    def from_iquv(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> Self:
        i, q, u, v = as_promoted_dtype((i, q, u, v))
        return cls(i, q, u, v)


StokesPyTreeType = StokesI | StokesQU | StokesIQU | StokesIQUV
