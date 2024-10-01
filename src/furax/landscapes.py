import math
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar, Literal, Union, cast, get_args, overload

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jax_healpy as jhp
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Integer, Key, PyTree, ScalarLike, Shaped

from .samplings import Sampling
from .tree import (
    as_promoted_dtype,
    dot,
    full_like,
    normal_like,
    ones_like,
    uniform_like,
    zeros_like,
)

# XXX Remove after https://github.com/google/jax/pull/19669 is accepted
NumberType = Union[
    jnp.float32, jnp.int32, jnp.int16
]  # to be completed with all jax scalar number types
ScalarType = Union[jnp.bool_, NumberType]
DTypeLike = Union[
    str,  # like 'float32', 'int32'
    type[Union[bool, int, float, complex, ScalarType, np.bool_, np.number]],  # type: ignore[type-arg]  # noqa: E501
    np.dtype,  # type: ignore[type-arg]
]

ValidStokesType = Literal['I', 'QU', 'IQU', 'IQUV']


@jdc.pytree_dataclass
class StokesPyTree(ABC):
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
        return self._operation(operator.add, other)

    def __sub__(self, other: Any) -> Self:
        return self._operation(operator.sub, other)

    def __mul__(self, other: Any) -> Self:
        return self._operation(operator.mul, other)

    def __truediv__(self, other: Any) -> Self:
        return self._operation(operator.truediv, other)

    def __pow__(self, other: Any) -> Self:
        return self._operation(operator.pow, other)

    def _operation(self, operation: Callable[[Any, Any], Any], right: Any) -> Self:
        result: Self
        if isinstance(right, type(self)):
            result = jax.tree.map(operation, self, right)
        elif jnp.isscalar(right) or isinstance(right, jax.Array):
            result = jax.tree.map(lambda leaf: operation(leaf, right), self)
        else:
            return NotImplemented
        return result

    def __radd__(self, other: Any) -> Self:
        return self._roperation(operator.add, other)

    def __rsub__(self, other: Any) -> Self:
        return self._roperation(operator.sub, other)

    def __rmul__(self, other: Any) -> Self:
        return self._roperation(operator.mul, other)

    def __rtruediv__(self, other: Any) -> Self:
        return self._roperation(operator.truediv, other)

    def __rpow__(self, other: Any) -> Self:
        return self._roperation(operator.pow, other)

    def _roperation(self, operation: Callable[[Any, Any], Any], left: Any) -> Self:
        result: Self
        if isinstance(left, type(self)):
            result = jax.tree.map(operation, left, self)
        elif jnp.isscalar(left) or isinstance(left, jax.Array):
            result = jax.tree.map(partial(operation, left), self)
        else:
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
    def class_for(cls, stokes: Literal['I']) -> type['StokesIPyTree']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['QU']) -> type['StokesQUPyTree']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['IQU']) -> type['StokesIQUPyTree']: ...

    @classmethod
    @overload
    def class_for(cls, stokes: Literal['IQUV']) -> type['StokesIQUVPyTree']: ...

    @classmethod
    def class_for(cls, stokes: str) -> type['StokesPyTreeType']:
        """Returns the StokesPyTree subclass associated to the specified Stokes types."""
        if stokes not in get_args(ValidStokesType):
            raise ValueError(f'Invalid Stokes parameters: {stokes!r}')
        requested_cls = {
            'I': StokesIPyTree,
            'QU': StokesQUPyTree,
            'IQU': StokesIQUPyTree,
            'IQUV': StokesIQUVPyTree,
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
    def from_stokes(cls, i: ArrayLike) -> 'StokesIPyTree': ...

    @classmethod
    @overload
    def from_stokes(cls, i: jax.ShapeDtypeStruct) -> 'StokesIPyTree': ...

    @classmethod
    @overload
    def from_stokes(cls, q: ArrayLike, u: ArrayLike) -> 'StokesQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(cls, q: jax.ShapeDtypeStruct, u: jax.ShapeDtypeStruct) -> 'StokesQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(cls, i: ArrayLike, q: ArrayLike, u: ArrayLike) -> 'StokesIQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(
        cls, i: jax.ShapeDtypeStruct, q: jax.ShapeDtypeStruct, u: jax.ShapeDtypeStruct
    ) -> 'StokesIQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(
        cls, i: ArrayLike, q: ArrayLike, u: ArrayLike, v: ArrayLike
    ) -> 'StokesIQUVPyTree': ...

    @classmethod
    @overload
    def from_stokes(
        cls,
        i: jax.ShapeDtypeStruct,
        q: jax.ShapeDtypeStruct,
        u: jax.ShapeDtypeStruct,
        v: jax.ShapeDtypeStruct,
    ) -> 'StokesIQUVPyTree': ...

    @classmethod
    def from_stokes(
        cls,
        *args: Any,
        **keywords: Any,
    ) -> 'StokesPyTree':
        """Returns a StokesPyTree according to the specified Stokes vectors.

        Examples:
            >>> tod_i = StokesPyTree.from_stokes(i)
            >>> tod_qu = StokesPyTree.from_stokes(q, u)
            >>> tod_iqu = StokesPyTree.from_stokes(i, q, u)
            >>> tod_iquv = StokesPyTree.from_stokes(i, q, u, v)
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
            return StokesIPyTree(*args)
        if len(args) == 2:
            return StokesQUPyTree(*args)
        if len(args) == 3:
            return StokesIQUPyTree(*args)
        if len(args) == 4:
            return StokesIQUVPyTree(*args)
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
class StokesIPyTree(StokesPyTree):
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
class StokesQUPyTree(StokesPyTree):
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
class StokesIQUPyTree(StokesPyTree):
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
class StokesIQUVPyTree(StokesPyTree):
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


StokesPyTreeType = StokesIPyTree | StokesQUPyTree | StokesIQUPyTree | StokesIQUVPyTree


@jax.tree_util.register_pytree_node_class
class Landscape(ABC):
    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = np.float64):
        self.shape = shape
        self.dtype = dtype

    def __len__(self) -> int:
        return math.prod(self.shape)

    @property
    def size(self) -> int:
        return len(self)

    @abstractmethod
    def normal(self, key: Key[Array, '']) -> PyTree[Shaped[Array, '...']]: ...

    @abstractmethod
    def uniform(
        self, key: Key[Array, ''], low: float = 0.0, high: float = 1.0
    ) -> PyTree[Shaped[Array, '...']]: ...

    @abstractmethod
    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, '...']]: ...

    def zeros(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(0)

    def ones(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(1)

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
        }  # static values
        return (), aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:  # type: ignore[no-untyped-def]
        return cls(**aux_data)


@jax.tree_util.register_pytree_node_class
class StokesLandscape(Landscape):
    """Class representing a multidimensional map of Stokes vectors.

    We assume that integer pixel values fall at the center of pixels (as in the FITS WCS standard,
    see Section 2.1.4 of Greisen et al., 2002, A&A 446, 747).

    Attributes:
        shape: The shape of the array that stores the map values. The dimensions are in the reverse
            order of the FITS NAXIS* keywords. For a 2-dimensional map, the shape corresponds to
            (NAXIS2, NAXIS1) or (:math:`n_row`, :math:`n_col`), i.e. (:math:`n_y`, :math:`n_x`).
        pixel_shape: The shape in reversed order. For a 2-dimensional map, the shape corresponds to
            (NAXIS1, NAXIS2) or (:math:`n_col`, :math:`n_row`), i.e. (:math:`n_x`, :math:`n_y`).
        stokes: The identifier for the Stokes vectors (`I`, `QU`, `IQU` or `IQUV`)
        dtype: The data type for the values of the landscape.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        stokes: ValidStokesType = 'IQU',
        dtype: DTypeLike = np.float64,
        pixel_shape: tuple[int, ...] | None = None,
    ):
        if shape is None and pixel_shape is None:
            raise TypeError('The shape is not specified.')
        if shape is not None and pixel_shape is not None:
            raise TypeError('Either the shape or pixel_shape should be specified.')
        shape = shape if pixel_shape is None else pixel_shape[::-1]
        assert shape is not None  # mypy assert
        super().__init__(shape, dtype)
        self.stokes = stokes
        self.pixel_shape = shape[::-1]

    @property
    def size(self) -> int:
        return len(self.stokes) * len(self)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = StokesPyTree.class_for(self.stokes)
        return cls.structure_for(self.shape, self.dtype)

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
            'stokes': self.stokes,
        }  # static values
        return (), aux_data

    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = StokesPyTree.class_for(self.stokes)
        return cls.full(self.shape, fill_value, self.dtype)

    def normal(self, key: Key[Array, '']) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = StokesPyTree.class_for(self.stokes)
        return cls.normal(key, self.shape, self.dtype)

    def uniform(
        self, key: Key[Array, ''], low: float = 0.0, high: float = 1.0
    ) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = StokesPyTree.class_for(self.stokes)
        return cls.uniform(self.shape, key, self.dtype, low, high)

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' 12*nside**2']:
        indices = self.world2index(arg.theta, arg.phi)
        unique_indices, counts = jnp.unique(indices, return_counts=True)
        coverage = jnp.zeros(len(self), dtype=np.int64)
        coverage = coverage.at[unique_indices].add(
            counts, indices_are_sorted=True, unique_indices=True
        )
        return coverage.reshape(self.shape)

    def world2index(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> Integer[Array, ' *dims']:
        pixels = self.world2pixel(theta, phi)
        return self.pixel2index(*pixels)

    @abstractmethod
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Float[Array, ' *dims'], ...]:
        r"""Converts angles from WCS to pixel coordinates

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            *floats: x, y, z, ... pixel coordinates
        """

    def pixel2index(self, *coords: Float[Array, ' *dims']) -> Integer[Array, ' *ndims']:
        r"""Converts multidimensional pixel coordinates into 1-dimensional indices.

        The order for the indices is row-major, i.e. from the leftmost to the rightmost argument,
        we walk from the fastest to the lowest dimensions. Example for a map of shape
        :math:`(n_y, n_x)`, the pixel with float coordinates :math:`(p_x, p_y)` has an index
        :math:`i = round(p_x) + n_x round(p_y)`.

        The indices travel from bottom to top, like the Y-coordinates.

        Integer values of the pixel coordinates correspond to the pixel centers. The points
        :math:`(p_x, p_y)` strictly inside a pixel centered on the integer coordinates
        :math:`(i_x, i_y)` verify
            - :math:`i_x - ½ < p_x < i_x + ½`
            - :math:`i_y - ½ < p_y < i_y + ½`

        The convention for pixels and indices is that the first one starts at zero.

        Arguments:
            *coords: The floating-point pixel coordinates along the X, Y, Z, ... axes.

        Returns:
            The 1-dimensional integer indices associated to the pixel coordinates. The data type is
            int32, unless the landscape largest index would overflow, in which case it is int64.
        """
        dtype: DTypeLike
        if len(self) - 1 <= np.iinfo(np.iinfo(np.int32)).max:
            dtype = np.int32
        else:
            dtype = np.int64
        if len(coords) == 0:
            raise TypeError('Pixel coordinates are not specified.')

        stride = self.pixel_shape[0]
        indices = jnp.round(coords[0]).astype(dtype)
        valid = (0 <= indices) & (indices < self.pixel_shape[0])
        for coord, dim in zip(coords[1:], self.pixel_shape[1:]):
            indices_axis = jnp.round(coord).astype(dtype)
            valid &= (0 <= indices_axis) & (indices_axis < dim)
            indices += indices_axis * stride
            stride *= dim
        return jnp.where(valid, indices, -1)


@jax.tree_util.register_pytree_node_class
class HealpixLandscape(StokesLandscape):
    """Class representing a Healpix-projected map of Stokes vectors."""

    def __init__(
        self, nside: int, stokes: ValidStokesType = 'IQU', dtype: DTypeLike = np.float64
    ) -> None:
        shape = (12 * nside**2,)
        super().__init__(shape, stokes, dtype)
        self.nside = nside

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
            'stokes': self.stokes,
            'nside': self.nside,
        }  # static values
        return (), aux_data

    @partial(jax.jit, static_argnums=0)
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], ...]:
        r"""Convert angles to HEALPix index for HEALPix ring ordering scheme.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.
        """
        return (jhp.ang2pix(self.nside, theta, phi),)
