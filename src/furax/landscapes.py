import sys
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal, Union, cast, get_args, overload

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jax_healpy as jhp
import numpy as np
from jaxtyping import Array, Float, Integer, PyTree, ScalarLike, Shaped

if TYPE_CHECKING:
    pass
from .samplings import Sampling

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
        return cast(tuple[int, ...], getattr(self, self.stokes[0]).shape)

    @property
    def dtype(self) -> DTypeLike:
        return cast(DTypeLike, getattr(self, self.stokes[0]).dtype)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.structure_for(self.shape, self.dtype)

    @classmethod
    def class_for(cls, stokes: ValidStokesType) -> type['StokesPyTree']:
        """Returns the StokesPyTree subclass associated to the specified Stokes types."""
        if stokes not in get_args(ValidStokesType):
            raise ValueError(f'Invalid Stokes parameters: {stokes!r}')
        return {
            'I': StokesIPyTree,
            'QU': StokesQUPyTree,
            'IQU': StokesIQUPyTree,
            'IQUV': StokesIQUVPyTree,
        }[stokes]

    @classmethod
    @overload
    def structure_for(cls, shape: tuple[int, ...], dtype: DTypeLike = np.float64): ...

    @classmethod
    @overload
    def structure_for(
        cls, stokes: ValidStokesType, shape: tuple[int, ...], dtype: DTypeLike = np.float64
    ): ...

    @classmethod
    def structure_for(
        cls,
        stokes: ValidStokesType | tuple[int, ...] | None = None,
        shape: tuple[int, ...] | DTypeLike | None = None,
        dtype: DTypeLike = np.float64,
    ) -> PyTree[jax.ShapeDtypeStruct]:
        if isinstance(stokes, str):
            cls = StokesPyTree.class_for(stokes)
        elif isinstance(stokes, tuple):
            if shape is not None:
                dtype = shape
            shape = stokes
        stokes_arrays = len(cls.stokes) * [jax.ShapeDtypeStruct(shape, dtype)]
        return cls(*stokes_arrays)

    def __getitem__(self, index: Integer[Array, '...']) -> Self:
        arrays = [getattr(self, stoke)[index] for stoke in self.stokes]
        return type(self)(*arrays)

    @classmethod
    @overload
    def from_stokes(cls, i: Float[Array, '...']) -> 'StokesIPyTree': ...

    @classmethod
    @overload
    def from_stokes(cls, q: Float[Array, '...'], u: Float[Array, '...']) -> 'StokesQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(
        cls, i: Float[Array, '...'], q: Float[Array, '...'], u: Float[Array, '...']
    ) -> 'StokesIQUPyTree': ...

    @classmethod
    @overload
    def from_stokes(
        cls,
        i: Float[Array, '...'],
        q: Float[Array, '...'],
        u: Float[Array, '...'],
        v: Float[Array, '...'],
    ) -> 'StokesIQUVPyTree': ...

    @classmethod
    def from_stokes(
        cls, *args: Float[Array, '...'], **keywords: Float[Array, '...']
    ) -> 'StokesPyTree':
        """Returns a StokesPyTree according to the specified Stokes vectors.

        Examples:
            >>> tod_i = StokesPyTree.from_stokes(I)
            >>> tod_qu = StokesPyTree.from_stokes(Q, U)
            >>> tod_iqu = StokesPyTree.from_stokes(I, Q, U)
            >>> tod_iquv = StokesPyTree.from_stokes(I, Q, U, V)
        """
        if len(args) == 1:
            return StokesIPyTree(*args)
        if len(args) == 2:
            return StokesQUPyTree(*args)
        if len(args) == 3:
            return StokesIQUPyTree(*args)
        if len(args) == 4:
            return StokesIQUVPyTree(*args)
        if len(args) > 4:
            raise TypeError(f'Unexpected number of Stokes parameters: {len(args)}.')

        if not keywords:
            raise TypeError(f'The Stokes vectors are not specified.')
        i = keywords.pop('I', None)
        q = keywords.pop('Q', None)
        u = keywords.pop('U', None)
        v = keywords.pop('V', None)
        if keywords:
            raise TypeError(f'Invalid keyword arguments: {", ".join(repr(_) for _ in keywords)}')
        if i is not None and q is None and u is None and v is None:
            return StokesIPyTree(i)
        if i is None and q is not None and u is not None and v is None:
            return StokesQUPyTree(q, u)
        if i is not None and q is not None and u is not None:
            if v is None:
                return StokesIQUPyTree(i, q, u)
            return StokesIQUVPyTree(i, q, u, v)

        invalid_stokes = ''.join(
            'IQUV'[index] if _ is not None else '' for index, _ in enumerate([i, q, u, v])
        )
        raise TypeError(
            f'Invalid Stokes vectors: {invalid_stokes!r}. I, QU, IQU or IQUV must be specified.'
        )

    @classmethod
    @abstractmethod
    def from_iquv(
        cls,
        I: Float[Array, '...'],
        Q: Float[Array, '...'],
        U: Float[Array, '...'],
        V: Float[Array, '...'],
    ) -> Self:
        """Returns a StokesPyTree ignoring the Stokes components not in the type."""

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: DTypeLike | float = float) -> Self:
        return cls.full(shape, 0, dtype)

    @classmethod
    def ones(cls, shape: tuple[int, ...], dtype: DTypeLike | float = float) -> Self:
        return cls.full(shape, 1, dtype)

    @classmethod
    def full(
        cls, shape: tuple[int, ...], fill_value: ScalarLike, dtype: DTypeLike | float = float
    ) -> Self:
        arrays = len(cls.stokes) * [jnp.full(shape, fill_value, dtype)]  # type: ignore[arg-type]
        return cls(*arrays)

    def ravel(self) -> Self:
        """Ravels each Stokes component."""
        return jax.tree.map(lambda x: x.ravel(), self)

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Reshape each Stokes component."""
        return jax.tree.map(lambda x: x.reshape(shape), self)


@jdc.pytree_dataclass
class StokesIPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'I'
    I: Array

    @classmethod
    def from_iquv(
        cls,
        I: Float[Array, '...'],
        Q: Float[Array, '...'],
        U: Float[Array, '...'],
        V: Float[Array, '...'],
    ) -> Self:
        return cls(I)


@jdc.pytree_dataclass
class StokesQUPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'QU'
    Q: Array
    U: Array

    @classmethod
    def from_iquv(
        cls,
        I: Float[Array, '...'],
        Q: Float[Array, '...'],
        U: Float[Array, '...'],
        V: Float[Array, '...'],
    ) -> Self:
        return cls(Q, U)


@jdc.pytree_dataclass
class StokesIQUPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'IQU'
    I: Array
    Q: Array
    U: Array

    @classmethod
    def from_iquv(
        cls,
        I: Float[Array, '...'],
        Q: Float[Array, '...'],
        U: Float[Array, '...'],
        V: Float[Array, '...'],
    ) -> Self:
        return cls(I, Q, U)


@jdc.pytree_dataclass
class StokesIQUVPyTree(StokesPyTree):
    stokes: ClassVar[ValidStokesType] = 'IQUV'
    I: Array
    Q: Array
    U: Array
    V: Array

    @classmethod
    def from_iquv(
        cls,
        I: Float[Array, '...'],
        Q: Float[Array, '...'],
        U: Float[Array, '...'],
        V: Float[Array, '...'],
    ) -> Self:
        return cls(I, Q, U, V)


StokesPyTreeType = StokesIPyTree | StokesQUPyTree | StokesIQUPyTree | StokesIQUVPyTree


@jax.tree_util.register_pytree_node_class
class Landscape(ABC):

    def __init__(self, shape: tuple[int, ...], dtype: DTypeLike = np.float64):
        self.shape = shape
        self.dtype = dtype

    def __len__(self) -> int:
        return np.prod(self.shape)

    @property
    def size(self) -> int:
        return len(self)

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

    We assume that integer pixel values fall at the center of pixels (as in the FITS WCS standard, see Section 2.1.4
    of Greisen et al., 2002, A&A 446, 747).

    Attributes:
        shape: The shape of the array that stores the map values. The dimensions are in the reverse order of
            the FITS NAXIS* keywords. For a 2-dimensional map, the shape corresponds to (NAXIS2, NAXIS1) or
            (:math:`n_row`, :math:`n_col`), i.e. (:math:`n_y`, :math:`n_x`).
        pixel_shape: The shape in reversed order. For a 2-dimensional map, the shape corresponds to (NAXIS1, NAXIS2) or
            (:math:`n_col`, :math:`n_row`), i.e. (:math:`n_x`, :math:`n_y`).
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
        shape = shape or pixel_shape[::-1]
        super().__init__(shape, dtype)
        self.stokes = stokes
        self.pixel_shape = shape[::-1]

    @property
    def size(self) -> int:
        return len(self.stokes) * len(self)

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.structure_for(self.stokes, self.shape, self.dtype)

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

        The order for the indices is row-major, i.e. from the leftmost to the rightmost argument, we walk from the
        fastest to the lowest dimensions. Example for a map of shape :math:`(n_y, n_x)`, the pixel with float
        coordinates :math:`(p_x, p_y)` has an index :math:`i = round(p_x) + n_x round(p_y)`.

        The indices travel from bottom to top, like the Y-coordinates.

        Integer values of the pixel coordinates correspond to the pixel centers. The points :math:`(p_x, p_y)` strictly
        inside a pixel centered on the integer coordinates :math:`(i_x, i_y)` verify
            - :math:`i_x - ½ < p_x < i_x + ½`
            - :math:`i_y - ½ < p_y < i_y + ½`

        The convention for pixels and indices is that the first one starts at zero.

        Arguments:
            *coords: The floating-point pixel coordinates along the X, Y, Z, ... axes.

        Returns:
            The 1-dimensional integer indices associated to the pixel coordinates. The data type is int32, unless
            the landscape largest index would overflow, in which case it is int64.
        """
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

    def __init__(self, nside: int, stokes: ValidStokesType = 'IQU', dtype: DTypeLike = np.float64):
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
    ) -> Integer[Array, ' *dims']:
        r"""Convert angles to HEALPix index for HEALPix ring ordering scheme.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            int: HEALPix map index for ring ordering scheme.
        """
        return (jhp.ang2pix(self.nside, theta, phi),)  # type: ignore[no-any-return]