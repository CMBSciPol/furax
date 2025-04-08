import math
import sys
from abc import ABC, abstractmethod
from functools import partial

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import u
from astropy.wcs import WCS
from jaxtyping import Array, DTypeLike, Float, Integer, Key, PyTree, ScalarLike, Shaped

from furax.math.quaternion import qrot_zaxis
from furax.obs._samplings import Sampling
from furax.obs.stokes import Stokes, ValidStokesType


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
        cls = Stokes.class_for(self.stokes)
        return cls.structure_for(self.shape, self.dtype)

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
            'stokes': self.stokes,
        }  # static values
        return (), aux_data

    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.full(self.shape, fill_value, self.dtype)

    def normal(self, key: Key[Array, '']) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.normal(key, self.shape, self.dtype)

    def uniform(
        self, key: Key[Array, ''], low: float = 0.0, high: float = 1.0
    ) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = Stokes.class_for(self.stokes)
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
    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        r"""Convert quaternion to HEALPix index in ring ordering.

        Args:
            quat (float): Quaternion.

        Returns:
            int: HEALPix map index for ring ordering scheme.
        """
        # we want the 3 dimensions on the left
        vec = jnp.moveaxis(qrot_zaxis(quat), -1, 0)
        pix: Integer[Array, ' *dims'] = jhp.vec2pix(self.nside, *vec)
        return pix

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


@jax.tree_util.register_pytree_node_class
class FrequencyLandscape(HealpixLandscape):
    def __init__(
        self,
        nside: int,
        frequencies: Array,
        stokes: ValidStokesType = 'IQU',
        dtype: DTypeLike = np.float64,
    ):
        super().__init__(nside, stokes, dtype)
        self.frequencies = frequencies
        self.shape = (len(frequencies), 12 * nside**2)

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
            'stokes': self.stokes,
            'nside': self.nside,
            'frequencies': self.frequencies,
        }  # static values
        return (), aux_data


@jax.tree_util.register_pytree_node_class
class WCSLandscape(StokesLandscape):
    """Class representing an astropy WCS map of Stokes vectors."""

    def __init__(
        self,
        shape: tuple[int, ...],
        wcs: WCS,
        stokes: ValidStokesType = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__(shape, stokes, dtype)
        self.wcs = wcs

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        aux_data = {
            'shape': self.shape,
            'dtype': self.dtype,
            'stokes': self.stokes,
            'wcs': self.wcs,
        }  # static values
        return (), aux_data

    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], ...]:
        r"""Convert angles to WCS map indices.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            WCS map index pairs
        """

        def f(theta, phi):  # type: ignore[no-untyped-def]
            # SkyCoord takes (lon,lat)
            pix_i, pix_j = self.wcs.world_to_pixel(SkyCoord(phi, (np.pi / 2 - theta), unit=u.rad))
            return tuple(np.array(np.round([pix_i, pix_j]), dtype=np.int64))

        struct = jax.ShapeDtypeStruct(theta.shape, jnp.int64)
        result_shape = (struct, struct)

        return jax.pure_callback(f, result_shape, theta, phi)  # type: ignore[no-any-return]
