import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from jax.tree_util import register_static
from jaxtyping import Array, DTypeLike, Float, Integer, Key, PyTree, ScalarLike, Shaped

from furax.math.quaternion import qrot_zaxis, to_iso_angles
from furax.obs._samplings import Sampling
from furax.obs.stokes import Stokes, ValidStokesType


@register_static
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


@register_static
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

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' {self.npixel}']:
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

        For a map of shape :math:`(n_y, n_x)`, the indices of the pixels are walked from the
        rightmost dimension :math:`n_x` to the leftmost dimension :math:`n_y` (row-major layout).
        In such a map, the pixel with float coordinates :math:`(p_x, p_y)` has an index
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

    def quat2world(
        self, quat: Float[Array, '*dims 4']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Converts quaternion to WCS angles (theta, phi)."""
        theta, phi, _psi = to_iso_angles(quat)
        return theta, phi

    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        """Converts quaternion to 1-dimensional pixel indices."""
        world = self.quat2world(quat)
        return self.world2index(*world)


@register_static
class HealpixLandscape(StokesLandscape):
    """Class representing a Healpix-projected map of Stokes vectors."""

    def __init__(
        self, nside: int, stokes: ValidStokesType = 'IQU', dtype: DTypeLike = np.float64
    ) -> None:
        shape = (12 * nside**2,)
        super().__init__(shape, stokes, dtype)
        self.nside = nside

    @jax.jit
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

    @jax.jit
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


@register_static
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


@register_static
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

    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], Integer[Array, ' *dims']]:
        r"""Convert angles to WCS map indices.

        Args:
            theta (float): Spherical :math:`\theta` angle.
            phi (float): Spherical :math:`\phi` angle.

        Returns:
            WCS map index pairs
        """

        def f(theta, phi):  # type: ignore[no-untyped-def]
            # SkyCoord takes (lon,lat)
            pix_i, pix_j = self.wcs.world_to_pixel(SkyCoord(phi, (np.pi / 2 - theta), unit='rad'))
            return tuple(np.array(np.round([pix_i, pix_j]), dtype=np.int64))

        struct = jax.ShapeDtypeStruct(theta.shape, jnp.int64)
        result_shape = (struct, struct)

        return jax.pure_callback(f, result_shape, theta, phi)  # type: ignore[no-any-return]


@register_static
class HorizonLandscape(StokesLandscape):
    """Class representing a map of Stokes vectors in horizon (ground) coordinates.
    Contains two axes:
        AXIS1: altitude (elevation) in radians
        AXIS2: azimuth in radians
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        altitude_limits: tuple[Array, Array],
        azimuth_limits: tuple[Array, Array],
        stokes: ValidStokesType = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__(shape, stokes, dtype)
        self.altitude_limits = altitude_limits
        self.azimuth_limits = azimuth_limits

    def bin_edges(self) -> tuple[Float[Array, ' bins+1'], Float[Array, ' bins+1']]:
        """Return the bin edges of the map."""
        alt_min, alt_max = self.altitude_limits
        az_min, az_max = self.azimuth_limits
        n_az, n_alt = self.shape
        return jnp.linspace(alt_min, alt_max, n_alt + 1), jnp.linspace(az_min, az_max, n_az + 1)

    def bin_centers(self) -> tuple[Float[Array, ' bins'], Float[Array, ' bins']]:
        """Return the bin centers of the map."""
        alt_edges, az_edges = self.bin_edges()
        alt_centers = 0.5 * (alt_edges[:-1] + alt_edges[1:])
        az_centers = 0.5 * (az_edges[:-1] + az_edges[1:])
        return alt_centers, az_centers

    @jax.jit
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], Integer[Array, ' *dims']]:
        r"""Convert angles in radians to Horizon map indices.

        Args:
            theta (float): Spherical :math:`\theta` angle = pi/2 - altitude.
            phi (float): Spherical :math:`\phi` angle = -azimuth.

        Returns:
            Ground map index pairs
        """
        alt_min, alt_max = self.altitude_limits
        az_min, az_max = self.azimuth_limits
        n_az, n_alt = self.shape  # Note self.shape has reverse order: [AXIS2, AXIS1]
        dalt = (alt_max - alt_min) / n_alt
        daz = (az_max - az_min) / n_az

        altitude = jnp.pi / 2 - theta
        azimuth = -phi

        pix_i = jnp.round((altitude - alt_min) / dalt - 0.5).astype(jnp.int64)
        pix_j = jnp.round(((azimuth - az_min) % (2 * jnp.pi)) / daz - 0.5).astype(jnp.int64)

        return pix_i, pix_j


@register_static
class LocalStokesLandscape:
    """A landscape representing a subset of pixels from a parent StokesLandscape.

    This class wraps a parent StokesLandscape and a sorted array of global pixel indices
    to define a reduced pixel domain. It provides methods to convert between global
    (parent) and local (subset) index spaces.

    The ``world2index`` and ``quat2index`` methods delegate to the parent and return
    global indices. Use ``world2local_index`` / ``quat2local_index`` or ``global2local``
    to obtain local indices that can be used to index into arrays of shape ``self.shape``.

    Attributes:
        parent: The parent StokesLandscape.
        global_indices: A sorted 1-D numpy array of unique global pixel indices.
    """

    def __init__(
        self,
        parent: StokesLandscape,
        global_indices: Integer[np.ndarray | Array, ' nlocal'],
    ) -> None:
        gi = jnp.asarray(global_indices)
        if gi.ndim != 1:
            raise ValueError('global_indices must be 1-dimensional.')
        if gi.size > 0 and not jnp.all(jnp.diff(gi) > 0):
            raise ValueError('global_indices must be sorted and contain no duplicates.')
        self.parent = parent
        self.global_indices = gi

    @property
    def shape(self) -> tuple[int]:
        return (len(self.global_indices),)

    @property
    def nlocal(self) -> int:
        return len(self.global_indices)

    @property
    def stokes(self) -> ValidStokesType:
        return self.parent.stokes

    @property
    def dtype(self) -> DTypeLike:
        return self.parent.dtype

    @property
    def size(self) -> int:
        return len(self.parent.stokes) * self.nlocal

    @property
    def structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = Stokes.class_for(self.stokes)
        return cls.structure_for(self.shape, self.dtype)

    # --- Index conversion ---

    def global2local(self, indices: Integer[Array, ' *dims']) -> Integer[Array, ' *dims']:
        """Convert global pixel indices to local indices.

        Uses ``jnp.searchsorted`` on the sorted ``global_indices``. Returns -1 for
        global indices that are not present in this local landscape.
        """
        pos = jnp.searchsorted(self.global_indices, indices)
        pos = jnp.clip(pos, 0, self.nlocal - 1)
        valid = self.global_indices[pos] == indices
        return jnp.where(valid, pos, -1)

    def local2global(self, indices: Integer[Array, ' *dims']) -> Integer[Array, ' *dims']:
        """Convert local indices to global pixel indices.

        Returns -1 for local indices that are out of bounds.
        """
        valid = (indices >= 0) & (indices < self.nlocal)
        safe = jnp.where(valid, indices, 0)
        return jnp.where(valid, self.global_indices[safe], -1)

    # --- Stokes conversion ---

    def restrict(self, sky: Stokes) -> Stokes:
        """Extract local pixels from a global Stokes sky map."""
        return sky[self.global_indices]

    def promote(self, sky: Stokes, fill_value: ScalarLike = 0) -> Stokes:
        """Scatter a local Stokes sky map into a global-shaped one.

        Pixels not in the local subset are filled with ``fill_value``.
        """
        global_sky = self.parent.full(fill_value)
        result: Stokes = jax.tree.map(
            lambda g, l: g.at[self.global_indices].set(l),
            global_sky,
            sky,
        )
        return result

    # --- Parent delegation ---

    def world2index(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> Integer[Array, ' *dims']:
        """Convert world coordinates to global pixel indices (delegates to parent)."""
        return self.parent.world2index(theta, phi)

    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        """Convert quaternions to global pixel indices (delegates to parent)."""
        return self.parent.quat2index(quat)

    # --- Convenience: world/quat → local ---

    def world2local_index(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> Integer[Array, ' *dims']:
        """Convert world coordinates to local pixel indices."""
        return self.global2local(self.world2index(theta, phi))

    def quat2local_index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        """Convert quaternions to local pixel indices."""
        return self.global2local(self.quat2index(quat))

    # --- Array creation (local shape) ---

    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, ' {self.nlocal}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.full(self.shape, fill_value, self.dtype)

    def zeros(self) -> PyTree[Shaped[Array, ' {self.nlocal}']]:
        return self.full(0)

    def ones(self) -> PyTree[Shaped[Array, ' {self.nlocal}']]:
        return self.full(1)

    def normal(self, key: Key[Array, '']) -> PyTree[Shaped[Array, ' {self.nlocal}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.normal(key, self.shape, self.dtype)

    def uniform(
        self, key: Key[Array, ''], low: float = 0.0, high: float = 1.0
    ) -> PyTree[Shaped[Array, ' {self.nlocal}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.uniform(self.shape, key, self.dtype, low, high)

    # --- Coverage ---

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' {self.nlocal}']:
        """Compute per-pixel hit counts in local index space."""
        global_indices = self.world2index(arg.theta, arg.phi)
        local_indices = self.global2local(global_indices)
        # Filter out pixels not in the local landscape
        valid = local_indices >= 0
        local_indices = jnp.where(valid, local_indices, 0)
        unique_indices, counts = jnp.unique(local_indices, return_counts=True)
        coverage = jnp.zeros(self.nlocal, dtype=np.int64)
        coverage = coverage.at[unique_indices].add(
            counts, indices_are_sorted=True, unique_indices=True
        )
        # Subtract out the count for invalid pixels that were mapped to index 0
        invalid_count = jnp.sum(~valid).astype(np.int64)
        coverage = coverage.at[0].add(-invalid_count)
        return coverage

    # --- Convenience constructors ---

    @classmethod
    def from_boolean_mask(
        cls, parent: StokesLandscape, mask: Integer[Array, ' npixel']
    ) -> 'LocalStokesLandscape':
        """Create a LocalStokesLandscape from a boolean mask over parent pixels."""
        (global_indices,) = np.nonzero(np.asarray(mask).ravel())
        return cls(parent, global_indices)

    @classmethod
    def from_sampling(cls, parent: StokesLandscape, sampling: Sampling) -> 'LocalStokesLandscape':
        """Create a LocalStokesLandscape from observed pixels in a Sampling."""
        global_indices = parent.world2index(sampling.theta, sampling.phi)
        global_indices = jnp.asarray(global_indices).ravel()
        global_indices = global_indices[global_indices >= 0]
        global_indices = jnp.unique(global_indices)
        return cls(parent, global_indices)
