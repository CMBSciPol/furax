import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeVar

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
from furax.obs.stokes import Stokes, ValidStokesLiteral

_StokesT = TypeVar('_StokesT', bound=Stokes)


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
        self, key: Key[Array, ''], minval: float = 0.0, maxval: float = 1.0
    ) -> PyTree[Shaped[Array, '...']]: ...

    @abstractmethod
    def full(self, fill_value: ScalarLike) -> PyTree[Shaped[Array, '...']]: ...

    def zeros(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(0)

    def ones(self) -> PyTree[Shaped[Array, '...']]:
        return self.full(1)


@register_static
class StokesLandscape(Landscape):
    r"""Class representing a multidimensional map of Stokes vectors.

    We assume that integer pixel values fall at the center of pixels (as in the FITS WCS standard,
    see Section 2.1.4 of Greisen et al., 2002, A&A 446, 747).

    Attributes:
        shape: The shape of the array that stores the map values. The dimensions are in the reverse
            order of the FITS NAXIS* keywords. For a 2-dimensional map, the shape corresponds to
            (NAXIS2, NAXIS1) or ($n_\mathrm{row}$, $n_\mathrm{col}$), i.e. ($n_y$, $n_x$).
        pixel_shape: The shape in reversed order. For a 2-dimensional map, the shape corresponds to
            (NAXIS1, NAXIS2) or ($n_\mathrm{col}$, $n_\mathrm{row}$), i.e. ($n_x$, $n_y$).
        stokes: The identifier for the Stokes vectors (`I`, `QU`, `IQU` or `IQUV`)
        dtype: The data type for the values of the landscape.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        stokes: ValidStokesLiteral = 'IQU',
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
        self, key: Key[Array, ''], minval: float = 0.0, maxval: float = 1.0
    ) -> PyTree[Shaped[Array, ' {self.npixel}']]:
        cls = Stokes.class_for(self.stokes)
        return cls.uniform(self.shape, key, self.dtype, minval, maxval)

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
        r"""Converts angles from WCS to pixel coordinates.

        Args:
            theta (float): Spherical $\theta$ angle.
            phi (float): Spherical $\phi$ angle.

        Returns:
            tuple[float, ...]: x, y, z, ... pixel coordinates
        """

    def pixel2index(self, *coords: Float[Array, ' *dims']) -> Integer[Array, ' *ndims']:
        r"""Converts multidimensional pixel coordinates into 1-dimensional indices.

        For a map of shape $(n_y, n_x)$, the indices of the pixels are walked from the
        rightmost dimension $n_x$ to the leftmost dimension $n_y$ (row-major layout).
        In such a map, the pixel with float coordinates $(p_x, p_y)$ has an index
        $i = round(p_x) + n_x round(p_y)$.

        The indices travel from bottom to top, like the Y-coordinates.

        Integer values of the pixel coordinates correspond to the pixel centers. The points
        $(p_x, p_y)$ strictly inside a pixel centered on the integer coordinates
        $(i_x, i_y)$ verify

        - $i_x - \frac{1}{2} < p_x < i_x + \frac{1}{2}$
        - $i_y - \frac{1}{2} < p_y < i_y + \frac{1}{2}$

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

    def quat2pixel(self, quat: Float[Array, '*dims 4']) -> tuple[Float[Array, ' *dims'], ...]:
        """Converts quaternion to floating-point pixel coordinates."""
        theta, phi, _ = to_iso_angles(quat)  # psi not needed
        return self.world2pixel(theta, phi)

    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        """Converts quaternion to 1-dimensional pixel indices."""
        return self.pixel2index(*self.quat2pixel(quat))

    def quat2world(
        self, quat: Float[Array, '*dims 4']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Converts quaternion to spherical world angles ``(theta, phi)``."""
        theta, phi, _ = to_iso_angles(quat)  # psi not needed
        return theta, phi

    def world2interp(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, '...'], Float[Array, '...']]:
        """Returns (indices, weights) with a trailing neighbor dimension for interpolation.

        Subclasses support interpolation by overriding [`pixel2interp`][] (2-D pixel grid, see
        [`WCSLandscape`][]) or this method directly (e.g. HEALPix via ``get_interp_weights``).
        """
        raise NotImplementedError(f'{type(self).__name__} does not support interpolation')

    def pixel2interp(
        self, pix_x: Float[Array, ' *dims'], pix_y: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, '...'], Float[Array, '...']]:
        """Returns (indices, weights) for interpolation from float pixel coordinates."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support pixel-space interpolation'
        )

    def quat2interp(
        self, quat: Float[Array, '*dims 4']
    ) -> tuple[Integer[Array, '...'], Float[Array, '...']]:
        """Converts quaternion to (indices, weights) for interpolation."""
        theta, phi, _ = to_iso_angles(quat)
        return self.world2interp(theta, phi)


class ProjectionType(IntEnum):
    """Supported WCS projection types."""

    CAR = 0


@register_static
@dataclass
class WCSProjection:
    """Class that holds basic WCS projection parameters."""

    crpix: tuple[float, float]
    """Reference pixel ``(crpix_x, crpix_y)`` in 1-indexed FITS convention."""
    crval: tuple[float, float]
    """Reference coordinate ``(lon_deg, lat_deg)`` in degrees."""
    cdelt: tuple[float, float]
    """Pixel scale ``(cdelt_x_deg, cdelt_y_deg)`` in degrees per pixel."""
    type: ProjectionType = ProjectionType.CAR
    """Projection type (only CAR is supported for now)."""

    def __post_init__(self) -> None:
        if self.cdelt[0] > 0:
            warnings.warn(
                f'cdelt_ra = {self.cdelt[0]} is positive; FITS convention requires cdelt_ra < 0'
                ' (RA decreases left-to-right).',
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_astropy(cls, wcs: WCS) -> 'WCSProjection':
        """Extract a WCSProjection from an astropy WCS object."""
        projection = ProjectionType[wcs.wcs.ctype[0].split('-')[-1]]
        return cls(
            crpix=(float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])),
            crval=(float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])),
            cdelt=(float(wcs.wcs.cdelt[0]), float(wcs.wcs.cdelt[1])),
            type=projection,
        )


@register_static
class WCSLandscape(StokesLandscape):
    """Base class for WCS-projected Stokes landscapes."""

    def __init__(
        self,
        shape: tuple[int, int],
        projection: WCSProjection,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__(shape, stokes, dtype)
        self.projection = projection

    @property
    def crpix(self) -> tuple[float, float]:
        return self.projection.crpix

    @property
    def crval(self) -> tuple[float, float]:
        return self.projection.crval

    @property
    def cdelt(self) -> tuple[float, float]:
        return self.projection.cdelt

    def world2interp(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, '*dims 4'], Float[Array, '*dims 4']]:
        """Returns (indices, weights) for bilinear interpolation (n=4)."""
        return self.pixel2interp(*self.world2pixel(theta, phi))

    def pixel2interp(
        self, pix_x: Float[Array, ' *dims'], pix_y: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, '*dims 4'], Float[Array, '*dims 4']]:
        """Bilinear (indices, weights) from float pixel coordinates (grid-agnostic)."""
        xs, ys, weights = _2d_bilinear_interp(pix_x, pix_y)
        indices = self.pixel2index(xs, ys)
        return indices, weights

    def to_wcs(self) -> WCS:
        """Reconstruct an astropy WCS object from the stored projection parameters."""
        proj = self.projection.type.name
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [f'RA---{proj}', f'DEC--{proj}']
        wcs.wcs.crpix = list(self.crpix)
        wcs.wcs.crval = list(self.crval)
        wcs.wcs.cdelt = list(self.cdelt)
        wcs.wcs.set()
        return wcs

    @classmethod
    def class_for(cls, projection_type: ProjectionType) -> type['WCSLandscape']:
        """Return the WCSLandscape subclass for the given projection type."""
        if projection_type == ProjectionType.CAR:
            return CARLandscape
        raise ValueError(f'Unsupported projection type: {projection_type!r}')

    @classmethod
    def from_wcs(
        cls,
        shape: tuple[int, int],
        wcs: WCS,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> 'WCSLandscape':
        """Create a WCSLandscape subclass instance from an astropy WCS object.

        Dispatches to the appropriate subclass based on the projection type.
        """
        projection = WCSProjection.from_astropy(wcs)
        return cls.class_for(projection.type)(shape, projection, stokes, dtype)


@register_static
class CARLandscape(WCSLandscape):
    r"""Class representing a CAR (Plate Carrée) projected map of Stokes vectors.

    CAR maps native coordinates linearly to the projection plane (``x = φ, y = θ``).
    Requiring ``phi_0 = 0`` aligns the native and celestial equators, reducing the
    celestial-to-native rotation to a pure RA shift and making the full world-to-pixel
    mapping linear in ``(RA, Dec)``:

    $$
    p_x = \frac{\Delta\lambda}{\delta_x} + (c_x - 1) \\
    p_y = \frac{\phi}{\delta_y} + (c_y - 1)
    $$

    where $\Delta\lambda = ((\lambda - \lambda_0 + 180) \bmod 360) - 180$ handles
    the 0°/360° RA wrap, $(\lambda_0, 0.0)$ is ``crval``, $(\delta_x, \delta_y)$
    is ``cdelt`` in degrees, and $(c_x, c_y)$ is ``crpix`` (1-indexed FITS convention).
    """

    def __init__(
        self,
        shape: tuple[int, int],
        projection: WCSProjection,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> None:
        if projection.crval[1] != 0.0:
            msg = (
                f'CAR projection requires crval_dec = 0 (native and celestial equators must'
                f' coincide), got {projection.crval[1]}'
            )
            raise ValueError(msg)
        super().__init__(shape, projection, stokes, dtype)

    @jax.jit
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        r"""Convert spherical angles to CAR pixel coordinates in pure JAX.

        Args:
            theta (float): Spherical $\theta$ angle (co-latitude, 0 at north pole).
            phi (float): Spherical $\phi$ angle (longitude).

        Returns:
            Pixel coordinate pair ``(pix_x, pix_y)`` as float arrays.
        """
        lon_deg = jnp.degrees(phi)
        lat_deg = jnp.degrees(jnp.pi / 2 - theta)
        lon_diff = ((lon_deg - self.crval[0]) + 180) % 360 - 180
        pix_x = lon_diff / self.cdelt[0] + (self.crpix[0] - 1)
        pix_y = lat_deg / self.cdelt[1] + (self.crpix[1] - 1)
        return pix_x, pix_y


def _2d_bilinear_interp(
    pix_x: Float[Array, ' *dims'], pix_y: Float[Array, ' *dims']
) -> tuple[Float[Array, '*dims 4'], Float[Array, '*dims 4'], Float[Array, '*dims 4']]:
    # Integer coordinates of the four neighbors
    x0 = jnp.floor(pix_x).astype(jnp.int32)
    y0 = jnp.floor(pix_y).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    fx = pix_x - x0
    fy = pix_y - y0

    # Bilinear weights: (bottom-left, bottom-right, top-left, top-right)
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Stack neighbor coords along a trailing axis, call pixel2index once
    # pixel2index handles bounds, returning -1 for out-of-bounds
    xs = jnp.stack([x0, x1, x0, x1], axis=-1).astype(pix_x.dtype)
    ys = jnp.stack([y0, y0, y1, y1], axis=-1).astype(pix_y.dtype)
    weights = jnp.stack([w00, w10, w01, w11], axis=-1)
    return xs, ys, weights


@register_static
class HealpixLandscape(StokesLandscape):
    """Class representing a Healpix-projected map of Stokes vectors."""

    def __init__(
        self,
        nside: int,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
        nested: bool = False,
    ) -> None:
        if nested:
            raise NotImplementedError('NESTED pixel ordering is not fully supported by jax-healpy.')
        shape = (12 * nside**2,)
        super().__init__(shape, stokes, dtype)
        self.nside = nside
        self.nested = nested

    @jax.jit
    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        r"""Convert quaternion to HEALPix pixel index.

        Args:
            quat (float): Quaternion.

        Returns:
            int: HEALPix pixel index.
        """
        # we want the 3 dimensions on the left
        vec = jnp.moveaxis(qrot_zaxis(quat), -1, 0)
        pix: Integer[Array, ' *dims'] = jhp.vec2pix(self.nside, *vec, nest=self.nested)
        return pix

    def world2interp(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, '*dims 4'], Float[Array, '*dims 4']]:
        """Returns (indices, weights) for bilinear HEALPix interpolation (n=4)."""
        # get_interp_weights returns (4, *dims) for both pixels and weights
        pixels, weights = jhp.get_interp_weights(self.nside, theta, phi, nest=self.nested)
        indices = jnp.moveaxis(pixels, 0, -1)
        weights = jnp.moveaxis(weights, 0, -1).astype(self.dtype)
        return indices, weights

    @jax.jit
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Integer[Array, ' *dims'], ...]:
        r"""Convert angles to HEALPix pixel index.

        Args:
            theta (float): Spherical $\theta$ angle.
            phi (float): Spherical $\phi$ angle.

        Returns:
            int: HEALPix pixel index.
        """
        return (jhp.ang2pix(self.nside, theta, phi, nest=self.nested),)


@register_static
class FrequencyLandscape(HealpixLandscape):
    def __init__(
        self,
        nside: int,
        frequencies: Array,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
    ):
        super().__init__(nside, stokes, dtype)
        self.frequencies = frequencies
        self.shape = (len(frequencies), 12 * nside**2)


@register_static
class AstropyWCSLandscape(StokesLandscape):
    """Class representing an astropy WCS map of Stokes vectors."""

    def __init__(
        self,
        shape: tuple[int, ...],
        wcs: WCS,
        stokes: ValidStokesLiteral = 'IQU',
        dtype: DTypeLike = np.float64,
    ) -> None:
        super().__init__(shape, stokes, dtype)
        self.wcs = wcs

    @jax.jit
    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        r"""Convert angles to WCS map indices.

        Args:
            theta (float): Spherical $\theta$ angle.
            phi (float): Spherical $\phi$ angle.

        Returns:
            WCS map index pairs
        """

        def f(theta, phi):  # type: ignore[no-untyped-def]
            # SkyCoord takes (lon,lat)
            pix_i, pix_j = self.wcs.world_to_pixel(SkyCoord(phi, (np.pi / 2 - theta), unit='rad'))
            return np.array(pix_i), np.array(pix_j)

        struct = jax.ShapeDtypeStruct(theta.shape, theta.dtype)
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
        stokes: ValidStokesLiteral = 'IQU',
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
            theta (float): Spherical $\theta$ angle = pi/2 - altitude.
            phi (float): Spherical $\phi$ angle = -azimuth.

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
class TangentialLandscape(StokesLandscape):
    r"""Class representing a flat map in the local tangent plane at zenith.

    Implements the gnomonic (tangential) projection: a ray from the telescope origin
    in direction $\hat{v}$ intersects the horizontal plane at height $h$ at

    $$
    x = h \frac{v_x}{v_z}, \quad y = h \frac{v_y}{v_z}
    $$

    where $v_z = \cos\theta$ is the elevation cosine (z-axis = zenith).

    The map is a 2D Cartesian grid in physical units (same units as ``height``), centered
    at the origin. Pixel spacing is ``dx`` along x and ``dy`` along y, so the map covers
    ``[-n_x*dx/2, n_x*dx/2]`` × ``[-n_y*dy/2, n_y*dy/2]``.

    Attributes:
        shape: Array shape ``(n_y, n_x)`` in row-major order.
        dx: Pixel spacing along the x-axis (same units as ``height``).
        dy: Pixel spacing along the y-axis (same units as ``height``).
        height: Height of the tangent plane above the telescope (same units as ``dx``/``dy``).
        stokes: Stokes components stored in the map (default ``'I'``).
        dtype: Data type for map values.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dx: float,
        dy: float,
        height: float,
        x0: float = 0.0,
        y0: float = 0.0,
        stokes: ValidStokesLiteral = 'I',
        dtype: DTypeLike = np.float64,
    ) -> None:
        if stokes != 'I':
            raise NotImplementedError
        super().__init__(shape, stokes, dtype)
        self.dx = dx
        self.dy = dy
        self.height = height
        self.x0 = x0
        self.y0 = y0

    @property
    def extent(self) -> tuple[float, float]:
        """Physical size of the map ``(x_size, y_size)`` in the same units as ``dx``/``dy``."""
        n_x, n_y = self.pixel_shape
        return n_x * self.dx, n_y * self.dy

    @classmethod
    def from_extent(
        cls,
        x_size: float,
        y_size: float,
        dx: float,
        dy: float,
        height: float,
        x0: float = 0.0,
        y0: float = 0.0,
        stokes: ValidStokesLiteral = 'I',
        dtype: DTypeLike = np.float64,
    ) -> 'TangentialLandscape':
        """Create a landscape from physical extent and pixel spacing.

        The map is centered at ``(x0, y0)`` and covers
        ``[x0 - x_size/2, x0 + x_size/2]`` × ``[y0 - y_size/2, y0 + y_size/2]``.
        The number of pixels is ``ceil(x_size / dx)`` × ``ceil(y_size / dy)``, so the
        actual covered area may be slightly larger than requested to accommodate whole
        pixels.

        Args:
            x_size: Total physical width of the map (same units as ``height``).
            y_size: Total physical height of the map (same units as ``height``).
            dx: Pixel spacing along x.
            dy: Pixel spacing along y.
            height: Height of the tangent plane above the telescope.
            x0: Physical x-coordinate of the map center (default 0).
            y0: Physical y-coordinate of the map center (default 0).
            stokes: Stokes components to store (default ``'I'``).
            dtype: Data type for map values.
        """
        n_x = int(np.ceil(x_size / dx))
        n_y = int(np.ceil(y_size / dy))
        return cls((n_y, n_x), dx, dy, height, x0, y0, stokes, dtype)

    def xy2pixel(
        self,
        x: Float[Array, ' *dims'],
        y: Float[Array, ' *dims'],
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Convert physical coordinates to floating-point pixel coordinates.

        The map is centered at ``(x0, y0)``. Integer pixel coordinates correspond to
        pixel centers.

        Args:
            x: Physical x-coordinates.
            y: Physical y-coordinates.

        Returns:
            Floating-point pixel coordinate pair ``(pix_x, pix_y)``.
        """
        n_x, n_y = self.pixel_shape
        pix_x = (x - self.x0) / self.dx + n_x / 2 - 0.5
        pix_y = (y - self.y0) / self.dy + n_y / 2 - 0.5
        return pix_x, pix_y

    @jax.jit
    def world2pixel(
        self,
        theta: Float[Array, ' *dims'],
        phi: Float[Array, ' *dims'],
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        r"""Convert ISO spherical angles to pixel coordinates via gnomonic projection.

        Args:
            theta: Co-latitude from zenith (0 at zenith, $\pi/2$ at horizon).
            phi: Azimuthal angle.

        Returns:
            Floating-point pixel coordinate pair ``(pix_x, pix_y)``.
        """
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        x = self.height * sin_theta * jnp.cos(phi) / cos_theta
        y = self.height * sin_theta * jnp.sin(phi) / cos_theta
        return self.xy2pixel(x, y)

    def quat2xy(
        self, quat: Float[Array, '*dims 4']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Convert quaternions to physical (x, y) coordinates on the tangent plane.

        Uses [`qrot_zaxis`][] to extract the pointing direction and applies the exact
        gnomonic projection. This is the primary conversion step used by
        [`AtmospherePointingOperator`][furax.obs.atmosphere.AtmospherePointingOperator] before
        adding wind displacement.

        Args:
            quat: Pointing quaternions (z-axis = zenith frame).

        Returns:
            Physical coordinate pair ``(x, y)``.
        """
        v = qrot_zaxis(quat)
        # cos(theta) = (a² + d²) - (b² + c²) = v[..., 2]
        # sin(theta) cos(phi) = 2 (ca + db) = v[..., 0]
        # sin(theta) sin(phi) = 2 (cd - ab) = v[..., 1]
        x = self.height * v[..., 0] / v[..., 2]
        y = self.height * v[..., 1] / v[..., 2]
        return x, y

    def quat2pixel(
        self, quat: Float[Array, '*dims 4']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Convert quaternions to floating-point pixel coordinates.

        Overrides the base-class implementation to use [`qrot_zaxis`][] directly,
        avoiding the round-trip through ISO angles.
        """
        x, y = self.quat2xy(quat)
        return self.xy2pixel(x, y)

    def xy2interp(
        self,
        x: Float[Array, ' *dims'],
        y: Float[Array, ' *dims'],
    ) -> tuple[Integer[Array, ' *dims 4'], Float[Array, ' *dims 4']]:
        """Bilinear interpolation weights and indices for physical coordinates.

        Returns the four neighbouring pixel indices and their bilinear weights.
        Out-of-bounds neighbours are marked with index ``-1`` and weight ``0``.

        Args:
            x: Physical x-coordinates.
            y: Physical y-coordinates.

        Returns:
            ``(indices, weights)`` where both have shape ``(*dims, 4)`` and the
            trailing axis runs over the four neighbours in order
            ``(i0,j0), (i1,j0), (i0,j1), (i1,j1)``.
        """
        pix_x, pix_y = self.xy2pixel(x, y)
        xs, ys, weights = _2d_bilinear_interp(pix_x, pix_y)
        indices = self.pixel2index(xs, ys)
        return indices, weights


@register_static
class LocalStokesLandscape(StokesLandscape):
    r"""A landscape over a subset of a parent landscape's pixels.

    Wraps a ``parent`` [`StokesLandscape`][] and an array of *global* (parent) flat pixel indices,
    defining a reduced pixel domain of ``nlocal`` real pixels. The indices are normalized on
    construction: any shape is flattened, duplicates are removed, negative entries (the
    out-of-bounds marker of ``world2index``) are dropped, and the result is sorted. Index-producing
    methods (``world2index`` / ``quat2index``) return **local** indices into the subset, so it is a
    drop-in landscape wherever a full-sky one is used.

    The pixel axis carries an extra trailing *sink* slot: ``shape == (nlocal + 1,)`` and the sink is
    index ``nlocal``. Coordinates outside the subset (and out-of-bounds inputs) map to the sink
    rather than ``-1``, so a plain gather from the length ``nlocal + 1`` buffer lands them on the
    dedicated discardable slot instead of wrapping onto a real pixel (as ``-1`` would). The map
    product space is the ``nlocal`` real pixels (no sink): [`restrict`][] and [`promote`][] move
    between the parent sky and that space.

    Attributes:
        parent: The parent landscape.
        global_indices: Sorted 1-D array of unique global (parent) flat pixel indices, length
            ``nlocal``.

    Examples:
        Restrict a 12-pixel HEALPix sky to two pixels (the shape gains a trailing sink slot):

        >>> parent = HealpixLandscape(1, stokes='I')
        >>> local = LocalStokesLandscape(parent, jnp.array([3, 7]))
        >>> local.nlocal, local.sink, local.shape
        (2, 2, (3,))

        Global indices map to local ones; unselected and out-of-bounds pixels fall in the sink:

        >>> local.global2local(jnp.array([3, 7, 5, -1]))
        Array([0, 1, 2, 2], dtype=int32)

        [`restrict`][] and [`promote`][] move between the parent sky and the real (sink-less)
        subset:

        >>> sky = parent.ones()
        >>> local.restrict(sky).shape
        (2,)
        >>> local.promote(local.restrict(sky)).shape
        (12,)
    """

    def __init__(
        self,
        parent: StokesLandscape,
        global_indices: Integer[np.ndarray | Array, '...'],
    ) -> None:
        # Deduplicate and drop negative entries to respect sorted-unique invariant.
        gi = jnp.unique(jnp.asarray(global_indices).reshape(-1))
        gi = gi[gi >= 0]
        npix = math.prod(parent.shape)
        if gi.size > 0 and int(gi[-1]) >= npix:
            raise ValueError(
                f'global_indices contains {int(gi[-1])}, out of range for a parent '
                f'with {npix} pixels.'
            )
        nlocal = int(gi.size)
        # The sink occupies index `nlocal`, so the buffer has shape `(nlocal + 1,)`.
        super().__init__(shape=(nlocal + 1,), stokes=parent.stokes, dtype=parent.dtype)
        self.parent = parent
        self.global_indices = gi
        # lut[global_flat_index] -> local index in [0, nlocal); unmapped pixels -> sink.
        lut = jnp.full(npix, nlocal, dtype=jnp.int32)
        lut = lut.at[gi].set(jnp.arange(nlocal, dtype=jnp.int32))
        self._global2local_lut = lut

    @property
    def nlocal(self) -> int:
        return len(self.global_indices)

    @property
    def sink(self) -> int:
        """The sink index that unmapped/out-of-bounds samples map to."""
        return len(self.global_indices)

    def world2index(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> Integer[Array, ' *dims']:
        """Convert world coordinates to local pixel indices (sink for unmapped)."""
        return self.global2local(self.parent.world2index(theta, phi))

    def quat2index(self, quat: Float[Array, '*dims 4']) -> Integer[Array, ' *dims']:
        """Convert quaternions to local pixel indices (sink for unmapped)."""
        return self.global2local(self.parent.quat2index(quat))

    def world2pixel(
        self, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
    ) -> tuple[Float[Array, ' *dims'], ...]:
        raise NotImplementedError('LocalStokesLandscape does not support world2pixel.')

    def global2local(self, indices: Integer[Array, ' *dims']) -> Integer[Array, ' *dims']:
        """Map global (parent) flat indices to local indices, sending unmapped ones to the sink.

        Out-of-bounds inputs (negative, e.g. ``-1`` from a missed sample) also map to the sink.
        """
        safe = jnp.where(indices >= 0, indices, 0)
        local: Integer[Array, ' *dims'] = jnp.where(
            indices >= 0, self._global2local_lut[safe], self.sink
        )
        return local

    def local2global(self, indices: Integer[Array, ' *dims']) -> Integer[Array, ' *dims']:
        """Map local indices to global (parent) flat indices, ``-1`` for out-of-range inputs."""
        valid = (indices >= 0) & (indices < self.nlocal)
        safe = jnp.where(valid, indices, 0)
        return jnp.where(valid, self.global_indices[safe], -1)

    def restrict(self, sky: _StokesT) -> _StokesT:
        """Extract the local subset from a parent-shaped Stokes map (``nlocal``-sized, no sink)."""
        flat = sky.data.reshape(sky.data.shape[0], -1)
        return type(sky).from_array(flat[:, self.global_indices])

    def promote(self, sky: _StokesT, fill_value: ScalarLike = 0) -> _StokesT:
        """Scatter a local (``nlocal``-sized) Stokes map back into a parent-shaped one.

        Pixels outside the local subset are set to ``fill_value``.
        """
        ncomp = sky.data.shape[0]
        npix = math.prod(self.parent.shape)
        flat = jnp.full((ncomp, npix), fill_value, dtype=sky.dtype)
        flat = flat.at[:, self.global_indices].set(sky.data)
        return type(sky).from_array(flat.reshape(ncomp, *self.parent.shape))

    def get_coverage(self, arg: Sampling) -> Integer[Array, ' nlocal']:
        """Per-pixel hit counts in local index space (samples outside the subset are ignored)."""
        local = self.world2index(arg.theta, arg.phi)
        valid = local < self.nlocal  # drop the sink
        safe = jnp.where(valid, local, 0)
        coverage = jnp.zeros(self.nlocal, dtype=np.int64)
        coverage = coverage.at[safe].add(valid.astype(np.int64))
        return coverage

    @classmethod
    def from_boolean_mask(
        cls, parent: StokesLandscape, mask: Integer[Array, ' *shape']
    ) -> 'LocalStokesLandscape':
        """Build from a boolean mask over parent pixels (any shape; raveled in C order)."""
        global_indices = jnp.flatnonzero(jnp.asarray(mask).reshape(-1))
        return cls(parent, global_indices)

    @classmethod
    def from_sampling(cls, parent: StokesLandscape, sampling: Sampling) -> 'LocalStokesLandscape':
        """Build from the pixels a [`Sampling`][furax.obs._samplings.Sampling] observes."""
        return cls(parent, parent.world2index(sampling.theta, sampling.phi))
