import healpy as hp
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pixell.enmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Any

from furax.obs.landscapes import WCSLandscape
from furax.obs.stokes import StokesPyTreeType


def ndmap_from_wcs_landscape(map: StokesPyTreeType, landscape: WCSLandscape) -> pixell.enmap.ndmap:
    """Convert a given Stokes pytree to pixell's ndmap"""
    if landscape.stokes == 'I':
        return pixell.enmap.ndmap(map.i, landscape.wcs)  # type: ignore[union-attr]
    if landscape.stokes == 'QU':
        return pixell.enmap.ndmap([map.q, map.u], landscape.wcs)  # type: ignore[union-attr]
    if landscape.stokes == 'IQU':
        return pixell.enmap.ndmap([map.i, map.q, map.u], landscape.wcs)  # type: ignore[union-attr]
    if landscape.stokes == 'IQUV':
        return pixell.enmap.ndmap([map.i, map.q, map.u, map.v], landscape.wcs)  # type: ignore[union-attr]
    else:
        raise NotImplementedError(f'Stokes {landscape.stokes} not supported')


def plot_ndmap(
    data: pixell.enmap.ndmap,
    title: str | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    cmap: str = 'RdBu_r',
    scale: float = 1.0,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes._axes.Axes | None = None,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Visualisation function for ndmap that replaces pixell.enplot.eshow for now.
    Inputs:
        title: title of the produced plot
        vmax: value above this are mapped to the colormap's upper end
        vmin: value below this are mapped to the colormap's lower end
        cmap: colormap name in matplotlib
        scale: value range scale factor of the colormap,
               overridden if vmin or vmax is provided
        fig, ax: matplotlib figure and axes
    Returns:
        fig, ax
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if vmax is None and vmin is None:
        vmax = scale * np.max(np.abs(data))
        vmin = -vmax

    wcs = data.wcs.wcs
    # Note that the data has axes [Dec, RA], unlike the wcs object's [RA, Dec]
    ra = wcs.crval[0] + wcs.cdelt[0] * (np.arange(data.shape[1] + 1) - wcs.crpix[0] - 0.5)
    dec = wcs.crval[1] + wcs.cdelt[1] * (np.arange(data.shape[0] + 1) - wcs.crpix[1] - 0.5)
    # Convert negative RA to positive values and unwrap if necessary
    ra = np.unwrap(ra % 360.0, period=360.0)

    # Plot and label
    im = ax.pcolormesh(ra, dec, data, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_xlabel('RA [deg]')
    ax.set_ylabel('Dec [deg]')
    ax.invert_xaxis()  # As per convention
    ax.grid(alpha=0.5)
    fig.colorbar(im)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')

    return fig, ax


def plot_cartview(
    input_maps: NDArray[Any],
    titles: list[str] | str | None = None,
    lonra: list[float] = [-180, 180],
    latra: list[float] = [-90, 90],
    xsize: int = 800,
    cmap: str = 'RdBu',
    vmaxs: list[float] | list[None] | None = None,
    vmins: list[float] | list[None] | None = None,
    vmax_quantile: float = 0.999,
    nside: int | None = None,
) -> tuple[matplotlib.figure.Figure, NDArray[matplotlib.axes.Axes]]:
    """Visualisation function for CAR projection of healpix maps.
    Unlike healpy.cartview, this function returns matplotlib Axes object, and one can have
    more control of the plot elements such as grids and axes labels.
    
    Args:
        input_maps: HEALPix map(s) to plot. Can be 1D (single map) or 2D (multiple maps)
        titles: Title(s) for the plot(s)
        lonra: Longitude range [min, max] in degrees
        latra: Latitude range [min, max] in degrees
        xsize: Number of pixels in longitude direction
        cmap: Colormap name
        vmaxs: Maximum values for color scale (one per map)
        vmins: Minimum values for color scale (one per map)
        vmax_quantile: Quantile to use for automatic vmax determination
        nside: HEALPix nside parameter. If None, inferred from input_maps
    
    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    if not isinstance(input_maps, list) and input_maps.ndim == 1:
        input_maps = input_maps[None, :]
    if titles is not None and not isinstance(titles, list):
        titles = [titles]
    if vmaxs is None:
        vmaxs = [None] * len(input_maps)
    if vmins is None:
        vmins = [None] * len(input_maps)
    
    # Infer nside if not provided
    if nside is None:
        npix = input_maps[0].shape[-1]  # Last dimension should be the number of pixels
        nside = hp.npix2nside(npix)

    ysize = int(np.round((latra[1] - latra[0]) * xsize / (lonra[1] - lonra[0])))
    lon_grid = np.linspace(lonra[0], lonra[1], xsize)
    lat_grid = np.linspace(latra[0], latra[1], ysize)

    pix_inds = hp.pixelfunc.ang2pix(nside, lon_grid[None, :], lat_grid[:, None], lonlat=True)

    n_maps = len(input_maps)
    fig, axs = plt.subplots(n_maps, 1, figsize=(10, 10 * (ysize / xsize) * n_maps))
    axs = np.atleast_1d(axs)

    for map_no in range(n_maps):
        ax = axs[map_no]

        proj_map = input_maps[map_no][pix_inds]

        vmax = vmaxs[map_no]
        vmin = vmins[map_no]
        if not vmax:
            vmax = np.quantile(np.abs(proj_map[np.abs(proj_map) > 0]), vmax_quantile)
        if not vmin and vmax is not None:
            vmin = -vmax

        im = ax.pcolor(
            lon_grid, lat_grid, proj_map, cmap=cmap, vmax=vmax, vmin=vmin, shading='nearest'
        )
        ax.xaxis.set_inverted(True) # Follow astronomical convention for inverted RA
        ax.set_xlabel('lon [deg]')
        ax.set_ylabel('lat [deg]')
        ax.grid(alpha=0.5)
        ax.set_aspect('equal')
        if titles is not None:
            ax.set_title(titles[map_no])

        the_divider = make_axes_locatable(ax)
        color_axis = the_divider.append_axes('right', size='5%', pad=0.1)
        _ = fig.colorbar(im, cax=color_axis, format='%.0e')
    fig.tight_layout()

    return fig, axs


def get_healpix_lonlat_ranges(
    healpix_map: NDArray[Any], 
    nside: int | None = None,
    padding_deg: float = 5.0,
    hit_threshold: float = 1e-18
) -> tuple[list[float], list[float]]:
    """Find appropriate longitude and latitude ranges for a HEALPix map.
    
    This function identifies the sky region covered by non-zero pixels in a HEALPix map
    and returns appropriate longitude and latitude ranges for visualization.
    
    Args:
        healpix_map: HEALPix map array
        nside: HEALPix nside parameter. If None, inferred from map length
        padding_deg: Padding in degrees to add around the data region
        hit_threshold: Minimum value to consider a pixel as having data
    
    Returns:
        tuple: (lonra, latra) where lonra=[lon_min, lon_max] and latra=[lat_min, lat_max]
               Ranges are in degrees, suitable for use with plot_cartview()
    
    Example:
        >>> import healpy as hp
        >>> import numpy as np
        >>> # Create a small patch map
        >>> nside = 64
        >>> npix = hp.nside2npix(nside)
        >>> hpx_map = np.zeros(npix)
        >>> # Add signal in a small region
        >>> theta, phi = hp.pix2ang(nside, np.arange(1000, 2000))
        >>> hpx_map[1000:2000] = 1.0
        >>> lonra, latra = get_healpix_lonlat_ranges(hpx_map, nside)
        >>> print(f"Longitude range: {lonra}")
        >>> print(f"Latitude range: {latra}")
    """
    # Infer nside if not provided
    if nside is None:
        npix = len(healpix_map)
        nside = hp.npix2nside(npix)
    
    # Find pixels with data above threshold
    data_mask = np.abs(healpix_map) > hit_threshold
    data_pixels = np.where(data_mask)[0]
    
    if len(data_pixels) == 0:
        # No data found, return full sky
        return [-180.0, 180.0], [-90.0, 90.0]
    
    # Convert pixel indices to longitude and latitude in degrees
    lon, lat = hp.pix2ang(nside, data_pixels, lonlat=True)
    
    # Convert longitude to [-180, 180] range
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
    lon_min = np.min(lon_wrapped) - padding_deg
    lon_max = np.max(lon_wrapped) + padding_deg

    # Clamp longitude to valid range [-180, 180]
    lon_min = max(lon_min, -180.0)
    lon_max = min(lon_max, 180.0)
    
    # Handle latitude range
    lat_min = np.min(lat) - padding_deg
    lat_max = np.max(lat) + padding_deg
    
    # Clamp latitude to valid range [-90, 90]
    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    
    return [lon_min, lon_max], [lat_min, lat_max]
