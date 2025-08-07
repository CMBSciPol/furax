import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pixell.enmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def plot_cartview(  # type: ignore[no-untyped-def]
    input_maps,
    titles=None,
    lonra=[-180, 180],
    latra=[-90, 90],
    xsize=800,
    cmap='RdBu',
    vmaxs=None,
    vmins=None,
    vmax_quantile=0.999,
    nside=512,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]:
    """Visualisation function for CAR projection of healpix maps.
    Unlike healpy.cartview, this function returns matplotlib Axes object, and one can have
    more control of the plot elements such as grids and axes labels.
    # TODO: write type definition and documentation
    """
    if not isinstance(input_maps, list) and input_maps.ndim == 1:
        input_maps = input_maps[None, :]
    if titles is not None and not isinstance(titles, list):
        titles = [titles]
    if vmaxs is None:
        vmaxs = [None] * len(input_maps)
    if vmins is None:
        vmins = [None] * len(input_maps)

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
        if not vmin:
            vmin = -vmax

        im = ax.pcolor(
            lon_grid, lat_grid, proj_map, cmap=cmap, vmax=vmax, vmin=vmin, shading='nearest'
        )
        ax.xaxis.set_inverted(True)
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
