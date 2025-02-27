import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pixell

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
