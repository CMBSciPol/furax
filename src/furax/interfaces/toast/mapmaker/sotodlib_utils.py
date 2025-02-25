import logging
import typing
from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pixell
from jax import Array, ShapeDtypeStruct
from jaxtyping import Bool, DTypeLike, Float, Integer, PyTree

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    BroadcastDiagonalOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    IndexOperator,
)
from furax.interfaces.sotodlib.observation import SotodlibObservationData
from furax.mapmaking.preconditioner import BJPreconditioner
from furax.mapmaking.utils import psd_to_invntt
from furax.obs import QURotationOperator
from furax.obs.landscapes import HealpixLandscape, WCSLandscape
from furax.obs.stokes import Stokes, StokesIQU, StokesPyTreeType, ValidStokesType

from . import templates

""" sotodlib data interface """


@typing.no_type_check
def get_landscape(
    obs: SotodlibObservationData,
    dtype: DTypeLike = np.float32,
    stokes: ValidStokesType = 'IQU',
    landscape_configs: dict[str, Any] = {},
) -> WCSLandscape | HealpixLandscape:
    """Create and return a WCSLandscape instance from the observation"""
    pass


@typing.no_type_check
def get_pointing_and_parallactic_angles(
    obs: SotodlibObservationData, landscape: WCSLandscape | HealpixLandscape
) -> tuple[Integer[Array, 'dets samps 2'], Float[Array, 'dets samps']]:
    """Obtain pointing information and parallactic angles from the observation"""
    pass


def get_invntt(
    obs: SotodlibObservationData, fmin: float, correlation_length: int, normalize: bool = True
) -> Float[Array, 'dets {correlation_length}']:
    """Compute the inverse covariance matrix from the noise psd fit"""

    noise_fits = obs.get_noise_fits(fmin=fmin)
    invntt = psd_to_invntt(noise_fits, correlation_length=correlation_length)
    if normalize:
        # Normalise to 1 at x=0
        invntt = invntt / invntt[:, [0]]

    return invntt  # type: ignore[no-any-return]


def select_pixel_indices(
    blocks: Float[Array, '... nstokes nstokes'], hits_cut: float, cond_cut: float
) -> Bool[Array, '...']:
    """Select indices of map pixels satisfying
    the minimum hits (hits_cut) and condition number (cond_cut) criteria"""
    eigs = jnp.linalg.eigvalsh(blocks)
    return jnp.logical_and(
        eigs[..., -1] > hits_cut * eigs[..., -1].max(), eigs[..., 0] > cond_cut * eigs[..., -1]
    )


def ndmap_from_wcs_landscape(map: StokesPyTreeType, landscape: WCSLandscape) -> pixell.enmap.ndmap:
    """Convert a given Stokes pytree to pixell's ndmap"""
    if landscape.stokes == 'I':
        return pixell.enmap.ndmap(map.i, landscape.wcs)  # type: ignore[union-attr]
    if landscape.stokes == 'IQU':
        return pixell.enmap.ndmap([map.i, map.q, map.u], landscape.wcs)  # type: ignore[union-attr]
    else:
        raise NotImplementedError(f'Stokes {landscape.stokes} not supported')


def safe_divide(
    x: StokesPyTreeType,
    y: StokesPyTreeType,
    abs_thres: float | None = None,
    rel_thres: float = 1e-5,
) -> pixell.enmap.ndmap:
    """Computes (x / y) in a numerically stable manner, and"""
    raise NotImplementedError()


""" Operators """


@typing.no_type_check
def get_pointing_operators(
    obs: SotodlibObservationData, landscape: WCSLandscape | HealpixLandscape
) -> tuple[IndexOperator, QURotationOperator]:
    pass


@typing.no_type_check
def get_acquisition(
    obs: SotodlibObservationData, demodulated: bool, landscape: WCSLandscape | HealpixLandscape
) -> AbstractLinearOperator:
    pass


@typing.no_type_check
def get_scanning_masker(
    obs: SotodlibObservationData, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> IndexOperator:
    pass


@typing.no_type_check
def get_template_operator(
    obs: SotodlibObservationData, name: str, configs: dict[str, Any]
) -> templates.TemplateOperator:
    """Create and return a template operator corresponding to the
    name and configuration provided.
    """
    pass


def template_operator_from_dict(
    obs: SotodlibObservationData, template_configs: dict[str, Any]
) -> BlockRowOperator:
    return BlockRowOperator(
        [
            get_template_operator(obs, name, template_configs[name])
            for name in template_configs.keys()
        ]
    )


""" Visualisation tools """


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


""" Mapmaking functions """


def binned_demod_mapmaker(
    observation: SotodlibObservationData, configs: dict[str, Any] | None, logger: logging.Logger
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'Binned Demod Mapmaker: {msg}') if logger else None

    # Set mapmaking config variables
    obs = observation.observation
    if configs is None:
        configs = dict()
    dtype = jnp.dtype(configs.get('dtype', 'float32'))
    landscape_configs = configs.get('landscape', {})
    scanning_mask = configs.get('scanning_mask', False)
    hits_cut = configs.get('hits_cut', 1e-5)

    # Data and landscape
    data = Stokes.from_stokes(
        obs.dsT.astype(dtype),
        obs.demodQ.astype(dtype),
        obs.demodU.astype(dtype),
    )
    struct = ShapeDtypeStruct(data.shape, dtype=dtype)
    data_struct = Stokes.from_stokes(struct, struct, struct)
    landscape = get_landscape(obs, dtype=dtype, stokes='IQU', landscape_configs=landscape_configs)

    # Pointing
    indexer, rotator = get_pointing_operators(obs, landscape)
    logger_info('Created pointing operators')

    # Optional mask for scanning
    if scanning_mask:
        one_masker = get_scanning_masker(obs, in_structure=struct)
        masker = BlockDiagonalOperator(blocks=StokesIQU(one_masker, one_masker, one_masker))
        logger_info('Created scan intervals masking operator')
        data_struct = masker.out_structure()  # Now with a subset of samples
    else:
        masker = IdentityOperator(data_struct)

    # Noise
    det_inv_var = 1.0 / np.var(data.q, axis=1)
    det_weighter = BroadcastDiagonalOperator(
        det_inv_var[:, None],
        in_structure=data_struct,
    )
    logger_info('Created inverse noise covariance operator')

    # Acquisition (I, Q, U Maps -> I, Q, U TODs)
    acquisition = masker @ rotator @ indexer
    logger_info('Created acquisition operator')

    # Define main mapmaking operators
    weighted_binner = (acquisition.T @ det_weighter @ masker).reduce()
    weight_computer = (acquisition.T @ det_weighter @ masker @ rotator).reduce()

    # Compute weights
    weights = weight_computer(StokesIQU.full(data.i.shape, 1.0, dtype=dtype))
    weights.i.block_until_ready()
    logger_info('Computed map weights')

    # Main computation
    weighted_map = weighted_binner(data)
    logger_info('Computed weighted maps')

    thres = hits_cut * jnp.max(weights.i)
    final_map = jax.tree_map(
        lambda weights, weighted_map: jnp.where(weights > thres, weighted_map / weights, 0.0),
        weights,
        weighted_map,
    )
    logger_info('Computed final maps')

    if isinstance(landscape, WCSLandscape):
        # Convert to ndmaps
        weighted_map = ndmap_from_wcs_landscape(weighted_map, landscape)
        weights = ndmap_from_wcs_landscape(weights, landscape)
        final_map = ndmap_from_wcs_landscape(final_map, landscape)
    else:
        weighted_map = np.array([weighted_map.i, weighted_map.q, weighted_map.u])
        weights = np.array([weights.i, weights.q, weights.u])
        final_map = np.array([final_map.i, final_map.q, final_map.u])

    return {'map': final_map, 'weighted_map': weighted_map, 'weight': weights}


@typing.no_type_check
def binned_mapmaker(
    observation: SotodlibObservationData,
    configs: dict[str, Any] | None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    pass


@typing.no_type_check
def ml_mapmaker(
    observation: SotodlibObservationData,
    configs: dict[str, Any] | None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    pass


def two_step_mapmaker(
    observation: SotodlibObservationData,
    configs: dict[str, Any] | None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'Two-Step Mapmaker: {msg}') if logger else None

    obs = observation.observation
    # Set mapmaking config variables
    if configs is None:
        configs = dict()
    dtype = jnp.dtype(configs.get('dtype', 'float64'))
    landscape_configs = configs.get('landscape', {})
    scanning_mask = configs.get('scanning_mask', False)
    hits_cut = configs.get('hits_cut', 1e-2)
    cond_cut = configs.get('cond_cut', 1e-2)
    solver = configs.get('solver', dict())
    rtol = solver.get('rtol', 1e-6)
    atol = solver.get('atol', 0)
    max_steps = solver.get('max_steps', 1000)
    template_configs = configs.get('template', {})
    template_names = list(template_configs.keys())

    # Data and landscape
    data = jnp.array(obs.signal, dtype=dtype)
    data_struct = ShapeDtypeStruct(data.shape, dtype)
    landscape = get_landscape(obs, dtype=dtype, stokes='IQU', landscape_configs=landscape_configs)

    # Acquisition (pointing operator): I, Q, U Maps -> TOD
    acquisition = get_acquisition(obs, demodulated=False, landscape=landscape)
    logger_info('Created acquisition operator')

    # Optional mask for scanning
    if scanning_mask:
        mask = jnp.array(observation.get_scanning_mask(), dtype=dtype)
        mask_projector = BroadcastDiagonalOperator(mask, in_structure=data_struct)
        logger_info('Created scan intervals masking operator')
        logger_info(f'{round(np.sum(mask))}/{mask.shape[0]} samples used')
    else:
        mask_projector = IdentityOperator(data_struct)

    # Noise
    white_noise = observation.get_white_noise_fit()
    diag_invntt_op = DiagonalOperator(
        (jnp.array(1.0 / white_noise)[:, None]), in_structure=data_struct
    )
    logger_info('Created inverse noise covariance operator')

    # System matrix
    system = BJPreconditioner.create(
        acquisition.T @ mask_projector @ diag_invntt_op @ mask_projector @ acquisition
    )
    logger_info('Created system operator')

    # Map pixel selection
    blocks = system.get_blocks()
    valid_inds = jnp.argwhere(select_pixel_indices(blocks, hits_cut=hits_cut, cond_cut=cond_cut))
    logger_info(f'Proceeding with {valid_inds.shape[0]}/{prod(landscape.shape)} pixels')

    if isinstance(landscape, WCSLandscape):
        selector = IndexOperator(
            (valid_inds[:, 0], valid_inds[:, 1]), in_structure=landscape.structure
        )
    elif isinstance(landscape, HealpixLandscape):
        selector = IndexOperator((valid_inds,), in_structure=landscape.structure)
    else:
        raise NotImplementedError
    # TODO: more efficient solution?
    system_inv = selector @ system.inverse() @ selector.T

    # Templates
    template_op = template_operator_from_dict(obs, template_configs)
    logger_info('Built template operators')

    # Define operators
    A = acquisition @ selector.T
    M = diag_invntt_op
    mp = mask_projector
    FA = M - M @ mp @ A @ system_inv @ A.T @ mp @ M

    solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_steps)
    with Config(solver=solver):
        template_estimator = (
            (template_op.T @ mp @ FA @ mp @ template_op).I @ template_op.T @ mp @ FA @ mp
        )
    map_estimator = system_inv @ A.T @ mp @ M @ mp

    @jax.jit
    def process(d):  # type: ignore[no-untyped-def]
        x = template_estimator(d)  # Template amplitude estimates
        s = map_estimator(d - template_op(x))  # Map estimates
        return s, x

    logger_info('Set up mapmaking operator')

    # Run mapmaking
    rec_map, tmpl_ampl = process(data)
    result_map = selector.T(rec_map)
    result_map.i.block_until_ready()
    logger_info('Finished mapmaking computation')

    # Format output and compute auxilary data
    if isinstance(landscape, WCSLandscape):
        final_map = ndmap_from_wcs_landscape(result_map, landscape)
    else:
        final_map = np.array([result_map.i, result_map.q, result_map.u])
    projs = {
        'proj_map': (mask_projector @ acquisition)(result_map),
        **{
            f'proj_{template_names[i]}': (mask_projector @ template_op.blocks[i])(tmpl_ampl[i])
            for i in range(len(template_names))
        },
    }

    return {'map': final_map, 'weights': blocks, **projs}
