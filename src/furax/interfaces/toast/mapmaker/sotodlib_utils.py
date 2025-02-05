import logging
from math import prod

import equinox
import jax
import jax.numpy as jnp
import lineax as lx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pixell
from astropy.wcs import WCS
from jax import Array, ShapeDtypeStruct
from jaxtyping import Bool, Float, Inexact, Integer, PyTree, DTypeLike
from sotodlib import coords
from sotodlib.core import AxisManager
from typing import Any
from numpy.typing import NDArray
from furax.obs.stokes import ValidStokesType

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    BroadcastDiagonalOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    IndexOperator,
    SymmetricBandToeplitzOperator,
)
from furax.obs import QURotationOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape
from furax.obs.stokes import Stokes, StokesIQU, StokesPyTreeType

from . import templates
from .preconditioner import BJPreconditioner
from .utils import psd_to_invntt

""" Custom FURAX classes and operators """


class WCSLandscape(StokesLandscape):
    """Stokes PyTree for WCS maps
    Not fully implemented yet, potentially should be added to FURAX
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        wcs: WCS,
        stokes: ValidStokesType,
        dtype: DTypeLike = np.float32,
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
        self, theta: Float[Array, '...'], phi: Float[Array, '...']
    ) -> tuple[Float[Array, '...'], ...]:
        raise NotImplementedError()


class StokesIndexOperator(AbstractLinearOperator):
    """Operator for integer index operation on Stokes PyTrees
    The indices are assumed to be identical for I, Q and U
    """

    indices: Integer[Array, '...'] | tuple[Integer[Array, '...'] | NDArray[Any], ...]
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _out_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        indices: Integer[Array, '...'] | tuple[Integer[Array, '...'] | NDArray[Any], ...],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        out_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
    ) -> None:
        self.indices = indices
        self._in_structure = in_structure
        self._out_structure = out_structure or AbstractLinearOperator.out_structure(self)

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(lambda leaf: leaf[self.indices], x)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure


class IQUModulationOperator(AbstractLinearOperator):
    """Class that adds the input Stokes signals to a single HWP-modulated signal
    Similar to LinearPolarizerOperator @ QURotationOperator(hwp_angle), except that
    only half of the QU rotation needs to be computed
    """

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    cos_hwp_angle: Float[Array, ' samps']
    sin_hwp_angle: Float[Array, ' samps']

    def __init__(
        self,
        shape: tuple[int, ...],
        hwp_angle: Float[Array, '...'],
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        self._in_structure = Stokes.class_for('IQU').structure_for(shape, dtype)
        self.cos_hwp_angle = jnp.cos(4 * hwp_angle.astype(dtype))
        self.sin_hwp_angle = jnp.sin(4 * hwp_angle.astype(dtype))

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        return x.i + self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u  # type: ignore[union-attr]

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


""" sotodlib data interface """


def get_landscape(
    obs: AxisManager,
    dtype: DTypeLike = np.float32,
    stokes: ValidStokesType = 'IQU',
    landscape_configs: dict[str, Any] = {},
) -> WCSLandscape | HealpixLandscape:
    """Create and return a WCSLandscape instance from the observation"""
    landscape_type = landscape_configs.get('type', 'WCS')

    if landscape_type.upper() == 'WCS':
        resolution = landscape_configs.get('resolution', 8.0)
        res = resolution * pixell.utils.arcmin

        # Base wcs object with CAR projection
        wcs_kernel_init = coords.get_wcs_kernel('car', 0, 0, res=res)

        # Adjust map boundaries to match the footprint of the observation
        wcs_shape, wcs_kernel = coords.get_footprint(obs, wcs_kernel_init)

        return WCSLandscape(wcs_shape, wcs_kernel, stokes=stokes, dtype=dtype)

    elif landscape_type.upper() == 'HEALPIX':
        nside = landscape_configs.get('nside', 512)

        return HealpixLandscape(nside=nside, stokes=stokes, dtype=dtype)

    else:
        raise NotImplementedError(f'Landscape type {landscape_type} not supported')


def get_pointing_and_parallactic_angles(
    obs: AxisManager, landscape: WCSLandscape | HealpixLandscape
) -> tuple[Integer[Array, 'dets samps 2'], Float[Array, 'dets samps']]:
    """Obtain pointing information and parallactic angles from the observation"""
    # Projection Matrix class instance for the observation
    if isinstance(landscape, WCSLandscape):
        # TODO: pass 'cuts' keyword here for time slices (glitches etc)?
        P = coords.P.for_tod(obs, wcs_kernel=landscape.wcs, comps='TQU', hwp=True)
    elif isinstance(landscape, HealpixLandscape):
        hp_geom = coords.healpix_utils.get_geometry(nside=landscape.nside)
        P = coords.P.for_tod(obs, geom=hp_geom)
    else:
        raise NotImplementedError(f'Landscape {landscape} not supported')

    # Projectionist object from P
    proj = P._get_proj()

    # Assembly containing the focal plane and boresight information
    assembly = P._get_asm()

    # Get the pixel indicies as before, but also obtain
    # the spin projection factors of size (n_samps,n_comps) for each detector,
    # which have 1, cos(2*p), sin(2*p) where p is the parallactic angle
    pixel_inds, spin_proj = proj.get_pointing_matrix(assembly)
    pixel_inds = np.array(pixel_inds)
    spin_proj = jnp.array(spin_proj, dtype=landscape.dtype)
    para_ang = jnp.arctan2(spin_proj[..., 2], spin_proj[..., 1]) / 2.0

    return pixel_inds, para_ang


def get_noise_fits(obs: AxisManager, fmin: float) -> NDArray[np.float_]:
    if 'psdT' in obs.preprocess.keys():
        f = obs.preprocess.psdT.freqs
        fit = obs.preprocess.noiseT_fit.fit  # columns: (fknee, w, alpha)
    elif 'Pxx_raw' in obs.preprocess.keys():
        f = obs.preprocess.Pxx_raw.freqs
        fit = obs.preprocess.noise_signal_fit.fit  # columns: (fknee, w, alpha)
    else:
        # Estimate psd
        raise NotImplementedError('Self-psd evaluation not implemented')
    imin = np.argmin(np.abs(f - fmin))
    noiseT_fit_eval = np.zeros((fit.shape[0], f.size), dtype=float)  # (dets, freqs)
    noiseT_fit_eval[:, imin:] = fit[:, [1]] * (1 + (fit[:, [0]] / f[None, imin:]) ** fit[:, [2]])
    noiseT_fit_eval[:, :imin] = noiseT_fit_eval[:, [imin]]

    return np.array(noiseT_fit_eval)


def get_white_noise_fit(
    obs: AxisManager,
) -> NDArray[np.float_]:
    if 'psdT' in obs.preprocess.keys():
        fit = obs.preprocess.noiseT_fit.fit  # columns: (fknee, w, alpha)
    elif 'Pxx_raw' in obs.preprocess.keys():
        fit = obs.preprocess.noise_signal_fit.fit  # columns: (fknee, w, alpha)
    return fit[:, 1]  # type: ignore[no-any-return]


def get_scanning_intervals(obs: AxisManager) -> NDArray[np.int_]:
    # Assumes that the detectors have identical scanning intervals,
    # and that the scanning intervals are the complement of turnaround intervals
    return obs.preprocess.turnaround_flags.turnarounds.ranges[0].complement().ranges()  # type: ignore[no-any-return]


def get_scanning_mask(obs: AxisManager) -> NDArray[np.bool_]:
    # Assumes that the detectors have identical scanning intervals,
    return obs.preprocess.turnaround_flags.turnarounds.ranges[0].complement().mask()  # type: ignore[no-any-return]


def get_timestamps(obs: AxisManager) -> NDArray[np.float_]:
    return obs.timestamps  # type: ignore[no-any-return]


def get_azimuth(obs: AxisManager) -> NDArray[np.float_]:
    return obs.boresight.az  # type: ignore[no-any-return]


def get_hwp_angles(obs: AxisManager) -> NDArray[np.float_]:
    return obs.hwp_angle  # type: ignore[no-any-return]


def get_invntt(
    obs: AxisManager, fmin: float, correlation_length: int, normalize: bool = True
) -> Float[Array, 'dets {correlation_length}']:
    """Compute the inverse covariance matrix from the noise psd fit"""

    noise_fits = get_noise_fits(obs, fmin=fmin)
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


def get_pointing_operators(
    obs: AxisManager, landscape: WCSLandscape | HealpixLandscape
) -> tuple[StokesIndexOperator, QURotationOperator]:
    pixel_inds, para_ang = get_pointing_and_parallactic_angles(obs, landscape)

    # Pointing, StokesPyTree-compatable
    if isinstance(landscape, WCSLandscape):
        indexer = StokesIndexOperator(
            (pixel_inds[..., 0], pixel_inds[..., 1]), in_structure=landscape.structure
        )
    elif isinstance(landscape, HealpixLandscape):
        indexer = StokesIndexOperator(pixel_inds[..., 0], in_structure=landscape.structure)
    else:
        raise NotImplementedError(f'Landscape {landscape} not supported')

    # Rotation due to coordinate transform
    # Note the minus sign on the rotation angle!
    tod_shape = pixel_inds.shape[:2]
    rotator = QURotationOperator.create(
        tod_shape, dtype=landscape.dtype, stokes='IQU', angles=-para_ang
    )

    return indexer, rotator


def get_acquisition(
    obs: AxisManager, demodulated: bool, landscape: WCSLandscape | HealpixLandscape
) -> AbstractLinearOperator:
    indexer, rotator = get_pointing_operators(obs, landscape)
    if demodulated:
        return (rotator @ indexer).reduce()
    else:
        hwp_angle = jnp.array(obs.hwp_angle, dtype=landscape.dtype)
        modulator = IQUModulationOperator(obs.signal.shape, hwp_angle, dtype=landscape.dtype)
        return (modulator @ rotator @ indexer).reduce()


def get_scanning_masker(
    obs: AxisManager, in_structure: PyTree[jax.ShapeDtypeStruct]
) -> IndexOperator:
    """Create and return a flag operator which selects only the scanning intervals
    of the given TOD of shape (ndets, nsamps).
    """
    mask = jnp.array(get_scanning_mask(obs))
    out_structure = ShapeDtypeStruct(
        shape=(in_structure.shape[0], np.sum(mask)), dtype=in_structure.dtype
    )
    masker = IndexOperator(
        (slice(None), mask), in_structure=in_structure, out_structure=out_structure
    )
    return masker


def get_template_operator(
    obs: AxisManager, name: str, configs: dict[str, Any]
) -> templates.TemplateOperator:
    """Create and return a template operator corresponding to the
    name and configuration provided.
    """
    n_dets = obs.dets.count

    if name == 'polynomial':
        max_poly_order: int = configs.get('max_poly_order', 0)
        return templates.PolynomialTemplateOperator.create(
            max_poly_order=max_poly_order,
            intervals=get_scanning_intervals(obs),
            times=jnp.array(get_timestamps(obs)),
            n_dets=n_dets,
        )

    if name == 'scan_synchronous':
        min_poly_order: int = configs.get('min_poly_order', 0)
        max_poly_order: int = configs.get('max_poly_order', 0)  # type: ignore[no-redef]
        return templates.ScanSynchronousTemplateOperator.create(
            min_poly_order=min_poly_order,
            max_poly_order=max_poly_order,
            azimuth=jnp.array(get_azimuth(obs)),
            n_dets=n_dets,
        )

    if name == 'hwp_synchronous':
        n_harmonics: int = configs.get('n_harmonics', 0)
        return templates.HWPSynchronousTemplateOperator.create(
            n_harmonics=n_harmonics, hwp_angles=jnp.array(get_hwp_angles(obs)), n_dets=n_dets
        )

    """
    if name == 'common_mode':
        # Assumes that the cross power spectral density is precomputed
        # and stored as '_cross_psd'
        freq, csd = observation_data._cross_psd
        freq_threshold: float = config.get('freq_threshold')
        n_modes: int = config.get('n_modes')

        return templates.CommonModeTemplateOperator.create(
            freq_threshold=freq_threshold,
            n_modes=n_modes,
            freq=freq,
            csd=csd,
            tods=observation_data.get_tods()
        )
    """

    raise NotImplementedError(f'Template {name} is not implemented')


def template_operator_from_dict(
    obs: AxisManager, template_configs: dict[str, Any]
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
    obs: AxisManager, configs: dict[str, Any] | None, logger: logging.Logger
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'Binned Demod Mapmaker: {msg}') if logger else None

    # Set mapmaking config variables
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


def binned_mapmaker(
    obs: AxisManager, configs: dict[str, Any] | None, logger: logging.Logger | None = None
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'Binned Mapmaker: {msg}') if logger else None

    # Set mapmaking config variables
    if configs is None:
        configs = dict()
    dtype = jnp.dtype(configs.get('dtype', 'float32'))
    landscape_configs = configs.get('landscape', {})
    scanning_mask = configs.get('scanning_mask', False)
    hits_cut = configs.get('hits_cut', 1e-5)
    cond_cut = configs.get('cond_cut', 1e-4)
    debug = configs.get('debug', True)

    # Data and landscape
    data = jnp.array(obs.signal, dtype=dtype)
    data_struct = ShapeDtypeStruct(data.shape, data.dtype)
    landscape = get_landscape(obs, dtype=dtype, stokes='IQU', landscape_configs=landscape_configs)

    # Acquisition (I, Q, U Maps -> TOD)
    acquisition = get_acquisition(obs, demodulated=False, landscape=landscape)
    logger_info('Created acquisition operator')

    # Optional mask for scanning
    if scanning_mask:
        masker = get_scanning_masker(obs, in_structure=data_struct)
        logger_info('Created scan intervals masking operator')
        data_struct = masker.out_structure()  # Now with a subset of samples
        acquisition = masker @ acquisition
    else:
        masker = IdentityOperator(data_struct)

    # Noise
    det_inv_var = 1.0 / np.var(data, axis=1)
    det_weighter = DiagonalOperator(det_inv_var[:, None], in_structure=data_struct)
    logger_info('Created inverse noise covariance operator')

    # System matrix
    system = BJPreconditioner.create((acquisition.T @ det_weighter @ acquisition).reduce())
    logger_info('Created system operator')

    # Mapmaking operator
    binner = acquisition.T @ det_weighter @ masker
    mapmaking_operator = system.inverse() @ binner

    @jax.jit
    def process(d):  # type: ignore[no-untyped-def]
        return mapmaking_operator.reduce()(d)

    logger_info('Set up mapmaking operator')

    # Run mapmaking
    res = process(data)
    res.i.block_until_ready()
    logger_info('Finished mapmaking')

    if debug:
        res = process(data)
        res.i.block_until_ready()
        logger_info('Test - second time - Finished mapmaking')

    # Convert to ndmaps
    if isinstance(landscape, WCSLandscape):
        final_map = ndmap_from_wcs_landscape(Stokes.from_stokes(res.i, res.q, res.u), landscape)
    else:
        final_map = np.array([res.i, res.q, res.u])
    weights = np.array(system.blocks)

    # Map pixel selection
    valid = select_pixel_indices(system.blocks, hits_cut=hits_cut, cond_cut=cond_cut)
    logger_info(f'Selecting {jnp.sum(valid)}/{valid.size} pixels')
    final_map[:, np.logical_not(valid)] = 0.0
    weights[np.logical_not(valid), :, :] = 0.0

    return {'map': final_map, 'weights': weights}


def ml_mapmaker(
    obs: AxisManager, configs: dict[str, Any] | None, logger: logging.Logger | None = None
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'ML Mapmaker: {msg}') if logger else None

    # Set mapmaking config variables
    if configs is None:
        configs = dict()
    dtype = jnp.dtype(configs.get('dtype', 'float64'))
    correlation_length = configs.get('correlation_length', 1000)
    psd_fmin = configs.get('psd_fmin', 1e-2)
    landscape_configs = configs.get('landscape', {})
    scanning_mask = configs.get('scanning_mask', False)
    hits_cut = configs.get('hits_cut', 1e-2)
    cond_cut = configs.get('cond_cut', 1e-2)
    solver = configs.get('solver', dict())
    rtol = solver.get('rtol', 1e-6)
    atol = solver.get('atol', 0)
    max_steps = solver.get('max_steps', 1000)
    has_templates = 'template' in configs.keys()
    if has_templates:
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
        mask = jnp.array(get_scanning_mask(obs), dtype=dtype)
        mask_projector = BroadcastDiagonalOperator(mask, in_structure=data_struct)
        logger_info('Created scan intervals masking operator')
        logger_info(f'{round(np.sum(mask))}/{mask.shape[0]} samples used')
    else:
        mask_projector = IdentityOperator(data_struct)

    # Noise
    invntt = jnp.array(
        get_invntt(obs, fmin=psd_fmin, correlation_length=correlation_length, normalize=True),
        dtype=dtype,
    )
    invntt_op = SymmetricBandToeplitzOperator(invntt, in_structure=data_struct)
    diag_invntt_op = DiagonalOperator(invntt[:, [0]], in_structure=data_struct)
    logger_info('Created inverse noise covariance operator')

    # System matrix with diagonal noise covariance and full map pixels
    diag_system = BJPreconditioner.create(
        (acquisition.T @ mask_projector @ diag_invntt_op @ mask_projector @ acquisition).reduce()
    )
    logger_info('Created diagonal system matrix')

    # Map pixel selection
    valid_inds = np.argwhere(
        select_pixel_indices(diag_system.blocks, hits_cut=hits_cut, cond_cut=cond_cut)
    )

    logger_info(f'Proceeding with {valid_inds.shape[0]}/{prod(landscape.shape)} pixels')

    # Preconditioner
    if isinstance(landscape, WCSLandscape):
        selector = StokesIndexOperator(
            (valid_inds[:, 0], valid_inds[:, 1]), in_structure=landscape.structure
        )
        preconditioner = BJPreconditioner(
            diag_system.blocks[valid_inds[:, 0], valid_inds[:, 1], :, :], selector.out_structure()
        ).inverse()
    elif isinstance(landscape, HealpixLandscape):
        selector = StokesIndexOperator((valid_inds,), in_structure=landscape.structure)
        preconditioner = BJPreconditioner(
            diag_system.blocks[valid_inds, :, :], selector.out_structure()
        ).inverse()

    logger_info('Created Block Jacobi preconditioner')

    # Templates (optional)
    if has_templates:
        template_op = template_operator_from_dict(obs, template_configs)
        logger_info('Built template operators')

    # Mapmaking operator
    if has_templates:
        p = BlockDiagonalOperator([preconditioner, IdentityOperator(template_op.in_structure())])
        h = BlockRowOperator([acquisition @ selector.T, template_op])
    else:
        p = preconditioner
        h = acquisition @ selector.T
    mp = mask_projector
    solver = lx.CG(rtol=rtol, atol=atol, max_steps=max_steps)
    solver_options = {'preconditioner': lx.TaggedLinearOperator(p, lx.positive_semidefinite_tag)}
    with Config(solver=solver, solver_options=solver_options):
        mapmaking_operator = (h.T @ mp @ invntt_op @ mp @ h).I @ h.T @ mp @ invntt_op @ mp

    @jax.jit
    def process(d):  # type: ignore[no-untyped-def]
        return mapmaking_operator.reduce()(d)

    logger_info('Completed setting up the solver')

    # Run mapmaking
    if has_templates:
        rec_map, tmpl_ampl = process(data)
    else:
        rec_map = process(data)
    result_map = selector.T(rec_map)
    result_map.i.block_until_ready()
    logger_info('Finished mapmaking computation')

    # Format output and compute auxilary data
    if isinstance(landscape, WCSLandscape):
        final_map = ndmap_from_wcs_landscape(result_map, landscape)
    else:
        final_map = np.array([result_map.i, result_map.q, result_map.u])
    weights = diag_system.blocks
    if has_templates:
        projs = {
            'proj_map': (mask_projector @ acquisition)(result_map),
            **{
                f'proj_{template_names[i]}': (mask_projector @ template_op.blocks[i])(tmpl_ampl[i])
                for i in range(len(template_names))
            },
        }
    else:
        projs = {'proj_map': (mask_projector @ acquisition)(result_map)}

    return {'map': final_map, 'weights': weights, **projs}


def two_step_mapmaker(
    obs: AxisManager, configs: dict[str, Any] | None, logger: logging.Logger | None = None
) -> dict[str, Any]:
    logger_info = lambda msg: logger.info(f'Two-Step Mapmaker: {msg}') if logger else None

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
        mask = jnp.array(get_scanning_mask(obs), dtype=dtype)
        mask_projector = BroadcastDiagonalOperator(mask, in_structure=data_struct)
        logger_info('Created scan intervals masking operator')
        logger_info(f'{round(np.sum(mask))}/{mask.shape[0]} samples used')
    else:
        mask_projector = IdentityOperator(data_struct)

    # Noise
    white_noise = get_white_noise_fit(obs)
    diag_invntt_op = DiagonalOperator((jnp.array(1.0 / white_noise)[:, None]), in_structure=data_struct)
    logger_info('Created inverse noise covariance operator')

    # System matrix
    system = BJPreconditioner.create(
        (acquisition.T @ mask_projector @ diag_invntt_op @ mask_projector @ acquisition).reduce()
    )
    logger_info('Created system operator')

    # Map pixel selection
    valid_inds = np.argwhere(
        select_pixel_indices(system.blocks, hits_cut=hits_cut, cond_cut=cond_cut)
    )
    logger_info(f'Proceeding with {valid_inds.shape[0]}/{prod(landscape.shape)} pixels')

    if isinstance(landscape, WCSLandscape):
        selector = StokesIndexOperator(
            (valid_inds[:, 0], valid_inds[:, 1]), in_structure=landscape.structure
        )
        system_inv = BJPreconditioner(
            system.blocks[valid_inds[:, 0], valid_inds[:, 1], :, :], selector.out_structure()
        ).inverse()
    elif isinstance(landscape, HealpixLandscape):
        selector = StokesIndexOperator((valid_inds,), in_structure=landscape.structure)
        system_inv = BJPreconditioner(
            system.blocks[valid_inds, :, :], selector.out_structure()
        ).inverse()

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
    weights = system.blocks
    projs = {
        'proj_map': (mask_projector @ acquisition)(result_map),
        **{
            f'proj_{template_names[i]}': (mask_projector @ template_op.blocks[i])(tmpl_ampl[i])
            for i in range(len(template_names))
        },
    }

    return {'map': final_map, 'weights': weights, **projs}
