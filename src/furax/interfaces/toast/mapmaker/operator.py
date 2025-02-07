from pathlib import Path
from typing import Any, TypeAlias

import healpy as hp
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import traitlets
import yaml
from jaxtyping import PyTree
from toast.observation import default_values as defaults
from toast.ops.operator import Operator as ToastOperator
from toast.traits import Bool, Float, Int, Unicode, trait_docs
from toast.utils import Logger

from furax import (
    AbstractLinearOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    IndexOperator,
    RavelOperator,
    SymmetricBandToeplitzOperator,
)
from furax.mapmaking.utils import (
    compute_cross_psd,
    estimate_filtered_psd,
    estimate_psd,
    interpolate_psd,
    next_fast_fft_size,
    psd_to_invntt,
)
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import StokesIQU

from ..observation import ToastObservationData
from .templates import TemplateOperator

ObservationKeysDict: TypeAlias = dict[str, list[str]]


@trait_docs
class MapMaker(ToastOperator):  # type: ignore[misc]
    """Operator which makes maps with the furax tools."""

    # Class traits
    API = Int(0, help='Internal interface version for this operator')

    # General configuration
    binned = Bool(False, help='Make a binned map')
    lagmax = Int(1_000, help='Maximum lag of the correlation function')
    nperseg = Int(1_024, help='Number of samples in each segment for fourier transforms')
    nside = Int(64, help='HEALPix nside parameter for the output maps')
    output_dir = Unicode('.', help='Write output data products to this directory')
    stokes = Unicode('IQU', help='Stokes parameters to reconstruct')

    # Solver configuration
    atol = Float(1e-6, help='Absolute tolerance for terminating solve')
    rtol = Float(1e-6, help='Relative tolerance for terminating solve')
    max_steps = Int(500, help='Maximum number of iterations to run the solver for')

    # TOAST buffer names
    det_data = Unicode(defaults.det_data, help='Observation detdata key for the timestream data')
    hwp_angle = Unicode(None, allow_none=True, help='Observation shared key for HWP angle')
    noise_model = Unicode(
        None, allow_none=True, help='Observation key containing a noise model to use'
    )
    pixels = Unicode(defaults.pixels, help='Observation detdata key for pixel indices')
    quats = Unicode(defaults.quats, help='Observation detdata key for detector quaternions')

    # Flagging and masking
    det_mask = Int(defaults.det_mask_nonscience, help='Bit mask value for per-detector flagging')

    # TODO support this?
    view = Unicode(None, allow_none=True, help='Use this view of the data in all observations')

    @traitlets.validate('stokes')
    def _check_stokes(self, proposal):  # type: ignore[no-untyped-def]
        value = proposal['value']
        if value != 'IQU':
            not_iqu_error_msg = f'Only IQU map-making is supported for now, got {value!r}'
            raise traitlets.TraitError(not_iqu_error_msg)
        return value

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._cached = False
        self._logprefix = 'Furax MapMaker:'

        # Enforce that nperseg is a power of 2 for performance
        new_nperseg = next_fast_fft_size(self.nperseg)
        if new_nperseg != self.nperseg:
            raise RuntimeWarning(
                f'Provided nperseg ({self.nperseg}) is not a power of 2. \
                  Using ({new_nperseg}) instead.'
            )
        self.nperseg = new_nperseg

    def clear(self) -> None:
        """Delete the underlying memory."""
        for attr in [
            '_tods',
            '_pixels',
            '_quats',
            '_det_angles',
            '_gamma',
            '_hwp_angles',
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    def __del__(self) -> None:
        self.clear()

    def _exec(self, data, detectors=None, logger=None, **kwargs):  # type: ignore[no-untyped-def]
        if logger is None:
            logger = Logger.get()

        if len(data.obs) == 0:
            raise RuntimeError('Every supplied data object must contain at least one observation.')

        if len(data.obs) > 1:
            raise RuntimeWarning(
                'Only one observation is supported for now. The first observation will be used.'
            )

        # Initialize the internal data interface
        self._data = ToastObservationData(
            observation=data.obs[0],
            det_selection=detectors,
            det_mask=self.det_mask,
            det_data=self.det_data,
            pixels=self.pixels,
            quats=self.quats,
            hwp_angle=self.hwp_angle,
            noise_model=self.noise_model,
        )

        # Check that we have at least one valid detector
        dets = self._data.dets
        ndet = len(dets)
        if ndet == 0:
            return

        # Stage data
        has_hwp = self.hwp_angle is not None
        self._stage(has_hwp)
        logger.info('Staged data')

        # Build the acquisition operator
        h = self._get_acquisition(has_hwp)
        tod_structure = h.out_structure()
        logger.info('Built acquisition')

        # Build the inverse noise covariance matrix
        invntt = self._get_invntt(tod_structure)
        logger.info('Built invntt')

        # preconditioner
        # TODO replace by block Jacobi when available
        coverage = h.T(jnp.ones(tod_structure.shape, tod_structure.dtype))
        m = BlockDiagonalOperator(
            StokesIQU(
                d := DiagonalOperator(
                    coverage.i, in_structure=jax.eval_shape(lambda _: _, coverage.i)
                ),
                d,
                d,
            )
        ).inverse()

        # solving
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        solver_options = {
            'preconditioner': lx.TaggedLinearOperator(m, lx.positive_semidefinite_tag)
        }
        with Config(solver=solver, solver_options=solver_options):
            A = (h.T @ invntt @ h).I @ h.T @ invntt

        @jax.jit
        def process(tod):  # type: ignore[no-untyped-def]
            return A.reduce()(tod)

        logger.info('Setup complete')
        estimate = process(self._tods)
        estimate.i.block_until_ready()
        logger.info('JIT 1 complete')

        estimate = process(self._tods)
        estimate.i.block_until_ready()
        logger.info('JIT 2 complete')

        logger.info('Mapped the data')

        # save the result
        out_dir = Path(self.output_dir)
        out_dir.mkdir(exist_ok=True)
        fname = out_dir / 'map.fits'
        map_estimate = np.array([estimate.i, estimate.q, estimate.u])
        hp.write_map(str(fname), map_estimate, nest=True, overwrite=True)
        logger.info('Saved outputs')

        return

    def _stage(self, hwp: bool) -> None:
        self._tods = self._data.get_tods()
        self._pixels = self._data.get_pixels()
        self._det_angles = self._data.get_det_angles()
        if not hwp:
            return
        self._hwp_angles = self._data.get_hwp_angles()
        self._gamma = self._data.get_det_offset_angles()

    def _get_invntt(self, structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        if self.binned:
            # we are making a binned map
            return IdentityOperator(structure)

        if self.lagmax > self._data.n_samples:
            raise RuntimeError(
                'Maximum correlation length should be less than the number of samples'
            )

        nperseg = self.nperseg
        sample_rate = self._data.sample_rate
        if self.noise_model is None:
            # estimate the noise covariance from the data
            psd = estimate_psd(self._tods, nperseg=nperseg, rate=sample_rate)
        else:
            # use an existing noise model
            freq, psd = self._data.get_psd_model()
            psd = interpolate_psd(freq, psd, fft_size=nperseg, rate=self.sample_rate)

        invntt = psd_to_invntt(psd, self.lagmax)
        return SymmetricBandToeplitzOperator(invntt, structure)

    def _get_acquisition(self, has_hwp: bool) -> AbstractLinearOperator:
        self._landscape = HealpixLandscape(self.nside, self.stokes)
        reshape = RavelOperator(in_structure=self._landscape.structure)
        sampling = IndexOperator(self._pixels, in_structure=reshape.out_structure())
        meta = {'shape': self._tods.shape, 'stokes': self.stokes}
        if has_hwp:
            polarizer = LinearPolarizerOperator.create(**meta, angles=self._gamma[:, None])
            hwp = HWPOperator.create(**meta, angles=self._hwp_angles[None, :])
            rotation = QURotationOperator.create(
                **meta, angles=self._det_angles - self._gamma[:, None]
            )
            acquisition = polarizer @ hwp @ rotation @ sampling
            reduced_acquisition = acquisition.reduce()
            return reduced_acquisition
        else:
            polarizer = LinearPolarizerOperator.create(**meta, angles=self._det_angles)
            # no need for reduction here
            return polarizer @ sampling

    def _finalize(self, data, **kwargs):  # type: ignore[no-untyped-def]
        self.clear()

    def _requires(self) -> ObservationKeysDict:
        req = {
            'meta': [],
            'shared': [],
            'detdata': [self.det_data, self.pixels, self.quats],
            'intervals': [],
        }
        for trait_name, destination in zip(
            ['noise_model', 'hwp_angle', 'view'], ['meta', 'shared', 'intervals']
        ):
            if (trait := getattr(self, trait_name)) is not None:
                req[destination].append(trait)
        return req

    def _provides(self) -> ObservationKeysDict:
        return dict()

    @classmethod
    def from_dict(cls, dict: dict[str, Any]) -> ToastOperator:
        """Create and return a MapMaker instance from a dictionary.
        TODO: Add checks on attributes
        """

        return cls(**dict)


@trait_docs
class TemplateMapMaker(MapMaker):
    """Operator for template mapmaking with the furax tools."""

    template_config: dict[str, Any]

    ## Template settings
    # max_poly_order = Int(4, help='Maximum order for polynomial templates')

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        if 'template' not in kwargs.keys():
            RuntimeError('No templates provided for the template mapmaker')
        self.template_config = kwargs.pop('template')
        super().__init__(**kwargs)
        self._cached = False
        self._logprefix = 'Furax TemplateMapMaker:'

    def _exec(self, data, detectors=None, logger=None, **kwargs):  # type: ignore[no-untyped-def]
        # Currently a lot of these are duplicates of the mapmaker above
        # TODO refactor to reduce duplicates and improve readability

        if logger is None:
            logger = Logger.get()

        if len(data.obs) == 0:
            raise RuntimeError('Every supplied data object must contain at least one observation.')

        if len(data.obs) > 1:
            raise RuntimeWarning(
                'Only one observation is supported for now. The first observation will be used.'
            )

        # Initialize the internal data interface
        self._data = ToastObservationData(
            observation=data.obs[0],
            det_selection=detectors,
            det_mask=self.det_mask,
            det_data=self.det_data,
            pixels=self.pixels,
            quats=self.quats,
            hwp_angle=self.hwp_angle,
            noise_model=self.noise_model,
        )

        # Check that we have at least one valid detector
        dets = self._data.dets
        ndet = len(dets)
        if ndet == 0:
            return

        # Stage data
        has_hwp = self.hwp_angle is not None
        self._stage(has_hwp)
        logger.info('Staged data')

        # Build the acquisition operator
        acquisition = self._get_acquisition(has_hwp)
        tod_structure = acquisition.out_structure()
        logger.info('Built acquisition')

        # Compute cross psd data for PCA related work, if needed
        # TODO move this to a better location
        if 'common_mode' in self.template_config.keys():
            nperseg = self.nperseg
            rate = self._data.sample_rate
            if self._data._cross_psd is None:
                self._data._cross_psd = compute_cross_psd(self._tods, nperseg=nperseg, rate=rate)

        # Build the template operators
        template_names = list(self.template_config.keys())
        self.template_operators = [
            TemplateOperator.from_dict(name, self.template_config[name], self._data)
            for name in template_names
        ]
        logger.info('Built template operators')

        # Combine the aquisition and template operators
        h = BlockRowOperator([acquisition, self.template_operators])

        # Build the inverse noise covariance matrix
        invntt = self._get_invntt(tod_structure)
        logger.info('Built invntt')

        # preconditioner
        # TODO replace by block Jacobi when available
        coverage, tmpls = h.T(jnp.ones_like(tod_structure))
        m = BlockDiagonalOperator(
            [
                StokesIQU(
                    i=coverage.i,
                    q=coverage.i,
                    u=coverage.i,
                ),
                [jnp.abs(tmpl) for tmpl in tmpls],
            ],
        ).inverse()

        # initial values
        y0 = [
            StokesIQU(
                i=jnp.zeros_like(coverage.i),
                q=jnp.zeros_like(coverage.i),
                u=jnp.zeros_like(coverage.i),
            ),
            [jnp.zeros(template.n_params) for template in self.template_operators],
        ]

        # solving
        solver = lx.CG(rtol=self.rtol, atol=self.atol, max_steps=self.max_steps)
        solver_options = {
            'y0': y0,
            'preconditioner': lx.TaggedLinearOperator(m, lx.positive_semidefinite_tag),
        }
        with Config(solver=solver, solver_options=solver_options):
            A = (h.T @ invntt @ h).I @ h.T @ invntt

        @jax.jit
        def process(tod):  # type: ignore[no-untyped-def]
            return A.reduce()(tod)

        logger.info('Setup complete')
        estimate, t_estimates = process(self._tods)
        estimate.i.block_until_ready()
        logger.info('JIT 1 complete')

        estimate, t_estimates = process(self._tods)
        estimate.i.block_until_ready()
        logger.info('JIT 2 complete')

        logger.info('Mapped the data')

        # save the result
        out_dir = Path(self.output_dir)
        out_dir.mkdir(exist_ok=True)

        fname = out_dir / 'map.fits'
        map_estimate = np.array([estimate.i, estimate.q, estimate.u])
        hp.write_map(str(fname), map_estimate, nest=True, overwrite=True)

        # Optional info
        fname = out_dir / 'tod_fit.npy'
        tod_fit = acquisition(estimate)
        np.save(str(fname), tod_fit)

        for name, template, t_estimate in zip(template_names, self.template_operators, t_estimates):
            fname = out_dir / f'{name}_template_estimate.npy'
            np.save(str(fname), t_estimate)

            fname = out_dir / f'{name}_template_fit.npy'
            np.save(str(fname), template(t_estimate))

        fname = out_dir / 'template_config.yaml'
        with open(str(fname), 'w') as outfile:
            yaml.dump(self.template_config, outfile, default_flow_style=False)

        logger.info('Saved outputs')

        return

    def _get_invntt(self, structure: PyTree[jax.ShapeDtypeStruct]) -> AbstractLinearOperator:
        if self.noise_model == 'pca':
            # PCA mode filtered covariance
            if 'common_mode' not in self.template_config.keys():
                msg = 'PCA mode-filtered noise covariance requires common mode template to be used'
                raise RuntimeError(msg)
            freq_threshold = self.template_config['common_mode'].get('freq_threshold')
            n_modes = self.template_config['common_mode'].get('n_modes')
            nperseg = self.nperseg
            rate = self._data.sample_rate

            if self._data._cross_psd is None:
                self._data._cross_psd = compute_cross_psd(self._tods, nperseg=nperseg, rate=rate)
            freq, csd = self._data._cross_psd

            psd = estimate_filtered_psd(
                self._tods,
                nperseg=nperseg,
                rate=rate,
                freq=freq,
                csd=csd,
                freq_threshold=freq_threshold,
                n_modes=n_modes,
            )
            invntt = psd_to_invntt(psd, self.lagmax)
            return SymmetricBandToeplitzOperator(invntt, structure)

        return super()._get_invntt(structure)

    @classmethod
    def from_dict(cls, dict: dict[str, Any]) -> ToastOperator:
        """Create and return a MapMaker instance from a dictionary.
        TODO: Add checks on attributes
        """

        return cls(**dict)
