import logging
from abc import abstractmethod
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from typing import Any

import equinox
import jax
import jax.numpy as jnp
import lineax
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, PyTree

from furax import AbstractLinearOperator, Config, DiagonalOperator, IdentityOperator
from furax.core import BlockDiagonalOperator, BlockRowOperator, IndexOperator
from furax.interfaces.toast.mapmaker import templates
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.operators import QURotationOperator
from furax.obs.stokes import Stokes, StokesPyTreeType, ValidStokesType

from ._observation_data import GroundObservationData
from .config import MapMakingConfig
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner


@dataclass
class MapMaker:
    """Class for generic mapmakers which consume GroundObservationData."""

    config: MapMakingConfig
    logger: logging.Logger

    def __post_init__(self) -> None:
        return

    @abstractmethod
    def make_maps(self, observation: GroundObservationData) -> dict[str, Any]: ...

    @classmethod
    def from_config(
        cls, config: MapMakingConfig, logger: logging.Logger | None = None
    ) -> 'MapMaker':
        if logger is None:
            logger = logging.getLogger()

        if config.method == 'Binned':
            return BinnedMapMaker(config=config, logger=logger)
        if config.method == 'ML':
            return MLMapmaker(config=config, logger=logger)
        if config.method == 'TwoStep':
            return TwoStepMapmaker(config=config, logger=logger)

        return ValueError('Invalid mapmaking method: {config.method}')

    @classmethod
    def from_yaml(cls, path: str | Path, logger: logging.Logger | None = None) -> 'MapMaker':
        return cls.from_config(config=MapMakingConfig.load_yaml(path), logger=logger)

    def get_landscape(
        self, observation: GroundObservationData, stokes: ValidStokesType = 'IQU'
    ) -> StokesLandscape:
        """Landscape used for mapmaking with given observation"""
        if self.config.landscape.type == 'WCS':
            wcs_shape, wcs_kernel = observation.get_wcs_shape_and_kernel(
                resolution=self.config.landscape.resolution, projection='car'
            )
            return WCSLandscape(wcs_shape, wcs_kernel, stokes=stokes, dtype=self.config.dtype)

        elif self.config.landscape.type == 'Healpix':
            return HealpixLandscape(
                nside=self.config.landscape.nside, stokes=stokes, dtype=self.config.dtype
            )

        else:
            raise TypeError('Landscape type not supported')

    def get_pointing_operators(
        self, observation: GroundObservationData, landscape: StokesLandscape
    ) -> tuple[IndexOperator, QURotationOperator]:
        """Operators containing pointing information with given observation"""

        pixel_inds, para_ang = observation.get_pointing_and_parallactic_angles(landscape)

        if isinstance(landscape, WCSLandscape):
            indexer = IndexOperator(
                (pixel_inds[..., 0], pixel_inds[..., 1]), in_structure=landscape.structure
            )
        elif isinstance(landscape, HealpixLandscape):
            indexer = IndexOperator(pixel_inds[..., 0], in_structure=landscape.structure)

        # Rotation due to coordinate transform
        # Note the minus sign on the rotation angle!
        tod_shape = pixel_inds.shape[:2]
        rotator = QURotationOperator.create(
            tod_shape, dtype=landscape.dtype, stokes='IQU', angles=-para_ang
        )

        return indexer, rotator

    def get_acquisition(
        self,
        observation: GroundObservationData,
        landscape: StokesLandscape,
    ) -> AbstractLinearOperator:
        """Acquisition operator mapping sky maps to time-ordered data"""
        indexer, rotator = self.get_pointing_operators(observation, landscape)

        if self.config.demodulated:
            return (rotator @ indexer).reduce()
        else:
            hwp_angle = observation.get_hwp_angles().astype(self.config.dtype)
            modulator = IQUModulationOperator(
                (observation.n_dets, observation.n_samples), hwp_angle, dtype=self.config.dtype
            )
            return (modulator @ rotator @ indexer).reduce()

    def get_scanning_masker(self, observation: GroundObservationData) -> AbstractLinearOperator:
        """Flag operator which selects only the scanning intervals
        of the given TOD of shape (ndets, nsamps).
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_dets, observation.n_samples), dtype=self.config.dtype
        )
        if self.config.scanning_mask:
            mask = observation.get_scanning_mask()
            out_structure = ShapeDtypeStruct(
                shape=(observation.n_dets, np.sum(mask)), dtype=self.config.dtype
            )
            masker = IndexOperator(
                (slice(None), jnp.array(mask)),
                in_structure=in_structure,
                out_structure=out_structure,
            )
            return masker

        else:
            return IdentityOperator(in_structure)

    def get_scanning_mask_projector(
        self, observation: GroundObservationData
    ) -> AbstractLinearOperator:
        """Flag operator which sets the values outside the scanning intervals
        of the given TOD (of shape (ndets, nsamps)) to zero.
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_dets, observation.n_samples), dtype=self.config.dtype
        )
        if self.config.scanning_mask:
            mask = observation.get_scanning_mask()
            masking_projector = DiagonalOperator(
                jnp.array(mask, dtype=self.config.dtype), in_structure=in_structure
            )
            return masking_projector

        else:
            return IdentityOperator(in_structure)

    def get_or_fit_noise_model(self, observation: GroundObservationData) -> NoiseModel:
        """Return a noise model for the observation, corresponding to
        the type (diagonal, toeplitz, ...) specified by the mapmaker.
        Attempts to load the noise model from the data if available,
        but otherwise fits a model to the data.
        """
        config = self.config

        # Load the noise model from data if available
        noise_model = observation.get_noise_model()
        if config.binned:
            if noise_model:
                if isinstance(noise_model, WhiteNoiseModel):
                    return noise_model
                elif isinstance(noise_model, AtmosphericNoiseModel):
                    return noise_model.to_white_noise_model()
        else:
            if isinstance(noise_model, AtmosphericNoiseModel):
                return noise_model

        # Compute the noise model from data
        f, Pxx = jax.scipy.signal.welch(
            observation.get_tods(), fs=observation.sample_rate, nperseg=config.nperseg
        )

        if config.binned:
            # Diagonal noise
            return WhiteNoiseModel(sigma=jnp.mean(Pxx[..., (f > 0)], axis=-1))
        else:
            # Non-diagonal noise
            n_dets = observation.n_dets
            dtype = config.dtype
            init_model = AtmosphericNoiseModel(
                sigma=jnp.mean(Pxx[..., (f >= f[-1] / 2)], axis=-1),
                alpha=(-3.0) * jnp.ones(n_dets, dtype=dtype),
                fk=(f[-1] / 2) * jnp.ones(n_dets, dtype=dtype),
                f0=(f[1]) * jnp.ones(n_dets, dtype=dtype),
            )
            return NoiseModel.fit_psd_model(
                f=f,
                Pxx=Pxx,
                init_model=init_model,
                max_iter=config.solver.max_steps,
                tol=config.solver.rtol,
            )

    def get_pixel_selector(
        self, blocks: Float[Array, '... nstokes nstokes'], landscape: StokesLandscape
    ) -> IndexOperator:
        """Select indices of map pixels satisfying
        the minimum fractional hits (hits_cut) and condition number (cond_cut) criteria"""
        config = self.config

        eigs = jnp.linalg.eigvalsh(blocks)
        valid = jnp.logical_and(
            eigs[..., -1] > config.hits_cut * eigs[..., -1].max(),
            eigs[..., 0] > config.cond_cut * eigs[..., -1],
        )
        valid_indices = jnp.argwhere(valid)

        if config.landscape.type == 'WCS':
            return IndexOperator(
                (valid_indices[:, 0], valid_indices[:, 1]), in_structure=landscape.structure
            )
        else:
            # Healpix
            return IndexOperator((valid_indices,), in_structure=landscape.structure)

    def get_template_operator(self, observation: GroundObservationData) -> AbstractLinearOperator:
        """Create and return a template operator corresponding to the
        name and configuration provided.
        """
        config = self.config
        assert config.templates is not None
        blocks = {}

        if _ := config.templates.polynomial:
            blocks['polynomial'] = templates.PolynomialTemplateOperator.create(
                max_poly_order=_.max_poly_order,
                intervals=observation.get_scanning_intervals(),
                times=observation.get_elapsed_time(),
                n_dets=observation.n_dets,
            )
        if _ := config.templates.scan_synchronous:
            blocks['scan_synchronous'] = templates.ScanSynchronousTemplateOperator.create(
                min_poly_order=_.min_poly_order,
                max_poly_order=_.max_poly_order,
                azimuth=jnp.array(observation.get_azimuth()),
                n_dets=observation.n_dets,
            )
        if _ := config.templates.hwp_synchronous:
            blocks['hwp_synchronous'] = templates.HWPSynchronousTemplateOperator.create(
                n_harmonics=_.n_harmonics,
                hwp_angles=observation.get_hwp_angles(),
                n_dets=observation.n_dets,
            )

        return BlockRowOperator(blocks=blocks)


class BinnedMapMaker(MapMaker):
    """Class for mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Binned Mapmaker is incompatible with binned=False')

    def make_maps(self, observation: GroundObservationData) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'Binned Mapmaker: {msg}')

        # Data and landscape
        data = observation.get_tods().astype(config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation, stokes='IQU')

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_masker(observation)
        acquisition = masker @ acquisition
        data_struct = masker.out_structure()  # Now with a subset of samples
        logger_info('Created scanning mask operator')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # TODO: JIT system matrix creation
        # System matrix
        system = BJPreconditioner.create((acquisition.T @ inv_noise @ acquisition).reduce())
        logger_info('Created system operator')

        # Mapmaking operator
        binner = acquisition.T @ inv_noise @ masker
        mapmaking_operator = system.inverse() @ binner

        @jax.jit
        def process(d):  # type: ignore[no-untyped-def]
            return mapmaking_operator.reduce()(d)

        logger_info('Set up mapmaking operator')

        # Run mapmaking
        res = process(data)
        res.i.block_until_ready()
        logger_info('Finished mapmaking')

        if config.debug:
            res = process(data)
            res.i.block_until_ready()
            logger_info('Test - second time - Finished mapmaking')

        final_map = np.array([res.i, res.q, res.u])
        weights = np.array(system.get_blocks())

        output = {'map': final_map, 'weights': weights}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs

        return output


class MLMapmaker(MapMaker):
    """Class for mapmaking with maximum likelihood (ML) estimator"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if self.config.binned:
            raise ValueError('ML Mapmaker is incompatible with binned=True')
        if self.config.demodulated:
            raise ValueError('ML Mapmaker is incompatible with demodulated=True')

    def make_maps(self, observation: GroundObservationData) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'ML Mapmaker: {msg}')

        # Data and landscape
        data = observation.get_tods().astype(config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation, stokes='IQU')

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_mask_projector(observation)
        logger_info('Created scanning mask operator')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(
            data_struct,
            nperseg=config.nperseg,
            sample_rate=observation.sample_rate,
            correlation_length=config.correlation_length,
        )
        logger_info('Created inverse noise covariance operator')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_inv_noise = DiagonalOperator(
            diagonal=inv_noise.band_values[..., [0]], in_structure=data_struct
        )
        diag_system = BJPreconditioner.create(
            acquisition.T @ masker @ diag_inv_noise @ masker @ acquisition
        )
        logger_info('Created approximate system matrix')

        # Map pixel selection
        blocks = diag_system.get_blocks()
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure().shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Preconditioner
        preconditioner = selector @ diag_system.inverse() @ selector.T

        # Templates (optional)
        if config.use_templates:
            template_op = self.get_template_operator(observation)
            logger_info('Built template operators')

        # Mapmaking operator
        if config.use_templates:
            p = BlockDiagonalOperator(
                [preconditioner, IdentityOperator(template_op.in_structure())]
            )
            h = BlockRowOperator([acquisition @ selector.T, template_op])
        else:
            p = preconditioner
            h = acquisition @ selector.T
        mp = masker

        solver = lineax.CG(**asdict(config.solver))
        solver_options = {
            'preconditioner': lineax.TaggedLinearOperator(p, lineax.positive_semidefinite_tag)
        }
        with Config(solver=solver, solver_options=solver_options):
            mapmaking_operator = (h.T @ mp @ inv_noise @ mp @ h).I @ h.T @ mp @ inv_noise @ mp

        @jax.jit
        def process(d):  # type: ignore[no-untyped-def]
            return mapmaking_operator.reduce()(d)

        logger_info('Completed setting up the solver')

        # Run mapmaking
        if config.use_templates:
            rec_map, tmpl_ampl = process(data)
        else:
            rec_map = process(data)
        result_map = selector.T(rec_map)
        result_map.i.block_until_ready()
        logger_info('Finished mapmaking computation')

        # Format output and compute auxilary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs
        if config.use_templates:
            output['template'] = tmpl_ampl
        if config.debug:
            proj_map = (mp @ acquisition)(result_map)
            if config.use_templates:
                projs = {
                    'proj_map': proj_map
                    ** {
                        f'proj_{tmpl}': (mp @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                        for tmpl in tmpl_ampl.keys()
                    },
                }
            else:
                projs = {'proj_map': proj_map}
            output['projs'] = projs

        return output


class TwoStepMapmaker(MapMaker):
    """Class for binned mapmaking with templates, using the two-step estimation method"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Two-Step Mapmaker is incompatible with binned=False')
        if self.config.demodulated:
            raise ValueError('Two-Step Mapmaker is incompatible with demodulated=True')
        if not self.config.use_templates:
            raise ValueError('Two-Step Mapmaker is incompatible with no templates')

    def make_maps(self, observation: GroundObservationData) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'Two-Step Mapmaker: {msg}')

        # Data and landscape
        data = observation.get_tods().astype(config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation, stokes='IQU')

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_mask_projector(observation)
        logger_info('Created scanning mask operator')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # System matrix
        system = BJPreconditioner.create(acquisition.T @ masker @ inv_noise @ masker @ acquisition)
        logger_info('Created system matrix')

        # Map pixel selection
        blocks = system.get_blocks()
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure().shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Templates
        template_op = self.get_template_operator(observation)
        logger_info('Built template operators')

        # Define operators
        system_inv = selector @ system.inverse() @ selector.T
        A = acquisition @ selector.T
        M = inv_noise
        mp = masker
        FA = M - M @ mp @ A @ system_inv @ A.T @ mp @ M

        solver = lineax.CG(**asdict(config.solver))
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

        logger_info('Completed setting up the solver')

        # Run mapmaking
        rec_map, tmpl_ampl = process(data)
        result_map = selector.T(rec_map)
        result_map.i.block_until_ready()
        logger_info('Finished mapmaking computation')

        # Format output and compute auxilary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks, 'template': tmpl_ampl}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs
        if config.debug:
            proj_map = (mp @ acquisition)(result_map)
            projs = {
                'proj_map': proj_map,
                **{
                    f'proj_{tmpl}': (mp @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                    for tmpl in tmpl_ampl.keys()
                },
            }
            output['projs'] = projs

        return output


class IQUModulationOperator(AbstractLinearOperator):
    """Class that adds the input Stokes signals to a single HWP-modulated signal
    Similar to LinearPolarizerOperator @ QURotationOperator(hwp_angle), except that
    only half of the QU rotation needs to be computed
    """

    _in_structure: PyTree[ShapeDtypeStruct] = equinox.field(static=True)
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

    def in_structure(self) -> PyTree[ShapeDtypeStruct]:
        return self._in_structure
