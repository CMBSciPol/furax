import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
from logging import Logger
from math import prod
from pathlib import Path
from typing import Any

import equinox
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import pixell.enmap
from astropy.io import fits
from astropy.wcs import WCS
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, PyTree

from furax import AbstractLinearOperator, Config, DiagonalOperator, IdentityOperator
from furax.core import BlockDiagonalOperator, BlockRowOperator, IndexOperator
from furax.interfaces.toast.mapmaker import templates
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.operators import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.stokes import Stokes, StokesPyTreeType, ValidStokesType

from ._logger import logger as furax_logger
from ._observation_data import GroundObservationData
from .config import MapMakingConfig
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .preconditioner import BJPreconditioner


@dataclass
class MapMaker:
    """Class for generic mapmakers which consume GroundObservationData."""

    config: MapMakingConfig
    logger: Logger = furax_logger

    def __post_init__(self) -> None:
        return

    @abstractmethod
    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]: ...

    def make_maps(self, observation: GroundObservationData, out_dir: str | None) -> dict[str, Any]:
        results = self.mapmake(observation)

        # Save output
        try:
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                for key, m in results.items():
                    if isinstance(m, jax.Array) or isinstance(m, np.ndarray):
                        np.save(f'{out_dir}/{key}.npy', np.array(m))
                    elif isinstance(m, pixell.enmap.ndmap):
                        pixell.enmap.write_map(f'{out_dir}/{key}.hdf', m, allow_modify=True)
                    elif isinstance(m, WCS):
                        header = m.to_header()
                        hdu = fits.PrimaryHDU(header=header)
                        hdu.writeto(f'{out_dir}/{key}.fits', overwrite=True)
                    else:
                        continue
                    self.logger.info(f'Mapmaking result [{key}] saved to file')
                self.config.dump_yaml(f'{out_dir}/mapmaking_config.yaml')
                self.logger.info('Mapmaking config saved to file')
        except Exception as err:
            self.logger.info(f'Error while saving output: {err}')

        return results

    @classmethod
    def from_config(cls, config: MapMakingConfig, logger: Logger | None = None) -> 'MapMaker':
        """Return the appropriate mapmaker based on the config's mapmaking method."""
        maker = {
            'Binned': BinnedMapMaker,
            'ML': MLMapmaker,
            'TwoStep': TwoStepMapmaker,
            'ATOP': ATOPMapMaker,
        }[config.method]

        if logger is None:
            return maker(config)
        else:
            return maker(config, logger=logger)

    @classmethod
    def from_yaml(cls, path: str | Path, logger: Logger | None = None) -> 'MapMaker':
        return cls.from_config(MapMakingConfig.load_yaml(path), logger=logger)

    def get_landscape(
        self, observation: GroundObservationData, stokes: ValidStokesType = 'IQU'
    ) -> StokesLandscape:
        """Landscape used for mapmaking with given observation"""
        if self.config.landscape.type == 'WCS':
            wcs_shape, wcs_kernel = observation.get_wcs_shape_and_kernel(
                resolution=self.config.landscape.resolution, projection='car'
            )
            return WCSLandscape(wcs_shape, wcs_kernel, stokes=stokes, dtype=self.config.dtype)

        if self.config.landscape.type == 'Healpix':
            return HealpixLandscape(
                nside=self.config.landscape.nside, stokes=stokes, dtype=self.config.dtype
            )

        raise TypeError('Landscape type not supported')

    def get_pointing_operators(
        self, observation: GroundObservationData, landscape: StokesLandscape
    ) -> tuple[IndexOperator, QURotationOperator]:
        """Operators containing pointing information with given observation"""

        pixel_inds, spin_ang = observation.get_pointing_and_spin_angles(landscape)

        if isinstance(landscape, WCSLandscape):
            assert pixel_inds.shape[-1] == 2, 'Wrong WCS landscape format'
            indexer = IndexOperator(
                (pixel_inds[..., 0], pixel_inds[..., 1]), in_structure=landscape.structure
            )
        elif isinstance(landscape, HealpixLandscape):
            if pixel_inds.shape[-1] == 1:
                pixel_inds = pixel_inds[..., 0]
            indexer = IndexOperator(pixel_inds, in_structure=landscape.structure)

        # Rotation due to coordinate transform
        tod_shape = pixel_inds.shape[:2]
        rotator = QURotationOperator.create(
            tod_shape, dtype=landscape.dtype, stokes=landscape.stokes, angles=spin_ang
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
            meta = {
                'shape': (observation.n_dets, observation.n_samples),
                'stokes': landscape.stokes,
                'dtype': self.config.dtype,
            }
            polarizer = LinearPolarizerOperator.create(**meta)  # type: ignore[arg-type]
            hwp = HWPOperator.create(
                **meta,  # type: ignore[arg-type]
                angles=observation.get_hwp_angles().astype(self.config.dtype),
            )

            return (polarizer @ hwp @ rotator @ indexer).reduce()

    def get_scanning_masker(self, observation: GroundObservationData) -> AbstractLinearOperator:
        """Flag operator which selects only the scanning intervals
        of the given TOD of shape (ndets, nsamps).
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_dets, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(in_structure)

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

    def get_scanning_mask_projector(
        self, observation: GroundObservationData
    ) -> AbstractLinearOperator:
        """Flag operator which sets the values outside the scanning intervals
        of the given TOD (of shape (ndets, nsamps)) to zero.
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_dets, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(in_structure)

        mask = observation.get_scanning_mask()
        masking_projector = DiagonalOperator(
            jnp.array(mask, dtype=self.config.dtype), in_structure=in_structure
        )
        return masking_projector

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
                if isinstance(noise_model, AtmosphericNoiseModel):
                    return noise_model.to_white_noise_model()

        if isinstance(noise_model, AtmosphericNoiseModel):
            return noise_model

        # Compute the noise model from data
        f, Pxx = jax.scipy.signal.welch(
            observation.get_tods(), fs=observation.sample_rate, nperseg=config.nperseg
        )

        if config.binned:
            # Diagonal noise
            return WhiteNoiseModel(sigma=jnp.mean(Pxx[..., (f > 0)], axis=-1))

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

        # eigs = jnp.linalg.eigvalsh(blocks)
        eigs = np.linalg.eigvalsh(blocks)
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

        if poly := config.templates.polynomial:
            blocks['polynomial'] = templates.PolynomialTemplateOperator.create(
                max_poly_order=poly.max_poly_order,
                intervals=observation.get_scanning_intervals(),
                times=observation.get_elapsed_time(),
                n_dets=observation.n_dets,
            )
        if sss := config.templates.scan_synchronous:
            blocks['scan_synchronous'] = templates.ScanSynchronousTemplateOperator.create(
                min_poly_order=sss.min_poly_order,
                max_poly_order=sss.max_poly_order,
                azimuth=jnp.array(observation.get_azimuth()),
                n_dets=observation.n_dets,
            )
        if hwpss := config.templates.hwp_synchronous:
            blocks['hwp_synchronous'] = templates.HWPSynchronousTemplateOperator.create(
                n_harmonics=hwpss.n_harmonics,
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

    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]:
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

    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]:
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

    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]:
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


class ATOPProjectionOperator(AbstractLinearOperator):
    tau: int = equinox.field(static=True)
    n_det: int = equinox.field(static=True)
    n_samp: int = equinox.field(static=True)
    _in_structure: jax.ShapeDtypeStruct = equinox.field(static=True)

    def __init__(self, tau: int, *, in_structure: PyTree[jax.ShapeDtypeStruct]):
        self.tau = tau
        self._in_structure = in_structure
        self.n_det, self.n_samp = in_structure.shape

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: Float[Array, 'det samp']) -> Float[Array, 'det samp']:
        if self.n_samp % self.tau == 0:
            y = x.reshape(self.n_det, self.n_samp // self.tau, self.tau)
            y = y - jnp.mean(y, axis=-1)[:, :, None]
            return y.reshape(self.n_det, self.n_samp)
        else:
            n_int = self.n_samp // self.tau
            y = x[:, : n_int * self.tau].reshape(self.n_det, n_int, self.tau)
            y = y - jnp.mean(y, axis=-1)[:, :, None]
            return jnp.concatenate(
                [y.reshape(self.n_det, n_int * self.tau), x[:, -(self.n_samp % self.tau) :]], axis=1
            )


class ATOPMapMaker(MapMaker):
    """Class for ATOP mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('ATOP Mapmaker is currently incompatible with binned=False')
        if self.config.atop_tau < 2:
            raise ValueError('ATOP tau should be at least 2')

    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'ATOP Mapmaker: {msg}')

        # Data and landscape
        data = observation.get_tods().astype(config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation, stokes='QU')

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # Optional mask for scanning
        masker = self.get_scanning_mask_projector(observation)
        logger_info('Created scanning mask operator')

        # Noise
        logger_info('Noise assumed to be identity')

        # ATOP projector
        atop_projector = ATOPProjectionOperator(self.config.atop_tau, in_structure=data_struct)

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_system = BJPreconditioner.create((acquisition.T @ masker @ acquisition).reduce())
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

        # Mapmaking operator
        p = preconditioner
        h = acquisition @ selector.T
        mp = masker
        ap = atop_projector

        solver = lineax.CG(**asdict(config.solver))
        solver_options = {
            'preconditioner': lineax.TaggedLinearOperator(p, lineax.positive_semidefinite_tag)
        }
        with Config(solver=solver, solver_options=solver_options):
            mapmaking_operator = (h.T @ mp @ ap @ mp @ h).I @ h.T @ mp @ ap @ mp

        @jax.jit
        def process(d):  # type: ignore[no-untyped-def]
            return mapmaking_operator.reduce()(d)

        logger_info('Completed setting up the solver')

        # Run mapmaking
        rec_map = process(data)
        result_map = selector.T(rec_map)
        result_map.q.block_until_ready()
        logger_info('Finished mapmaking computation')

        # Format output and compute auxilary data
        final_map = np.array([result_map.q, result_map.u])

        output = {'map': final_map, 'weights': blocks}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs
        if config.debug:
            proj_map = (mp @ acquisition)(result_map)
            output['proj_map'] = proj_map

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


class QUModulationOperator(AbstractLinearOperator):
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
        self._in_structure = Stokes.class_for('QU').structure_for(shape, dtype)
        self.cos_hwp_angle = jnp.cos(4 * hwp_angle.astype(dtype))
        self.sin_hwp_angle = jnp.sin(4 * hwp_angle.astype(dtype))

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        return self.cos_hwp_angle[None, :] * x.q + self.sin_hwp_angle[None, :] * x.u  # type: ignore[union-attr]

    def in_structure(self) -> PyTree[ShapeDtypeStruct]:
        return self._in_structure
