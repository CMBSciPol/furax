import os
import pickle
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

import furax.tree
from furax import (
    AbstractLinearOperator,
    Config,
    DiagonalOperator,
    IdentityOperator,
    MaskOperator,
)
from furax.core import BlockDiagonalOperator, BlockRowOperator, IndexOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.operators import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.stokes import Stokes, StokesIQU, StokesPyTreeType, ValidStokesType

from . import templates
from ._logger import logger as furax_logger
from ._observation import AbstractGroundObservation
from .config import Landscapes, MapMakingConfig, Methods
from .noise import AtmosphericNoiseModel, NoiseModel, WhiteNoiseModel
from .pointing import PointingOperator
from .preconditioner import BJPreconditioner


@dataclass
class MapMaker:
    """Class for generic mapmakers which consume GroundObservationData."""

    config: MapMakingConfig
    logger: Logger = furax_logger

    def __post_init__(self) -> None:
        return

    @abstractmethod
    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]: ...

    def run(
        self, observation: AbstractGroundObservation[Any], out_dir: str | None
    ) -> dict[str, Any]:
        results = self.make_map(observation)

        # Save output
        if out_dir is not None:
            self._save(results, out_dir)
            self.config.dump_yaml(f'{out_dir}/mapmaking_config.yaml')
            self.logger.info('Mapmaking config saved to file')

        return results

    def _save(self, results: dict[str, Any], out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        for key, m in results.items():
            if isinstance(m, jax.Array) or isinstance(m, np.ndarray):
                np.save(f'{out_dir}/{key}.npy', np.array(m))
            elif isinstance(m, StokesIQU):
                np.save(f'{out_dir}/{key}.npy', np.stack([m.i, m.q, m.u], axis=0))
            elif isinstance(m, pixell.enmap.ndmap):
                pixell.enmap.write_map(f'{out_dir}/{key}.hdf', m, allow_modify=True)
            elif isinstance(m, WCS):
                header = m.to_header()
                hdu = fits.PrimaryHDU(header=header)
                hdu.writeto(f'{out_dir}/{key}.fits', overwrite=True)
            elif isinstance(m, StokesLandscape):
                with open(f'{out_dir}/{key}.pkl', 'wb') as f:
                    pickle.dump(m, f)
            elif isinstance(m, dict):
                self._save(m, out_dir)
                continue
            else:
                continue
            self.logger.info(f'Mapmaking result [{key}] saved to file')

    @classmethod
    def from_config(cls, config: MapMakingConfig, logger: Logger | None = None) -> 'MapMaker':
        """Return the appropriate mapmaker based on the config's mapmaking method."""
        maker = {
            Methods.BINNED: BinnedMapMaker,
            Methods.MAXL: MLMapmaker,
            Methods.TWOSTEP: TwoStepMapmaker,
            Methods.ATOP: ATOPMapMaker,
        }[config.method]

        if logger is None:
            return maker(config)  # type: ignore[abstract]
        else:
            return maker(config, logger=logger)  # type: ignore[abstract]

    @classmethod
    def from_yaml(cls, path: str | Path, logger: Logger | None = None) -> 'MapMaker':
        return cls.from_config(MapMakingConfig.load_yaml(path), logger=logger)

    def get_landscape(
        self, observation: AbstractGroundObservation[Any], stokes: ValidStokesType = 'IQU'
    ) -> StokesLandscape:
        """Landscape used for mapmaking with given observation"""
        if self.config.landscape.type == Landscapes.WCS:
            wcs_shape, wcs_kernel = observation.get_wcs_shape_and_kernel(
                resolution=self.config.landscape.resolution, projection='car'
            )
            return WCSLandscape(wcs_shape, wcs_kernel, stokes=stokes, dtype=self.config.dtype)

        if self.config.landscape.type == Landscapes.HPIX:
            return HealpixLandscape(
                nside=self.config.landscape.nside, stokes=stokes, dtype=self.config.dtype
            )

        raise TypeError('Landscape type not supported')

    def get_pointing(
        self, observation: AbstractGroundObservation[Any], landscape: StokesLandscape
    ) -> AbstractLinearOperator:
        """Operator containing pointing information for given observation"""

        det_off_ang = observation.get_detector_offset_angles().astype(landscape.dtype)

        if self.config.pointing_on_the_fly:
            pointing = PointingOperator(
                landscape=landscape,
                qbore=observation.get_boresight_quaternions(),
                qdet=observation.get_detector_quaternions(),
                det_gamma=det_off_ang,
                _in_structure=landscape.structure,
                _out_structure=Stokes.class_for(landscape.stokes).structure_for(
                    (observation.n_detectors, observation.n_samples), dtype=landscape.dtype
                ),
                chunk_size=self.config.pointing_chunk_size,
            )
            return pointing

        else:
            pixel_inds, spin_ang = observation.get_pointing_and_spin_angles(landscape)
            point_ang = spin_ang + det_off_ang[:, None]

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
                tod_shape, dtype=landscape.dtype, stokes=landscape.stokes, angles=point_ang
            )

            return (rotator @ indexer).reduce()

    def get_acquisition(
        self,
        observation: AbstractGroundObservation[Any],
        landscape: StokesLandscape,
    ) -> AbstractLinearOperator:
        """Acquisition operator mapping sky maps to time-ordered data"""
        pointing = self.get_pointing(observation, landscape)

        if self.config.demodulated:
            return pointing
        else:
            meta = {
                'shape': (observation.n_detectors, observation.n_samples),
                'stokes': landscape.stokes,
                'dtype': self.config.dtype,
            }
            polarizer = LinearPolarizerOperator.create(
                **meta,  # type: ignore[arg-type]
                angles=observation.get_detector_offset_angles().astype(self.config.dtype)[:, None],
            )
            hwp = HWPOperator.create(
                **meta,  # type: ignore[arg-type]
                angles=observation.get_hwp_angles().astype(self.config.dtype),
            )

            return (polarizer @ hwp @ pointing).reduce()

    def get_scanning_masker(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Flag operator which selects only the scanning intervals
        of the given TOD of shape (ndets, nsamps).
        """
        in_structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(in_structure)

        mask = observation.get_scanning_mask()
        out_structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, np.sum(mask)), dtype=self.config.dtype
        )
        masker = IndexOperator(
            (slice(None), jnp.array(mask)),
            in_structure=in_structure,
            out_structure=out_structure,
        )
        return masker

    def get_scanning_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Flag operator which sets the values outside the scanning intervals
        of the given TOD (of shape (ndets, nsamps)) to zero.
        """
        structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.scanning_mask:
            return IdentityOperator(structure)

        mask = observation.get_scanning_mask()
        # broadcast along detector axis
        return MaskOperator.from_boolean_mask(mask[None, :], in_structure=structure)

    def get_sample_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Flag operator which sets the values of the given TOD (of shape (ndets, nsamps)) to
        zero at masked (flagged) samples.
        """
        structure = ShapeDtypeStruct(
            shape=(observation.n_detectors, observation.n_samples), dtype=self.config.dtype
        )
        if not self.config.sample_mask:
            return IdentityOperator(structure)

        # Note the mask value is 1 at valid (unmasked) samples
        mask = observation.get_sample_mask()
        return MaskOperator.from_boolean_mask(mask, in_structure=structure)

    def get_mask_projector(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
        """Mask operator which incorporates both the scanning and sample mask projectors."""
        return (
            self.get_scanning_mask_projector(observation)
            @ self.get_sample_mask_projector(observation)
        ).reduce()

    def get_or_fit_noise_model(self, observation: AbstractGroundObservation[Any]) -> NoiseModel:
        """Return a noise model for the observation, corresponding to
        the type (diagonal, toeplitz, ...) specified by the mapmaker.
        Attempts to load the noise model from the data if available,
        but otherwise fits a model to the data.
        """
        config = self.config
        Model = WhiteNoiseModel if config.binned else AtmosphericNoiseModel

        if not config.fit_noise_model:
            # Load the noise model from data if available
            noise_model = observation.get_noise_model()
            if noise_model:
                self.logger.info('Loading noise model from data')
                if isinstance(noise_model, Model):
                    return noise_model
                if config.binned and isinstance(noise_model, AtmosphericNoiseModel):
                    return noise_model.to_white_noise_model()

        # Otherwise, fit the noise model from data
        self.logger.info('Fitting noise model from data')
        f, Pxx = jax.scipy.signal.welch(
            observation.get_tods(), fs=observation.sample_rate, nperseg=config.nperseg
        )
        return Model.fit_psd_model(f, Pxx)

    def get_pixel_selector(
        self, blocks: Float[Array, '... nstokes nstokes'], landscape: StokesLandscape
    ) -> IndexOperator:
        """Select indices of map pixels satisfying
        the minimum fractional hits (hits_cut) and condition number (cond_cut) criteria"""
        config = self.config

        # eigs = jnp.linalg.eigvalsh(blocks)
        eigs = np.linalg.eigvalsh(blocks)
        hits_quantile = np.quantile(eigs[(eigs[..., -1] > 0),], q=0.95)
        valid = jnp.logical_and(
            eigs[..., -1] > config.hits_cut * hits_quantile,
            eigs[..., 0] > config.cond_cut * eigs[..., -1],
        )
        valid_indices = jnp.argwhere(valid)

        if config.landscape.type == Landscapes.WCS:
            return IndexOperator(
                (valid_indices[:, 0], valid_indices[:, 1]), in_structure=landscape.structure
            )
        else:
            # Healpix
            return IndexOperator((valid_indices,), in_structure=landscape.structure)

    def get_template_operator(
        self, observation: AbstractGroundObservation[Any]
    ) -> AbstractLinearOperator:
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
                times=observation.get_elapsed_times(),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if sss := config.templates.scan_synchronous:
            blocks['scan_synchronous'] = templates.ScanSynchronousTemplateOperator.create(
                min_poly_order=sss.min_poly_order,
                max_poly_order=sss.max_poly_order,
                azimuth=jnp.array(observation.get_azimuth()),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if hwpss := config.templates.hwp_synchronous:
            blocks['hwp_synchronous'] = templates.HWPSynchronousTemplateOperator.create(
                n_harmonics=hwpss.n_harmonics,
                hwp_angles=observation.get_hwp_angles(),
                n_dets=observation.n_detectors,
                dtype=config.dtype,
            )
        if azhwpss := config.templates.azhwp_synchronous:
            if azhwpss.split_scans:
                blocks['azhwp_synchronous_left'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=observation.get_left_scan_mask(),
                    )
                )
                blocks['azhwp_synchronous_right'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                        scan_mask=observation.get_right_scan_mask(),
                    )
                )
            else:
                blocks['azhwp_synchronous'] = (
                    templates.AzimuthHWPSynchronousTemplateOperator.create(
                        n_polynomials=azhwpss.n_polynomials,
                        n_harmonics=azhwpss.n_harmonics,
                        azimuth=observation.get_azimuth(),
                        hwp_angles=observation.get_hwp_angles(),
                        n_dets=observation.n_detectors,
                        dtype=config.dtype,
                    )
                )
        if binazhwpss := config.templates.binazhwp_synchronous:
            blocks['binazhwp_synchronous'] = (
                templates.BinAzimuthHWPSynchronousTemplateOperator.create(
                    n_azimuth_bins=binazhwpss.n_azimuth_bins,
                    n_harmonics=binazhwpss.n_harmonics,
                    interpolate_azimuth=binazhwpss.interpolate_azimuth,
                    smooth_interpolation=binazhwpss.smooth_interpolation,
                    azimuth=observation.get_azimuth(),
                    hwp_angles=observation.get_hwp_angles(),
                    n_dets=observation.n_detectors,
                    dtype=config.dtype,
                )
            )
        if ground := config.templates.ground:
            self._ground_landscape = templates.GroundTemplateOperator.get_landscape(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=observation.get_azimuth(),
                boresight_elevation=observation.get_elevation(),
                detector_quaternions=observation.get_detector_quaternions(),
                stokes='IQU',
                dtype=config.dtype,
            )
            ground_op = templates.GroundTemplateOperator.create(
                azimuth_resolution=ground.azimuth_resolution,
                elevation_resolution=ground.elevation_resolution,
                boresight_azimuth=observation.get_azimuth(),
                boresight_elevation=observation.get_elevation(),
                boresight_rotation=jnp.zeros_like(observation.get_azimuth()),
                detector_quaternions=observation.get_detector_quaternions(),
                hwp_angles=observation.get_hwp_angles(),
                stokes='IQU',
                dtype=config.dtype,
                landscape=self._ground_landscape,
                chunk_size=config.pointing_chunk_size,
            )
            ones_tod = jnp.ones((observation.n_detectors, observation.n_samples), dtype=jnp.float64)
            self._ground_coverage = ground_op.T(ones_tod)
            nonzero_hits = jnp.argwhere(self._ground_coverage.i > 0)
            indexer = IndexOperator(
                (nonzero_hits[:, 0], nonzero_hits[:, 1]),
                in_structure=furax.tree.as_structure(self._ground_coverage),
            )
            flattener = templates.StokesIQUFlattenOperator(in_structure=indexer.out_structure())
            self._ground_selector = flattener @ indexer

            blocks['ground'] = ground_op @ self._ground_selector.T

        return BlockRowOperator(blocks=blocks)


class BinnedMapMaker(MapMaker):
    """Class for mapmaking with diagonal noise covariance."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validation on config
        if not self.config.binned:
            raise ValueError('Binned Mapmaker is incompatible with binned=False')

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
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
        if config.fit_noise_model:
            output['noise_fit'] = noise_model.to_array()  # type: ignore[assignment]
        if config.debug:
            proj_map = (masker.T @ acquisition)(res)
            output['proj_map'] = proj_map

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

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
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
        masker = self.get_mask_projector(observation)
        valid_sample_fraction = (
            1.0
            if isinstance(masker, IdentityOperator)
            else float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        )
        logger_info('Created mask operator')
        logger_info(f'Valid sample fraction: {valid_sample_fraction:.4f}')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(
            data_struct,
            nperseg=config.nperseg,
            sample_rate=observation.sample_rate,
            correlation_length=config.correlation_length,
        )
        noise = noise_model.operator(
            data_struct,
            nperseg=config.nperseg,
            sample_rate=observation.sample_rate,
            correlation_length=config.correlation_length,
        )
        logger_info('Created noise and inverse noise covariance operators')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_inv_noise = DiagonalOperator(
            diagonal=inv_noise.band_values[..., [0]], in_structure=data_struct
        )
        diag_system = BJPreconditioner.create(acquisition.T @ diag_inv_noise @ masker @ acquisition)
        logger_info('Created approximate system matrix')

        # Map pixel selection
        blocks = diag_system.get_blocks()
        selector = self.get_pixel_selector(blocks, landscape)
        logger_info(
            f'Selected {prod(selector.out_structure().shape)}\
                            /{prod(landscape.shape)} pixels'
        )

        # Adjust the sample mask according to the new pixel selection
        positive_sample_hits = (
            (masker @ acquisition @ selector.T)(
                StokesIQU.from_iquv(
                    i=jnp.ones(selector.out_structure().shape, dtype=data.dtype),
                    q=jnp.zeros(selector.out_structure().shape, dtype=data.dtype),
                    u=jnp.zeros(selector.out_structure().shape, dtype=data.dtype),
                    v=None,  # type: ignore[arg-type]
                )
            )
            > 0
        ).astype(data.dtype)
        masker = DiagonalOperator(diagonal=positive_sample_hits, in_structure=data_struct)
        logger_info(f'Updated valid sample fraction: {jnp.mean(masker._diagonal):.4f}')

        # Preconditioner
        # We use the approximate diagonal system matrix before the mask update
        preconditioner = selector @ diag_system.inverse() @ selector.T

        # Templates (optional)
        if config.use_templates:
            template_op = self.get_template_operator(observation)
            logger_info('Built template operators')
            REGVAL = config.templates.regularization  # type: ignore[union-attr]
            tmpl_inv_sys = {}
            regs = {}
            for tmpl, tmpl_op in template_op.blocks.items():
                tmpl_sys = (tmpl_op.T @ diag_inv_noise @ masker @ tmpl_op).reduce()
                # Approximation to the diagonal of the matrix
                norm_sys = jnp.abs(
                    jax.jit(lambda x: tmpl_sys(x))(furax.tree.ones_like(tmpl_op.in_structure()))
                )
                # Regualrisation value is REGVAL times the smallest non-zero eigenvalue
                regs[tmpl] = REGVAL * jnp.min(norm_sys[norm_sys > 0])
                tmpl_inv_sys[tmpl] = DiagonalOperator(
                    diagonal=(norm_sys + regs[tmpl]),
                    in_structure=tmpl_op.in_structure(),
                ).inverse()
            template_preconditioner = BlockDiagonalOperator(tmpl_inv_sys)
            logger_info('Built template preconditioner')
            template_reg_op = BlockDiagonalOperator(
                [
                    DiagonalOperator(
                        diagonal=jnp.array([0.0]), in_structure=selector.out_structure()
                    ),
                    {
                        tmpl: regs[tmpl]
                        * IdentityOperator(in_structure=template_op.blocks[tmpl].in_structure())
                        for tmpl in template_op.blocks.keys()
                    },
                ]
            )
            logger_info('Built template regularizer')
            print(f'Template operator input structure: {template_op.in_structure()}')

        # Mapmaking operator
        if config.use_templates:
            p = BlockDiagonalOperator([preconditioner, template_preconditioner])
            h = BlockRowOperator([acquisition @ selector.T, template_op])
            reg = template_reg_op
        else:
            p = preconditioner
            h = acquisition @ selector.T

        if not config.nested_pcg:
            M = masker @ inv_noise @ masker
        else:
            nested_solver = lineax.CG(
                rtol=config.solver.rtol,
                atol=config.solver.atol,
                max_steps=30,
            )
            nested_solver_options = {
                'preconditioner': lineax.TaggedLinearOperator(
                    masker @ inv_noise @ masker, lineax.positive_semidefinite_tag
                )
            }
            M = (
                masker
                @ (masker @ noise @ masker).I(
                    solver=nested_solver,
                    solver_options=nested_solver_options,
                )
                @ masker
            )
            logger_info('Set up nested PCG for the noise inverse')

        solver = lineax.CG(**asdict(config.solver))
        solver_options = {
            'preconditioner': lineax.TaggedLinearOperator(p, lineax.positive_semidefinite_tag)
        }
        options = {'solver': solver, 'solver_options': solver_options}
        if config.use_templates:
            mapmaking_operator = (h.T @ M @ h + reg).I(**options) @ h.T @ M
        else:
            mapmaking_operator = (h.T @ M @ h).I(**options) @ h.T @ M

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

        # Get weights after pixel selection
        weights = jnp.zeros_like(blocks)
        weights = weights.at[selector.indices + (slice(None), slice(None))].add(
            blocks[selector.indices + (slice(None), slice(None))]
        )

        # Format output and compute auxilary data
        final_map = np.array([result_map.i, result_map.q, result_map.u])

        output = {'map': final_map, 'weights': weights, 'weights_uncut': blocks}
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs
        if config.fit_noise_model:
            output['noise_fit'] = noise_model.to_array()
        if config.use_templates:
            for key in tmpl_ampl.keys():
                output[f'template_{key}'] = tmpl_ampl[key]
                output[f'template_reg_{key}'] = np.array(regs[key])
                aux_data = template_op.blocks[key].compute_auxiliary_data(tmpl_ampl[key])
                for aux_key in aux_data.keys():
                    output[f'template_{key}_{aux_key}'] = aux_data[aux_key]
            if 'ground' in tmpl_ampl.keys():
                output['ground_landscape'] = self._ground_landscape
                output['ground_coverage'] = self._ground_coverage
                output['ground_map'] = self._ground_selector.T(tmpl_ampl['ground'])
        if config.debug:
            proj_map = (masker @ acquisition)(result_map)
            if config.use_templates:
                projs = {
                    'proj_map': proj_map,
                    **{
                        f'proj_{tmpl}': (masker @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                        for tmpl in tmpl_ampl
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

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
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

        output = {'map': final_map, 'weights': blocks}
        for key in tmpl_ampl.keys():
            output[f'template_{key}'] = tmpl_ampl[key]
        if isinstance(landscape, WCSLandscape):
            output['wcs'] = landscape.wcs
        if config.fit_noise_model:
            output['noise_fit'] = noise_model.to_array()
        if config.debug:
            proj_map = (mp @ acquisition)(result_map)
            projs = {
                'proj_map': proj_map,
                **{
                    f'proj_{tmpl}': (mp @ template_op.blocks[tmpl])(tmpl_ampl[tmpl])
                    for tmpl in tmpl_ampl
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

    def make_map(self, observation: AbstractGroundObservation[Any]) -> dict[str, Any]:
        config = self.config
        logger_info = lambda msg: self.logger.info(f'ATOP Mapmaker: {msg}')

        # Data and landscape
        data = observation.get_tods().astype(config.dtype)
        data_struct = ShapeDtypeStruct(data.shape, data.dtype)
        landscape = self.get_landscape(observation, stokes='QU')

        # Acquisition (I, Q, U Maps -> TOD)
        acquisition = self.get_acquisition(observation, landscape=landscape)
        logger_info('Created acquisition operator')

        # ATOP projector
        atop_projector = ATOPProjectionOperator(self.config.atop_tau, in_structure=data_struct)

        # Optional mask for scanning
        masker = self.get_mask_projector(observation)
        valid_sample_fraction = (
            1.0
            if isinstance(masker, IdentityOperator)
            else float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        )
        logger_info('Created mask operator')
        logger_info(f'Valid sample fraction: {valid_sample_fraction:.4f}')

        # Additionally, mask all tau-intervals with any masked samples
        tau_mask = jnp.abs(atop_projector(masker(jnp.ones_like(data)))) < 0.5 / config.atop_tau
        masker @= MaskOperator.from_boolean_mask(tau_mask, in_structure=data_struct)
        valid_sample_fraction = float(jnp.mean(masker(jnp.ones(data.shape, data.dtype))))
        logger_info(f'Updated valid sample fraction: {valid_sample_fraction:.4f}')

        # Noise
        noise_model = self.get_or_fit_noise_model(observation)
        inv_noise = noise_model.inverse_operator(data_struct)
        logger_info('Created inverse noise covariance operator')

        # Approximate system matrix with diagonal noise covariance and full map pixels
        diag_system = BJPreconditioner.create(
            (acquisition.T @ inv_noise @ masker @ acquisition).reduce()
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

        # Mapmaking operator
        p = preconditioner
        h = acquisition @ selector.T
        mp = masker
        ap = inv_noise @ atop_projector

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
        if config.fit_noise_model:
            output['noise_fit'] = noise_model.to_array()
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
