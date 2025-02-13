import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, DTypeLike, Float, PyTree

from furax import AbstractLinearOperator, DiagonalOperator, IdentityOperator
from furax.core import IndexOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape, WCSLandscape
from furax.obs.operators import QURotationOperator
from furax.obs.stokes import Stokes, StokesPyTreeType, ValidStokesType

from ._observation_data import GroundObservationData
from .config import MapMakingConfig
from .preconditioner import BJPreconditioner


@dataclass
class MapMaker:
    """Class for generic mapmaker which consumes GroundObservationData."""

    config: MapMakingConfig
    logger: logging.Logger

    @abstractmethod
    def mapmake(self, observation: GroundObservationData) -> dict[str, Any]: ...

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'MapMaker':
        return cls(config=MapMakingConfig.load_yaml(path), logger=logging.getLogger())

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
        """Create and return a flag operator which selects only the scanning intervals
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


class BinnedMapMaker(MapMaker):
    """Class for mapmaking with diagonal noise covariance."""

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

        if self.config.debug:
            res = process(data)
            res.i.block_until_ready()
            logger_info('Test - second time - Finished mapmaking')

        final_map = np.array([res.i, res.q, res.u])
        weights = np.array(system.blocks)

        return {'map': final_map, 'weights': weights}


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
