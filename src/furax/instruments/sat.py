import numpy as np

from furax.detectors import DetectorArray
from furax.landscapes import HealpixLandscape
from furax.operators import AbstractLinearOperator
from furax.operators.hwp import RotatingHWPOperator
from furax.operators.polarizers import LinearPolarizerOperator
from furax.operators.projections import create_projection_operator
from furax.samplings import Sampling

FOV_DEG = 35
DIAMETER = 0.42
FREQUENCY = 150e9
LAMBDA = 3e8 / FREQUENCY
FWHM = LAMBDA / DIAMETER
FWHM_DEG = np.rad2deg(FWHM)
DETECTOR_ARRAY_SHAPE = (30, 30)
NDIR_PER_DETECTOR = 1
NDIR_PER_DETECTOR_ALONG_AXIS = int(np.sqrt(NDIR_PER_DETECTOR))
NDIR_IN_DETECTOR_UNIT = 2
FOCAL_PLANE_SIZE_METERS = 0.36
FOCAL_LENGTH = FOCAL_PLANE_SIZE_METERS / 2 / np.tan(np.deg2rad(FOV_DEG) / 2)


def create_detector_array() -> DetectorArray:
    shape = DETECTOR_ARRAY_SHAPE
    detector_size = FOCAL_PLANE_SIZE_METERS / max(shape)
    offset = (np.array(shape) - 1) / 2
    grid = (np.mgrid[: shape[0], : shape[1]] - offset[:, None, None]) * detector_size
    return DetectorArray(FOCAL_LENGTH, grid[0], grid[1])


def create_detector_directions() -> DetectorArray:
    if NDIR_PER_DETECTOR_ALONG_AXIS**2 != NDIR_PER_DETECTOR:
        raise ValueError('NDIR_PER_DETECTOR should be a square number.')
    array_shape = DETECTOR_ARRAY_SHAPE
    detector_size = FOCAL_PLANE_SIZE_METERS / max(array_shape)
    array_offset = (np.array(array_shape) - 1) / 2
    grid_centers = (
        np.mgrid[: array_shape[0], : array_shape[1]] - array_offset[:, None, None]
    ) * detector_size
    xcenters, ycenters = grid_centers.reshape((2, np.prod(DETECTOR_ARRAY_SHAPE)))

    detector_offset = (NDIR_PER_DETECTOR - 1) / 2
    grid_deltas = (
        (np.mgrid[:NDIR_PER_DETECTOR_ALONG_AXIS, :NDIR_PER_DETECTOR_ALONG_AXIS] - detector_offset)
        / NDIR_PER_DETECTOR_ALONG_AXIS
        * detector_size
    )
    xdeltas, ydeltas = grid_deltas.reshape((2, NDIR_PER_DETECTOR))

    xdirs = xcenters[:, None] + xdeltas[None, :]
    ydirs = ycenters[:, None] + ydeltas[None, :]

    return DetectorArray(xdirs, ydirs, FOCAL_LENGTH)


def create_acquisition(
    landscape: HealpixLandscape, samplings: Sampling, detector_dirs: DetectorArray
) -> AbstractLinearOperator:
    tod_shape = len(detector_dirs), len(samplings)
    proj = create_projection_operator(landscape, samplings, detector_dirs)
    hwp = RotatingHWPOperator.create(tod_shape, landscape.stokes, 0.0)
    polarizer = LinearPolarizerOperator(tod_shape, landscape.stokes)
    acquisition: AbstractLinearOperator = polarizer @ hwp @ proj
    return acquisition
