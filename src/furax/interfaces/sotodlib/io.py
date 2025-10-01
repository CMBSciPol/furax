from pathlib import Path

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree
from sotodlib.core import AxisManager

from furax.io.readers import AbstractReader

from .observation import SOTODLibObservation


@jax.tree_util.register_static
class SOTODLibReader(AbstractReader):
    """Reader for SOTODlib observations.

    The reader is set up with a list of filenames. The observation data can be accessed by
    the filename index. The observation data is padded so that all observations have the same
    structure.

        >>> reader = SOTODLibReader(['obs1.h5', 'obs2.h5'])
        >>> data, padding = reader.read(0)
    """

    def _read_structure_impure(self, filename: Path | str) -> PyTree[jax.ShapeDtypeStruct]:
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)
        return {
            'signal': jax.ShapeDtypeStruct((obs.n_detectors, obs.n_samples), jnp.float32),
            'mask': jax.ShapeDtypeStruct((obs.n_detectors, obs.n_samples), jnp.float64),
            'timestamps': jax.ShapeDtypeStruct((obs.n_samples,), jnp.float64),
            'detector_quaternions': jax.ShapeDtypeStruct((obs.n_detectors, 4), jnp.float64),
            'boresight_quaternions': jax.ShapeDtypeStruct((obs.n_samples, 4), jnp.float64),
        }

    def _read_data_impure(self, filename: Path | str) -> PyTree[Array]:
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)
        return {
            'signal': obs.get_tods(),
            'mask': obs.get_sample_mask(),
            'timestamps': obs.get_timestamps(),
            'detector_quaternions': obs.get_detector_quaternions(),
            'boresight_quaternions': obs.get_boresight_quaternions(),
        }
