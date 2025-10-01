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
    """Jittable reader for SOTODlib observations.

    The reader is set up with a list of filenames. Individual files can be loaded by passing
    an index in this list to the `read` method. The observation data is padded so that all
    observations have the same structure.

    Attributes:
        count (int): The number of data to read.
        args (list[tuple[Any, ...]]): For each data, the filename to be read.
        keywords (list[dict[str, Any]]): Not used.
        common_keywords (dict[str, Any]): Not used.
        out_structure (dict[str, jax.ShapeDtypeStruct]): The structure of the data that is returned
            by the read function. The structure is the same for all data.
        paddings (list[PyTree[tuple[int, ...]]): For each data, the padding that is applied to
            the data that is returned by the read function.

    Usage:
        >>> reader = SOTODLibReader(['obs1.h5', 'obs2.h5'])
        >>> data1, padding1 = reader.read(0)
        >>> data2, padding2 = reader.read(1)
    """

    def __init__(self, filenames: list[Path | str]) -> None:
        """Initializes the SOTODLib reader with a list of filenames.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
        """
        filenames = [Path(name) if isinstance(name, str) else name for name in filenames]
        super().__init__(filenames)

    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array]]:  # type: ignore[override]
        """Reads one SOTODLib observation from the list of filenames specified in the reader.

        This method is jittable.

        Args:
            data_index: The index of the data to read.

        Returns:
            A pair of PyTrees, the first one containing the data and the second one containing the
            padding. The structure of the data is the same as the structure of the padding.

            The data is a dictionary with the following keys:
                - signal: the detector read-outs.
                - mask: the mask indicating which samples are valid.
                - timestamps: the timestamps of the samples.
                - detector_quaternions: the detector quaternions.
                - boresight_quaternions: the boresight quaternions.
            The padding is a dictionary with the following keys:
                - signal: the padding for the signal.
                - mask: the padding for the mask.
                - timestamps: the padding for the timestamps.
                - detector_quaternions: the padding for the detector quaternions.
                - boresight_quaternions: the padding for the boresight quaternions.
        """
        return super().read(data_index)  # type: ignore[no-any-return]

    def _read_structure_impure(self, path: Path) -> PyTree[jax.ShapeDtypeStruct]:
        filename = path.as_posix()
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)
        return {
            'signal': jax.ShapeDtypeStruct((obs.n_detectors, obs.n_samples), jnp.float32),
            'mask': jax.ShapeDtypeStruct((obs.n_detectors, obs.n_samples), jnp.float64),
            'timestamps': jax.ShapeDtypeStruct((obs.n_samples,), jnp.float64),
            'detector_quaternions': jax.ShapeDtypeStruct((obs.n_detectors, 4), jnp.float64),
            'boresight_quaternions': jax.ShapeDtypeStruct((obs.n_samples, 4), jnp.float64),
        }

    def _read_data_impure(self, path: Path) -> PyTree[Array]:
        filename = path.as_posix()
        manager = AxisManager.load(filename)
        obs = SOTODLibObservation(manager)
        return {
            'signal': obs.get_tods(),
            'mask': obs.get_sample_mask(),
            'timestamps': obs.get_timestamps(),
            'detector_quaternions': obs.get_detector_quaternions(),
            'boresight_quaternions': obs.get_boresight_quaternions(),
        }
