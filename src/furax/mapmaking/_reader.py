from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_static
from jaxtyping import PyTree

from furax.io.readers import AbstractReader

from ._observation import AbstractGroundObservation


@register_static
class AbstractGroundObservationReader(AbstractReader):
    """Jittable reader for ground observations.

    The reader is set up with a list of filenames and data field names. Individual files can be
    loaded by passing an index in this list to the `read` method. The observation data is padded
    so that all observations have the same structure.

    Attributes:
        count (int): The number of data to read.
        args (list[tuple[Any, ...]]): For each data, the filename to be read.
        keywords (list[dict[str, Any]]): Not used.
        data_field_names (list[str]): For all data, load the listed data fields only. If None,
            read all possible data fields
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

    DATA_FIELD_NAMES = [
        'sample_data',  # the detector read-outs.
        'valid_sample_masks',  # the (boolean) mask indicating which samples are valid (=True).
        'valid_scanning_masks',  # the (boolean) mask indicating which samples are taken during
        # scans and not turnarounds.
        'timestamps',  # the timestamps of the samples.
        'hwp_angles',  # the half-wave-plate angles in radians.
        'detector_quaternions',  # the detector quaternions.
        'boresight_quaternions',  # the boresight quaternions.
    ]

    def __init__(
        self, filenames: list[Path | str], data_field_names: list[str] | None = None
    ) -> None:
        """Initializes the SOTODLib reader with a list of filenames.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
            data_field_names: A list of data fields to load. If None, read all available data fields
                available for SOTODLib observations.
        """
        filenames = [Path(name) if isinstance(name, str) else name for name in filenames]
        if data_field_names is None:
            data_field_names = self.DATA_FIELD_NAMES
        else:
            data_field_names = list(set(data_field_names))  # Remove duplicates if any
            for name in data_field_names:
                # Validate field names
                if name not in self.DATA_FIELD_NAMES:
                    raise ValueError(
                        f'Data field "{name}" NOT supported for ground observation data format. '
                        f'Supported fields: {self.DATA_FIELD_NAMES}'
                    )

        super().__init__(filenames, common_keywords={'data_field_names': data_field_names})

    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array]]:  # type: ignore[override]
        """Reads one ground observation from the list of filenames specified in the reader.

        This method is jittable.

        Args:
            data_index: The index of the data to read.

        Returns:
            A pair of PyTrees, the first one containing the data and the second one containing the
            padding. The structure of the data is the same as the structure of the padding.

            The data is a dictionary with the following keys:
                - sample_data: the detector read-outs.
                - valid_sample_masks: the (boolean) mask indicating which samples are valid (=True).
                - valid_scanning_masks: the (boolean) mask indicating which samples are taken
                    during scans (and not turnarounds).
                - timestamps: the timestamps of the samples.
                - hwp_angles: the half-wave plate angle measured at each sample
                - detector_quaternions: the detector quaternions.
                - boresight_quaternions: the boresight quaternions.
            The padding is a dictionary with the following keys:
                - sample_data: the padding for the detector read-outs.
                - valid_sample_masks: the padding for the sample mask.
                - valid_scanning_masks: the padding for the scanning mask.
                - timestamps: the padding for the timestamps.
                - hwp_angles: the padding for the half-wave plate angles
                - detector_quaternions: the padding for the detector quaternions.
                - boresight_quaternions: the padding for the boresight quaternions.
        """
        return super().read(data_index)  # type: ignore[no-any-return]

    @classmethod
    def _get_data_field_structures_for(
        cls, n_detectors: int, n_samples: int
    ) -> dict[str, jax.ShapeDtypeStruct]:
        return {
            'sample_data': jax.ShapeDtypeStruct((n_detectors, n_samples), jnp.float64),
            'valid_sample_masks': jax.ShapeDtypeStruct((n_detectors, n_samples), jnp.bool),
            'valid_scanning_masks': jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            'timestamps': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'hwp_angles': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'detector_quaternions': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
            'boresight_quaternions': jax.ShapeDtypeStruct((n_samples, 4), jnp.float64),
        }

    @classmethod
    def _get_data_field_readers(cls) -> dict[str, Callable[[AbstractGroundObservation[Any]], Any]]:
        return {
            'sample_data': lambda obs: obs.get_tods().astype(jnp.float64),
            'valid_sample_masks': lambda obs: obs.get_sample_mask(),
            'valid_scanning_masks': lambda obs: obs.get_scanning_mask(),
            'timestamps': lambda obs: obs.get_timestamps(),
            'hwp_angles': lambda obs: obs.get_hwp_angles(),
            'detector_quaternions': lambda obs: obs.get_detector_quaternions(),
            'boresight_quaternions': lambda obs: obs.get_boresight_quaternions(),
        }

    @abstractmethod
    def _read_structure_impure(
        self, path: Path, data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]: ...

    @abstractmethod
    def _read_data_impure(self, path: Path, data_field_names: list[str]) -> PyTree[Array]: ...
