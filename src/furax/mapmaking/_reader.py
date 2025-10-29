from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

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

    The available data fields for ground observations are:
        - sample_data: the detector read-outs.
        - valid_sample_masks: the (boolean) mask indicating which samples are valid (=True).
        - valid_scanning_masks: the (boolean) mask indicating which samples are taken
            during scans (and not turnarounds).
        - timestamps: the timestamps of the samples.
        - hwp_angles: the half-wave plate angle measured at each sample.
        - detector_quaternions: the detector quaternions.
        - boresight_quaternions: the boresight quaternions.
        - noise_model_fits: the fitted parameters for the noise model (1/f noise by default).
    """

    DATA_FIELD_NAMES: ClassVar[list[str]] = [
        'sample_data',  # the detector read-outs.
        'valid_sample_masks',  # the (boolean) mask indicating which samples are valid (=True).
        'valid_scanning_masks',  # the (boolean) mask indicating which samples are taken during
        # scans and not turnarounds.
        'timestamps',  # the timestamps of the samples.
        'hwp_angles',  # the half-wave-plate angles in radians.
        'detector_quaternions',  # the detector quaternions.
        'boresight_quaternions',  # the boresight quaternions.
        'noise_model_fits', # the fitted parameters for the noise model
    ]
    OPTIONAL_DATA_FIELD_NAMES: ClassVar[list[str]] = [
        'noise_model_fits',
    ]
    """Supported data field names for ground observations"""

    def __init__(
        self, filenames: list[Path | str], *, data_field_names: list[str] | None = None
    ) -> None:
        """Initializes the reader with a list of filenames and optional list of field names.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
            data_field_names: Optional list of fields to load. If None, read all non-optional fields
        """
        filenames = [Path(name) if isinstance(name, str) else name for name in filenames]
        if data_field_names is None:
            data_field_names = [field for field in self.DATA_FIELD_NAMES
                if field not in self.OPTIONAL_DATA_FIELD_NAMES]
        else:
            data_field_names = list(set(data_field_names))  # Remove duplicates if any
            # Validate field names
            for name in data_field_names:
                if name not in self.DATA_FIELD_NAMES:
                    msg = (
                        f'Data field "{name}" NOT supported for ground observation data format. '
                        f'Supported fields: {self.DATA_FIELD_NAMES}'
                    )
                    raise ValueError(msg)

        super().__init__(filenames, common_keywords={'data_field_names': data_field_names})

    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array]]:  # type: ignore[override]
        """Reads one ground observation from the list of filenames specified in the reader.
        The data is padded differently depending on the key:
            - sample_data, timestamps, hwp_angles: padded with 0.0 outside the valid samples
            - valid_sample_masks, valid_scanning_masks : padded with 0 (False) outside
                the valid samples
            - detector_quaternions: padded with (1, 0, 0, 0) for invalid detectors, as if they
                are located at the centre of the focal plane.
            - boresight_quaternions: padded with the last valid sample's quaternion, as if
                the telescoped stopped moving since then.
            - noise_model_fits: padded with (sigma, alpha, fknee, f0) = (0., 0., 1., 0.1)
        """
        # First, read them and pad them with 0 by default
        data, padding = super().read(data_index)

        # Handle fields with non-zero padding
        data_field_names = self.common_keywords['data_field_names']
        if 'detector_quaternions' in data_field_names:
            # Pad with (1, 0, 0, 0), corresponding to xi=eta=gamma=0.
            zero_padded = jnp.linalg.norm(data['detector_quaternions'], axis=-1) == 0.0
            data['detector_quaternions'] = jnp.where(
                zero_padded[:, None],
                jnp.array([[1.0, 0.0, 0.0, 0.0]]),
                data['detector_quaternions'],
            )
        if 'boresight_quaternions' in data_field_names:
            # Pad with the last non-zero quaternion provided.
            pad_size = padding['boresight_quaternions'][0]  # samples axis
            last_quaternion = data['boresight_quaternions'][-pad_size - 1, :]
            zero_padded = jnp.linalg.norm(data['boresight_quaternions'], axis=-1) == 0.0
            data['boresight_quaternions'] = jnp.where(
                zero_padded[:, None], last_quaternion[None, :], data['boresight_quaternions']
            )
        if 'noise_model_fits' in data_field_names:
            zero_padded = data['noise_model_fits'][:,0] == 0.0
            data['noise_model_fits'] = jnp.where(
                zero_padded[:, None],
                jnp.array([[0.0, 0.0, 1.0, 0.1]]),
                data['noise_model_fits'],
            )

        return data, padding

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
            'noise_model_fits': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
            'noise_model_fits': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
        }

    @classmethod
    def _get_data_field_readers(cls) -> dict[str, Callable[[AbstractGroundObservation[Any]], Any]]:
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        return {
            'sample_data': lambda obs: obs.get_tods().astype(jnp.float64),
            'valid_sample_masks': lambda obs: obs.get_sample_mask(),
            'valid_scanning_masks': lambda obs: obs.get_scanning_mask(),
            'timestamps': lambda obs: obs.get_timestamps(),
            'hwp_angles': lambda obs: obs.get_hwp_angles(),
            'detector_quaternions': lambda obs: obs.get_detector_quaternions(),
            'boresight_quaternions': lambda obs: obs.get_boresight_quaternions(),
            'noise_model_fits': lambda obs: if_none_raise_error(obs.get_noise_model()).to_array()
        }

    @abstractmethod
    def _read_structure_impure(
        self, path: Path, data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]: ...

    @abstractmethod
    def _read_data_impure(self, path: Path, data_field_names: list[str]) -> PyTree[Array]: ...

    @abstractmethod
    def update_data_field_names(
        self, data_field_names: list[str]
    ) -> 'AbstractGroundObservationReader':
        """Returns a new reader with a new list of data fields to read."""
