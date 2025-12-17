from collections.abc import Callable
from hashlib import sha1
from typing import Any, ClassVar, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_static
from jaxtyping import PyTree, UInt32

from furax.io.readers import AbstractReader

from ._observation import (
    AbstractGroundObservation,
    AbstractLazyObservation,
    HashedObservationMetadata,
)

T = TypeVar('T')


@register_static
class GroundObservationReader(AbstractReader, Generic[T]):
    """Jittable reader for ground observations.

    The reader is set up with a list of filenames and data field names. Individual files can be
    loaded by passing an index in this list to the `read` method. The observation data is padded
    so that all observations have the same structure.

    The available data fields for ground observations are:
        - metadata: observation, telescope and detector uids.
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
        'metadata',
        'sample_data',
        'valid_sample_masks',
        'valid_scanning_masks',
        'timestamps',
        'hwp_angles',
        'detector_quaternions',
        'boresight_quaternions',
        'noise_model_fits',
    ]
    """Supported data field names for ground observations"""

    OPTIONAL_DATA_FIELD_NAMES: ClassVar[list[str]] = [
        'noise_model_fits',
    ]

    def __init__(
        self,
        observations: list[AbstractLazyObservation[T]],
        *,
        data_field_names: list[str] | None = None,
    ) -> None:
        """Initializes the reader with a list of filenames and optional list of field names.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
            data_field_names: Optional list of fields to load. If None, read all non-optional fields
        """
        if data_field_names is None:
            data_field_names = [
                field
                for field in self.DATA_FIELD_NAMES
                if field not in self.OPTIONAL_DATA_FIELD_NAMES
            ]
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

        super().__init__(observations, common_keywords={'data_field_names': data_field_names})

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
            zero_padded = data['noise_model_fits'][:, 0] == 0.0
            data['noise_model_fits'] = jnp.where(
                zero_padded[:, None],
                jnp.array([[0.0, 0.0, 1.0, 0.1]]),
                data['noise_model_fits'],
            )

        return data, padding

    @classmethod
    def _get_data_field_structures_for(
        cls, n_detectors: int, n_samples: int
    ) -> PyTree[jax.ShapeDtypeStruct]:
        return {
            'metadata': HashedObservationMetadata(
                uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),  # type: ignore[arg-type]
                telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),  # type: ignore[arg-type]
                detector_uids=jax.ShapeDtypeStruct((n_detectors,), dtype=jnp.uint32),  # type: ignore[arg-type]
            ),
            'sample_data': jax.ShapeDtypeStruct((n_detectors, n_samples), jnp.float64),
            'valid_sample_masks': jax.ShapeDtypeStruct((n_detectors, n_samples), jnp.bool),
            'valid_scanning_masks': jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            'timestamps': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'hwp_angles': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'detector_quaternions': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
            'boresight_quaternions': jax.ShapeDtypeStruct((n_samples, 4), jnp.float64),
            'noise_model_fits': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
        }

    @classmethod
    def _get_data_field_readers(cls) -> dict[str, Callable[[AbstractGroundObservation[Any]], Any]]:
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        def get_metadata(obs: AbstractGroundObservation[T]) -> HashedObservationMetadata:
            return HashedObservationMetadata(
                uid=jnp.asarray(_names_to_uids(obs.name)),
                telescope_uid=jnp.asarray(_names_to_uids(obs.telescope)),
                detector_uids=jnp.asarray(_names_to_uids(obs.detectors)),
            )

        return {
            'metadata': lambda obs: get_metadata(obs),
            'sample_data': lambda obs: obs.get_tods().astype(jnp.float64),
            'valid_sample_masks': lambda obs: obs.get_sample_mask(),
            'valid_scanning_masks': lambda obs: obs.get_scanning_mask(),
            'timestamps': lambda obs: obs.get_timestamps(),
            'hwp_angles': lambda obs: obs.get_hwp_angles(),
            'detector_quaternions': lambda obs: obs.get_detector_quaternions(),
            'boresight_quaternions': lambda obs: obs.get_boresight_quaternions(),
            'noise_model_fits': lambda obs: if_none_raise_error(obs.get_noise_model()).to_array(),
        }

    def _read_structure_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: list[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        # request an empty list
        # this loads sufficient info to determine the structure
        data = observation.get_data([])

        # find the data shape
        n_detectors = data.n_detectors
        n_samples = data.n_samples

        field_structure = GroundObservationReader._get_data_field_structures_for(
            n_detectors=n_detectors, n_samples=n_samples
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: list[str]
    ) -> PyTree[Array]:
        data = observation.get_data(data_field_names)
        field_reader = GroundObservationReader._get_data_field_readers()
        return {field: field_reader[field](data) for field in data_field_names}


def _names_to_uids(names: str | list[str] | np.ndarray) -> UInt32[np.ndarray, '...']:
    """Converts names to unsigned 32-bit integers using hashing.

    This is typically used to generate deterministic uids for detectors based on their names.
    """
    # hashing + converting to int + keeping only 7 bytes
    name_to_int = np.vectorize(lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0xEFFFFFFF)
    return name_to_int(names).astype(np.uint32)  # type: ignore[no-any-return]
