from collections.abc import Sequence
from hashlib import sha1
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_static
from jaxtyping import PyTree, UInt32

from furax.io.readers import AbstractReader

from ._observation import (
    AbstractLazyObservation,
    AbstractObservation,
    HashedObservationMetadata,
)

T = TypeVar('T')


@register_static
class ObservationReader(AbstractReader, Generic[T]):
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

    def __init__(
        self,
        observations: Sequence[AbstractLazyObservation[T]],
        *,
        requested_fields: list[str] | None = None,
    ) -> None:
        """Initializes the reader with a list of filenames and optional list of field names.

        Args:
            filenames: A list of filenames. Each filename can be a string or a Path object.
            requested_fields: Optional list of fields to load. If None, read all non-optional fields.
        """
        interface = observations[0].interface_class
        available = set(interface.AVAILABLE_READER_FIELDS)
        optional = set(interface.OPTIONAL_READER_FIELDS)
        if requested_fields is None:
            fields = available - optional
        else:
            fields = set(requested_fields)
            unsupported = fields - available
            if len(unsupported) > 0:
                msg = f'Requested data fields {unsupported} are not supported by the interface.'
                raise ValueError(msg)

        super().__init__(observations, common_keywords={'data_field_names': list(fields)})

    def read(self, data_index: int) -> tuple[PyTree[Array], PyTree[Array]]:  # type: ignore[override]
        """Reads one ground observation from the list of filenames specified in the reader.
        The data is padded differently depending on the key:
            - sample_data: padded with 0.0 outside the valid samples
            - timestamps, hwp_angles: extrapolated in the padded region so that
                the sample rate and the hwp rotation frequency remain consistent
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
        if 'timestamps' in data_field_names:
            # Extrapolate in the padded region for constant sample rate
            timestamps = data['timestamps']
            pad_size = padding['timestamps'][0]  # Padded length along samples axis
            data_size = timestamps.size - pad_size  # Unpadded length
            dt = (timestamps[data_size - 1] - timestamps[0]) / (data_size - 1)  # Mean time spacing
            extrapolated = (
                timestamps[data_size - 1]
                + (jnp.arange(timestamps.size, dtype=timestamps.dtype) - (data_size - 1)) * dt
            )  # Extrapolate from the last non-zero data entry
            data['timestamps'] = jnp.where(
                jnp.arange(timestamps.size) < data_size,
                timestamps,
                extrapolated,
            )
        if 'hwp_angles' in data_field_names:
            # Extrapolate in the padded region for constant hwp roation frequency
            hwp_angles = data['hwp_angles']
            pad_size = padding['hwp_angles'][0]  # Padded length along samples axis
            data_size = hwp_angles.size - pad_size  # Unpadded length
            dphi = (jnp.unwrap(hwp_angles)[data_size - 1] - hwp_angles[0]) / (
                data_size - 1
            )  # Mean angle spacing
            extrapolated = (
                hwp_angles[data_size - 1]
                + (jnp.arange(hwp_angles.size, dtype=hwp_angles.dtype) - (data_size - 1)) * dphi
            )  # Extrapolate from the last non-zero data entry
            data['hwp_angles'] = jnp.where(
                jnp.arange(hwp_angles.size) < data_size,
                hwp_angles,
                extrapolated,
            ) % (2 * jnp.pi)
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
    def _get_data_field_readers(cls):  # type: ignore[no-untyped-def]
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        def get_metadata(obs: AbstractObservation[T]) -> HashedObservationMetadata:
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

        field_structure = ObservationReader._get_data_field_structures_for(
            n_detectors=n_detectors, n_samples=n_samples
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: list[str]
    ) -> PyTree[Array]:
        data = observation.get_data(data_field_names)
        field_reader = ObservationReader._get_data_field_readers()
        return {field: field_reader[field](data) for field in data_field_names}


def _names_to_uids(names: str | list[str] | np.ndarray) -> UInt32[np.ndarray, '...']:
    """Converts names to unsigned 32-bit integers using hashing.

    This is typically used to generate deterministic uids for detectors based on their names.
    """
    # hashing + converting to int + keeping only 7 bytes
    name_to_int = np.vectorize(lambda s: int(sha1(s.encode()).hexdigest(), 16) & 0xEFFFFFFF)
    return name_to_int(names).astype(np.uint32)  # type: ignore[no-any-return]
