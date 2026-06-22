from collections.abc import Collection, Sequence
from typing import Any, Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils as mhu
from jax.tree_util import register_static
from jax.typing import DTypeLike
from jaxtyping import PyTree

from furax.io.readers import AbstractReader
from furax.obs.stokes import Stokes, ValidStokesType

from ._observation import (
    AbstractLazyObservation,
    AbstractObservation,
    HashedObservationMetadata,
    ReaderField,
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
        *args: Sequence[Any],
        demodulated: bool,
        stokes: ValidStokesType,
        dtype: DTypeLike = jnp.float64,
        common_keywords: dict[str, Any] | None = None,
        structures: list[PyTree[jax.ShapeDtypeStruct]] | None = None,
        **keywords: Sequence[Any],
    ) -> None:
        # Set before super().__init__ so _read_structure_impure can use them
        self.demodulated = demodulated
        self.stokes = stokes
        self.dtype = dtype
        super().__init__(*args, common_keywords=common_keywords, structures=structures, **keywords)

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[AbstractLazyObservation[T]],
        *,
        read_indices: Sequence[int] | None = None,
        requested_fields: Collection[str] | None = None,
        demodulated: bool = False,
        stokes: ValidStokesType = 'IQU',
        dtype: DTypeLike = jnp.float64,
    ) -> Self:
        """Create a reader, performing I/O to infer data structures.

        Args:
            observations: Full list of lazy observations.
            read_indices: Optional indices into ``observations``; when set, only
                those are opened on this process to infer shapes, and shapes are
                synchronised across processes (distributed-mode shortcut).
            requested_fields: Optional list of fields to load. If None, read all non-optional fields.
            demodulated: Whether to read demodulated TODs.
            stokes: Stokes components to read when demodulated.
            dtype: Floating-point dtype applied to every floating-point field the reader
                returns: sample data, noise model fits and the geometry (timestamps, HWP
                angles, quaternions). Use jnp.float32 to run a float32 mapmaking pipeline
                (MapMakingConfig.double_precision=False). Casting the geometry is also
                required there: under jax_enable_x64=False a float64 array is illegal, so
                no field may stay float64. Timestamps are rebased to a per-observation
                zero origin (in float64, before the downcast) so the float32 cast does not
                collapse the absolute POSIX epoch onto a single value; see the timestamps
                reader in ``_get_data_field_readers``.
        """
        fields = cls._resolve_fields(observations, requested_fields)
        common_keywords = {'data_field_names': fields}
        if read_indices is None:
            # Default path: AbstractReader.__init__ will call _read_structures(),
            # opening every observation on this process to infer its structure.
            return cls(
                list(observations),
                common_keywords=common_keywords,
                demodulated=demodulated,
                stokes=stokes,
                dtype=dtype,
            )

        # Distributed-mode shortcut. Each process opens only its subset; an
        # all-gather then makes every rank agree on the full structures list so
        # padding / out_structure / etc. are consistent. Bypasses the superclass's
        # per-item I/O by passing the assembled `structures` directly.
        def _shape(idx: int) -> tuple[int, int, int]:
            return idx, *observations[idx].probe_shape()

        local_shapes = np.array([_shape(idx) for idx in read_indices], dtype=np.int64)

        # All-gather → (n_procs * n_local, 3). Padding (see ``get_padded_indices``,
        # which uses ``np.pad(..., mode='edge')``) makes some indices repeat across
        # ranks; keep only one entry per obs index.
        all_shapes = mhu.process_allgather(local_shapes).reshape(-1, 3)
        by_idx = {int(idx): (int(n_det), int(n_samp)) for idx, n_det, n_samp in all_shapes}

        def _struct_for(n_det: int, n_samp: int) -> PyTree[jax.ShapeDtypeStruct]:
            field_struct = cls._get_data_field_structures_for(
                n_det, n_samp, demodulated=demodulated, stokes=stokes, dtype=dtype
            )
            return {field: field_struct[field] for field in fields}

        structures = [_struct_for(*by_idx[idx]) for idx in sorted(by_idx)]
        return cls(
            list(observations),
            common_keywords=common_keywords,
            demodulated=demodulated,
            stokes=stokes,
            dtype=dtype,
            structures=structures,
        )

    @staticmethod
    def _resolve_fields(
        observations: Sequence[AbstractLazyObservation[T]],
        requested_fields: Collection[str] | None,
    ) -> set[str]:
        interface = observations[0].interface_class
        available = set(interface.AVAILABLE_READER_FIELDS)
        optional = set(interface.OPTIONAL_READER_FIELDS)
        if requested_fields is None:
            return available - optional
        fields = set(requested_fields)
        unsupported = fields - available
        if unsupported:
            raise ValueError(
                f'Requested data fields {unsupported} are not supported by the interface.'
            )
        return fields

    def _pad(
        self, data: PyTree[np.ndarray], padding: PyTree[tuple[int, ...]]
    ) -> PyTree[np.ndarray]:
        """Pads one ground observation to the common structure, on the host (numpy).

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
        # First, pad them with 0 by default
        data = super()._pad(data, padding)

        # Handle fields with non-zero padding
        data_field_names = self.common_keywords['data_field_names']
        if ReaderField.TIMESTAMPS in data_field_names:
            # Extrapolate in the padded region for constant sample rate
            pad_size = padding[ReaderField.TIMESTAMPS][0]
            valid = data[ReaderField.TIMESTAMPS][: data[ReaderField.TIMESTAMPS].size - pad_size]
            dt = (valid[-1] - valid[0]) / (valid.size - 1)  # Mean time spacing
            data[ReaderField.TIMESTAMPS] = np.pad(
                valid, (0, pad_size), mode='linear_ramp', end_values=valid[-1] + dt * pad_size
            )
        if ReaderField.HWP_ANGLES in data_field_names:
            # Extrapolate in the padded region for constant hwp rotation frequency
            pad_size = padding[ReaderField.HWP_ANGLES][0]
            valid = data[ReaderField.HWP_ANGLES][: data[ReaderField.HWP_ANGLES].size - pad_size]
            dphi = (np.unwrap(valid)[-1] - valid[0]) / (valid.size - 1)  # Mean angle spacing
            data[ReaderField.HWP_ANGLES] = np.pad(
                valid, (0, pad_size), mode='linear_ramp', end_values=valid[-1] + dphi * pad_size
            ) % (2 * np.pi)
        if ReaderField.DETECTOR_QUATERNIONS in data_field_names:
            # Pad with (1, 0, 0, 0), corresponding to xi=eta=gamma=0.
            quats = data[ReaderField.DETECTOR_QUATERNIONS]
            zero_padded = np.linalg.norm(quats, axis=-1) == 0.0
            data[ReaderField.DETECTOR_QUATERNIONS] = np.where(
                zero_padded[:, None],
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=quats.dtype),
                quats,
            )
        if ReaderField.BORESIGHT_QUATERNIONS in data_field_names:
            # Pad with the last non-zero quaternion provided.
            quats = data[ReaderField.BORESIGHT_QUATERNIONS]
            pad_size = padding[ReaderField.BORESIGHT_QUATERNIONS][0]  # samples axis
            last_quaternion = quats[-pad_size - 1, :]
            zero_padded = np.linalg.norm(quats, axis=-1) == 0.0
            data[ReaderField.BORESIGHT_QUATERNIONS] = np.where(
                zero_padded[:, None], last_quaternion[None, :], quats
            )
        if ReaderField.NOISE_MODEL_FITS in data_field_names:

            def _pad_noise_fits(arr: np.ndarray) -> np.ndarray:
                default = np.array([[0.0, 0.0, 1.0, 0.1]], dtype=arr.dtype)
                zero_padded = arr[:, 0] == 0.0
                return np.where(zero_padded[:, None], default, arr)

            data[ReaderField.NOISE_MODEL_FITS] = jax.tree.map(
                _pad_noise_fits, data[ReaderField.NOISE_MODEL_FITS]
            )

        return data

    @staticmethod
    def _get_data_field_structures_for(
        n_detectors: int,
        n_samples: int,
        *,
        demodulated: bool,
        stokes: ValidStokesType,
        dtype: DTypeLike = jnp.float64,
    ) -> PyTree[jax.ShapeDtypeStruct]:
        tod_shape = (n_detectors, n_samples)
        sample_data_structure = (
            Stokes.class_for(stokes).structure_for(tod_shape, dtype)
            if demodulated
            else jax.ShapeDtypeStruct(tod_shape, dtype)
        )

        return {
            ReaderField.METADATA: HashedObservationMetadata.structure_for(n_detectors),
            ReaderField.SAMPLE_DATA: sample_data_structure,
            ReaderField.VALID_SAMPLE_MASKS: jax.ShapeDtypeStruct(
                (n_detectors, n_samples), jnp.bool
            ),
            ReaderField.VALID_SCANNING_MASKS: jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            ReaderField.TIMESTAMPS: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.HWP_ANGLES: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.DETECTOR_QUATERNIONS: jax.ShapeDtypeStruct((n_detectors, 4), dtype),
            ReaderField.BORESIGHT_QUATERNIONS: jax.ShapeDtypeStruct((n_samples, 4), dtype),
            ReaderField.NOISE_MODEL_FITS: (
                Stokes.class_for(stokes).structure_for((n_detectors, 4), dtype)
                if demodulated
                else jax.ShapeDtypeStruct((n_detectors, 4), dtype)
            ),
        }

    def _get_data_field_readers(self):  # type: ignore[no-untyped-def]
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        demodulated = self.demodulated
        stokes = self.stokes
        target_dtype = self.dtype

        def get_sample_data(obs: AbstractObservation[T]) -> Any:
            if demodulated:
                tods = obs.get_demodulated_tods(stokes=stokes)
            else:
                tods = obs.get_tods()
            return jax.tree.map(lambda x: x.astype(target_dtype), tods)

        def get_noise_model_fits(obs: AbstractObservation[T]) -> Any:
            if demodulated:
                fits = obs.get_demodulated_noise_models(stokes=stokes)
            else:
                fits = if_none_raise_error(obs.get_noise_model()).to_array()
            return jax.tree.map(lambda x: x.astype(target_dtype), fits)

        def get_timestamps(obs: AbstractObservation[T]) -> Any:
            # The pipeline uses only time differences, so anchor each observation at its
            # own start. Subtracting in float64 keeps the samples resolved when cast to
            # float32; the absolute epoch is read from the interface where needed.
            timestamps = np.asarray(obs.get_timestamps(), dtype=np.float64)
            return (timestamps - timestamps[0]).astype(target_dtype)

        return {
            ReaderField.METADATA: lambda obs: HashedObservationMetadata.from_observation(obs),
            ReaderField.SAMPLE_DATA: get_sample_data,
            ReaderField.VALID_SAMPLE_MASKS: lambda obs: obs.get_sample_mask(),
            ReaderField.VALID_SCANNING_MASKS: lambda obs: obs.get_scanning_mask(),
            ReaderField.TIMESTAMPS: get_timestamps,
            ReaderField.HWP_ANGLES: lambda obs: obs.get_hwp_angles().astype(target_dtype),
            ReaderField.DETECTOR_QUATERNIONS: lambda obs: obs.get_detector_quaternions().astype(
                target_dtype
            ),
            ReaderField.BORESIGHT_QUATERNIONS: lambda obs: obs.get_boresight_quaternions().astype(
                target_dtype
            ),
            ReaderField.NOISE_MODEL_FITS: get_noise_model_fits,
        }

    def _read_structure_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        n_detectors, n_samples = observation.probe_shape()
        field_structure = self._get_data_field_structures_for(
            n_detectors,
            n_samples,
            demodulated=self.demodulated,
            stokes=self.stokes,
            dtype=self.dtype,
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str]
    ) -> PyTree[Array]:
        data = observation.get_data(data_field_names)
        field_reader = self._get_data_field_readers()
        return {field: field_reader[field](data) for field in data_field_names}
