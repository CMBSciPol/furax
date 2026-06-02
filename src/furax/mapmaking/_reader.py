import time
from collections.abc import Collection, Sequence
from typing import Any, Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils as mhu
from jax.tree_util import register_static
from jaxtyping import PyTree

from furax.io.readers import AbstractReader
from furax.obs.stokes import Stokes, ValidStokesType

from ._logger import logger
from ._observation import (
    AbstractLazyObservation,
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
        *args: Sequence[Any],
        demodulated: bool,
        stokes: ValidStokesType,
        common_keywords: dict[str, Any] | None = None,
        structures: list[PyTree[jax.ShapeDtypeStruct]] | None = None,
        **keywords: Sequence[Any],
    ) -> None:
        # Set before super().__init__ so _read_structure_impure can use them
        self.demodulated = demodulated
        self.stokes = stokes
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
            )

        # Distributed-mode shortcut. Each process opens only its subset; an
        # all-gather then makes every rank agree on the full structures list so
        # padding / out_structure / etc. are consistent. Bypasses the superclass's
        # per-item I/O by passing the assembled `structures` directly.
        needs_intervals = 'scanning_intervals' in fields
        probe_fields = ['scanning_intervals'] if needs_intervals else []

        def _shape(idx: int) -> tuple[int, int, int, int]:
            data = observations[idx].get_data(probe_fields)
            # scanning_intervals is ground-only; requested only for ground observations
            n_int = data.get_scanning_intervals().shape[0] if needs_intervals else 0  # type: ignore[attr-defined]
            return idx, data.n_detectors, data.n_samples, n_int

        local_shapes = np.array([_shape(idx) for idx in read_indices], dtype=np.int64)

        # All-gather → (n_procs * n_local, 4). Padding (see ``get_padded_indices``,
        # which uses ``np.pad(..., mode='edge')``) makes some indices repeat across
        # ranks; keep only one entry per obs index.
        all_shapes = mhu.process_allgather(local_shapes).reshape(-1, 4)
        by_idx = {
            int(idx): (int(n_det), int(n_samp), int(n_int))
            for idx, n_det, n_samp, n_int in all_shapes
        }

        def _struct_for(n_det: int, n_samp: int, n_int: int) -> PyTree[jax.ShapeDtypeStruct]:
            field_struct = cls._get_data_field_structures_for(
                n_det, n_samp, demodulated=demodulated, stokes=stokes, n_intervals=n_int
            )
            return {field: field_struct[field] for field in fields}

        structures = [_struct_for(*by_idx[idx]) for idx in sorted(by_idx)]
        return cls(
            list(observations),
            common_keywords=common_keywords,
            demodulated=demodulated,
            stokes=stokes,
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
            - valid_sample_masks, valid_scanning_masks, left_scan_mask, right_scan_mask :
                padded with 0 (False) outside the valid samples
            - azimuth: held at the last valid value, as if the telescope stopped moving.
                Constant padding keeps the min/peak-to-peak range (used for basis
                normalisation) equal to that of the real scan.
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
        if 'timestamps' in data_field_names:
            # Extrapolate in the padded region for constant sample rate
            pad_size = padding['timestamps'][0]
            valid = data['timestamps'][: data['timestamps'].size - pad_size]
            dt = (valid[-1] - valid[0]) / (valid.size - 1)  # Mean time spacing
            data['timestamps'] = np.pad(
                valid, (0, pad_size), mode='linear_ramp', end_values=valid[-1] + dt * pad_size
            )
        if 'hwp_angles' in data_field_names:
            # Extrapolate in the padded region for constant hwp rotation frequency
            pad_size = padding['hwp_angles'][0]
            valid = data['hwp_angles'][: data['hwp_angles'].size - pad_size]
            dphi = (np.unwrap(valid)[-1] - valid[0]) / (valid.size - 1)  # Mean angle spacing
            data['hwp_angles'] = np.pad(
                valid, (0, pad_size), mode='linear_ramp', end_values=valid[-1] + dphi * pad_size
            ) % (2 * np.pi)
        if 'azimuth' in data_field_names:
            # Hold the last valid azimuth in the padded region (telescope stopped moving).
            # Keeps min(azimuth)/ptp(azimuth) unchanged, so basis normalisation is unaffected.
            pad_size = padding['azimuth'][0]
            valid = data['azimuth'][: data['azimuth'].size - pad_size]
            data['azimuth'] = np.pad(valid, (0, pad_size), mode='edge')
        if 'detector_quaternions' in data_field_names:
            # Pad with (1, 0, 0, 0), corresponding to xi=eta=gamma=0.
            zero_padded = np.linalg.norm(data['detector_quaternions'], axis=-1) == 0.0
            data['detector_quaternions'] = np.where(
                zero_padded[:, None],
                np.array([[1.0, 0.0, 0.0, 0.0]]),
                data['detector_quaternions'],
            )
        if 'boresight_quaternions' in data_field_names:
            # Pad with the last non-zero quaternion provided.
            pad_size = padding['boresight_quaternions'][0]  # samples axis
            last_quaternion = data['boresight_quaternions'][-pad_size - 1, :]
            zero_padded = np.linalg.norm(data['boresight_quaternions'], axis=-1) == 0.0
            data['boresight_quaternions'] = np.where(
                zero_padded[:, None], last_quaternion[None, :], data['boresight_quaternions']
            )
        if 'noise_model_fits' in data_field_names:
            default = np.array([[0.0, 0.0, 1.0, 0.1]])

            def _pad_noise_fits(arr: np.ndarray) -> np.ndarray:
                zero_padded = arr[:, 0] == 0.0
                return np.where(zero_padded[:, None], default, arr)

            data['noise_model_fits'] = jax.tree.map(_pad_noise_fits, data['noise_model_fits'])

        return data

    @staticmethod
    def _get_data_field_structures_for(
        n_detectors: int,
        n_samples: int,
        *,
        demodulated: bool,
        stokes: ValidStokesType,
        n_intervals: int = 0,
    ) -> PyTree[jax.ShapeDtypeStruct]:
        tod_shape = (n_detectors, n_samples)
        sample_data_structure = (
            Stokes.class_for(stokes).structure_for(tod_shape, jnp.float64)
            if demodulated
            else jax.ShapeDtypeStruct(tod_shape, jnp.float64)
        )

        return {
            'metadata': HashedObservationMetadata.structure_for(n_detectors),
            'sample_data': sample_data_structure,
            'valid_sample_masks': jax.ShapeDtypeStruct((n_detectors, n_samples), jnp.bool),
            'valid_scanning_masks': jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            'timestamps': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'hwp_angles': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'detector_quaternions': jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64),
            'boresight_quaternions': jax.ShapeDtypeStruct((n_samples, 4), jnp.float64),
            'noise_model_fits': (
                Stokes.class_for(stokes).structure_for((n_detectors, 4), jnp.float64)
                if demodulated
                else jax.ShapeDtypeStruct((n_detectors, 4), jnp.float64)
            ),
            'azimuth': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'elevation': jax.ShapeDtypeStruct((n_samples,), jnp.float64),
            'left_scan_mask': jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            'right_scan_mask': jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            'scanning_intervals': jax.ShapeDtypeStruct((n_intervals, 2), jnp.int64),
        }

    def _get_data_field_readers(self):  # type: ignore[no-untyped-def]
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        demodulated = self.demodulated
        stokes = self.stokes

        return {
            'metadata': lambda obs: HashedObservationMetadata.from_observation(obs),
            'sample_data': lambda obs: (
                obs.get_demodulated_tods(stokes=stokes) if demodulated else obs.get_tods()
            ),
            'valid_sample_masks': lambda obs: obs.get_sample_mask(),
            'valid_scanning_masks': lambda obs: obs.get_scanning_mask(),
            'timestamps': lambda obs: obs.get_timestamps(),
            'hwp_angles': lambda obs: obs.get_hwp_angles(),
            'detector_quaternions': lambda obs: obs.get_detector_quaternions(),
            'boresight_quaternions': lambda obs: obs.get_boresight_quaternions(),
            'noise_model_fits': (
                (lambda obs: obs.get_demodulated_noise_models(stokes=stokes))
                if demodulated
                else (lambda obs: if_none_raise_error(obs.get_noise_model()).to_array())
            ),
            'azimuth': lambda obs: obs.get_azimuth(),
            'elevation': lambda obs: obs.get_elevation(),
            'left_scan_mask': lambda obs: obs.get_left_scan_mask(),
            'right_scan_mask': lambda obs: obs.get_right_scan_mask(),
            'scanning_intervals': lambda obs: np.asarray(
                obs.get_scanning_intervals(), dtype=np.int64
            ),
        }

    def _read_structure_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str]
    ) -> PyTree[jax.ShapeDtypeStruct]:
        # Request the minimum needed to determine buffer shapes. ``scanning_intervals``
        # has a variable leading dimension (the per-observation interval count), so when
        # it is requested we must read it here to size the padded buffer.
        needs_intervals = 'scanning_intervals' in data_field_names
        probe_fields = ['scanning_intervals'] if needs_intervals else []
        data = observation.get_data(probe_fields)
        # scanning_intervals is ground-only; requested only for ground observations
        n_intervals = data.get_scanning_intervals().shape[0] if needs_intervals else 0  # type: ignore[attr-defined]
        field_structure = self._get_data_field_structures_for(
            data.n_detectors,
            data.n_samples,
            demodulated=self.demodulated,
            stokes=self.stokes,
            n_intervals=n_intervals,
        )
        return {field: field_structure[field] for field in data_field_names}

    def _read_data_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str]
    ) -> PyTree[Array]:
        t0 = time.perf_counter()
        data = observation.get_data(data_field_names)
        field_reader = self._get_data_field_readers()
        result = {field: field_reader[field](data) for field in data_field_names}
        dt = time.perf_counter() - t0
        nbytes = sum(np.asarray(v).nbytes for v in result.values())
        mbps = nbytes / 1e6 / dt if dt > 0 else float('nan')
        logger.debug(
            'read timing: rank=%d obs=%s read=%.3fs bytes=%.1fMB rate=%.1fMB/s',
            jax.process_index(),
            observation.file.stem,
            dt,
            nbytes / 1e6,
            mbps,
        )
        return result
