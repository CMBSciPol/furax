import logging
import mmap
import threading
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from math import prod
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_static
from jax.typing import DTypeLike
from jaxtyping import PyTree

from furax.io.readers import AbstractReader
from furax.obs import TODFormat
from furax.obs.stokes import Stokes, ValidStokesLiteral

from ._observation import (
    AbstractLazyObservation,
    AbstractObservation,
    HashedObservationMetadata,
    ObservationBufferShape,
    ReaderField,
)

logger = logging.getLogger(__name__)


@dataclass
class ObservationRecord:
    """What was read of one observation.

    The detectors are listed in the order of the detector axis of the data read: detector `j` is
    at index `j` along that axis, and the indices past `n_detectors` are padding.
    """

    name: str
    """The observation's name."""
    detectors: list[str]
    """The names of the detectors read, in the order of the detector axis."""
    quaternions: np.ndarray | None = None
    """The detector quaternions, `(n_detectors, 4)`; `None` if they were not read."""
    n_samples: int = 0
    """The number of samples read."""
    valid_sample_fraction: float | None = None
    """The fraction of the samples not flagged; `None` if the flags were not read."""
    load_seconds: float | None = None
    """The time taken to load the observation; `None` if it is not known."""
    error: str | None = None
    """The exception that stopped the record being made; `None` if there was none."""

    @property
    def n_detectors(self) -> int:
        return len(self.detectors)

    @classmethod
    def from_fields(
        cls,
        observation: AbstractObservation[Any],
        fields: dict[str, Any],
        load_seconds: float,
        name: str,
    ) -> Self:
        """The record of an observation read, from the fields read of it.

        The record is named `name`, as the observation was listed: the loaded data can be named
        otherwise, e.g. without the wafer the observation is limited to.
        """
        quaternions = fields.get(ReaderField.DETECTOR_QUATERNIONS)
        masks = fields.get(ReaderField.VALID_SAMPLE_MASKS)
        return cls(
            name=name,
            detectors=list(observation.detectors),
            quaternions=None if quaternions is None else np.array(quaternions, dtype=np.float64),
            n_samples=observation.n_samples,
            valid_sample_fraction=None if masks is None else float(np.mean(masks)),
            load_seconds=load_seconds,
        )


@register_static
class ObservationReader[T](AbstractReader):
    """Jittable reader for ground observations.

    The reader is set up with a list of filenames and data field names. Individual files can be
    loaded by passing an index in this list to the `read` method. The observation data is padded
    so that all observations have the same structure.

    These are the data fields available for ground observations, named by [`ReaderField`][]:

    - `metadata`: observation, telescope and detector uids.
    - `sample_data`: the detector read-outs.
    - `valid_sample_masks`: the (boolean) mask indicating which samples are valid (=True).
    - `valid_scanning_masks`: the (boolean) mask indicating which samples are taken during scans
      (and not turnarounds).
    - `timestamps`: the timestamps of the samples.
    - `hwp_angles`: the half-wave plate angle measured at each sample.
    - `detector_quaternions`: the detector quaternions.
    - `boresight_quaternions`: the boresight quaternions.
    - `noise_model_fits`: the fitted parameters for the noise model (1/f noise by default).
    - `azimuth`, `elevation`: the boresight scan angles (ground observations).
    - `left_scan_mask`, `right_scan_mask`: per-sample masks splitting left/right-going scans.
    - `scanning_intervals`: per-scan [start, end) sample-index pairs (ground observations).

    Attributes:
        stokes: Stokes components of the demodulated TOD; `None` for detector timestreams.
        dtype: Floating-point dtype every floating-point field is cast to.
        sample_dtype: Floating-point dtype the sample data is cast to instead.
        records: What was read of each item read in this pass, keyed by its index: its
            detectors, in the order of the detector axis of the data, and a summary.
    """

    def __init__(
        self,
        *args: Sequence[Any],
        stokes: ValidStokesLiteral | None = None,
        dtype: DTypeLike = jnp.float64,
        sample_dtype: DTypeLike | None = None,
        common_keywords: dict[str, Any] | None = None,
        shapes: list[tuple[int, ...]] | None = None,
        known_failures: Sequence[int] | None = None,
        **keywords: Sequence[Any],
    ) -> None:
        """Initialize the reader.

        Prefer [`from_observations`][furax.mapmaking.ObservationReader.from_observations], which
        resolves the data fields of the observations and passes them on as `common_keywords`.

        Args:
            *args: One list per positional argument to the read function, one element per
                observation.
            stokes: Stokes components of the demodulated TOD; `None` for detector timestreams.
            dtype: Floating-point dtype applied to every floating-point field the reader returns.
            sample_dtype: Floating-point dtype applied to the sample data instead; `None` for
                `dtype`.
            common_keywords: Keyword arguments shared by all observations, in particular
                `data_field_names`, the fields to read.
            shapes: Per-observation buffer shapes `(n_detectors, n_samples, n_intervals)`, in the
                order of the observations. When given, no observation is opened to size the
                buffers; otherwise every observation is opened once here.
            known_failures: Positions known to be unreadable (their probe failed); read as
                filler, never loaded.
            **keywords: One list per keyword argument to the read function, one element per
                observation.
        """
        # Set before super().__init__ so the structure/reader builders can use them
        self.stokes = stokes
        self.dtype = dtype
        self.sample_dtype = sample_dtype or dtype
        # Distributed mode passes pre-gathered (n_detectors, n_samples) shapes; turn them into
        # structures here, now that self carries stokes/dtype. Otherwise leave them unset and
        # super() opens every observation via _read_structure_impure.
        structures = None
        if shapes is not None:
            fields = (common_keywords or {})['data_field_names']
            structures = [self._get_data_field_structures_for(shape, fields) for shape in shapes]
        super().__init__(
            *args,
            common_keywords=common_keywords,
            structures=structures,
            known_failures=known_failures,
            # each read is told its item, under which it records what it read
            item=range(len(args[0])),
            **keywords,
        )

    def reset_failures(self) -> None:
        """Forget the read failures, and the records of the items read, recorded so far."""
        super().reset_failures()
        self.records: dict[int, ObservationRecord] = {}

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[AbstractLazyObservation[T]],
        *,
        requested_fields: Collection[str] | None = None,
        stokes: ValidStokesLiteral | None = None,
        dtype: DTypeLike = jnp.float64,
        sample_dtype: DTypeLike | None = None,
        shapes: Sequence[ObservationBufferShape] | None = None,
        known_failures: Sequence[int] | None = None,
    ) -> Self:
        """Create a reader over the given observations.

        Args:
            observations: The lazy observations to read, one item each.
            requested_fields: Optional list of fields to load. If None, read all non-optional
                fields.
            stokes: Stokes components of the demodulated TOD; `None` for detector timestreams.
            dtype: Floating-point dtype applied to every floating-point field the reader
                returns: sample data, noise model fits and the geometry (timestamps, HWP
                angles, quaternions). Use `jnp.float32` to run a float32 mapmaking pipeline
                ([`MapMakingConfig.double_precision`][] set to `False`). Casting the geometry
                is also required there: under `jax_enable_x64=False` a float64 array is
                illegal, so no field may stay float64. Timestamps are rebased to a
                per-observation zero origin (in float64, before the downcast) so the float32
                cast does not collapse the absolute POSIX epoch onto a single value.
            sample_dtype: Floating-point dtype applied to the sample data instead; `None` for
                `dtype`. A float32 TOD takes half the memory of a float64 one, and loses nothing
                when it was recorded in float32.
            shapes: Per-observation buffer shapes, in the order of `observations`. When given,
                no observation is opened to size the buffers (the caller has already probed them,
                see [`AbstractLazyObservation.probe`][]); otherwise every observation is
                opened once here.
            known_failures: Positions in `observations` known to be unreadable (their probe
                failed); read as filler, never loaded.

        Returns:
            A reader whose items are `observations`, in order.
        """
        fields = cls._resolve_fields(observations, requested_fields)
        return cls(
            observations,
            common_keywords={'data_field_names': fields},
            stokes=stokes,
            dtype=dtype,
            sample_dtype=sample_dtype,
            shapes=list(shapes) if shapes is not None else None,
            known_failures=known_failures,
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

        - `sample_data`: padded with 0.0 outside the valid samples
        - `timestamps`, `hwp_angles`: extrapolated in the padded region so that the sample rate
          and the hwp rotation frequency remain consistent
        - `valid_sample_masks`, `valid_scanning_masks`: padded with 0 (False) outside the valid
          samples
        - `detector_quaternions`, `boresight_quaternions`: padded with zero quaternions, which
          mark samples without pointing: the pointing operator neither reads nor bins them.
        - `noise_model_fits`: padded with (sigma, alpha, fknee, f0) = (0., 0., 1., 0.1)

        Args:
            data: The fields read for one observation.
            padding: Per-field padding widths, as returned by the base reader.

        Returns:
            The padded fields.
        """
        # The sample data already sits in the leading corner of a zeroed buffer of the padded
        # shape (see `get_sample_data`): the corner starts the buffer and has its strides, so a
        # view of the padded shape from there spans the whole buffer.
        sample_data = data.get(ReaderField.SAMPLE_DATA)
        others = [field for field in data if field != ReaderField.SAMPLE_DATA]

        # First, pad them with 0 by default
        data = super()._pad(
            {field: data[field] for field in others}, {field: padding[field] for field in others}
        )
        if sample_data is not None:
            (struct,) = jax.tree.leaves(self.out_structure[ReaderField.SAMPLE_DATA])
            data[ReaderField.SAMPLE_DATA] = jax.tree.map(
                lambda tod: np.lib.stride_tricks.as_strided(tod, struct.shape, tod.strides),
                sample_data,
            )

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
        for angle_field in (ReaderField.AZIMUTH, ReaderField.ELEVATION):
            if angle_field in data_field_names:
                # Hold the last valid value (telescope stopped moving). Keeps min/ptp of the
                # scan unchanged, so Legendre/bin basis normalisation is unaffected.
                pad_size = padding[angle_field][0]
                valid = data[angle_field][: data[angle_field].size - pad_size]
                data[angle_field] = np.pad(valid, (0, pad_size), mode='edge')
        if ReaderField.SCANNING_INTERVALS in data_field_names:
            # Pad with zero-width [end, end) intervals anchored at the last real interval's
            # end, not [0, 0]: keeps `starts` non-decreasing, and a zero-width interval never
            # contains a real sample.
            pad_size = padding[ReaderField.SCANNING_INTERVALS][0]
            intervals = data[ReaderField.SCANNING_INTERVALS]
            valid = intervals[: intervals.shape[0] - pad_size]
            last_end = valid[-1, 1]
            data[ReaderField.SCANNING_INTERVALS] = np.concatenate(
                [valid, np.full((pad_size, 2), last_end, dtype=intervals.dtype)]
            )
        if ReaderField.NOISE_MODEL_FITS in data_field_names:

            def _pad_noise_fits(arr: np.ndarray) -> np.ndarray:
                default = np.array([0.0, 0.0, 1.0, 0.1], dtype=arr.dtype)
                zero_padded = arr[..., 0] == 0.0
                return np.where(zero_padded[..., None], default, arr)

            data[ReaderField.NOISE_MODEL_FITS] = jax.tree.map(
                _pad_noise_fits, data[ReaderField.NOISE_MODEL_FITS]
            )

        return data

    def _get_data_field_structures_for(
        self,
        shape: tuple[int, ...],
        fields: Collection[str] | None = None,
    ) -> PyTree[jax.ShapeDtypeStruct]:
        """Build the padded-buffer structures for one observation from its probed shape.

        Restricted to `fields` when given, else returns every supported field.
        """
        n_detectors, n_samples, n_intervals = shape
        stokes = self.stokes
        dtype = self.dtype
        tod_format = TODFormat((n_detectors, n_samples), self.sample_dtype, stokes=stokes)

        structures: dict[str, PyTree[jax.ShapeDtypeStruct]] = {
            ReaderField.METADATA: HashedObservationMetadata.structure_for(n_detectors),
            ReaderField.SAMPLE_DATA: tod_format.structure,
            ReaderField.VALID_SAMPLE_MASKS: jax.ShapeDtypeStruct(
                (n_detectors, n_samples), jnp.bool
            ),
            ReaderField.VALID_SCANNING_MASKS: jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            ReaderField.TIMESTAMPS: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.HWP_ANGLES: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.DETECTOR_QUATERNIONS: jax.ShapeDtypeStruct((n_detectors, 4), dtype),
            ReaderField.BORESIGHT_QUATERNIONS: jax.ShapeDtypeStruct((n_samples, 4), dtype),
            ReaderField.NOISE_MODEL_FITS: (
                jax.ShapeDtypeStruct((n_detectors, 4), dtype)
                if stokes is None
                else jax.ShapeDtypeStruct((len(stokes), n_detectors, 4), dtype)
            ),
            ReaderField.AZIMUTH: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.ELEVATION: jax.ShapeDtypeStruct((n_samples,), dtype),
            ReaderField.LEFT_SCAN_MASK: jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            ReaderField.RIGHT_SCAN_MASK: jax.ShapeDtypeStruct((n_samples,), jnp.bool),
            ReaderField.SCANNING_INTERVALS: jax.ShapeDtypeStruct((n_intervals, 2), jnp.int32),
        }
        if fields is None:
            return structures
        return {field: structures[field] for field in fields}

    def _get_data_field_readers(self):
        def if_none_raise_error(x: Any) -> Any:
            if x is None:
                raise ValueError('Data field not available')
            return x

        stokes = self.stokes
        target_dtype = self.dtype
        sample_dtype = self.sample_dtype

        def cast(x: Any) -> Any:
            return x.astype(target_dtype, copy=False)

        def cast_bool(x: Any) -> Any:
            return x.astype(bool, copy=False)

        def get_sample_data(obs: AbstractObservation[T]) -> Any:
            # The getter writes straight into a zeroed buffer of the padded shape, at the sample
            # dtype, so that the TOD is copied once. `_pad` takes the whole buffer back.
            (struct,) = jax.tree.leaves(self.out_structure[ReaderField.SAMPLE_DATA])
            buffer = _aligned_zeros(struct.shape, sample_dtype)
            tods: Stokes | np.ndarray
            if stokes is None:
                tods = obs.get_tods(out=buffer)
            else:
                tods = obs.get_demodulated_tods(stokes=stokes, out=buffer)

            def in_buffer(tod: np.ndarray) -> np.ndarray:
                corner = buffer[tuple(slice(n) for n in tod.shape)]
                if not np.may_share_memory(corner, tod):  # a getter that ignored `out`
                    corner[...] = tod
                return corner

            return jax.tree.map(in_buffer, tods)

        def get_noise_model_fits(obs: AbstractObservation[T]) -> Any:
            if stokes is None:
                model = if_none_raise_error(obs.get_noise_model())
            else:
                model = obs.get_demodulated_noise_model(stokes=stokes)
            fits = model.to_array()
            return jax.tree.map(cast, fits)

        def get_timestamps(obs: AbstractObservation[T]) -> Any:
            # The pipeline uses only time differences, so anchor each observation at its
            # own start. Subtracting in float64 keeps the samples resolved when cast to
            # float32; the absolute epoch is read from the interface where needed.
            timestamps = np.asarray(obs.get_timestamps(), dtype=np.float64)
            return cast(timestamps - timestamps[0])

        return {
            ReaderField.METADATA: lambda obs: HashedObservationMetadata.from_observation(obs),
            ReaderField.SAMPLE_DATA: get_sample_data,
            ReaderField.VALID_SAMPLE_MASKS: lambda obs: obs.get_sample_mask(),
            ReaderField.VALID_SCANNING_MASKS: lambda obs: obs.get_scanning_mask(),
            ReaderField.TIMESTAMPS: get_timestamps,
            ReaderField.HWP_ANGLES: lambda obs: cast(obs.get_hwp_angles()),
            ReaderField.DETECTOR_QUATERNIONS: lambda obs: cast(obs.get_detector_quaternions()),
            ReaderField.BORESIGHT_QUATERNIONS: lambda obs: cast(obs.get_boresight_quaternions()),
            ReaderField.NOISE_MODEL_FITS: get_noise_model_fits,
            ReaderField.AZIMUTH: lambda obs: cast(obs.get_azimuth()),
            ReaderField.ELEVATION: lambda obs: cast(obs.get_elevation()),
            ReaderField.LEFT_SCAN_MASK: lambda obs: cast_bool(obs.get_left_scan_mask()),
            ReaderField.RIGHT_SCAN_MASK: lambda obs: cast_bool(obs.get_right_scan_mask()),
            ReaderField.SCANNING_INTERVALS: lambda obs: np.asarray(
                obs.get_scanning_intervals(), dtype=np.int32
            ),
        }

    def _failure_filler(self) -> dict[str, Any]:
        """Finite, `out_structure`-shaped data for an observation that could not be read.

        The values are finite and non-degenerate (identity quaternions, a strictly increasing time
        vector, a broadband signal, unit white-noise fits) so every operator built from this
        observation stays finite. The observation is gated out (all-False masker), so these values
        never enter the maps; finiteness only matters so the gated contribution is `0` and not
        `NaN`.
        """
        rng = np.random.default_rng(0)

        def identity_quaternions(s: jax.ShapeDtypeStruct) -> np.ndarray:
            quats = np.zeros(s.shape, s.dtype)
            quats[..., 0] = 1.0  # (1, 0, 0, 0)
            return quats

        def unit_noise_fit(s: jax.ShapeDtypeStruct) -> np.ndarray:
            fit = np.zeros(s.shape, s.dtype)  # columns: white_noise, alpha, fknee, f0
            fit[..., 0] = 1.0  # unit white noise so the inverse noise covariance stays finite
            fit[..., 2] = 1.0
            fit[..., 3] = 0.1
            return fit

        def zeros(s: jax.ShapeDtypeStruct) -> np.ndarray:
            return np.zeros(s.shape, s.dtype)

        def fill(field: str, struct: PyTree[jax.ShapeDtypeStruct]) -> Any:
            if field in (ReaderField.TIMESTAMPS, ReaderField.AZIMUTH, ReaderField.ELEVATION):
                # strictly increasing: non-zero range so Legendre/bin normalisation of azimuth (and
                # the time axis) stays finite when template operators are built from this filler
                return np.arange(struct.shape[0], dtype=struct.dtype)
            if field in (ReaderField.DETECTOR_QUATERNIONS, ReaderField.BORESIGHT_QUATERNIONS):
                return identity_quaternions(struct)
            if field == ReaderField.SAMPLE_DATA:
                # broadband so a fitted noise model has non-zero white-noise level
                return jax.tree.map(lambda s: rng.standard_normal(s.shape).astype(s.dtype), struct)
            if field == ReaderField.NOISE_MODEL_FITS:
                return jax.tree.map(unit_noise_fit, struct)
            # metadata, masks (all-False), hwp angles: plain zeros per leaf
            return jax.tree.map(zeros, struct)

        return {field: fill(field, struct) for field, struct in self.out_structure.items()}

    def _read_structure_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str], item: int
    ) -> PyTree[jax.ShapeDtypeStruct]:
        return self._get_data_field_structures_for(
            observation.probe(intervals=ReaderField.SCANNING_INTERVALS in data_field_names).shape,
            data_field_names,
        )

    def _read_data_impure(
        self, observation: AbstractLazyObservation[T], data_field_names: Collection[str], item: int
    ) -> PyTree[Array]:
        start = time.perf_counter()
        data = observation.get_data(data_field_names)
        field_reader = self._get_data_field_readers()
        fields = {field: field_reader[field](data) for field in data_field_names}
        # Items are read concurrently, each recording under its own key. The data was read, so a
        # failure to record it must not make the read fail.
        try:
            record = ObservationRecord.from_fields(
                data, fields, time.perf_counter() - start, observation.name
            )
        except Exception as error:
            logger.exception('Read item %d: could not record what was read', item)
            record = ObservationRecord(
                observation.name, [], error=f'{type(error).__name__}: {error}'
            )
        self.records[item] = record
        logger.info(f'Read {observation.name} ({time.perf_counter() - start:.1f} s)')
        return fields


def _aligned_zeros(shape: tuple[int, ...], dtype: DTypeLike) -> np.ndarray:
    """A zeroed array on fresh memory pages, which a CPU device can use without copying it.

    The device copies a host buffer unless it is suitably aligned (64 bytes), which a NumPy
    allocation is not guaranteed to be; a page-aligned one always is.
    """
    dtype = np.dtype(dtype)
    pages = mmap.mmap(-1, prod(shape) * dtype.itemsize)  # anonymous: zeroed by the system
    return np.frombuffer(pages, dtype).reshape(shape)


class TileReader:
    """Reads the detector tiles of the observations of an [`ObservationReader`][].

    An observation is read when the first of its tiles is requested, and kept on the host until
    the last one has been: its tiles are views of its fields. The tiles of an observation are
    requested one after the other, but requests for different observations may come from several
    threads, and the per-observation locks keep a read from starting twice.

    Args:
        reader: The reader of the observations.
        tile_size: Detectors per tile.
        tile_counts: The number of tiles that will be read from each observation of the reader,
            in its order; an observation is released after that many.

    Attributes:
        structure: The structure of what [`read`][furax.mapmaking._reader.TileReader.read]
            returns: the fields of a tile, their padding widths, and whether the read succeeded.
        strides: The stride of each leaf of `structure` in an ensemble of tiles.
        filler: What a tile that holds no detectors reads: finite filler.
    """

    def __init__(
        self, reader: ObservationReader[Any], tile_size: int, tile_counts: Sequence[int]
    ) -> None:
        self.reader = reader
        self.tile_size = tile_size
        self._remaining = list(tile_counts)
        self._held: dict[int, tuple[Any, Any, np.ndarray]] = {}
        self._locks = [threading.Lock() for _ in tile_counts]
        self._lock = threading.Lock()

        fields, field_strides = detector_tile_structure(reader.out_structure, tile_size)
        padding = reader._padding_structure()
        self.structure = (fields, padding, jax.ShapeDtypeStruct((), bool))
        # the padding widths count the tile's detectors, and the read succeeds or fails per
        # observation
        self.strides = (field_strides, jax.tree.map(lambda _: 1, padding), 'segment')
        self.filler = self._tile(reader.read_filler_host(), 0)

    def read(self, observation: int, start: int) -> tuple[Any, Any, np.ndarray]:
        """The tile of `tile_size` detectors from `start` of an observation of the reader.

        Args:
            observation: The position of the observation in the reader.
            start: The first detector of the tile.

        Returns:
            The tile, as described by `structure`.
        """
        with self._locks[observation]:
            read = self._held.get(observation)
            if read is None:
                read = self.reader.read_host(observation)
                self._held[observation] = read
        with self._lock:
            self._remaining[observation] -= 1
            if self._remaining[observation] == 0:
                del self._held[observation]
        return self._tile(read, start)

    def _tile(self, read: tuple[Any, Any, np.ndarray], start: int) -> tuple[Any, Any, np.ndarray]:
        data, padding, use = read
        data, padding = detector_tile(data, padding, start, self.tile_size)
        return data, padding, use


def detector_tile(
    data: dict[str, Any], padding: dict[str, Any], start: int, size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The fields read for one observation restricted to `size` detectors from `start`, on the host.

    Shared fields (timestamps, boresight, scan masks, ...) are kept whole. The per-detector fields
    are sliced along their detector axis, as views, and their padding widths along it count the
    detectors of the tile past those read.

    Args:
        data: The fields read for one observation, padded to the reader's buffers.
        padding: Their padding widths, one per axis of each field, as read.
        start: The first detector of the tile.
        size: The detectors of the tile.

    Returns:
        The tile's fields and their padding widths.
    """

    def past_read(read: int) -> int:
        """The detectors of the tile past the `read` ones of the observation."""
        return size - min(max(read - start, 0), size)

    def tile_padding(widths: np.ndarray, leaf: np.ndarray) -> np.ndarray:
        widths = np.array(widths)
        widths[-2] = past_read(leaf.shape[-2] - widths[-2])
        return widths

    fields, widths = {}, {}
    for field, value in data.items():
        if ReaderField(field).per_detector:
            fields[field] = jax.tree.map(lambda x: x[..., start : start + size, :], value)
            widths[field] = jax.tree.map(tile_padding, padding[field], value)
        else:
            fields[field] = value
            widths[field] = padding[field]
    if (metadata := data.get(ReaderField.METADATA)) is not None:
        uids = metadata.detector_uids
        fields[ReaderField.METADATA] = replace(metadata, detector_uids=uids[start : start + size])
        uid_widths = np.array(padding[ReaderField.METADATA].detector_uids)
        uid_widths[-1] = past_read(len(uids) - uid_widths[-1])
        widths[ReaderField.METADATA] = replace(
            padding[ReaderField.METADATA], detector_uids=uid_widths
        )
    return fields, widths


def detector_tile_structure(
    structure: dict[str, Any], size: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The structure of a tile of `size` detectors of the fields of `structure`, with strides.

    The stride of each leaf is that of an ensemble of tiles: 1 for a leaf of the tile alone,
    `'segment'` for one the tiles of its observation share.
    """
    # the structure of `detector_tile`'s output; the buffers hold whole tiles, so the first is full
    padding = jax.tree.map(lambda leaf: np.zeros(len(leaf.shape), dtype=np.int64), structure)
    tiles = jax.eval_shape(lambda data: detector_tile(data, padding, 0, size)[0], structure)

    strides = {}
    for field, value in tiles.items():
        stride = 1 if ReaderField(field).per_detector else 'segment'
        strides[field] = jax.tree.map(lambda _, stride=stride: stride, value)
    if ReaderField.METADATA in tiles:
        strides[ReaderField.METADATA] = replace(strides[ReaderField.METADATA], detector_uids=1)
    return tiles, strides
