"""What a multi-observation run knows of its observations and detectors.

Each observation is probed and read by one process. The rows each process fills are gathered by
`gather_rows`, so that every process holds the same catalog.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from fastquat import Quaternion
from jax.experimental import multihost_utils as mhu
from jaxtyping import Bool, Float64, Int64

from furax.math.coords import XiEtaAngles

from ._observation import ObservationBufferShape
from ._reader import ObservationRecord
from .results import _OBSERVATIONS_SCHEMA

type Read = ObservationRecord | str
"""What a process read of an observation, or the exception that stopped its load."""


@dataclass
class ObservationCatalog:
    """What a run knows of each of its observations, keyed by observation index.

    The probes are known from the start; each read pass replaces `reads`.
    """

    names: list[str]
    """The names of the observations, as given, in observation order."""
    probes: dict[int, ObservationBufferShape | str]
    """The shape probed of each observation, or the exception its probe raised."""
    reads: dict[int, Read] = field(default_factory=dict)
    """What the last pass read of each observation, or the exception that stopped its load; an
    observation not loaded, such as one whose probe failed, is absent."""

    @property
    def probed(self) -> Int64[np.ndarray, 'obs 3']:
        """The detector, sample and interval counts probed; zero for a failed probe."""
        probed = np.zeros((len(self.names), 3), np.int64)
        for i, probe in self.probes.items():
            if not isinstance(probe, str):
                probed[i] = probe
        return probed

    @property
    def probe_failed(self) -> Bool[np.ndarray, ' obs']:
        """Per observation, whether its probe raised."""
        return np.array([isinstance(self.probes.get(i), str) for i in range(len(self.names))])

    @property
    def n_failed(self) -> int:
        """The number of observations not read."""
        n_read = sum(isinstance(read, ObservationRecord) for read in self.reads.values())
        return len(self.names) - n_read

    def to_table(self) -> pl.DataFrame:
        """The catalog as `MapMakingResults.observations` holds it."""
        rows = [self._row(i) for i in range(len(self.names))]
        return pl.DataFrame(rows, schema=_OBSERVATIONS_SCHEMA, orient='row')

    def _row(self, i: int) -> dict[str, Any]:
        probe = self.probes.get(i)
        read = self.reads.get(i)
        row: dict[str, Any] = {
            'name': self.names[i],
            'loaded': False,
            'n_detectors': 0,
            'n_samples': 0,
            'probed_detectors': 0,
            'probed_samples': 0,
            'valid_sample_fraction': None,
            'load_seconds': None,
            'error': None,
        }
        if isinstance(probe, ObservationBufferShape):
            row |= {'probed_detectors': probe.detector_count, 'probed_samples': probe.sample_count}
        if isinstance(read, ObservationRecord):
            row |= {
                'name': read.name,
                'loaded': True,
                'n_detectors': read.n_detectors,
                'n_samples': read.n_samples,
                'valid_sample_fraction': read.valid_sample_fraction,
                'load_seconds': read.load_seconds,
                'error': read.error,
            }
        if isinstance(probe, str):
            row['error'] = f'probe failed: {probe}'
        elif isinstance(read, str):
            row['error'] = f'load failed: {read}'
        return row


def detector_table(
    reads: dict[int, Read],
    noise: dict[str, np.ndarray] | None = None,
    stokes: str | None = None,
) -> pl.DataFrame:
    """One row per detector read, in observation then detector order.

    This is the table `MapMakingResults.detectors` holds. A detector is keyed by its indices along
    the observation and detector axes of the per-detector products, such as the amplitudes of a
    template with one set per detector.

    Args:
        reads: What was read of each observation, keyed by observation index.
        noise: Each noise model parameter, `(n_observations, [stream,] detector)`.
        stokes: The streams of a demodulated TOD, which suffix the noise parameters; `None`
            otherwise.
    """
    records = [
        (i, r)
        for i, r in sorted(reads.items())
        if isinstance(r, ObservationRecord) and r.n_detectors
    ]
    n = [r.n_detectors for _, r in records]
    observation = np.repeat(np.array([i for i, _ in records], np.int64), n)
    detector = np.concatenate([np.arange(k) for k in n] or [np.zeros(0, np.int64)])
    columns: dict[str, Any] = {
        'observation': observation,
        'detector': detector,
        'name': [name for _, r in records for name in r.detectors],
    }
    coordinates = _focal_plane_coordinates([r for _, r in records])
    for axis, values in zip(('xi', 'eta', 'gamma'), coordinates, strict=True):
        columns[axis] = pl.Series(axis, values).fill_nan(None)  # null where not read
    for name, values in (noise or {}).items():
        if values.ndim == 2:
            columns[name] = values[observation, detector]
            continue
        assert stokes is not None and values.shape[1] == len(stokes)  # one row per stream
        for k, stream in enumerate(stokes.lower()):
            columns[f'{name}_{stream}'] = values[observation, k, detector]
    return pl.DataFrame(columns)


def gather_rows[R](local: dict[int, R]) -> dict[int, R]:
    """The rows held by every process, merged, on every process.

    Each row is held by one process, under its observation index; a process may hold none. The
    rows are pickled, so they can be any Python values.
    """
    payload = np.frombuffer(pickle.dumps(local), dtype=np.uint8)
    # every process sends a buffer of the same length, the longest
    n_proc = jax.process_count()
    sizes = np.asarray(mhu.process_allgather(np.array([payload.size]))).reshape(n_proc)
    padded = np.zeros(int(sizes.max()), dtype=np.uint8)
    padded[: payload.size] = payload
    buffers = np.asarray(mhu.process_allgather(padded)).reshape(n_proc, -1)
    rows: dict[int, R] = {}
    for buffer, size in zip(buffers, sizes, strict=True):
        rows |= pickle.loads(buffer[:size].tobytes())
    return rows


def _focal_plane_coordinates(records: list[ObservationRecord]) -> Float64[np.ndarray, '3 det']:
    """The `(xi, eta, gamma)` of the detectors of all `records`.

    NaN for the detectors of an observation whose quaternions were not read. The quaternions are
    converted in one call, so the conversion compiles once, not once per number of detectors.
    """
    read = np.concatenate(
        [np.full(r.n_detectors, r.quaternions is not None) for r in records] or [np.zeros(0, bool)]
    )
    coordinates = np.full((3, read.size), np.nan)
    quaternions = [r.quaternions for r in records if r.quaternions is not None]
    if quaternions:
        q = Quaternion.from_array(jnp.asarray(np.concatenate(quaternions)))
        coordinates[:, read] = np.stack([np.asarray(a) for a in XiEtaAngles.from_quaternion(q)])
    return coordinates
