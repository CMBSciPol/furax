"""Logic to group observations into buckets of similar buffer shapes.

[`MultiObservationMapMaker`][] runs one jit-compiled program over multiple observations, so they
are required to share a common buffer shape. In reality they do not: they differ in detector and
sample count. This module provides tools to group them into *buckets*, i.e., groups of observations
padded to a common buffer shape. The mapmaker then streams each bucket separately through the
available devices.

Two kinds of padding are required:

- *shape padding*: every observation of the group grows to the bucket's **envelope**, the per-axis
  maximum over the group. A 10-minute scan bucketed with a 60-minute one occupies a 60-minute
  buffer;
- *slot padding*: the group is rounded up to a whole number of **slots** per device, so the stream
  axis divides evenly over them. Nine observations on four devices take twelve slots, three of
  them empty.

The **observation index** is the global position of an observation in the list given to the
mapmaker. Within a bucket, the local index of that observation is called the **item index**.
A **slot** is where one observation's data lives once the bucket is sharded over the devices.
A slot may contain a fake observation with the same bucket-level envelope (slot padding).

!!! example

    Six observations of two very different lengths, in two buckets, on four devices held by two
    processes:

    ```text
    observation shapes (detectors, samples):
      0: (100, 4900)   1: (100, 5000)   2: ( 90, 1200)
      3: (100, 4800)   4: (110, 5000)   5: (100,  900)

    bucket A, envelope (100, 1200)              bucket B, envelope (110, 5000)
    slot  process  device  item  observation    slot  process  device  item  observation
      0    proc 0   dev 0    0        2           0    proc 0   dev 0    0        0
      1    proc 0   dev 1    1        5           1    proc 0   dev 1    1        1
      2    proc 1   dev 2    -        -           2    proc 1   dev 2    2        3
      3    proc 1   dev 3    -        -           3    proc 1   dev 3    3        4
    ```

!!! tip "Choosing the number of buckets"

    [`partition_padded`][] chooses the number of buckets within the `max_buckets` budget, given the
    available device count, optimising for total padded volume. A larger budget can only lower the
    volume it settles on, but additional buckets cost compile time: each one is traced and compiled
    for its own envelope. A rule of thumb is to allow one bucket per distinct observation shape.

!!! tip "Choosing the device count"

    Although more devices can in general increase throughput, remember that each bucket rounds up
    its slots to the device count (slot padding). A device count just above a divisor of the bucket
    sizes wastes almost a full round of slots, and a run with fewer observations than devices leaves
    empty slots in every bucket. To keep slot padding low, try to size the job so its device count
    divides the number of observations.
"""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np

from ._observation import ObservationBufferShape

__all__ = [
    'Bucket',
    'SlotLayout',
    'padded_volume',
    'partition_padded',
    'real_volume',
]


@dataclass(frozen=True, eq=False)
class Bucket:
    """One group of observations, streamed together through every device.

    Every slot of the bucket holds a buffer of the same shape, the `envelope` of the group.
    `n_slots` is set to a value that fills the device count evenly, so its value is at least
    the number of real observations. Empty (fake) slots are always at the end.

    Buckets are normally built for a whole run by [`SlotLayout.create`][], not one by one.

    Attributes:
        observations: Observation indices (global), sorted.
        n_slots: The number of slots; a multiple of the device count.
        envelope: The per-axis maximum buffer shape over the group.
    """

    observations: np.ndarray
    n_slots: int
    envelope: ObservationBufferShape

    @classmethod
    def create(
        cls, shapes: Sequence[ObservationBufferShape], group: Collection[int], n_devices: int = 1
    ) -> Self:
        """Bucket a group of observations, padding them to their common envelope.

        Args:
            shapes: Per-observation buffer shapes.
            group: Indices of the observations in the bucket.
            n_devices: Number of devices the bucket is sharded over.

        Returns:
            The bucket, with its observations sorted.

        Raises:
            ValueError: If `group` is empty.

        Examples:
            A short and a long observation bucketed together: both slots are as long as the
            longest and as wide as the widest.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> bucket = Bucket.create([Shape(2, 100), Shape(3, 10)], [1, 0])
            >>> bucket.envelope
            ObservationBufferShape(detector_count=3, sample_count=100, interval_count=0)
            >>> bucket.padded_volume, real_volume([Shape(2, 100), Shape(3, 10)])
            (600, 230)
        """
        if len(group) == 0:
            raise ValueError('a bucket needs at least one observation')
        envelope = ObservationBufferShape(
            max(shapes[i].detector_count for i in group),
            max(shapes[i].sample_count for i in group),
            max(shapes[i].interval_count for i in group),
        )
        observations = np.sort(np.asarray(group, dtype=np.int64))
        return cls(observations, cls.slot_count(len(group), n_devices), envelope)

    @staticmethod
    def slot_count(size: int, n_devices: int = 1) -> int:
        """Slots a bucket of `size` observations takes: rounded up to fill `n_devices` evenly."""
        return -(-size // n_devices) * n_devices

    @property
    def n_real(self) -> int:
        """Number of observations in the bucket, i.e. of slots holding real data."""
        return len(self.observations)

    @property
    def n_pad(self) -> int:
        """Number of empty slots."""
        return self.n_slots - self.n_real

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the bucket occupies once padded."""
        return self.n_slots * self.envelope.volume

    @cached_property
    def is_real(self) -> np.ndarray:
        """Per slot, whether it holds an observation (`False` for the empty slots at the end)."""
        return np.arange(self.n_slots) < self.n_real

    def summary(self, n_devices: int = 1) -> str:
        """The bucket's slot bookkeeping, as `obs=... slots=... pad=... envelope=...`."""
        return (
            f'obs={self.n_real} slots={self.n_slots} pad={self.n_pad} '
            f'slots_per_dev={self.n_slots // n_devices} '
            f'envelope=({self.envelope.detector_count}, {self.envelope.sample_count})'
        )

    @cached_property
    def item_of_slot(self) -> np.ndarray:
        """Per slot, the reader item to load there.

        Empty slots repeat the last item so the index is always valid.
        """
        return np.minimum(np.arange(self.n_slots), self.n_real - 1)


def real_volume(shapes: Sequence[ObservationBufferShape]) -> int:
    """Total volume of the observations, without padding.

    Args:
        shapes: Per-observation buffer shapes.
    """
    return sum(s.volume for s in shapes)


def padded_volume(
    shapes: Sequence[ObservationBufferShape], groups: Iterable[Collection[int]], n_devices: int = 1
) -> int:
    """Total padded volume of a partition, summed over its buckets.

    This is the quantity [`partition_padded`][] minimises.

    Args:
        shapes: Per-observation buffer shapes.
        groups: The groups of observation indices, e.g. as returned by [`partition_padded`][].
        n_devices: Number of devices each group is sharded over.
    """
    return sum(Bucket.create(shapes, g, n_devices).padded_volume for g in groups)


def partition_padded(
    shapes: Sequence[ObservationBufferShape], max_groups: int, *, n_devices: int = 1
) -> list[list[int]]:
    r"""Split observations into at most `max_groups` groups, minimising padding overhead.

    A group $b$ of $n_b$ observations is padded to a common envelope, and rounded up to a whole
    number of slots per device, so it occupies $n_b^\text{slots} d_b s_b$ elements, where
    $n_b^\text{slots} = \lceil n_b / n_\text{dev} \rceil \, n_\text{dev}$ and $d_b$, $s_b$ are
    the largest detector and sample counts in the group. Every device processes every group, so
    the run pays $\sum_b n_b^\text{slots} d_b s_b$, which is the quantity minimised here.

    For efficiency reasons, candidate groups are runs of the observations sorted by shape, so the
    partition is not always the best possible.

    Args:
        shapes: Per-observation buffer shapes.
        max_groups: Largest number of groups allowed (at most one per observation).
        n_devices: Number of devices each group is sharded over.

    Returns:
        A list of groups, from one up to `max_groups` of them. Each group is a sorted list of
        original observation indices. The groups are ordered by increasing detector count, then
        sample count, and together they cover `range(len(shapes))`.

    Raises:
        ValueError: If `max_groups < 1`, `n_devices < 1` or there are no observations.

    Examples:
        Three short scans, one long scan: the long one is left alone rather than padding the
        short ones up to it.

        >>> from furax.mapmaking import ObservationBufferShape as Shape
        >>> shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
        >>> partition_padded(shapes, max_groups=2)
        [[0, 1, 2], [3]]

        On four devices a group of one costs four slots of the long scan, so one group is cheaper:

        >>> partition_padded(shapes, max_groups=2, n_devices=4)
        [[0, 1, 2, 3]]

        Observations are grouped by detector count first, so the wide observation is not padded
        onto the narrow ones:

        >>> shapes = [Shape(1, 10), Shape(10, 20), Shape(1, 30)]
        >>> partition_padded(shapes, max_groups=2)
        [[0, 2], [1]]
    """
    if max_groups < 1:
        raise ValueError(f'max_groups must be >= 1, got {max_groups}')
    if n_devices < 1:
        raise ValueError(f'n_devices must be >= 1, got {n_devices}')
    n = len(shapes)
    if n == 0:
        raise ValueError('need at least one observation')
    max_groups = min(max_groups, n)

    # Sort by detector count first, then by sample count. Observations with the same detector
    # count stay together, so a group usually pads along the sample axis only. When every
    # observation has the same detector count, this is just sorting by length. Considering only
    # consecutive runs keeps the search polynomial; the unrestricted problem is NP-hard.
    order = sorted(range(n), key=lambda i: (shapes[i].detector_count, shapes[i].sample_count))
    det = np.array([shapes[i].detector_count for i in order], dtype=np.int64)
    samp = np.array([shapes[i].sample_count for i in order], dtype=np.int64)
    positions = np.arange(n)

    # dp[b, j] is the lowest total cost of splitting the first j observations into b groups;
    # cut[b, j] records where the last of those groups starts. Unreachable states hold `inf`,
    # kept far from the int64 limit so that adding a group cost to one cannot overflow.
    inf = np.int64(1) << 62
    dp = np.full((max_groups + 1, n + 1), inf, dtype=np.int64)
    cut = np.zeros((max_groups + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0

    for b in range(1, max_groups + 1):
        for j in range(b, n + 1):
            # Cost of a last group [i, j) for every start i < j at once, i.e. `Bucket.create`
            # vectorised over the candidates: the envelope is a suffix maximum of the sorted
            # shapes, the slot count the rounding of `Bucket.slot_count` applied to j - i.
            run_det = np.maximum.accumulate(det[j - 1 :: -1])[::-1]
            run_samp = np.maximum.accumulate(samp[j - 1 :: -1])[::-1]
            slots = -(-(j - positions[:j]) // n_devices) * n_devices
            total = dp[b - 1, :j] + slots * run_det * run_samp
            i = int(np.argmin(total[b - 1 :])) + b - 1  # earlier groups need b - 1 observations
            dp[b, j] = total[i]
            cut[b, j] = i

    # Fewest groups among the cheapest: argmin returns the first minimum.
    n_groups = int(np.argmin(dp[1:, n])) + 1

    groups: list[list[int]] = []
    j = n
    for b in range(n_groups, 0, -1):
        i = int(cut[b, j])
        groups.append(sorted(order[i:j]))
        j = i
    groups.reverse()
    return groups


@dataclass(frozen=True, eq=False)
class SlotLayout:
    """Complete layout of slots for a given dataset and job resources."""

    buckets: tuple[Bucket, ...]
    """The buckets, ordered by increasing envelope shape."""
    n_observations: int
    """Total number of (real) observations."""
    n_devices: int
    """Total number of devices in the whole job."""

    @classmethod
    def create(
        cls, shapes: Sequence[ObservationBufferShape], *, n_devices: int, max_buckets: int
    ) -> Self:
        """Choose the layout to minimise padding; see [`partition_padded`][].

        Args:
            shapes: Per-observation buffer shapes.
            n_devices: Number of devices (`jax.device_count()`).
            max_buckets: Largest number of buckets allowed.

        Examples:
            Three short scans and one long one on two devices.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
            >>> layout = SlotLayout.create(shapes, n_devices=2, max_buckets=4)
            >>> print(layout)
            SlotLayout: obs=4 buckets=2 slots=4 devices=2 slot_overhead=+0.0%
              bucket 0: obs=2 slots=2 pad=0 slots_per_dev=1 envelope=(2, 10)
              bucket 1: obs=2 slots=2 pad=0 slots_per_dev=1 envelope=(2, 90)

            The dataset is heterogeneous enough that the layout still costs two thirds more
            memory than the observations themselves:

            >>> round(layout.padded_volume / real_volume(shapes), 2)
            1.67
        """
        groups = partition_padded(shapes, max_buckets, n_devices=n_devices)
        buckets = tuple(Bucket.create(shapes, group, n_devices) for group in groups)
        return cls(buckets, len(shapes), n_devices)

    @property
    def n_slots(self) -> int:
        """Total number of slots, over every bucket."""
        return sum(bucket.n_slots for bucket in self.buckets)

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the run occupies once padded, summed over the buckets."""
        return sum(bucket.padded_volume for bucket in self.buckets)

    @property
    def slot_overhead(self) -> float:
        """Empty slots as a fraction of the observation count."""
        return (self.n_slots - self.n_observations) / self.n_observations

    def summary(self) -> str:
        """The run's totals, as `obs=... buckets=... slots=... devices=... slot_overhead=...`."""
        return (
            f'obs={self.n_observations} buckets={len(self.buckets)} slots={self.n_slots} '
            f'devices={self.n_devices} slot_overhead=+{self.slot_overhead:.1%}'
        )

    def __str__(self) -> str:
        lines = [f'{type(self).__name__}: {self.summary()}']
        lines += [
            f'  bucket {b}: {bucket.summary(self.n_devices)}'
            for b, bucket in enumerate(self.buckets)
        ]
        return '\n'.join(lines)

    def local_slots(self, bucket: int, *, process_index: int, n_local: int) -> slice:
        """The slots of a bucket held by one process's devices.

        The mesh orders devices by process, so a process's shards of the stream axis are one
        contiguous block: ``n_local`` shards of ``n_slots / n_devices`` slots each.

        Args:
            bucket: Index of the bucket.
            process_index: The process.
            n_local: Number of devices on every process.
        """
        n_per_device = self.buckets[bucket].n_slots // self.n_devices
        start = process_index * n_local * n_per_device
        return slice(start, start + n_local * n_per_device)

    def to_observation_order(self, per_bucket: Sequence[np.ndarray]) -> np.ndarray:
        """Reorder per-slot arrays, one per bucket, into a single per-observation array.

        This undoes the bucketing for results the caller wants per observation, in their original
        order -- per-observation diagnostics, for instance.

        The leading axis of each input runs over the bucket's slots; the empty slots are dropped.
        Trailing axes may differ between buckets (each pads to its own shape), so the result is
        padded with zeros to the largest.

        Args:
            per_bucket: One array per bucket, of shape ``(n_slots, ...)``.

        Returns:
            An array of shape ``(n_observations, ...)`` in observation order.
        """
        if len(per_bucket) != len(self.buckets):
            raise ValueError(f'expected {len(self.buckets)} arrays, got {len(per_bucket)}')
        trailing = tuple(max(dims) for dims in zip(*(a.shape[1:] for a in per_bucket), strict=True))
        dtype = np.result_type(*(a.dtype for a in per_bucket))
        out = np.zeros((self.n_observations, *trailing), dtype=dtype)
        for bucket, values in zip(self.buckets, per_bucket, strict=True):
            real = values[: bucket.n_real]
            region = (bucket.observations, *(slice(0, n) for n in real.shape[1:]))
            out[region] = real
        return out
