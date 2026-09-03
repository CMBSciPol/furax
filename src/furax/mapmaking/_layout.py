"""Where every observation sits on the device mesh, and what that costs in padding.

The multi-observation mapmaker streams observations through the devices in *buckets*: groups of
observations padded to a common buffer shape and laid out along one mesh axis that spans every
device of the job. A bucket pads twice: every observation to the bucket's envelope (the per-axis
maximum over the group), and the group to a whole number of slots per device. Both cost compute
and memory, so the grouping decides how much of a run is wasted on them.

This module owns that bookkeeping, so nothing else has to re-derive it. [`Bucket`][] is one
group with its padding cost, [`partition_padded`][] chooses the groups that waste the least,
[`PaddingReport`][] measures a grouping, and [`SlotLayout`][] maps between the index spaces of a
run:

- the *observation* index, the position in the mapmaker's list of observations;
- the *item* index of a bucket, the position in that bucket's reader (its observations in
  observation order);
- the *slot*, the position along the bucket's stream axis, which pads the items with empty slots
  up to a multiple of the device count;
- the *local block* of slots, the slots held by one process's devices.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple, Self

import numpy as np

from ._observation import ObservationBufferShape


@dataclass(frozen=True)
class Bucket:
    """One group of observations, streamed together through every device.

    Attributes:
        observations: Global indices of the observations, sorted. Their position here is their
            item index in the bucket's reader.
        n_slots: Length of the stream axis; a multiple of the device count, at least `n_real`.
        shape: The envelope every slot is padded to: the per-axis maximum over the group.
    """

    observations: np.ndarray
    n_slots: int
    shape: ObservationBufferShape

    @classmethod
    def create(
        cls, shapes: Sequence[ObservationBufferShape], group: Sequence[int], n_devices: int = 1
    ) -> Self:
        """Bucket a group of observations.

        Args:
            shapes: Per-observation buffer shapes, in observation order.
            group: Indices of the observations in the bucket.
            n_devices: Number of devices the bucket is sharded over.
        """
        if len(group) == 0:
            raise ValueError('a bucket needs at least one observation')
        shape = ObservationBufferShape(
            max(shapes[i].detector_count for i in group),
            max(shapes[i].sample_count for i in group),
            max(shapes[i].interval_count for i in group),
        )
        observations = np.sort(np.asarray(group, dtype=np.int64))
        return cls(observations, cls.slot_count(len(group), n_devices), shape)

    @staticmethod
    def slot_count(size: int, n_devices: int = 1) -> int:
        """Slots a bucket of ``size`` observations takes: rounded up to fill ``n_devices`` evenly."""
        return -(-size // n_devices) * n_devices

    @property
    def n_real(self) -> int:
        """Number of observations in the bucket."""
        return len(self.observations)

    @property
    def n_pad(self) -> int:
        """Number of empty slots."""
        return self.n_slots - self.n_real

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the bucket occupies once padded: every slot holds the envelope."""
        return self.n_slots * self.shape.volume

    @cached_property
    def is_real(self) -> np.ndarray:
        """Per slot, whether it holds an observation (``False`` for the empty slots at the end)."""
        return np.arange(self.n_slots) < self.n_real

    @cached_property
    def item_of_slot(self) -> np.ndarray:
        """Per slot, the reader item to load there.

        Empty slots repeat the last item so the index is always valid; they are never read (the
        accumulation gates the load on `is_real`).
        """
        return np.minimum(np.arange(self.n_slots), self.n_real - 1)


def real_volume(shapes: Sequence[ObservationBufferShape]) -> int:
    """Total volume of the observations, before any padding."""
    return sum(s.volume for s in shapes)


def padded_volume(
    shapes: Sequence[ObservationBufferShape], groups: Sequence[Sequence[int]], n_devices: int = 1
) -> int:
    """Total padded volume of a partition, summed over its buckets."""
    return sum(Bucket.create(shapes, g, n_devices).padded_volume for g in groups)


class PaddingReport(NamedTuple):
    """Padding cost of a grouping, relative to zero-padding.

    Attributes:
        n_groups: Number of groups in the partition.
        real: Volume before padding.
        padded: Volume after padding, which is what gets processed.
    """

    n_groups: int
    real: int
    padded: int

    @classmethod
    def create(
        cls,
        shapes: Sequence[ObservationBufferShape],
        groups: Sequence[Sequence[int]],
        n_devices: int = 1,
    ) -> Self:
        """Measure what a partition of the observations costs in padding.

        Args:
            shapes: Per-observation buffer shapes.
            groups: Partition of observation indices into groups.
            n_devices: Number of devices each group is sharded over (its slot count is rounded
                up to a multiple of it).

        Returns:
            The padding cost of that partition.

        Examples:
            Grouping everything together pads every observation to the largest one:

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 100), Shape(2, 100), Shape(2, 10)]
            >>> PaddingReport.create(shapes, [[0, 1, 2]]).overhead  # padded 600 vs real 420
            0.4285714285714286

            Sharding over four devices adds an empty slot as well:

            >>> PaddingReport.create(shapes, [[0, 1, 2]], n_devices=4).padded
            800
        """
        return cls(len(groups), real_volume(shapes), padded_volume(shapes, groups, n_devices))

    @property
    def overhead(self) -> float:
        """Fraction of wasted work, ``padded / real - 1``; 0.0 means no padding."""
        return (self.padded / self.real - 1.0) if self.real > 0 else 0.0


def partition_padded(
    shapes: Sequence[ObservationBufferShape], max_groups: int, *, n_devices: int = 1
) -> list[list[int]]:
    r"""Split observations into at most ``max_groups`` groups, wasting the least on padding.

    Observations in a group $b$ are padded to a common shape and the group to a whole number of
    slots per device, so it occupies
    $\lceil |b| / n_\text{dev} \rceil \, n_\text{dev} \cdot \max_{i \in b} d_i \cdot \max_{i \in b} s_i$
    elements, where observation $i$ has $d_i$ detectors and $s_i$ samples. Every device processes
    every group, so the run pays the sum over groups, and that is what the partition minimises.
    More groups pad less in shape but more in slots, so the best count is chosen here as well.

    Candidate groups are runs of the observations sorted by shape, so the partition is not always
    the best possible.

    Args:
        shapes: Per-observation buffer shapes.
        max_groups: Largest number of groups allowed (at most one per observation).
        n_devices: Number of devices each group is sharded over.

    Returns:
        A list of groups, from one up to ``max_groups`` of them. Each group is a sorted list of
        original observation indices. The groups are ordered by increasing detector count, then
        sample count, and together they cover ``range(len(shapes))``.

    Raises:
        ValueError: If ``max_groups < 1``, ``n_devices < 1`` or there are no observations.

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


@dataclass(frozen=True)
class SlotLayout:
    """The buckets of a run and the slot ranges each process holds.

    Built identically on every process from the same probe shapes, so every process agrees on
    the layout without communicating.

    Attributes:
        buckets: The buckets, ordered by increasing envelope shape.
        n_observations: Total number of observations.
        n_devices: Number of devices the stream axis spans (the whole job).
    """

    buckets: tuple[Bucket, ...]
    n_observations: int
    n_devices: int

    @classmethod
    def create(
        cls, shapes: Sequence[ObservationBufferShape], *, n_devices: int, max_buckets: int
    ) -> Self:
        """Choose the buckets that pad the least; see [`partition_padded`][].

        Args:
            shapes: Per-observation buffer shapes, in observation order.
            n_devices: Number of devices each bucket is sharded over.
            max_buckets: Largest number of buckets allowed.
        """
        groups = partition_padded(shapes, max_buckets, n_devices=n_devices)
        buckets = tuple(Bucket.create(shapes, group, n_devices) for group in groups)
        return cls(buckets, len(shapes), n_devices)

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

    def scatter(self, per_bucket: Sequence[np.ndarray]) -> np.ndarray:
        """Reorder per-slot arrays, one per bucket, into a single per-observation array.

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
            region: tuple[Any, ...] = (bucket.observations, *(slice(0, n) for n in real.shape[1:]))
            out[region] = real
        return out
