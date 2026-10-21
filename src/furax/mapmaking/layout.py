"""Logic to group observations into buckets of similar buffer shapes.

[`MultiObservationMapMaker`][] runs one jit-compiled program over multiple observations, so they
are required to share a common buffer shape. In reality they do not: they differ in detector and
sample count. This module provides tools to group them into *buckets*, i.e., groups of observations
padded to a common buffer shape. The mapmaker then processes the buckets one after the other,
each spread over all the available devices.

Two kinds of padding are required:

- *shape padding*: every observation of the group grows to the bucket's **envelope**, the per-axis
  maximum over the group. A 10-minute scan in a bucket with a 60-minute one occupies a 60-minute
  buffer;
- *slot padding*: the group is rounded up to a whole number of **slots** per device, so the slots
  divide evenly over the devices. Nine observations on four devices take twelve slots, three of
  them empty.

Each observation has an **observation index**: its position in the list given to the mapmaker.
A bucket lists its observations ([`Bucket.observations`][]) by increasing index. Each observation's
buffers occupy one **slot**. The slots after the last observation are **empty**
([`Bucket.is_empty`][]): they only hold filler data and are gated out of computations. Each bucket
has its own [`ObservationReader`][furax.mapmaking.ObservationReader] over its observations.

The detector axis of a bucket's envelope may further be split into `n_batches` **detector
batches** of `batch_size` detectors each, the same for every slot, and processed one after the
other. The mapmaker then builds the model of a bucket (its operators, noise model, weights and
templates) one batch of one slot at a time, and stacks the results along a leading axis of
`n_slots × n_batches` **entries**. They are ordered slot by slot, batch `j` of slot `s` at entry
`s · n_batches + j`, so the entries of a slot stay on the device that holds it. What the batches
of a slot share, such as the boresight pointing, is stacked once per slot instead.

The **detector index** runs along the detector axis of a slot's buffer, padded to the envelope;
the indices past the detectors read are padding.

The indices convert as follows, within a bucket:

| Index | Converts to | With |
| --- | --- | --- |
| slot `s`, not empty | observation | `Bucket.observations[s]` |
| entry `e` | slot, batch | `divmod(e, n_batches)` |
| detector `d` of batch `j` | detector of the slot | `j · batch_size + d` |
| process | its block of slots | [`SlotLayout.local_slots`][] |

A product computed per entry or per slot, such as the noise parameters or the template
amplitudes, is returned per observation. Per entry, its batches are first merged into slots
([`Bucket.merge_detector_batches`][]); then the empty slots are dropped and the others reordered
by observation index ([`SlotLayout.to_observation_order`][]). The result has the axes
`(observation, [stream], [detector], ...)`:

- the **stream** axis, over the Stokes components of a demodulated TOD, if the product has one
  value per stream;
- the detector axis, if the product has one value per detector. It is as long as the widest
  envelope of all buckets; past the detectors read of an observation, its values are padding.

!!! example

    Six observations of two very different lengths, on four devices held by two processes, with
    detector batches of 55 (`detector_batch_size=55`).

    **Partition.** Each shape is first padded to whole batches
    ([`ObservationBufferShape.whole_detector_batches`][]), then the observations are grouped
    into two buckets, each padded to its envelope:

    ```text
    observation   shape read     whole batches   bucket   slot
         0        (100, 4900)    (110, 4900)       B        0
         1        (100, 5000)    (110, 5000)       B        1
         2        ( 90, 1200)    (110, 1200)       A        0
         3        (100, 4800)    (110, 4800)       B        2
         4        (110, 5000)    (110, 5000)       B        3
         5        (100,  900)    (110,  900)       A        1

    bucket A: 2 observations, envelope (110, 1200), 4 slots (2 empty), 2 batches → 8 entries
    bucket B: 4 observations, envelope (110, 5000), 4 slots (0 empty), 2 batches → 8 entries
    ```

    **Slots.** Each bucket is sharded over the devices ([`SlotPlacement`][]), ordered by process
    so that each process holds a contiguous block of slots ([`SlotLayout.local_slots`][]). An
    empty slot holds filler:

    ```text
                    ┌──────── process 0 ────────┐   ┌──────── process 1 ────────┐
                        dev 0          dev 1           dev 2          dev 3
    bucket A   slot     0              1               2              3
               obs      2              5               (empty)        (empty)
               entries  0, 1           2, 3            4, 5           6, 7

    bucket B   slot     0              1               2              3
               obs      0              1               3              4
               entries  0, 1           2, 3            4, 5           6, 7
    ```

    **Entries.** The buffer of a slot has the bucket's envelope. Its detector axis is split into
    batches, batch `j` of slot `s` at entry `s · n_batches + j`. Slot 2 of bucket B, holding
    observation 3:

    ```text
                     sample 0 ··························· 4799   4800 ··· 4999
    detector   0  ┐  █████████████████████████████████████████   ░░░░░░░░░░░   batch 0 → entry 4
              54  ┘  █████████████████████████████████████████   ░░░░░░░░░░░
              55  ┐  █████████████████████████████████████████   ░░░░░░░░░░░   batch 1 → entry 5
              99  │  █████████████████████████████████████████   ░░░░░░░░░░░
             100  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ░░░░░░░░░░░
             109  ┘  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ░░░░░░░░░░░

                     █ data read    ░ padding to the envelope
    ```

    **Back to observations.** [`SlotLayout.to_observation_order`][] brings a product computed
    per entry back per observation, given its detector axis: the batches of each slot are merged
    ([`Bucket.merge_detector_batches`][]), then the empty slots are dropped and the slots
    reordered:

    ```text
    per entry    bucket A (8, 55, ...)   bucket B (8, 55, ...)
                          │ merge batches         │
    per slot     bucket A (4, 110, ...)  bucket B (4, 110, ...)
                          │ drop empty, reorder   │
                          └───────────┬───────────┘
    per observation              (6, 110, ...)   rows 0..5 = observations 0..5
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

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, ClassVar, Self

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Int, PyTree, Shaped

from ._observation import ObservationBufferShape
from .streaming import StreamLayout

__all__ = [
    'Bucket',
    'SlotLayout',
    'SlotPlacement',
    'padded_volume',
    'partition_padded',
    'unpadded_volume',
]


@dataclass(frozen=True, eq=False)
class Bucket:
    """One group of observations, processed together over all the devices.

    Every slot of the bucket holds a buffer of the same shape, the `envelope` of the group.
    `n_slots` is set to a value that fills the device count evenly, so its value is at least
    the number of observations. The empty slots come after those of the observations.

    Buckets are normally built for a whole run by [`SlotLayout.create`][], not one by one.

    Attributes:
        observations: The observation index held by each slot that is not empty, sorted.
        n_slots: The number of slots; a multiple of the device count.
        envelope: The per-axis maximum buffer shape over the group.
        n_batches: The number of detector batches each slot is split into.
    """

    observations: Int[np.ndarray, ' bucket_observation']
    n_slots: int
    envelope: ObservationBufferShape
    n_batches: int = 1

    @classmethod
    def create(
        cls,
        shapes: Sequence[ObservationBufferShape],
        group: Collection[int],
        n_devices: int = 1,
        detector_batch_size: int = 0,
    ) -> Self:
        """Bucket a group of observations, padding them to their common envelope.

        Args:
            shapes: Per-observation buffer shapes.
            group: Indices of the observations in the bucket.
            n_devices: Number of devices the bucket is sharded over.
            detector_batch_size: Detectors per batch; 0 for a single batch. The shapes must
                then be padded by [`ObservationBufferShape.whole_detector_batches`][].

        Returns:
            The bucket, with its observations sorted.

        Raises:
            ValueError: If `group` is empty, or the envelope's detectors are not whole batches.

        Examples:
            A short and a long observation bucketed together: both slots are as long as the
            longest and as wide as the widest.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> bucket = Bucket.create([Shape(2, 100), Shape(3, 10)], [1, 0])
            >>> bucket.envelope
            ObservationBufferShape(detector_count=3, sample_count=100, interval_count=0)
            >>> bucket.padded_volume, unpadded_volume([Shape(2, 100), Shape(3, 10)])
            (600, 230)
        """
        if len(group) == 0:
            raise ValueError('a bucket needs at least one observation')
        envelope = ObservationBufferShape(
            max(shapes[i].detector_count for i in group),
            max(shapes[i].sample_count for i in group),
            max(shapes[i].interval_count for i in group),
        )
        if envelope != envelope.whole_detector_batches(detector_batch_size):
            raise ValueError(
                f'{envelope.detector_count} detectors do not split into batches of '
                f'{detector_batch_size}: pad the shapes with `whole_detector_batches` first'
            )
        batch_size = detector_batch_size or envelope.detector_count
        # an envelope no wider than one batch is a single batch
        n_batches = max(1, envelope.detector_count // batch_size)
        observations = np.sort(np.asarray(group, dtype=np.int64))
        n_slots = cls.slot_count(len(group), n_devices)
        return cls(observations, n_slots, envelope, n_batches)

    @staticmethod
    def slot_count(size: int, n_devices: int = 1) -> int:
        """Slots a bucket of `size` observations takes: rounded up to fill `n_devices` evenly."""
        return -(-size // n_devices) * n_devices

    @property
    def n_observations(self) -> int:
        """Number of observations in the bucket, i.e. of slots that are not empty."""
        return len(self.observations)

    @property
    def n_empty(self) -> int:
        """Number of empty slots."""
        return self.n_slots - self.n_observations

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the bucket occupies once padded."""
        return self.n_slots * self.envelope.volume

    @property
    def batch_size(self) -> int:
        """Detectors per batch."""
        return self.envelope.detector_count // self.n_batches

    @property
    def n_entries(self) -> int:
        """Number of entries, one per detector batch of each slot."""
        return self.n_slots * self.n_batches

    @property
    def stream_layout(self) -> StreamLayout:
        """The entries as a stream lays them out, the batches of a slot sharing its common data."""
        return StreamLayout(self.n_entries, self.n_batches)

    @property
    def slot_of_entry(self) -> Int[np.ndarray, ' entry']:
        """The slot of each entry: entry `e` is batch `e % n_batches` of slot `e // n_batches`."""
        return np.arange(self.n_entries) // self.n_batches

    def merge_detector_batches(
        self, per_entry: Shaped[np.ndarray, 'entry batch_size *rest']
    ) -> Shaped[np.ndarray, 'slot det *rest']:
        """Per-slot values from per-entry ones: `(n_entries, batch_size, ...)` to `(n_slots, ...)`.

        The detector axis follows the entry axis; batch `j` of a slot holds its detectors from
        `j · batch_size` on.
        """
        return per_entry.reshape(
            self.n_slots, self.n_batches * per_entry.shape[1], *per_entry.shape[2:]
        )

    def sum_detector_batches(
        self, per_entry: Shaped[Array, ' entry *rest']
    ) -> Shaped[Array, ' slot *rest']:
        """Per-slot values from per-entry ones, summed over the detector batches of each slot."""
        return per_entry.reshape(self.n_slots, self.n_batches, *per_entry.shape[1:]).sum(axis=1)

    @cached_property
    def is_empty(self) -> Bool[np.ndarray, ' slot']:
        """Per slot, whether it is empty (`True` for the slots past the observations)."""
        return np.arange(self.n_slots) >= self.n_observations

    def summary(self, n_devices: int = 1) -> str:
        """The bucket's slot bookkeeping, as `obs=... slots=... empty=... envelope=...`."""
        batches = f' det_batches={self.n_batches}' if self.n_batches > 1 else ''
        return (
            f'obs={self.n_observations} slots={self.n_slots} empty={self.n_empty} '
            f'slots_per_dev={self.n_slots // n_devices} '
            f'envelope=({self.envelope.detector_count}, {self.envelope.sample_count}){batches}'
        )


def unpadded_volume(shapes: Sequence[ObservationBufferShape]) -> int:
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
    """Total number of observations."""
    n_devices: int
    """Total number of devices in the whole job."""

    @classmethod
    def create(
        cls,
        shapes: Sequence[ObservationBufferShape],
        *,
        n_devices: int,
        max_buckets: int,
        detector_batch_size: int = 0,
    ) -> Self:
        """Choose the layout to minimise padding; see [`partition_padded`][].

        Args:
            shapes: Per-observation buffer shapes, padded by
                [`ObservationBufferShape.whole_detector_batches`][] when
                `detector_batch_size` is given.
            n_devices: Number of devices (`jax.device_count()`).
            max_buckets: Largest number of buckets allowed.
            detector_batch_size: Detectors per batch; 0 for a single batch.

        Examples:
            Three short scans and one long one on two devices.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
            >>> layout = SlotLayout.create(shapes, n_devices=2, max_buckets=4)
            >>> print(layout)
            SlotLayout: obs=4 buckets=2 slots=4 devices=2 slot_overhead=+0.0%
              bucket 0: obs=2 slots=2 empty=0 slots_per_dev=1 envelope=(2, 10)
              bucket 1: obs=2 slots=2 empty=0 slots_per_dev=1 envelope=(2, 90)

            The dataset is heterogeneous enough that the layout still costs two thirds more
            memory than the observations themselves:

            >>> round(layout.padded_volume / unpadded_volume(shapes), 2)
            1.67
        """
        groups = partition_padded(shapes, max_buckets, n_devices=n_devices)
        buckets = tuple(
            Bucket.create(shapes, group, n_devices, detector_batch_size) for group in groups
        )
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

        The mesh orders devices by process, so a process's shards of the slot axis are one
        contiguous block: `n_local` shards of `n_slots / n_devices` slots each.

        Args:
            bucket: Index of the bucket.
            process_index: The process.
            n_local: Number of devices on every process.
        """
        n_per_device = self.buckets[bucket].n_slots // self.n_devices
        start = process_index * n_local * n_per_device
        return slice(start, start + n_local * n_per_device)

    def to_observation_order(
        self,
        per_bucket: Sequence[Shaped[np.ndarray, 'entry_or_slot ...']],
        detector_axis: int | None = None,
    ) -> Shaped[np.ndarray, 'obs ...']:
        """Bring per-entry or per-slot arrays, one per bucket, back into a per-observation array.

        This undoes the bucketing for results the caller wants per observation, in their original
        order, such as the noise parameters or the template amplitudes. Given `detector_axis`, the
        detector batches of each slot are first merged along it. Then the empty slots are dropped
        and the others reordered by observation index.

        Args:
            per_bucket: One array per bucket, of shape `(n_entries, ...)` with a detector axis,
                `(n_slots, ...)` without.
            detector_axis: The detector axis of the arrays, which holds one batch of detectors per
                entry; `None` for values per slot.

        Returns:
            An array of shape `(n_observations, ...)` in observation order. Trailing axes may
            differ between buckets, each padded to its own envelope, so they are zero-padded to
            the largest.
        """
        if len(per_bucket) != len(self.buckets):
            raise ValueError(f'expected {len(self.buckets)} arrays, got {len(per_bucket)}')
        if detector_axis is not None:
            per_bucket = [
                np.moveaxis(
                    bucket.merge_detector_batches(np.moveaxis(a, detector_axis, 1)),
                    1,
                    detector_axis,
                )
                for bucket, a in zip(self.buckets, per_bucket, strict=True)
            ]
        trailing = tuple(max(dims) for dims in zip(*(a.shape[1:] for a in per_bucket), strict=True))
        dtype = np.result_type(*(a.dtype for a in per_bucket))
        out = np.zeros((self.n_observations, *trailing), dtype=dtype)
        for bucket, values in zip(self.buckets, per_bucket, strict=True):
            held = values[: bucket.n_observations]
            region = (bucket.observations, *(slice(0, n) for n in held.shape[1:]))
            out[region] = held
        return out


@dataclass(frozen=True, eq=False)
class SlotPlacement:
    """The slots of a layout, placed on the devices of the job.

    The devices are ordered by process along one mesh axis, so that each process's shards of the
    slot axis form one contiguous block ([`SlotLayout.local_slots`][]).
    """

    layout: SlotLayout
    """The layout to place, planned for as many devices as the mesh has."""
    mesh: Mesh
    """The devices of the whole job, along the one axis `axis`."""

    axis: ClassVar[str] = 'slot'
    """The name of the mesh axis, which shards the slots (or the entries) of a bucket."""

    def __post_init__(self) -> None:
        if self.mesh.axis_names != (self.axis,):
            raise ValueError(f'expected a mesh along {self.axis!r}, got {self.mesh.axis_names}')
        if self.mesh.size != self.layout.n_devices:
            raise ValueError(
                f'layout planned for {self.layout.n_devices} devices, the mesh has {self.mesh.size}'
            )

    @classmethod
    def create(cls, layout: SlotLayout) -> Self:
        """Place `layout` on all the devices of the job, which it must have been planned for."""
        # `Auto` rather than `Explicit`: under explicit axis types an array built on this mesh
        # carries it in its sharding and then cannot be closed over inside `shard_map`, which is
        # how the sky landscape reaches the pointing. The price is that every `shard_map` must
        # spell out its `in_specs` instead of having them inferred.
        devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        return cls(layout, Mesh(np.array(devices), (cls.axis,), axis_types=(AxisType.Auto,)))

    @cached_property
    def sharding(self) -> NamedSharding:
        """The sharding of an array that leads with the slots, or the entries, of a bucket."""
        return NamedSharding(self.mesh, P(self.axis))

    @cached_property
    def replicated(self) -> NamedSharding:
        """The sharding of an array that every device holds whole."""
        return NamedSharding(self.mesh, P())

    @cached_property
    def local_devices(self) -> list[jax.Device]:
        """This process's devices, in the order of the mesh."""
        return [d for d in self.mesh.devices.flat if d.process_index == jax.process_index()]

    def device_slots(self, bucket: int) -> Int[np.ndarray, 'device round']:
        """The slots of a bucket held by each of this process's devices, one per round.

        Row `k` holds the slots of the `k`-th of the local devices, a contiguous run of
        `n_slots / n_devices` slots, as the sharding of the slot axis places them.
        """
        n_local = len(self.local_devices)
        local = self.layout.local_slots(bucket, process_index=jax.process_index(), n_local=n_local)
        return np.arange(local.start, local.stop).reshape(n_local, -1)

    def to_devices(self, per_device: Sequence[PyTree[Any]]) -> PyTree[Array]:
        """Host pytrees, one per local device, as arrays sharded over a leading device axis.

        On a CPU device, a suitably aligned host buffer is used in place rather than copied.
        """

        def stack(*leaves: Any) -> Array:
            shards = [
                jax.device_put(np.asarray(leaf)[None], device, may_alias=True)
                for leaf, device in zip(leaves, self.local_devices, strict=True)
            ]
            shape = (self.mesh.size, *np.shape(leaves[0]))
            return jax.make_array_from_single_device_arrays(shape, self.sharding, shards)

        return jax.tree.map(stack, *per_device)

    def broadcast(self, tree: PyTree[Any]) -> PyTree[Array]:
        """Host values, the same on every process, as arrays every device holds whole."""
        place = partial(jax.make_array_from_process_local_data, self.replicated)
        return jax.tree.map(lambda x: place(np.asarray(x)), tree)

    def gather_per_observation(
        self, per_bucket: Sequence[dict[str, Array]], detector_axes: Mapping[str, int | None]
    ) -> dict[str, np.ndarray]:
        """Arrays stacked over the entries or slots of each bucket, gathered per observation.

        Args:
            per_bucket: For each bucket, arrays by name, leading with its entries or slots.
            detector_axes: For each name, the detector axis along which the detector batches are
                merged (see [`SlotLayout.to_observation_order`][]), or `None` for arrays without one.
        """
        gathered = [self.gather(arrays) for arrays in per_bucket]
        return {
            name: self.layout.to_observation_order([g[name] for g in gathered], axis)
            for name, axis in detector_axes.items()
        }

    def gather(self, tree: PyTree[Array]) -> PyTree[np.ndarray]:
        """Arrays sharded over the devices, brought whole to the host of every process."""
        return jax.tree.map(np.asarray, self.replicate(tree))

    def replicate(
        self, tree: PyTree[Array], fn: Callable[[Array], Array] = lambda x: x
    ) -> PyTree[Array]:
        """`fn` of every array of `tree`, held whole by every device."""
        return jax.jit(partial(jax.tree.map, fn), out_shardings=self.replicated)(tree)

    def per_slot(
        self, tree: PyTree[Array], fn: Callable[[Array], Array], *, donate: bool = False
    ) -> PyTree[Array]:
        """`fn` of every array of `tree`, sharded over the slots (or entries) it leads with.

        Args:
            tree: Arrays sharded over the devices along their leading axis.
            fn: The function to apply to every array.
            donate: Whether the buffers of `tree` may be reused for the result.
        """
        apply = jax.jit(
            partial(jax.tree.map, fn),
            out_shardings=self.sharding,
            donate_argnums=0 if donate else (),
        )
        return apply(tree)
