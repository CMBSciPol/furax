"""Logic to split observations into detector tiles, grouped into buckets of similar length.

[`MultiObservationMapMaker`][] runs one jit-compiled program over many observations, so they are
required to share a common buffer shape. In reality they do not: they differ in detector and
sample count. This module splits each observation into **tiles** of a fixed number of detectors,
and groups the observations into **buckets** of similar length, whose tiles share one buffer
shape. The mapmaker then processes the buckets one after the other, each spread over all the
available devices.

Three kinds of padding are required:

- *sample padding*: every tile of a bucket is as long as its longest observation, the bucket's
  **envelope**. A 10-minute scan in a bucket with a 60-minute one occupies 60-minute tiles;
- *detector padding*: the last tile of an observation is completed with empty detectors;
- *tile padding*: every device holds the same number of tiles, so the devices with fewer are
  completed with **padding** tiles, which only hold filler data and are gated out of
  computations.

An observation's tiles are never split over two devices: they live on the device of its
**segment**, which holds what they share, such as the boresight pointing. Each device holds a
contiguous block of tiles and a contiguous block of segments; a device with fewer observations has
**empty** segments, and its padding tiles own one of them. Each observation has an **observation
index**: its position in the list given to the mapmaker. Each bucket has its own
[`ObservationReader`][furax.mapmaking.ObservationReader] over its observations.

The model of a bucket (its operators, noise model, weights and templates) is stacked along a
leading axis: of tiles for what differs between the tiles of an observation, of segments for what
they share. The **detector index** of a tile runs over its own detectors; tile `t` holds detectors
`tile_starts[t] + [0, tile_size)` of its observation, the indices past the detectors read being
padding.

The indices convert as follows, within a bucket:

| Index | Converts to | With |
| --- | --- | --- |
| tile `t` | its segment | [`TileBucket.owner`][]`[t]` |
| segment `s`, not empty | observation | [`TileBucket.segment_observations`][]`[s]` |
| detector `d` of tile `t` | detector of the observation | `tile_starts[t] + d` |
| process | its block of tiles | [`TileLayout.local_tiles`][] |

A product computed per tile or per segment, such as the noise parameters or the template
amplitudes, is returned per observation by [`TileLayout.to_observation_order`][], which
concatenates the tiles of each observation along the detector axis. The result has the axes
`(observation, [stream], [detector], ...)`:

- the **stream** axis, over the Stokes components of a demodulated TOD, if the product has one
  value per stream;
- the detector axis, if the product has one value per detector. It is as long as the widest
  observation's tiles; past the detectors read of an observation, its values are padding.

!!! example

    Five observations of two lengths, on two devices held by one process each, in tiles of 50
    detectors (`detector_tile_size=50`).

    **Partition.** The observations are grouped by length into two buckets, and each is split
    into tiles:

    ```text
    observation   shape read     tiles   bucket
         0        (100, 4900)      2       B
         1        ( 40, 1200)      1       A
         2        (140, 5000)      3       B
         3        ( 90, 1100)      2       A
         4        ( 50, 4800)      1       B

    bucket A: 2 observations in 3 tiles of (50, 1200)
    bucket B: 3 observations in 6 tiles of (50, 5000)
    ```

    **Devices.** The observations of a bucket go to the devices from the widest down, each to the
    device with the fewest tiles so far, or packed onto the first device each fits on under a
    lower cap if that lowers the fullest device ([`TileBucket.create`][]). Every device then holds
    the same number of tiles, completed with padding:

    ```text
                       ┌─── process 0 ───┐   ┌─── process 1 ───┐
                             dev 0                 dev 1
    bucket A   tiles     3  3                  1  ·
               segments  3  ∅                  1  ∅

    bucket B   tiles     2  2  2               0  0  4
               segments  2  ∅                  0  4

                         n: a tile or the segment of observation n
                         ·: a padding tile    ∅: an empty segment
    ```

    **Back to observations.** [`TileLayout.to_observation_order`][] brings a product computed
    per tile back per observation, given its detector axis, by concatenating the tiles of each
    observation; a product per segment needs no detector axis:

    ```text
    per tile      bucket A (4, 50, ...)   bucket B (6, 50, ...)
                          │ concatenate the tiles of each observation
                          └───────────┬───────────┘
    per observation              (5, 150, ...)   rows 0..4 = observations 0..4
    ```

!!! tip "Choosing the number of buckets"

    [`partition_tiles`][] chooses the number of buckets within the `max_buckets` budget, given the
    available device count, optimising for total padded volume. A larger budget can only lower the
    volume it settles on, but additional buckets cost compile time: each one is traced and compiled
    for its own envelope. Since the tiles absorb the differences in detector count, a rule of
    thumb is to allow one bucket per distinct observation length.

!!! tip "Choosing the tile size"

    Smaller tiles pad the detector axis less and balance the devices better, but each one is a
    step of the loop over the tiles, which costs a fixed overhead. Every device runs as many tiles
    as the most loaded one: an observation is never split over two devices, so a device count
    above the total tiles of a bucket over its widest observation's leaves devices idle.
"""

import heapq
import itertools
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, ClassVar, Self

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Int, PyTree, Shaped

from ._observation import ObservationBufferShape

__all__ = [
    'TileBucket',
    'TileLayout',
    'TilePlacement',
    'partition_tiles',
]


@dataclass(frozen=True, eq=False)
class TileBucket:
    """One group of observations of similar length, split into detector tiles.

    A **tile** holds `tile_size` detectors of one observation over the bucket's sample envelope:
    an observation of `d` detectors has `ceil(d / tile_size)` tiles, its last one padded with
    empty detectors. Each observation occupies one **segment**, which holds what its tiles share,
    such as the boresight pointing, and every tile is owned by the segment of its observation.

    The tiles are spread over the devices with whole observations: the tiles of an observation
    and its segment live on one device. Every device holds `tiles_per_device` tiles and
    `segments_per_device` segments, in contiguous blocks: device `d` holds tiles
    `d * tiles_per_device + [0, tiles_per_device)` and segments
    `d * segments_per_device + [0, segments_per_device)`. A device with fewer tiles is filled
    with **padding** tiles (`valid` is `False`), and one with fewer observations with **empty**
    segments (observation `-1`). The padding tiles of a device own one of its empty segments,
    never an observation's: what they hold, filler, stays out of the observations' segments.

    Buckets are normally built for a whole run by [`TileLayout.create`][], not one by one.

    Attributes:
        segment_observations: The observation index of each segment, `-1` for an empty one.
        owner: The segment of each tile.
        tile_starts: The first detector of each tile in its observation (0 for padding).
        valid: Per tile, whether it holds detectors of an observation.
        envelope: The shape of one tile: `tile_size` detectors over the longest observation of
            the bucket.
        n_devices: The number of devices the bucket is spread over.
    """

    segment_observations: Int[np.ndarray, ' segment']
    owner: Int[np.ndarray, ' tile']
    tile_starts: Int[np.ndarray, ' tile']
    valid: Bool[np.ndarray, ' tile']
    envelope: ObservationBufferShape
    n_devices: int

    @classmethod
    def create(
        cls,
        shapes: Sequence[ObservationBufferShape],
        group: Collection[int],
        n_devices: int = 1,
        detector_tile_size: int = 0,
    ) -> Self:
        """Split a group of observations into tiles, spread over the devices.

        The observations are spread to keep the tiles of the fullest device low. From the widest
        down, each goes to the device holding the fewest tiles so far. That can leave the fullest
        device a tile or so above what is possible, so a second packing is tried under each lower
        cap on the tiles per device: from the widest down, each observation goes to the first
        device it still fits on under the cap. The lowest cap that works is kept, and it fills
        the first devices up and leaves the last ones lighter.

        Args:
            shapes: Per-observation buffer shapes.
            group: Indices of the observations in the bucket.
            n_devices: Number of devices the bucket is spread over.
            detector_tile_size: Detectors per tile; 0 for one tile per observation. A bucket
                whose widest observation is narrower takes its width instead.

        Raises:
            ValueError: If `group` is empty.

        Examples:
            Three observations of 2, 5 and 3 detectors in tiles of 2, on two devices: the widest
            observation, of three tiles, has a device of its own.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 100), Shape(5, 100), Shape(3, 90)]
            >>> bucket = TileBucket.create(shapes, [0, 1, 2], n_devices=2, detector_tile_size=2)
            >>> bucket.segment_observations.tolist()
            [1, -1, 0, 2]
            >>> bucket.owner.tolist(), bucket.valid.tolist()
            ([0, 0, 0, 2, 3, 3], [True, True, True, True, True, True])
        """
        if len(group) == 0:
            raise ValueError('a bucket needs at least one observation')
        observations = sorted(group)
        widest = max(shapes[i].detector_count for i in observations)
        tile_size = min(detector_tile_size, widest) if detector_tile_size else widest
        counts = {i: -(-shapes[i].detector_count // tile_size) for i in observations}

        by_width = sorted(observations, key=lambda i: (-counts[i], i))
        devices, loads = _spread([counts[i] for i in by_width], n_devices)
        held: list[list[int]] = [[] for _ in range(n_devices)]
        for i, d in zip(by_width, devices, strict=True):
            held[d].append(i)
        tiles_per_device = max(loads)
        # a device with padding tiles keeps an empty segment for them
        spare = [int(load < tiles_per_device) for load in loads]
        segments_per_device = max(len(h) + s for h, s in zip(held, spare, strict=True))

        segment_observations = np.full(n_devices * segments_per_device, -1, dtype=np.int64)
        owner, starts, valid = [], [], []
        for d, on_device in enumerate(held):
            first = d * segments_per_device
            for j, i in enumerate(sorted(on_device)):
                segment_observations[first + j] = i
                owner += [first + j] * counts[i]
                starts += [k * tile_size for k in range(counts[i])]
                valid += [True] * counts[i]
            n_padding = tiles_per_device - loads[d]
            owner += [first + len(on_device)] * n_padding  # the first empty segment
            starts += [0] * n_padding
            valid += [False] * n_padding
        envelope = ObservationBufferShape(
            tile_size,
            max(shapes[i].sample_count for i in observations),
            max(shapes[i].interval_count for i in observations),
        )
        return cls(
            segment_observations,
            np.asarray(owner, dtype=np.int64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(valid, dtype=bool),
            envelope,
            n_devices,
        )

    @property
    def tile_size(self) -> int:
        """Detectors per tile."""
        return self.envelope.detector_count

    @property
    def n_tiles(self) -> int:
        """Number of tiles, padding included; a multiple of the device count."""
        return len(self.owner)

    @property
    def n_segments(self) -> int:
        """Number of segments, empty ones included; a multiple of the device count."""
        return len(self.segment_observations)

    @property
    def tiles_per_device(self) -> int:
        """Tiles held by each device."""
        return self.n_tiles // self.n_devices

    @property
    def segments_per_device(self) -> int:
        """Segments held by each device."""
        return self.n_segments // self.n_devices

    @cached_property
    def observations(self) -> Int[np.ndarray, ' bucket_observation']:
        """The observation indices of the bucket, sorted."""
        return np.sort(self.segment_observations[self.segment_observations >= 0])

    @property
    def n_observations(self) -> int:
        """Number of observations in the bucket."""
        return len(self.observations)

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the bucket occupies once padded."""
        return self.n_tiles * self.envelope.volume

    def tiles_of(self, segment: int) -> Int[np.ndarray, ' observation_tile']:
        """The tiles of a segment that hold detectors, by increasing first detector."""
        return np.flatnonzero((self.owner == segment) & self.valid)

    def summary(self) -> str:
        """The bucket's bookkeeping, as `obs=... tiles=... padding=... envelope=...`."""
        return (
            f'obs={self.n_observations} tiles={self.n_tiles} '
            f'padding={self.n_tiles - int(self.valid.sum())} '
            f'tiles_per_dev={self.tiles_per_device} segments_per_dev={self.segments_per_device} '
            f'envelope=({self.envelope.detector_count}, {self.envelope.sample_count})'
        )


_MAX_CUTS = 128


def partition_tiles(
    shapes: Sequence[ObservationBufferShape],
    max_groups: int,
    *,
    n_devices: int = 1,
    detector_tile_size: int = 0,
) -> list[list[int]]:
    r"""Split observations into at most `max_groups` groups of tiles, minimising padding.

    A group $b$ is split into tiles of $t_b$ detectors over its longest observation, $s_b$
    samples. The tiles are spread over the devices with whole observations, and the fullest device
    holds $T_b$ tiles. Every device is padded to that load, so the group occupies
    $n_\text{dev} T_b t_b s_b$ elements. The sum over the groups is the quantity minimised here.
    Since observations are not split over devices, $T_b$ can be far above an even share of the
    tiles when a group holds about as many observations as devices.

    $T_b$ is estimated by giving each observation, from the widest down, to the least loaded
    device. [`TileBucket.create`][] starts from the same rule and may then pack the tiles tighter,
    so a bucket never holds more tiles per device than estimated here.

    Since the tiles absorb the differences in detector count, candidate groups are runs of the
    observations sorted by length. Costing every run is too slow for thousands of observations,
    so a group may only start at one of a fixed number of evenly spaced observations (at any
    observation if there are fewer). On real observation sets, the padded volume of the result is
    within about half a percent of the best split's.

    Args:
        shapes: Per-observation buffer shapes.
        max_groups: Largest number of groups allowed (at most one per observation).
        n_devices: Number of devices each group is spread over.
        detector_tile_size: Detectors per tile; 0 for one tile per observation.

    Returns:
        A list of groups, from one up to `max_groups` of them. Each group is a sorted list of
        original observation indices. The groups are ordered by increasing length, and together
        they cover `range(len(shapes))`.

    Raises:
        ValueError: If `max_groups < 1`, `n_devices < 1` or there are no observations.

    Examples:
        Observations of very different widths but similar lengths share a group: their tiles
        absorb the difference.

        >>> from furax.mapmaking import ObservationBufferShape as Shape
        >>> shapes = [Shape(100, 10), Shape(800, 11), Shape(300, 90)]
        >>> partition_tiles(shapes, max_groups=2, detector_tile_size=100)
        [[0, 1], [2]]
    """
    if max_groups < 1:
        raise ValueError(f'max_groups must be >= 1, got {max_groups}')
    if n_devices < 1:
        raise ValueError(f'n_devices must be >= 1, got {n_devices}')
    n = len(shapes)
    if n == 0:
        raise ValueError('need at least one observation')
    max_groups = min(max_groups, n)

    order = sorted(range(n), key=lambda i: (shapes[i].sample_count, shapes[i].detector_count))
    det = np.array([shapes[i].detector_count for i in order], dtype=np.int64)
    samp = np.array([shapes[i].sample_count for i in order], dtype=np.int64)
    tiles = -(-det // detector_tile_size) if detector_tile_size else np.ones(n, dtype=np.int64)

    # Costing a group takes a spread over the devices, so costing all O(n²) runs is cubic in the
    # number of observations. The cost varies slowly with where a run starts, so the groups are
    # only allowed to start at about `_MAX_CUTS` evenly spaced positions: every one of them for
    # runs of up to that many observations.
    cuts = np.unique(np.linspace(0, n, min(n, _MAX_CUTS) + 1).round().astype(np.int64))
    m = len(cuts)
    max_groups = min(max_groups, m - 1)

    # cost[a, b] is that of the group [cuts[a], cuts[b]), sized as `TileBucket.create` does, with
    # the fullest device estimated by `_lpt`, the first spread it tries
    cost = np.full((m, m), np.inf)
    for a in range(m - 1):
        for b in range(a + 1, m):
            i, j = cuts[a], cuts[b]
            widest = int(det[i:j].max())
            tile_size = min(detector_tile_size, widest) if detector_tile_size else widest
            per_device = _fullest_device(tiles[i:j], n_devices)
            cost[a, b] = float(n_devices * per_device * tile_size * samp[i:j].max())

    # dp[g, b] is the lowest total cost of splitting the first cuts[b] observations into g
    # groups; start[g, b] records where the last of those groups starts.
    dp = np.full((max_groups + 1, m), np.inf)
    start = np.zeros((max_groups + 1, m), dtype=np.int64)
    dp[0, 0] = 0
    for g in range(1, max_groups + 1):
        total = dp[g - 1][:, None] + cost
        start[g] = np.argmin(total, axis=0)
        dp[g] = np.min(total, axis=0)

    n_groups = int(np.argmin(dp[1:, -1])) + 1  # fewest groups among the cheapest
    bounds = [m - 1]
    for g in range(n_groups, 0, -1):
        bounds.append(int(start[g, bounds[-1]]))
    bounds = [int(cuts[b]) for b in reversed(bounds)]
    return [sorted(order[i:j]) for i, j in itertools.pairwise(bounds)]


def _fullest_device(tiles: Int[np.ndarray, ' observation'], n_devices: int) -> int:
    """The tiles of the fullest device when the observations are spread by `_lpt`."""
    return max(_lpt(sorted(tiles.tolist(), reverse=True), n_devices)[1])


def _lpt(counts: Sequence[int], n_devices: int) -> tuple[list[int], list[int]]:
    """Spread observations, longest processing time first.

    Each observation goes to the device with the fewest tiles so far, then the fewest
    observations, then the lowest index.

    Args:
        counts: Tiles of each observation, from the widest down.
        n_devices: Number of devices.

    Returns:
        The device of each observation, and the tiles of each device.
    """
    heap = [(0, 0, d) for d in range(n_devices)]
    devices = []
    for count in counts:
        load, held, d = heap[0]
        heapq.heapreplace(heap, (load + count, held + 1, d))
        devices.append(d)
    loads = [0] * n_devices
    for load, _, d in heap:
        loads[d] = load
    return devices, loads


def _spread(counts: Sequence[int], n_devices: int) -> tuple[list[int], list[int]]:
    """Spread observations over the devices, tightening the tiles of the fullest one.

    Args:
        counts: Tiles of each observation, from the widest down.
        n_devices: Number of devices.

    Returns:
        The device of each observation, and the tiles of each device.
    """
    devices, loads = _lpt(counts, n_devices)
    if len(counts) <= n_devices:
        return devices, loads

    # `_lpt` can end a tile or so above the best possible. The fullest device holds at least
    # the floor below: the widest observation, an even share, or two of the n_devices + 1 widest
    # observations, which cannot all be alone. First-fit decreasing under each cap from there up
    # to the load of `_lpt` may find a tighter one.
    floor = max(counts[0], -(-sum(counts) // n_devices), counts[n_devices - 1] + counts[n_devices])
    for cap in range(floor, max(loads)):
        fitted = np.zeros(n_devices, dtype=np.int64)
        fits = []
        for count in counts:
            room = fitted <= cap - count
            d = int(room.argmax())
            if not room[d]:
                break
            fitted[d] += count
            fits.append(d)
        else:
            return fits, fitted.tolist()
    return devices, loads


@dataclass(frozen=True, eq=False)
class TileLayout:
    """Complete layout of detector tiles for a given dataset and job resources."""

    buckets: tuple[TileBucket, ...]
    """The buckets, ordered by increasing length."""
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
        detector_tile_size: int = 0,
    ) -> Self:
        """Choose the layout to minimise padding; see [`partition_tiles`][].

        Args:
            shapes: Per-observation buffer shapes.
            n_devices: Number of devices (`jax.device_count()`).
            max_buckets: Largest number of buckets allowed.
            detector_tile_size: Detectors per tile; 0 for one tile per observation.

        Examples:
            Five observations of two lengths, on two devices, in tiles of two detectors. The
            two narrow short observations share a device, the wide one has the other.

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 10), Shape(2, 10), Shape(4, 10), Shape(4, 90), Shape(4, 90)]
            >>> layout = TileLayout.create(
            ...     shapes, n_devices=2, max_buckets=2, detector_tile_size=2
            ... )
            >>> print(layout)
            TileLayout: obs=5 buckets=2 tiles=8 devices=2 tile_overhead=+0.0%
              bucket 0: obs=3 tiles=4 padding=0 tiles_per_dev=2 segments_per_dev=2 envelope=(2, 10)
              bucket 1: obs=2 tiles=4 padding=0 tiles_per_dev=2 segments_per_dev=1 envelope=(2, 90)
        """
        groups = partition_tiles(
            shapes, max_buckets, n_devices=n_devices, detector_tile_size=detector_tile_size
        )
        buckets = tuple(
            TileBucket.create(shapes, group, n_devices, detector_tile_size) for group in groups
        )
        return cls(buckets, len(shapes), n_devices)

    @property
    def n_tiles(self) -> int:
        """Total number of tiles, over every bucket."""
        return sum(bucket.n_tiles for bucket in self.buckets)

    @property
    def padded_volume(self) -> int:
        """Time-ordered elements the run occupies once padded, summed over the buckets."""
        return sum(bucket.padded_volume for bucket in self.buckets)

    @property
    def tile_overhead(self) -> float:
        """Padding tiles as a fraction of the tiles that hold detectors."""
        used = sum(int(bucket.valid.sum()) for bucket in self.buckets)
        return (self.n_tiles - used) / used

    def summary(self) -> str:
        """The run's totals, as `obs=... buckets=... tiles=... devices=... tile_overhead=...`."""
        return (
            f'obs={self.n_observations} buckets={len(self.buckets)} tiles={self.n_tiles} '
            f'devices={self.n_devices} tile_overhead=+{self.tile_overhead:.1%}'
        )

    def __str__(self) -> str:
        lines = [f'{type(self).__name__}: {self.summary()}']
        lines += [f'  bucket {b}: {bucket.summary()}' for b, bucket in enumerate(self.buckets)]
        return '\n'.join(lines)

    def local_tiles(self, bucket: int, *, process_index: int, n_local: int) -> slice:
        """The tiles of a bucket held by one process's devices.

        The mesh orders devices by process, so a process's shards of the tile axis are one
        contiguous block: `n_local` shards of `tiles_per_device` tiles each.

        Args:
            bucket: Index of the bucket.
            process_index: The process.
            n_local: Number of devices on every process.
        """
        n_per_device = self.buckets[bucket].tiles_per_device
        start = process_index * n_local * n_per_device
        return slice(start, start + n_local * n_per_device)

    def local_segments(self, bucket: int, *, process_index: int, n_local: int) -> slice:
        """The segments of a bucket held by one process's devices, as [`local_tiles`][]."""
        n_per_device = self.buckets[bucket].segments_per_device
        start = process_index * n_local * n_per_device
        return slice(start, start + n_local * n_per_device)

    def to_observation_order(
        self,
        per_bucket: Sequence[Shaped[np.ndarray, 'entry ...']],
        detector_axis: int | None = None,
    ) -> Shaped[np.ndarray, 'obs ...']:
        """Bring per-tile or per-segment arrays, one per bucket, back into a per-observation array.

        This undoes the tiling for results the caller wants per observation, in their original
        order, such as the noise parameters or the template amplitudes. Given `detector_axis`,
        the arrays hold one entry per tile, and the tiles of each observation are concatenated
        along that axis. Without it, they hold one entry per segment.

        Args:
            per_bucket: One array per bucket, of shape `(n_tiles, ...)` with a detector axis,
                `(n_segments, ...)` without.
            detector_axis: The detector axis of the arrays, which holds the detectors of one
                tile; `None` for values per segment. It is the detector axis of the result too.

        Returns:
            An array of shape `(n_observations, ...)` in observation order. Trailing axes may
            differ between observations, each padded to its own tiles or bucket, so they are
            zero-padded to the largest.

        Raises:
            ValueError: If there is not one array per bucket, or if `detector_axis` does not hold
                the detectors of a tile.
        """
        if len(per_bucket) != len(self.buckets):
            raise ValueError(f'expected {len(self.buckets)} arrays, got {len(per_bucket)}')
        rows: dict[int, np.ndarray] = {}
        for bucket, values in zip(self.buckets, per_bucket, strict=True):
            for segment, observation in enumerate(bucket.segment_observations):
                if observation < 0:
                    continue
                if detector_axis is None:
                    rows[int(observation)] = values[segment]
                    continue
                axis = detector_axis % values.ndim - 1  # in the entries, after the tile axis
                if (n := values.shape[axis + 1]) != bucket.tile_size:
                    msg = (
                        f'axis {detector_axis} has {n} entries, '
                        f'not the {bucket.tile_size} detectors of a tile'
                    )
                    raise ValueError(msg)
                tiles = [values[t] for t in bucket.tiles_of(segment)]
                rows[int(observation)] = np.concatenate(tiles, axis=axis)
        held = list(rows.values())
        trailing = tuple(max(dims) for dims in zip(*(a.shape for a in held), strict=True))
        dtype = np.result_type(*(a.dtype for a in held))
        out = np.zeros((self.n_observations, *trailing), dtype=dtype)
        for observation, values in rows.items():
            out[(observation, *(slice(0, n) for n in values.shape))] = values
        return out


@dataclass(frozen=True, eq=False)
class TilePlacement:
    """The tiles of a layout, placed on the devices of the job.

    The devices are ordered by process along one mesh axis, so that each process's shards of the
    tile axis form one contiguous block ([`TileLayout.local_tiles`][]).
    """

    layout: TileLayout
    """The layout to place, planned for as many devices as the mesh has."""
    mesh: Mesh
    """The devices of the whole job, along the one axis `axis`."""

    axis: ClassVar[str] = 'tile'
    """The name of the mesh axis, which shards the tiles and the segments of a bucket."""

    def __post_init__(self) -> None:
        if self.mesh.axis_names != (self.axis,):
            raise ValueError(f'expected a mesh along {self.axis!r}, got {self.mesh.axis_names}')
        if self.mesh.size != self.layout.n_devices:
            raise ValueError(
                f'layout planned for {self.layout.n_devices} devices, the mesh has {self.mesh.size}'
            )

    @classmethod
    def create(cls, layout: TileLayout) -> Self:
        """Place `layout` on all the devices of the job, which it must have been planned for."""
        # `Auto` rather than `Explicit`: under explicit axis types an array built on this mesh
        # carries it in its sharding and then cannot be closed over inside `shard_map`, which is
        # how the sky landscape reaches the pointing. The price is that every `shard_map` must
        # spell out its `in_specs` instead of having them inferred.
        devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        return cls(layout, Mesh(np.array(devices), (cls.axis,), axis_types=(AxisType.Auto,)))

    @cached_property
    def sharding(self) -> NamedSharding:
        """The sharding of an array that leads with the tiles or the segments of a bucket."""
        return NamedSharding(self.mesh, P(self.axis))

    @cached_property
    def replicated(self) -> NamedSharding:
        """The sharding of an array that every device holds whole."""
        return NamedSharding(self.mesh, P())

    @cached_property
    def local_devices(self) -> list[jax.Device]:
        """This process's devices, in the order of the mesh."""
        return [d for d in self.mesh.devices.flat if d.process_index == jax.process_index()]

    def shard(self, values: np.ndarray) -> Array:
        """Host values, the same on every process, split over the devices along their leading axis."""
        return jax.make_array_from_callback(
            values.shape, self.sharding, lambda index: values[index]
        )

    def broadcast(self, tree: PyTree[Any]) -> PyTree[Array]:
        """Host values, the same on every process, as arrays every device holds whole."""
        place = partial(jax.make_array_from_process_local_data, self.replicated)
        return jax.tree.map(lambda x: place(np.asarray(x)), tree)

    def gather(self, tree: PyTree[Array]) -> PyTree[np.ndarray]:
        """Arrays sharded over the devices, brought whole to the host of every process."""
        return jax.tree.map(np.asarray, self.replicate(tree))

    def replicate(
        self, tree: PyTree[Array], fn: Callable[[Array], Array] = lambda x: x
    ) -> PyTree[Array]:
        """`fn` of every array of `tree`, held whole by every device."""
        return jax.jit(partial(jax.tree.map, fn), out_shardings=self.replicated)(tree)
