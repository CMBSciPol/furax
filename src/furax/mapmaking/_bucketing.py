"""Group observations of similar size to cut buffer padding.

A single [`jax.shard_map`][] over the observation axis forces **one** global buffer shape:
[`ObservationReader`][furax.mapmaking.ObservationReader] pads every read to
``_get_common_structure``, the per-axis max over *all* observations. A short observation stacked
next to a long one is therefore padded up to the long one, wasting compute and memory on the
padding.

Splitting the observations into size-homogeneous *buckets* and running one accumulate/solve pass
per bucket lets each pass pad only to its own bucket's max. The TOD-shaped stacks never cross
process boundaries (only the ``psum``-reduced sky map does), so per-bucket shapes are safe; the
sole reason a single pass cannot do this is ``shard_map``'s uniform-shape rule. The cost is one
``shard_map`` pass per bucket (extra compilation + collective launches).

This module only computes the *partition* and its padding cost. Wiring the multi-pass accumulate
into the mapmaker is a separate step; use [`padding_report`][furax.mapmaking._bucketing.padding_report]
to decide whether the win justifies it for a given dataset.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

# A per-observation buffer shape, as returned by ``AbstractLazyObservation.probe_shape``:
# ``(detector_count, sample_count, interval_count)``. We only size on detectors x samples; the
# interval axis is negligible and often zero.
ShapeLike = Sequence[int]


def buffer_volume(shape: ShapeLike) -> int:
    """Padded-buffer volume charged to one observation: ``detector_count * sample_count``."""
    return int(shape[0]) * int(shape[1])


def real_volume(shapes: Sequence[ShapeLike]) -> int:
    """Total un-padded volume across observations (the ideal, zero-padding cost)."""
    return sum(buffer_volume(s) for s in shapes)


def bucket_padded_volume(shapes: Sequence[ShapeLike], bucket: Sequence[int]) -> int:
    """Padded volume of one bucket: ``count * max_detectors * max_samples`` over its members."""
    if len(bucket) == 0:
        return 0
    max_det = max(int(shapes[i][0]) for i in bucket)
    max_samp = max(int(shapes[i][1]) for i in bucket)
    return len(bucket) * max_det * max_samp


def padded_volume(shapes: Sequence[ShapeLike], buckets: Sequence[Sequence[int]]) -> int:
    """Total padded volume over a partition (sum of per-bucket padded volumes)."""
    return sum(bucket_padded_volume(shapes, b) for b in buckets)


class PaddingReport(NamedTuple):
    """Padding cost of a bucketing, relative to zero-padding.

    Attributes:
        n_buckets: Number of buckets in the partition.
        real: Un-padded volume (ideal).
        padded: Padded volume actually processed.
        overhead: ``padded / real - 1`` (fraction of wasted work); 0.0 means no padding.
    """

    n_buckets: int
    real: int
    padded: int
    overhead: float


def padding_report(shapes: Sequence[ShapeLike], buckets: Sequence[Sequence[int]]) -> PaddingReport:
    """Summarise the padding overhead of a partition.

    Args:
        shapes: Per-observation ``(detector_count, sample_count, ...)`` shapes.
        buckets: Partition of observation indices into buckets.

    Returns:
        A [`PaddingReport`][furax.mapmaking._bucketing.PaddingReport].

    Examples:
        The single-bucket partition reproduces the current global-max padding:

        >>> shapes = [(2, 100), (2, 100), (2, 10)]
        >>> padding_report(shapes, [[0, 1, 2]]).overhead  # padded 3*2*100=600 vs real 420
        0.4285714285714286
    """
    real = real_volume(shapes)
    padded = padded_volume(shapes, buckets)
    overhead = (padded / real - 1.0) if real > 0 else 0.0
    return PaddingReport(len(buckets), real, padded, overhead)


def partition_by_size(
    shapes: Sequence[ShapeLike], n_buckets: int, min_bucket: int = 1
) -> list[list[int]]:
    r"""Partition observation indices into ``n_buckets`` size-homogeneous buckets.

    Observations are sorted by sample count, then split into contiguous segments that **minimise
    the total padded volume** $\sum_b |b| \cdot \max_{i \in b} d_i \cdot \max_{i \in b} s_i$
    (dynamic programming over the sorted order). Sorting by sample count is exact for the sample
    axis (a segment's sample-max is its last element); the detector-max is taken over the actual
    segment members, so a mixed detector count is handled correctly.

    Each bucket is later sharded across the full process mesh, so every bucket must hold at least
    ``min_bucket`` observations (typically the process count). ``n_buckets`` is reduced when the
    dataset is too small to fill that many buckets at ``min_bucket`` each.

    Args:
        shapes: Per-observation ``(detector_count, sample_count, ...)`` shapes.
        n_buckets: Desired number of buckets (upper bound; reduced if the dataset is too small).
        min_bucket: Minimum observations per bucket.

    Returns:
        A list of buckets, each a sorted list of original observation indices. Buckets are ordered
        by increasing sample count and their union is ``range(len(shapes))``.

    Raises:
        ValueError: If ``n_buckets < 1``, ``min_bucket < 1``, or the dataset cannot fill even one
            bucket of ``min_bucket`` observations.

    Examples:
        Two clusters of sizes split cleanly, eliminating the padding:

        >>> shapes = [(2, 10), (2, 10), (2, 100), (2, 100)]
        >>> partition_by_size(shapes, n_buckets=2)
        [[0, 1], [2, 3]]
    """
    if n_buckets < 1:
        raise ValueError(f'n_buckets must be >= 1, got {n_buckets}')
    if min_bucket < 1:
        raise ValueError(f'min_bucket must be >= 1, got {min_bucket}')
    n = len(shapes)
    if n < min_bucket:
        raise ValueError(f'need at least min_bucket={min_bucket} observations, got {n}')

    # Cap the bucket count at what the dataset can fill with >= min_bucket observations each.
    n_buckets = min(n_buckets, n // min_bucket)

    det = np.array([int(s[0]) for s in shapes], dtype=np.int64)
    samp = np.array([int(s[1]) for s in shapes], dtype=np.int64)
    order = np.argsort(samp, kind='stable')  # sample-ascending; segment sample-max is the last item
    det_s = det[order]
    samp_s = samp[order]

    if n_buckets == 1:
        return [sorted(int(i) for i in order)]

    # DP over the sorted order. dp[b][j] = min padded volume to split the first j items into b
    # buckets (each of size >= min_bucket). cut[b][j] records the start of the last segment.
    inf = np.iinfo(np.int64).max
    dp = np.full((n_buckets + 1, n + 1), inf, dtype=np.int64)
    cut = np.zeros((n_buckets + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0

    for b in range(1, n_buckets + 1):
        for j in range(b * min_bucket, n + 1):
            # Last segment is [i, j); grow it downward from j, tracking its detector/sample max.
            best = inf
            best_i = -1
            run_det = 0
            for i in range(j - 1, (b - 1) * min_bucket - 1, -1):
                run_det = max(run_det, int(det_s[i]))
                seg_len = j - i
                if seg_len < min_bucket:
                    continue
                prev = dp[b - 1, i]
                if prev == inf:
                    continue
                seg_cost = seg_len * run_det * int(samp_s[j - 1])  # sorted: sample-max at j-1
                total = prev + seg_cost
                if total < best:
                    best = total
                    best_i = i
            dp[b, j] = best
            cut[b, j] = best_i

    # Backtrack the optimal cuts into contiguous segments of the sorted order.
    buckets: list[list[int]] = []
    j = n
    for b in range(n_buckets, 0, -1):
        i = int(cut[b, j])
        buckets.append(sorted(int(k) for k in order[i:j]))
        j = i
    buckets.reverse()
    return buckets


def partition_balanced(
    shapes: Sequence[ShapeLike], n_segments: int, min_bucket: int = 1
) -> list[list[int]]:
    r"""Split observations into ``n_segments`` size-contiguous, volume-balanced segments.

    The assignment target for the *bucket-per-process* execution (each process owns one segment and
    processes it on its local devices): observations are sorted by sample count and cut into
    contiguous segments that **minimise the largest segment's real volume** (a makespan / bin-balance
    objective, $\min \max_b \sum_{i \in b} d_i s_i$, via dynamic programming). Contiguity keeps each
    segment size-homogeneous, so a process pads only to its own segment's max; balancing the real
    volume keeps per-process work even, so the slowest process does not dominate wall time.

    A short-observation process therefore holds *many* observations and a long-observation process
    *few*, both doing comparable real work -- unlike the count-balanced index split, which gives a
    long-observation process far more work.

    Args:
        shapes: Per-observation ``(detector_count, sample_count, ...)`` shapes.
        n_segments: Desired number of segments (typically the process count; reduced if the dataset
            is too small to fill that many at ``min_bucket`` each).
        min_bucket: Minimum observations per segment.

    Returns:
        A list of segments, each a sorted list of original observation indices, ordered by
        increasing sample count; their union is ``range(len(shapes))``.

    Raises:
        ValueError: If ``n_segments < 1``, ``min_bucket < 1``, or the dataset cannot fill even one
            segment of ``min_bucket`` observations.

    Examples:
        Balances real volume rather than count: the short-scan segment takes more observations.

        >>> shapes = [(2, 10), (2, 10), (2, 10), (2, 90)]
        >>> partition_balanced(shapes, n_segments=2)
        [[0, 1, 2], [3]]
    """
    if n_segments < 1:
        raise ValueError(f'n_segments must be >= 1, got {n_segments}')
    if min_bucket < 1:
        raise ValueError(f'min_bucket must be >= 1, got {min_bucket}')
    n = len(shapes)
    if n < min_bucket:
        raise ValueError(f'need at least min_bucket={min_bucket} observations, got {n}')

    n_segments = min(n_segments, n // min_bucket)

    vol = np.array([buffer_volume(s) for s in shapes], dtype=np.int64)
    samp = np.array([int(s[1]) for s in shapes], dtype=np.int64)
    order = np.argsort(samp, kind='stable')
    vol_s = vol[order]
    prefix = np.concatenate([[0], np.cumsum(vol_s)])  # prefix[j] = real volume of first j items

    def seg_volume(i: int, j: int) -> int:
        return int(prefix[j] - prefix[i])

    if n_segments == 1:
        return [sorted(int(i) for i in order)]

    # dp[b][j] = min achievable largest-segment volume splitting the first j items into b segments
    # (each of size >= min_bucket). cut[b][j] records the start of the last segment.
    inf = np.iinfo(np.int64).max
    dp = np.full((n_segments + 1, n + 1), inf, dtype=np.int64)
    cut = np.zeros((n_segments + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0

    for b in range(1, n_segments + 1):
        for j in range(b * min_bucket, n + 1):
            best = inf
            best_i = -1
            for i in range((b - 1) * min_bucket, j - min_bucket + 1):
                prev = dp[b - 1, i]
                if prev == inf:
                    continue
                bottleneck = max(int(prev), seg_volume(i, j))  # largest segment so far
                if bottleneck < best:
                    best = bottleneck
                    best_i = i
            dp[b, j] = best
            cut[b, j] = best_i

    segments: list[list[int]] = []
    j = n
    for b in range(n_segments, 0, -1):
        i = int(cut[b, j])
        segments.append(sorted(int(k) for k in order[i:j]))
        j = i
    segments.reverse()
    return segments
