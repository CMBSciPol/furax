"""Assign observations to processes, keeping buffer padding down.

Observations read together are padded to a common shape, the per-axis maximum over the group, so
a short observation grouped with a long one is padded up to the long one. That padding costs
compute and memory, so the grouping decides how much of a run is wasted on it.

[`partition_balanced`][] splits the observations into one group per process.
[`PaddingReport`][] measures what a grouping costs.
"""

from collections.abc import Sequence
from typing import NamedTuple, Self

import numpy as np

from ._observation import ObservationBufferShape


def real_volume(shapes: Sequence[ObservationBufferShape]) -> int:
    """Total volume of the observations, before any padding."""
    return sum(s.volume for s in shapes)


def group_padded_volume(shapes: Sequence[ObservationBufferShape], group: Sequence[int]) -> int:
    """Volume one group occupies once its observations are padded to a common shape.

    Every observation is padded to the largest detector count and the largest sample count in the
    group, so the group holds that many elements per observation.
    """
    if len(group) == 0:
        return 0
    envelope = ObservationBufferShape(
        max(shapes[i].detector_count for i in group),
        max(shapes[i].sample_count for i in group),
    )
    return len(group) * envelope.volume


def padded_volume(shapes: Sequence[ObservationBufferShape], groups: Sequence[Sequence[int]]) -> int:
    """Total padded volume of a partition, summed over its groups."""
    return sum(group_padded_volume(shapes, g) for g in groups)


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
        cls, shapes: Sequence[ObservationBufferShape], groups: Sequence[Sequence[int]]
    ) -> Self:
        """Measure what a partition of the observations costs in padding.

        Args:
            shapes: Per-observation buffer shapes.
            groups: Partition of observation indices into groups.

        Returns:
            The padding cost of that partition.

        Examples:
            Grouping everything together pads every observation to the largest one:

            >>> from furax.mapmaking import ObservationBufferShape as Shape
            >>> shapes = [Shape(2, 100), Shape(2, 100), Shape(2, 10)]
            >>> PaddingReport.create(shapes, [[0, 1, 2]]).overhead  # padded 600 vs real 420
            0.4285714285714286
        """
        return cls(len(groups), real_volume(shapes), padded_volume(shapes, groups))

    @property
    def overhead(self) -> float:
        """Fraction of wasted work, ``padded / real - 1``; 0.0 means no padding."""
        return (self.padded / self.real - 1.0) if self.real > 0 else 0.0


def partition_balanced(
    shapes: Sequence[ObservationBufferShape], n_groups: int, min_group: int = 1
) -> list[list[int]]:
    r"""Split observations into ``n_groups`` groups of comparable cost.

    Observations in a group $b$ are padded to a common shape, so the group occupies
    $|b| \cdot \max_{i \in b} d_i \cdot \max_{i \in b} s_i$ elements, where observation $i$ has
    $d_i$ detectors and $s_i$ samples. Processing cost is roughly proportional to that volume, and
    the groups minimise the largest of them.

    Candidate groups are runs of the observations sorted by shape, so the partition is not always
    the best possible.

    Args:
        shapes: Per-observation buffer shapes.
        n_groups: Desired number of groups (typically the process count; reduced if the dataset
            is too small to fill that many at ``min_group`` each).
        min_group: Minimum observations per group.

    Returns:
        A list of groups. Each group is a sorted list of original observation indices. The groups
        are ordered by increasing detector count, then sample count, and together they cover
        ``range(len(shapes))``.

    Raises:
        ValueError: If ``n_groups < 1``, ``min_group < 1``, or the dataset cannot fill even one
            group of ``min_group`` observations.

    Examples:
        Three short scans, one long scan:

        >>> from furax.mapmaking import ObservationBufferShape as Shape
        >>> shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
        >>> partition_balanced(shapes, n_groups=2)
        [[0, 1, 2], [3]]

        Observations are grouped by detector count first, so the wide observation is not padded
        onto the narrow ones:

        >>> shapes = [Shape(1, 10), Shape(10, 20), Shape(1, 30)]
        >>> partition_balanced(shapes, n_groups=2)
        [[0, 2], [1]]
    """
    if n_groups < 1:
        raise ValueError(f'n_groups must be >= 1, got {n_groups}')
    if min_group < 1:
        raise ValueError(f'min_group must be >= 1, got {min_group}')
    n = len(shapes)
    if n < min_group:
        raise ValueError(f'need at least min_group={min_group} observations, got {n}')

    n_groups = min(n_groups, n // min_group)

    if n_groups == 1:
        return [list(range(n))]

    # Sort by detector count first, then by sample count. Observations with the same detector
    # count stay together, so a group usually pads along the sample axis only. When every
    # observation has the same detector count, this is just sorting by length. Considering only
    # consecutive runs keeps the search polynomial; the unrestricted problem is NP-hard.
    order = sorted(range(n), key=lambda i: (shapes[i].detector_count, shapes[i].sample_count))
    det_s = [shapes[i].detector_count for i in order]
    samp_s = [shapes[i].sample_count for i in order]

    # dp[b][j] is the lowest cost the most expensive group can have when the first j
    # observations are split into b groups of at least min_group each. cut[b][j] records where
    # the last of those groups starts.
    inf = np.iinfo(np.int64).max
    dp = np.full((n_groups + 1, n + 1), inf, dtype=np.int64)
    cut = np.zeros((n_groups + 1, n + 1), dtype=np.int64)
    dp[0, 0] = 0

    for b in range(1, n_groups + 1):
        for j in range(b * min_group, n + 1):
            best = inf
            best_i = -1
            # Try every start i for the last group [i, j), tracking the shape it pads to.
            run_det = 0
            run_samp = 0
            for i in range(j - 1, (b - 1) * min_group - 1, -1):
                run_det = max(run_det, det_s[i])
                run_samp = max(run_samp, samp_s[i])
                if j - i < min_group:
                    continue
                prev = dp[b - 1, i]
                if prev == inf:
                    continue
                bottleneck = max(int(prev), (j - i) * run_det * run_samp)
                if bottleneck < best:
                    best = bottleneck
                    best_i = i
            dp[b, j] = best
            cut[b, j] = best_i

    groups: list[list[int]] = []
    j = n
    for b in range(n_groups, 0, -1):
        i = int(cut[b, j])
        groups.append(sorted(order[i:j]))
        j = i
    groups.reverse()
    return groups
