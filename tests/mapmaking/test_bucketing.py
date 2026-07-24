import itertools

import pytest

from furax.mapmaking._bucketing import (
    bucket_padded_volume,
    buffer_volume,
    padded_volume,
    padding_report,
    partition_balanced,
    partition_by_size,
    real_volume,
)


def _brute_force_min(shapes, n_buckets, min_bucket):
    """Reference minimum padded volume over *all* contiguous partitions of the sorted order."""
    samp = sorted(range(len(shapes)), key=lambda i: shapes[i][1])
    n = len(shapes)
    best = None
    # Choose n_buckets-1 internal cut points respecting the min-size constraint.
    for cuts in itertools.combinations(range(1, n), n_buckets - 1):
        bounds = [0, *cuts, n]
        segs = [samp[bounds[k] : bounds[k + 1]] for k in range(n_buckets)]
        if any(len(s) < min_bucket for s in segs):
            continue
        vol = padded_volume(shapes, segs)
        best = vol if best is None else min(best, vol)
    return best


def test_volume_helpers():
    shapes = [(2, 100), (3, 10)]
    assert real_volume(shapes) == 2 * 100 + 3 * 10
    assert bucket_padded_volume(shapes, [0, 1]) == 2 * 3 * 100  # count * max_det * max_samp
    assert bucket_padded_volume(shapes, []) == 0


def test_single_bucket_matches_global_padding():
    shapes = [(2, 100), (2, 100), (2, 10)]
    report = padding_report(shapes, [[0, 1, 2]])
    assert report.n_buckets == 1
    assert report.real == 420
    assert report.padded == 600  # 3 * 2 * 100
    assert report.overhead == pytest.approx(600 / 420 - 1)


def test_partition_is_a_valid_partition():
    shapes = [(2, s) for s in (10, 50, 12, 90, 11, 88, 49)]
    buckets = partition_by_size(shapes, n_buckets=3)
    flat = sorted(i for b in buckets for i in b)
    assert flat == list(range(len(shapes)))  # covers every index exactly once
    assert all(b == sorted(b) for b in buckets)


def test_separates_size_clusters_with_zero_overhead():
    # Two tight clusters -> two buckets remove all padding.
    shapes = [(2, 10), (2, 10), (2, 100), (2, 100)]
    buckets = partition_by_size(shapes, n_buckets=2)
    assert buckets == [[0, 1], [2, 3]]
    assert padding_report(shapes, buckets).overhead == pytest.approx(0.0)


def test_more_buckets_never_increase_padding():
    shapes = [(2, s) for s in (10, 20, 30, 40, 55, 60, 90, 95, 12, 33)]
    vols = [padded_volume(shapes, partition_by_size(shapes, b)) for b in range(1, 6)]
    assert all(later <= earlier for earlier, later in zip(vols, vols[1:]))


@pytest.mark.parametrize('n_buckets', [1, 2, 3, 4])
@pytest.mark.parametrize('min_bucket', [1, 2])
def test_dp_matches_brute_force(n_buckets, min_bucket):
    shapes = [(d, s) for d, s in [(2, 10), (2, 90), (3, 12), (2, 88), (3, 50), (2, 49), (2, 30)]]
    if len(shapes) < n_buckets * min_bucket:
        pytest.skip('dataset too small for this bucket count')
    buckets = partition_by_size(shapes, n_buckets, min_bucket)
    assert all(len(b) >= min_bucket for b in buckets)
    assert padded_volume(shapes, buckets) == _brute_force_min(shapes, n_buckets, min_bucket)


def test_min_bucket_respected_and_bucket_count_capped():
    shapes = [(2, s) for s in (10, 20, 30, 40, 50)]
    # Ask for 5 buckets but require >=2 each: only 2 buckets fit (5 // 2).
    buckets = partition_by_size(shapes, n_buckets=5, min_bucket=2)
    assert len(buckets) == 2
    assert all(len(b) >= 2 for b in buckets)


def test_invalid_arguments():
    shapes = [(2, 10), (2, 20)]
    with pytest.raises(ValueError, match='n_buckets must be'):
        partition_by_size(shapes, n_buckets=0)
    with pytest.raises(ValueError, match='min_bucket must be'):
        partition_by_size(shapes, n_buckets=1, min_bucket=0)
    with pytest.raises(ValueError, match='at least min_bucket'):
        partition_by_size(shapes, n_buckets=1, min_bucket=5)


def _brute_force_makespan(shapes, n_segments, min_bucket):
    """Reference min largest-segment real volume over contiguous size-sorted partitions."""
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][1])
    n = len(shapes)
    best = None
    for cuts in itertools.combinations(range(1, n), n_segments - 1):
        bounds = [0, *cuts, n]
        segs = [order[bounds[k] : bounds[k + 1]] for k in range(n_segments)]
        if any(len(s) < min_bucket for s in segs):
            continue
        makespan = max(sum(buffer_volume(shapes[i]) for i in s) for s in segs)
        best = makespan if best is None else min(best, makespan)
    return best


def _segment_makespan(shapes, segments):
    return max(sum(buffer_volume(shapes[i]) for i in s) for s in segments)


def test_balanced_is_a_valid_partition():
    shapes = [(2, s) for s in (10, 50, 12, 90, 11, 88, 49)]
    segments = partition_balanced(shapes, n_segments=3)
    flat = sorted(i for s in segments for i in s)
    assert flat == list(range(len(shapes)))
    assert all(s == sorted(s) for s in segments)


def test_balanced_favours_volume_not_count():
    # Three short + one long: volume balance puts all shorts in one segment, the long alone.
    shapes = [(2, 10), (2, 10), (2, 10), (2, 90)]
    assert partition_balanced(shapes, n_segments=2) == [[0, 1, 2], [3]]


@pytest.mark.parametrize('n_segments', [1, 2, 3, 4])
@pytest.mark.parametrize('min_bucket', [1, 2])
def test_balanced_matches_brute_force(n_segments, min_bucket):
    shapes = [(d, s) for d, s in [(2, 10), (2, 90), (3, 12), (2, 88), (3, 50), (2, 49), (2, 30)]]
    if len(shapes) < n_segments * min_bucket:
        pytest.skip('dataset too small for this segment count')
    segments = partition_balanced(shapes, n_segments, min_bucket)
    assert all(len(s) >= min_bucket for s in segments)
    assert _segment_makespan(shapes, segments) == _brute_force_makespan(
        shapes, n_segments, min_bucket
    )


def test_balanced_beats_count_split_on_makespan():
    # Skewed sizes: contiguous count-split (index order already size-sorted here) is worse.
    shapes = [(2, s) for s in (5, 6, 7, 8, 9, 10, 200)]
    n = len(shapes)
    half = n // 2
    count_split = [list(range(half)), list(range(half, n))]
    balanced = partition_balanced(shapes, n_segments=2)
    assert _segment_makespan(shapes, balanced) <= _segment_makespan(shapes, count_split)


def test_balanced_caps_and_validates():
    shapes = [(2, s) for s in (10, 20, 30, 40, 50)]
    assert len(partition_balanced(shapes, n_segments=5, min_bucket=2)) == 2
    with pytest.raises(ValueError, match='n_segments must be'):
        partition_balanced(shapes, n_segments=0)
    with pytest.raises(ValueError, match='min_bucket must be'):
        partition_balanced(shapes, n_segments=2, min_bucket=0)
    with pytest.raises(ValueError, match='at least min_bucket'):
        partition_balanced(shapes, n_segments=1, min_bucket=99)
