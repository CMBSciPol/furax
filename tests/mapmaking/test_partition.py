from itertools import combinations

import pytest

from furax.mapmaking import ObservationBufferShape as Shape
from furax.mapmaking._partition import (
    PaddingReport,
    group_padded_volume,
    padded_volume,
    partition_balanced,
    real_volume,
)


def test_volume_helpers():
    shapes = [Shape(2, 100), Shape(3, 10)]
    assert real_volume(shapes) == 2 * 100 + 3 * 10
    assert group_padded_volume(shapes, [0, 1]) == 2 * 3 * 100  # count * max_det * max_samp
    assert group_padded_volume(shapes, []) == 0
    assert padded_volume(shapes, [[0], [1]]) == real_volume(shapes)  # singletons never pad
    assert padded_volume(shapes, [[0, 1]]) == 2 * 3 * 100  # grouped, both pad to (3, 100)


def test_interval_axis_is_not_charged_for():
    # Scanning intervals are metadata, not time-ordered samples.
    assert Shape(2, 100, 7).volume == Shape(2, 100).volume


def test_single_group_pads_to_the_largest_observation():
    shapes = [Shape(2, 100), Shape(2, 100), Shape(2, 10)]
    report = PaddingReport.create(shapes, [[0, 1, 2]])
    assert report.n_groups == 1
    assert report.real == 420
    assert report.padded == 600  # 3 * 2 * 100
    assert report.overhead == pytest.approx(600 / 420 - 1)


def _brute_force_makespan(shapes, n_groups, min_group):
    """Reference min largest-group padded volume over runs of the lexicographic order."""
    order = sorted(
        range(len(shapes)), key=lambda i: (shapes[i].detector_count, shapes[i].sample_count)
    )
    n = len(shapes)
    best = None
    for cuts in combinations(range(1, n), n_groups - 1):
        bounds = [0, *cuts, n]
        segs = [order[bounds[k] : bounds[k + 1]] for k in range(n_groups)]
        if any(len(s) < min_group for s in segs):
            continue
        makespan = _group_makespan(shapes, segs)
        best = makespan if best is None else min(best, makespan)
    return best


def _group_makespan(shapes, groups):
    """The cost of the busiest process: what it pads to, times how many slots it carries."""
    return max(group_padded_volume(shapes, s) for s in groups)


def test_partition_respects_the_return_contract():
    shapes = [Shape(2, s) for s in (10, 50, 12, 90, 11, 88, 49)]
    groups = partition_balanced(shapes, n_groups=3)
    assert len(groups) == 3  # as many groups as asked for
    assert min(len(s) for s in groups) >= 1  # none of them empty
    assert sorted(i for s in groups for i in s) == list(range(len(shapes)))  # exhaustive, no dupes
    assert groups == [sorted(s) for s in groups]  # each group sorted
    keys = [max((shapes[i].detector_count, shapes[i].sample_count) for i in s) for s in groups]
    assert keys == sorted(keys)  # groups ordered by increasing shape


def test_partition_favours_volume_not_count():
    # Three short + one long: volume balance puts all shorts in one group, the long alone.
    shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
    assert partition_balanced(shapes, n_groups=2) == [[0, 1, 2], [3]]


@pytest.mark.parametrize('n_groups', [1, 2, 3, 4])
@pytest.mark.parametrize('min_group', [1, 2])
def test_partition_matches_brute_force(n_groups, min_group):
    shapes = [
        Shape(2, 10),
        Shape(2, 90),
        Shape(3, 12),
        Shape(2, 88),
        Shape(3, 50),
        Shape(2, 49),
        Shape(2, 30),
    ]
    if len(shapes) < n_groups * min_group:
        pytest.skip('dataset too small for this group count')
    groups = partition_balanced(shapes, n_groups, min_group)
    assert all(len(s) >= min_group for s in groups)
    assert _group_makespan(shapes, groups) == _brute_force_makespan(shapes, n_groups, min_group)


def test_partition_groups_on_detector_count_before_length():
    # Sorting on length alone would put the wide observation next to a narrow one and pad it onto
    # the group; leading with the detector count keeps the two narrow ones together instead.
    shapes = [Shape(1, 10), Shape(10, 20), Shape(1, 30)]
    groups = partition_balanced(shapes, n_groups=2)
    assert groups == [[0, 2], [1]]
    by_length = [[0, 1], [2]]  # what the sample-sorted order yields
    assert _group_makespan(shapes, groups) < _group_makespan(shapes, by_length)


def test_partition_caps_and_validates():
    shapes = [Shape(2, s) for s in (10, 20, 30, 40, 50)]
    # Ask for 5 groups but require >=2 each: only 2 groups fit (5 // 2).
    assert len(partition_balanced(shapes, n_groups=5, min_group=2)) == 2
    with pytest.raises(ValueError, match='n_groups must be'):
        partition_balanced(shapes, n_groups=0)
    with pytest.raises(ValueError, match='min_group must be'):
        partition_balanced(shapes, n_groups=2, min_group=0)
    with pytest.raises(ValueError, match='at least min_group'):
        partition_balanced(shapes, n_groups=1, min_group=99)
