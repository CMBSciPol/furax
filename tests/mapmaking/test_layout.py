from itertools import combinations

import numpy as np
import pytest

from furax.mapmaking import ObservationBufferShape as Shape
from furax.mapmaking._layout import (
    Bucket,
    PaddingReport,
    SlotLayout,
    padded_volume,
    partition_padded,
    real_volume,
)

SHAPES = [Shape(2, 10), Shape(2, 90), Shape(3, 12), Shape(2, 88), Shape(3, 50), Shape(2, 49)]


# ---------------------------------------------------------------------------
# Bucket: one group and its padding cost
# ---------------------------------------------------------------------------


def test_bucket_envelope_and_volume():
    shapes = [Shape(2, 100), Shape(3, 10)]
    bucket = Bucket.create(shapes, [1, 0])
    np.testing.assert_array_equal(bucket.observations, [0, 1])  # sorted
    assert bucket.shape == Shape(3, 100)
    assert bucket.n_slots == 2
    assert bucket.padded_volume == 2 * 3 * 100  # count * max_det * max_samp
    assert real_volume(shapes) == 2 * 100 + 3 * 10
    assert padded_volume(shapes, [[0], [1]]) == real_volume(shapes)  # singletons never pad
    with pytest.raises(ValueError, match='at least one observation'):
        Bucket.create(shapes, [])


def test_bucket_slot_rounding_is_charged_for():
    assert Bucket.slot_count(5, n_devices=1) == 5
    assert Bucket.slot_count(5, n_devices=4) == 8
    assert Bucket.slot_count(8, n_devices=4) == 8
    shapes = [Shape(2, 100), Shape(3, 10)]
    # one group of two on four devices: four slots of the (3, 100) envelope
    assert Bucket.create(shapes, [0, 1], n_devices=4).padded_volume == 4 * 3 * 100
    # two singletons on four devices: four slots each of their own shape
    assert padded_volume(shapes, [[0], [1]], n_devices=4) == 4 * 200 + 4 * 30


def test_bucket_envelope_covers_the_interval_axis():
    bucket = Bucket.create([Shape(2, 100, 7), Shape(2, 50, 9)], [0, 1])
    assert bucket.shape == Shape(2, 100, 9)
    # Scanning intervals are metadata, not time-ordered samples.
    assert bucket.padded_volume == 2 * 2 * 100


def test_bucket_slot_bookkeeping():
    bucket = Bucket(observations=np.array([1, 4]), n_slots=4, shape=Shape(3, 90))
    assert (bucket.n_real, bucket.n_pad) == (2, 2)
    np.testing.assert_array_equal(bucket.is_real, [True, True, False, False])
    # empty slots point at a valid item; they are gated out by `is_real`, never read
    np.testing.assert_array_equal(bucket.item_of_slot, [0, 1, 1, 1])


def test_single_group_pads_to_the_largest_observation():
    shapes = [Shape(2, 100), Shape(2, 100), Shape(2, 10)]
    report = PaddingReport.create(shapes, [[0, 1, 2]])
    assert report.n_groups == 1
    assert report.real == 420
    assert report.padded == 600  # 3 * 2 * 100
    assert report.overhead == pytest.approx(600 / 420 - 1)
    assert PaddingReport.create(shapes, [[0, 1, 2]], n_devices=2).padded == 800


# ---------------------------------------------------------------------------
# partition_padded
# ---------------------------------------------------------------------------


def _brute_force_total(shapes, max_groups, n_devices):
    """Reference least total padded volume over runs of the lexicographic order."""
    order = sorted(
        range(len(shapes)), key=lambda i: (shapes[i].detector_count, shapes[i].sample_count)
    )
    n = len(shapes)
    best = None
    for n_groups in range(1, min(max_groups, n) + 1):
        for cuts in combinations(range(1, n), n_groups - 1):
            bounds = [0, *cuts, n]
            segs = [order[bounds[k] : bounds[k + 1]] for k in range(n_groups)]
            total = padded_volume(shapes, segs, n_devices)
            best = total if best is None else min(best, total)
    return best


def test_partition_respects_the_return_contract():
    shapes = [Shape(2, s) for s in (10, 50, 12, 90, 11, 88, 49)]
    groups = partition_padded(shapes, max_groups=3, n_devices=2)
    assert 1 <= len(groups) <= 3
    assert min(len(s) for s in groups) >= 1  # none of them empty
    assert sorted(i for s in groups for i in s) == list(range(len(shapes)))  # exhaustive, no dupes
    assert groups == [sorted(s) for s in groups]  # each group sorted
    keys = [max((shapes[i].detector_count, shapes[i].sample_count) for i in s) for s in groups]
    assert keys == sorted(keys)  # groups ordered by increasing shape


def test_partition_isolates_the_outlier():
    # Three short + one long: the long one alone, rather than padding the shorts up to it.
    shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
    assert partition_padded(shapes, max_groups=2) == [[0, 1, 2], [3]]


def test_partition_weighs_slot_rounding_against_shape_padding():
    shapes = [Shape(2, 10), Shape(2, 10), Shape(2, 10), Shape(2, 90)]
    # On four devices a singleton costs four slots of the long scan: one group is cheaper.
    assert partition_padded(shapes, max_groups=2, n_devices=4) == [[0, 1, 2, 3]]
    # On two devices a group of three pads to four slots: moving one short scan in with the long
    # one fills its second slot for free (2 * 10 + 2 * 90 < 4 * 10 + 2 * 90).
    assert partition_padded(shapes, max_groups=2, n_devices=2) == [[0, 1], [2, 3]]


def test_partition_never_uses_more_groups_than_helps():
    # Identical observations: any split only adds slot rounding, so one group is returned.
    shapes = [Shape(2, 10)] * 6
    assert partition_padded(shapes, max_groups=4, n_devices=4) == [list(range(6))]
    # With no rounding, splitting identical shapes changes nothing: the fewest groups win ties.
    assert partition_padded(shapes, max_groups=4, n_devices=1) == [list(range(6))]


@pytest.mark.parametrize('max_groups', [1, 2, 3, 4])
@pytest.mark.parametrize('n_devices', [1, 2, 3])
def test_partition_matches_brute_force(max_groups, n_devices):
    shapes = [Shape(2, 30), *SHAPES]
    groups = partition_padded(shapes, max_groups, n_devices=n_devices)
    assert len(groups) <= max_groups
    assert padded_volume(shapes, groups, n_devices) == _brute_force_total(
        shapes, max_groups, n_devices
    )


def test_partition_groups_on_detector_count_before_length():
    # Sorting on length alone would put the wide observation next to a narrow one and pad it onto
    # the group; leading with the detector count keeps the two narrow ones together instead.
    shapes = [Shape(1, 10), Shape(10, 20), Shape(1, 30)]
    groups = partition_padded(shapes, max_groups=2)
    assert groups == [[0, 2], [1]]
    by_length = [[0, 1], [2]]  # what the sample-sorted order yields
    assert padded_volume(shapes, groups) < padded_volume(shapes, by_length)


def test_partition_caps_and_validates():
    shapes = [Shape(2, s) for s in (10, 20, 30, 40, 50)]
    # Ask for more groups than observations: capped at one per observation.
    assert partition_padded(shapes, max_groups=9) == [[i] for i in range(5)]
    with pytest.raises(ValueError, match='max_groups must be'):
        partition_padded(shapes, max_groups=0)
    with pytest.raises(ValueError, match='n_devices must be'):
        partition_padded(shapes, max_groups=2, n_devices=0)
    with pytest.raises(ValueError, match='at least one observation'):
        partition_padded([], max_groups=1)


# ---------------------------------------------------------------------------
# SlotLayout: buckets on the mesh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('n_devices', [1, 2, 4])
@pytest.mark.parametrize('max_buckets', [1, 2, 4])
def test_layout_covers_every_observation_once(n_devices, max_buckets):
    layout = SlotLayout.create(SHAPES, n_devices=n_devices, max_buckets=max_buckets)
    assert 1 <= len(layout.buckets) <= max_buckets
    covered = np.concatenate([b.observations for b in layout.buckets])
    assert sorted(covered.tolist()) == list(range(len(SHAPES)))
    for bucket in layout.buckets:
        assert bucket.n_slots % n_devices == 0
        assert bucket.n_real <= bucket.n_slots < bucket.n_real + n_devices
        # the envelope covers every member
        for i in bucket.observations:
            assert SHAPES[i].detector_count <= bucket.shape.detector_count
            assert SHAPES[i].sample_count <= bucket.shape.sample_count


def test_local_slots_tile_the_axis_by_process():
    # 6 observations, 4 devices on 2 processes: 8 slots, 2 per device, 4 per process
    layout = SlotLayout.create(SHAPES, n_devices=4, max_buckets=1)
    (bucket,) = layout.buckets
    assert bucket.n_slots == 8
    blocks = [layout.local_slots(0, process_index=p, n_local=2) for p in range(2)]
    assert blocks == [slice(0, 4), slice(4, 8)]
    # the second process holds the two empty slots, which is a valid (partly empty) block
    np.testing.assert_array_equal(bucket.is_real[blocks[1]], [True, True, False, False])


def test_scatter_restores_observation_order_and_pads_trailing_axes():
    layout = SlotLayout(
        buckets=(
            Bucket(observations=np.array([0, 3]), n_slots=2, shape=Shape(2, 10)),
            Bucket(observations=np.array([1, 2]), n_slots=4, shape=Shape(3, 10)),
        ),
        n_observations=4,
        n_devices=2,
    )
    narrow = np.array([[10.0, 11.0], [30.0, 31.0]])  # (n_slots=2, n_det=2)
    wide = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    out = layout.scatter([narrow, wide])
    expected = np.array([[10.0, 11.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [30.0, 31.0, 0.0]])
    np.testing.assert_array_equal(out, expected)  # empty slots dropped, narrow rows zero-padded
    with pytest.raises(ValueError, match='expected 2 arrays'):
        layout.scatter([narrow])
