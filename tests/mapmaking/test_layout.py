from itertools import combinations

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from furax.mapmaking import ObservationBufferShape as Shape
from furax.mapmaking.layout import (
    Bucket,
    SlotLayout,
    SlotPlacement,
    padded_volume,
    partition_padded,
    unpadded_volume,
)

SHAPES = [Shape(2, 10), Shape(2, 90), Shape(3, 12), Shape(2, 88), Shape(3, 50), Shape(2, 49)]


# ---------------------------------------------------------------------------
# Bucket: one group and its padding cost
# ---------------------------------------------------------------------------


def test_bucket_envelope_and_volume():
    shapes = [Shape(2, 100), Shape(3, 10)]
    bucket = Bucket.create(shapes, [1, 0])
    np.testing.assert_array_equal(bucket.observations, [0, 1])  # sorted
    assert bucket.envelope == Shape(3, 100)
    assert bucket.n_slots == 2
    assert bucket.padded_volume == 2 * 3 * 100  # count * max_det * max_samp
    assert unpadded_volume(shapes) == 2 * 100 + 3 * 10
    assert padded_volume(shapes, [[0], [1]]) == unpadded_volume(shapes)  # singletons never pad
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
    assert bucket.envelope == Shape(2, 100, 9)
    # Scanning intervals are metadata, not time-ordered samples.
    assert bucket.padded_volume == 2 * 2 * 100


def test_bucket_slot_bookkeeping():
    bucket = Bucket(observations=np.array([1, 4]), n_slots=4, envelope=Shape(3, 90))
    assert (bucket.n_observations, bucket.n_empty) == (2, 2)
    np.testing.assert_array_equal(bucket.is_empty, [False, False, True, True])


@pytest.mark.parametrize(
    ('n_dets', 'batch_size', 'padded'),
    [(100, 64, 128), (128, 64, 128), (64, 64, 64), (50, 64, 50), (100, 0, 100)],
)
def test_whole_detector_batches(n_dets, batch_size, padded):
    # a single batch is never padded; more detectors fill whole batches
    shape = Shape(n_dets, 10, 3).whole_detector_batches(batch_size)
    assert shape == Shape(padded, 10, 3)


@pytest.mark.parametrize(
    ('n_dets', 'batch_size', 'n_batches', 'per_batch'),
    [(100, 64, 2, 64), (50, 64, 1, 50), (100, 0, 1, 100)],
)
def test_bucket_detector_batches(n_dets, batch_size, n_batches, per_batch):
    shapes = [Shape(n_dets, 10).whole_detector_batches(batch_size)] * 3
    bucket = Bucket.create(shapes, [0, 1, 2], n_devices=2, detector_batch_size=batch_size)
    assert (bucket.n_batches, bucket.batch_size) == (n_batches, per_batch)


def test_bucket_rejects_an_envelope_of_partial_batches():
    with pytest.raises(ValueError, match='do not split into batches'):
        Bucket.create([Shape(100, 10)], [0], detector_batch_size=64)


def test_merge_detector_batches_restores_the_slots():
    # batch j of slot s holds detectors j·batch_size on of slot s
    bucket = Bucket.create([Shape(6, 10)] * 2, [0, 1], detector_batch_size=2)
    per_slot = np.arange(2 * 6 * 4).reshape(2, 6, 4)
    per_batch = per_slot.reshape(2, 3, 2, 4)
    np.testing.assert_array_equal(bucket.merge_detector_batches(per_batch), per_slot)


def test_bucket_and_layout_use_identity_equality():
    bucket = Bucket.create([Shape(2, 10)], [0])
    equivalent = Bucket.create([Shape(2, 10)], [0])
    assert bucket != equivalent

    layout = SlotLayout((bucket,), n_observations=1, n_devices=1)
    equivalent_layout = SlotLayout((bucket,), n_observations=1, n_devices=1)
    assert layout != equivalent_layout


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
        assert bucket.n_observations <= bucket.n_slots < bucket.n_observations + n_devices
        # the envelope covers every member
        for i in bucket.observations:
            assert SHAPES[i].detector_count <= bucket.envelope.detector_count
            assert SHAPES[i].sample_count <= bucket.envelope.sample_count


def test_local_slots_tile_the_axis_by_process():
    # 6 observations, 4 devices on 2 processes: 8 slots, 2 per device, 4 per process
    layout = SlotLayout.create(SHAPES, n_devices=4, max_buckets=1)
    (bucket,) = layout.buckets
    assert bucket.n_slots == 8
    blocks = [layout.local_slots(0, process_index=p, n_local=2) for p in range(2)]
    assert blocks == [slice(0, 4), slice(4, 8)]
    # the second process holds the two empty slots, which is a valid (partly empty) block
    np.testing.assert_array_equal(bucket.is_empty[blocks[1]], [False, False, True, True])


# ---------------------------------------------------------------------------
# SlotPlacement: the slots on the devices of the job
# ---------------------------------------------------------------------------


def _placement() -> SlotPlacement:
    layout = SlotLayout.create(SHAPES, n_devices=jax.device_count(), max_buckets=1)
    return SlotPlacement.create(layout)


def test_placement_assigns_each_local_device_a_run_of_slots():
    placement = _placement()
    assert placement.local_devices == jax.local_devices()
    n_slots = placement.layout.buckets[0].n_slots
    np.testing.assert_array_equal(
        placement.device_slots(0), np.arange(n_slots).reshape(jax.device_count(), -1)
    )


def test_placement_moves_arrays_between_host_and_devices():
    placement = _placement()
    n = jax.device_count()
    sharded = placement.to_devices([{'x': np.arange(6.0).reshape(2, 3)} for _ in range(n)])
    assert sharded['x'].sharding == placement.sharding
    whole = placement.broadcast({'x': np.ones(3)})
    assert whole['x'].sharding == placement.replicated

    summed = placement.replicate(sharded, lambda x: x.sum(axis=0))
    assert summed['x'].sharding == placement.replicated
    np.testing.assert_array_equal(summed['x'], n * np.arange(6.0).reshape(2, 3))
    gathered = placement.gather(sharded)
    assert isinstance(gathered['x'], np.ndarray)
    assert gathered['x'].shape == (n, 2, 3)

    merged = placement.per_slot(sharded, lambda x: x.reshape(-1, 3), donate=True)
    assert merged['x'].sharding == placement.sharding
    assert merged['x'].shape == (2 * n, 3)
    assert sharded['x'].is_deleted()


def test_placement_needs_the_devices_the_layout_was_planned_for():
    layout = SlotLayout.create(SHAPES, n_devices=jax.device_count() + 1, max_buckets=1)
    with pytest.raises(ValueError, match='planned for'):
        SlotPlacement.create(layout)


def test_placement_shards_the_slots_over_one_mesh_axis():
    layout = SlotLayout.create(SHAPES, n_devices=1, max_buckets=1)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('slot', 'other'))
    with pytest.raises(ValueError, match="along 'slot'"):
        SlotPlacement(layout, mesh)


@pytest.mark.parametrize('detector_axis', [1, 2, -1])
@pytest.mark.parametrize('batch_size', [0, 2], ids=['one-batch', 'batches'])
def test_to_observation_order_merges_detector_batches(detector_axis: int, batch_size: int):
    # two buckets of different widths, with empty slots
    shapes = [Shape(4, 10), Shape(6, 10), Shape(6, 10)]
    buckets = (
        Bucket.create(shapes, [1], n_devices=3, detector_batch_size=batch_size),
        Bucket.create(shapes, [0, 2], n_devices=3, detector_batch_size=batch_size),
    )
    layout = SlotLayout(buckets, n_observations=3, n_devices=3)
    rng = np.random.default_rng(0)
    per_slot = []
    per_batch = []
    # the detector axis of the per-batch arrays, after their slot and batch axes
    batched_axis = detector_axis + 1 if detector_axis >= 0 else detector_axis
    for bucket in buckets:
        n_det = bucket.envelope.detector_count
        # (slot, stream, det, mode) with the detector axis moved to `detector_axis`
        values = np.moveaxis(rng.standard_normal((bucket.n_slots, 2, n_det, 3)), 2, detector_axis)
        per_slot.append(values)
        batches = np.moveaxis(values, detector_axis, 1).reshape(
            bucket.n_slots,
            bucket.n_batches,
            bucket.batch_size,
            *np.delete(values.shape, [0, detector_axis]),
        )
        per_batch.append(np.moveaxis(batches, 2, batched_axis))
    out = layout.to_observation_order(per_batch, detector_axis=batched_axis)
    np.testing.assert_array_equal(out, layout.to_observation_order(per_slot))


def test_scatter_restores_observation_order_and_pads_trailing_axes():
    layout = SlotLayout(
        buckets=(
            Bucket(observations=np.array([0, 3]), n_slots=2, envelope=Shape(2, 10)),
            Bucket(observations=np.array([1, 2]), n_slots=4, envelope=Shape(3, 10)),
        ),
        n_observations=4,
        n_devices=2,
    )
    narrow = np.array([[10.0, 11.0], [30.0, 31.0]])  # (n_slots=2, n_det=2)
    wide = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    out = layout.to_observation_order([narrow, wide])
    expected = np.array([[10.0, 11.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [30.0, 31.0, 0.0]])
    np.testing.assert_array_equal(out, expected)  # empty slots dropped, narrow rows zero-padded
    with pytest.raises(ValueError, match='expected 2 arrays'):
        layout.to_observation_order([narrow])
