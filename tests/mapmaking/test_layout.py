from itertools import combinations, pairwise

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from furax.mapmaking import ObservationBufferShape as Shape
from furax.mapmaking.layout import TileBucket, TileLayout, TilePlacement, partition_tiles

# ---------------------------------------------------------------------------
# TileBucket and partition_tiles: observations split into tiles, spread over the devices
# ---------------------------------------------------------------------------

TILE_SHAPES = [
    Shape(5, 10),
    Shape(2, 12),
    Shape(7, 11),
    Shape(1, 10),
    Shape(4, 90),
    Shape(3, 88),
    Shape(6, 12),
]


@pytest.mark.parametrize('n_devices', [1, 2, 3, 4])
@pytest.mark.parametrize('batch_size', [0, 2, 3])
def test_tile_bucket_holds_each_observation_on_one_device(n_devices, batch_size):
    group = range(len(TILE_SHAPES))
    bucket = TileBucket.create(TILE_SHAPES, group, n_devices, batch_size)
    widest = max(shape.detector_count for shape in TILE_SHAPES)
    assert bucket.tile_size == (min(batch_size, widest) if batch_size else widest)
    assert bucket.envelope.sample_count == max(shape.sample_count for shape in TILE_SHAPES)
    assert bucket.n_tiles % n_devices == 0 and bucket.n_segments % n_devices == 0
    np.testing.assert_array_equal(bucket.observations, list(group))

    tile_device = np.arange(bucket.n_tiles) // bucket.tiles_per_device
    np.testing.assert_array_equal(bucket.owner // bucket.segments_per_device, tile_device)
    assert np.all(bucket.segment_observations[bucket.owner[~bucket.valid]] == -1)
    for segment, observation in enumerate(bucket.segment_observations):
        tiles = bucket.tiles_of(segment)
        if observation < 0:
            assert len(tiles) == 0  # an empty segment owns padding tiles at most
            continue
        n_tiles = -(-TILE_SHAPES[observation].detector_count // bucket.tile_size)
        np.testing.assert_array_equal(
            bucket.tile_starts[tiles], bucket.tile_size * np.arange(n_tiles)
        )
    assert bucket.valid.sum() == sum(
        -(-shape.detector_count // bucket.tile_size) for shape in TILE_SHAPES
    )


def test_tile_bucket_balances_the_tiles_over_the_devices():
    # 3 + 2 + 1 + 1 + 1 tiles: two devices of four, where whole observations in input order
    # would give five and three
    shapes = [Shape(6, 10), Shape(4, 10), Shape(2, 10), Shape(2, 10), Shape(2, 10)]
    bucket = TileBucket.create(shapes, range(5), n_devices=2, detector_batch_size=2)
    assert bucket.tiles_per_device == 4
    assert bucket.valid.all()  # no padding tile
    assert bucket.padded_volume == 8 * 2 * 10


def test_tile_bucket_pads_devices_and_segments():
    # one observation of three tiles, one of one tile: two devices, the second one padded, its
    # padding tiles owning an empty segment of their own
    shapes = [Shape(6, 10), Shape(2, 10)]
    bucket = TileBucket.create(shapes, [0, 1], n_devices=2, detector_batch_size=2)
    assert bucket.valid.tolist() == [True, True, True, True, False, False]
    assert bucket.segment_observations.tolist() == [0, -1, 1, -1]
    assert bucket.owner.tolist() == [0, 0, 0, 2, 3, 3]
    # three observations on two devices: the device with one observation has an empty segment
    bucket = TileBucket.create([Shape(4, 10)] * 3, [0, 1, 2], n_devices=2, detector_batch_size=2)
    assert bucket.segment_observations.tolist() == [0, 2, 1, -1]
    with pytest.raises(ValueError, match='at least one observation'):
        TileBucket.create(shapes, [])


def _tile_cost(shapes, group, n_devices, batch_size):
    """The padded volume `partition_tiles` charges a group: an even share of tiles per device."""
    widest = max(shapes[i].detector_count for i in group)
    tile_size = min(batch_size, widest) if batch_size else widest
    tiles = [-(-shapes[i].detector_count // tile_size) for i in group]
    per_device = max(-(-sum(tiles) // n_devices), max(tiles))
    return n_devices * per_device * tile_size * max(shapes[i].sample_count for i in group)


@pytest.mark.parametrize('max_groups', [1, 2, 3])
@pytest.mark.parametrize('n_devices', [1, 3])
@pytest.mark.parametrize('batch_size', [0, 2])
def test_partition_tiles_matches_brute_force(max_groups, n_devices, batch_size):
    shapes = TILE_SHAPES
    groups = partition_tiles(
        shapes, max_groups, n_devices=n_devices, detector_batch_size=batch_size
    )
    assert 1 <= len(groups) <= max_groups
    assert sorted(i for g in groups for i in g) == list(range(len(shapes)))
    order = sorted(
        range(len(shapes)), key=lambda i: (shapes[i].sample_count, shapes[i].detector_count)
    )
    best = min(
        sum(
            _tile_cost(shapes, order[bounds[k] : bounds[k + 1]], n_devices, batch_size)
            for k in range(len(bounds) - 1)
        )
        for n_groups in range(1, max_groups + 1)
        for cuts in combinations(range(1, len(shapes)), n_groups - 1)
        for bounds in [[0, *cuts, len(shapes)]]
    )
    assert sum(_tile_cost(shapes, g, n_devices, batch_size) for g in groups) == best


def test_partition_tiles_groups_on_length_only():
    # the tiles absorb the difference in width: the wide observation joins the one of its length
    shapes = [Shape(100, 10), Shape(800, 10), Shape(300, 90)]
    assert partition_tiles(shapes, max_groups=2, detector_batch_size=100) == [[0, 1], [2]]


def test_partition_tiles_caps_and_validates():
    shapes = [Shape(2, s) for s in (10, 20, 30)]
    assert partition_tiles(shapes, max_groups=9) == [[0], [1], [2]]
    with pytest.raises(ValueError, match='max_groups must be'):
        partition_tiles(shapes, max_groups=0)
    with pytest.raises(ValueError, match='n_devices must be'):
        partition_tiles(shapes, max_groups=2, n_devices=0)
    with pytest.raises(ValueError, match='at least one observation'):
        partition_tiles([], max_groups=1)


@pytest.mark.parametrize('n_devices', [1, 2, 4])
def test_tile_layout_covers_every_observation_once(n_devices):
    layout = TileLayout.create(
        TILE_SHAPES, n_devices=n_devices, max_buckets=3, detector_batch_size=2
    )
    covered = np.concatenate([b.observations for b in layout.buckets])
    assert sorted(covered.tolist()) == list(range(len(TILE_SHAPES)))
    assert layout.n_tiles == sum(b.n_tiles for b in layout.buckets)
    for b, bucket in enumerate(layout.buckets):
        tiles = [
            layout.local_tiles(b, process_index=p, n_local=n_devices // 2 or 1)
            for p in range(min(n_devices, 2))
        ]
        segments = [
            layout.local_segments(b, process_index=p, n_local=n_devices // 2 or 1)
            for p in range(min(n_devices, 2))
        ]
        # consecutive processes hold consecutive blocks that cover the bucket
        assert tiles[0].start == 0 and tiles[-1].stop == bucket.n_tiles
        assert segments[0].start == 0 and segments[-1].stop == bucket.n_segments
        for blocks in (tiles, segments):
            assert all(a.stop == b.start for a, b in pairwise(blocks))


@pytest.mark.parametrize('detector_axis', [1, 2, -1])
@pytest.mark.parametrize('batch_size', [0, 2], ids=['one-tile', 'tiles'])
def test_to_observation_order_concatenates_the_tiles(detector_axis, batch_size):
    layout = TileLayout.create(
        TILE_SHAPES, n_devices=2, max_buckets=2, detector_batch_size=batch_size
    )
    rng = np.random.default_rng(0)
    widest = max(
        len(bucket.tiles_of(s)) * bucket.tile_size
        for bucket in layout.buckets
        for s in range(bucket.n_segments)
    )
    # (obs, stream, det, mode), the detector axis moved to `detector_axis`
    reference = rng.standard_normal((len(TILE_SHAPES), 2, widest, 3))
    per_tile, per_segment = [], []
    for bucket in layout.buckets:
        tiles = np.zeros((bucket.n_tiles, 2, bucket.tile_size, 3))
        segments = np.zeros((bucket.n_segments, 2))
        for segment, observation in enumerate(bucket.segment_observations):
            if observation < 0:
                continue
            segments[segment] = reference[observation, :, 0, 0]
            for t in bucket.tiles_of(segment):
                start = bucket.tile_starts[t]
                tiles[t] = reference[observation, :, start : start + bucket.tile_size]
        per_tile.append(np.moveaxis(tiles, 2, detector_axis))
        per_segment.append(segments)

    out = layout.to_observation_order(per_tile, detector_axis=detector_axis)
    for observation, shape in enumerate(TILE_SHAPES):
        held = np.moveaxis(out[observation], detector_axis - (detector_axis > 0), 1)
        detectors = shape.detector_count
        np.testing.assert_array_equal(held[:, :detectors], reference[observation, :, :detectors])
    np.testing.assert_array_equal(layout.to_observation_order(per_segment), reference[:, :, 0, 0])
    with pytest.raises(ValueError, match='expected 2 arrays'):
        layout.to_observation_order(per_tile[:1], detector_axis=detector_axis)


# ---------------------------------------------------------------------------
# TilePlacement: the tiles on the devices of the job
# ---------------------------------------------------------------------------


def _placement() -> TilePlacement:
    layout = TileLayout.create(TILE_SHAPES, n_devices=jax.device_count(), max_buckets=1)
    return TilePlacement.create(layout)


def test_placement_moves_arrays_between_host_and_devices():
    placement = _placement()
    assert placement.local_devices == jax.local_devices()
    n = jax.device_count()
    sharded = placement.shard(np.arange(6.0 * n).reshape(2 * n, 3))
    assert sharded.sharding == placement.sharding
    whole = placement.broadcast({'x': np.ones(3)})
    assert whole['x'].sharding == placement.replicated

    summed = placement.replicate(sharded, lambda x: x.sum(axis=0))
    assert summed.sharding == placement.replicated
    np.testing.assert_array_equal(summed, np.arange(6.0 * n).reshape(2 * n, 3).sum(axis=0))
    gathered = placement.gather(sharded)
    assert isinstance(gathered, np.ndarray)
    np.testing.assert_array_equal(gathered, np.arange(6.0 * n).reshape(2 * n, 3))


def test_placement_needs_the_devices_the_layout_was_planned_for():
    layout = TileLayout.create(TILE_SHAPES, n_devices=jax.device_count() + 1, max_buckets=1)
    with pytest.raises(ValueError, match='planned for'):
        TilePlacement.create(layout)


def test_placement_shards_the_tiles_over_one_mesh_axis():
    layout = TileLayout.create(TILE_SHAPES, n_devices=1, max_buckets=1)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('tile', 'other'))
    with pytest.raises(ValueError, match="along 'tile'"):
        TilePlacement(layout, mesh)
