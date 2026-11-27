import threading

import jax
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax.mapmaking import ObservationBufferShape, ObservationReader
from furax.mapmaking._reader import ObservationRecord, TileReader, tile_reads
from furax.mapmaking.layout import TileBucket
from tests.mapmaking.helpers import FakeLazyObservation, FakeObservation


class _IgnoresOut(FakeObservation):
    """A getter written before `out` existed: it returns new arrays."""

    def get_tods(self, out=None):
        return super().get_tods()

    def get_demodulated_tods(self, stokes='IQU', out=None):
        return super().get_demodulated_tods(stokes)


class _LazyIgnoresOut(FakeLazyObservation):
    interface_class = _IgnoresOut

    def get_data(self, requested_fields=None) -> _IgnoresOut:
        return _IgnoresOut(**self._kwargs)


@pytest.mark.parametrize(
    'lazy', [FakeLazyObservation, _LazyIgnoresOut], ids=['writes-into-out', 'ignores-out']
)
@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
def test_sample_data_is_zero_padded(lazy, demodulated: bool):
    # the second observation is larger on both axes, so the first is padded on both
    small = {'n_dets': 2, 'n_samples': 500}
    observations = [lazy(**small), lazy(n_dets=3, n_samples=600)]
    reader = ObservationReader.from_observations(
        observations,
        requested_fields=['sample_data'],
        stokes='IQU' if demodulated else None,
        sample_dtype=np.float32,
    )
    data, padding, _ = reader.read(0)
    obs = FakeObservation(**small)
    expected = obs.get_demodulated_tods().data if demodulated else obs.get_tods()
    (tods,) = jax.tree.leaves(data['sample_data'])
    assert tods.dtype == np.float32
    assert_array_equal(tods[..., :2, :500], expected)
    assert not np.any(tods[..., 2:, :])
    assert not np.any(tods[..., 500:])
    (sample_padding,) = jax.tree.leaves(padding['sample_data'])
    assert tuple(sample_padding[-2:]) == (1, 100)


def test_quaternions_are_zero_padded():
    """A zero quaternion marks the padded samples and detectors as having no pointing."""
    small = {'n_dets': 2, 'n_samples': 500}
    observations = [FakeLazyObservation(**small), FakeLazyObservation(n_dets=3, n_samples=600)]
    reader = ObservationReader.from_observations(
        observations, requested_fields=['q_sky_bore', 'q_bore_det']
    )
    data, _, _ = reader.read(0)
    obs = FakeObservation(**small)
    q_sky_bore, q_bore_det = data['q_sky_bore'], data['q_bore_det']
    assert_array_equal(q_sky_bore[:500], obs.get_q_sky_bore())
    assert_array_equal(q_bore_det[:2], obs.get_q_bore_det())
    assert not np.any(q_sky_bore[500:])
    assert not np.any(q_bore_det[2:])


def test_wafer_slots_have_a_detector_axis():
    observations = [FakeLazyObservation(n_dets=2), FakeLazyObservation(n_dets=3)]
    reader = ObservationReader.from_observations(observations, requested_fields=['wafer_slots'])
    data, _, _ = reader.read(0)
    # the fakes alternate between two wafers; padded to the largest observation with slot 0
    assert_array_equal(data['wafer_slots'], [[0], [1], [0]])


@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
def test_sample_data_reaches_a_cpu_device_without_a_copy(demodulated: bool):
    observations = [FakeLazyObservation(n_dets=2, n_samples=500), FakeLazyObservation(n_dets=3)]
    reader = ObservationReader.from_observations(
        observations, requested_fields=['sample_data'], stokes='IQU' if demodulated else None
    )
    data, _, _ = reader.read_host(0)
    (tod,) = jax.tree.leaves(data['sample_data'])
    on_device = jax.device_put(tod[None], jax.devices('cpu')[0], may_alias=True)
    assert on_device.unsafe_buffer_pointer() == tod.ctypes.data


def test_tile_reader_cuts_tiles_out_of_the_observations():
    # buffers of whole tiles: 3 detectors read of 4, in tiles of 2
    observations = [FakeLazyObservation(n_dets=3, n_samples=500), FakeLazyObservation(n_dets=2)]
    reader = ObservationReader.from_observations(
        observations, requested_fields=['sample_data'], shapes=[ObservationBufferShape(4, 500)] * 2
    )
    tiles = TileReader(reader, tile_size=2, tile_counts=[2, 1])
    whole, _, _ = reader.read_host(0)
    (tod,) = jax.tree.leaves(whole['sample_data'])

    for start, past_read in [(0, 0), (2, 1)]:
        data, padding, read = tiles.read(0, start)
        (tile,) = jax.tree.leaves(data['sample_data'])
        (widths,) = jax.tree.leaves(padding['sample_data'])
        assert_array_equal(tile, tod[..., start : start + 2, :])
        assert widths[-2] == past_read  # detectors of the tile past those the observation has
        assert read
    _, _, read = tiles.filler
    assert not read
    assert jax.tree.map(np.shape, tiles.read(1, 0)[0]) == jax.tree.map(np.shape, tiles.filler[0])


def test_tile_reads_map_each_tile_of_a_bucket_to_its_detectors():
    # two tiles of the first observation on one device, one tile and a padding tile on the other
    shapes = [ObservationBufferShape(3, 500), ObservationBufferShape(2, 500)]
    bucket = TileBucket.create(shapes, [0, 1], n_devices=2, detector_tile_size=2)
    assert bucket.valid.tolist() == [True, True, True, False]
    reader = ObservationReader.from_observations(
        [FakeLazyObservation(n_dets=n, n_samples=500) for n in (3, 2)],
        requested_fields=['sample_data'],
        shapes=[shape.whole_tiles(bucket.tile_size) for shape in shapes],
    )
    reads = tile_reads(bucket, reader, 'tile', prefetch=False)
    tods = [jax.tree.leaves(reader.read_host(i)[0]['sample_data'])[0] for i in range(2)]

    for tile, (observation, start) in enumerate([(0, 0), (0, 2), (1, 0)]):
        data, _, read = reads.read(tile)
        assert read
        assert_array_equal(
            jax.tree.leaves(data['sample_data'])[0], tods[observation][..., start : start + 2, :]
        )
    _, _, read = reads.read(3)
    assert not read  # the padding tile reads filler


def test_tile_reader_slices_the_detector_uids_with_the_tiles():
    reader = ObservationReader.from_observations(
        [FakeLazyObservation(n_dets=3, n_samples=500)],
        requested_fields=['sample_data', 'metadata'],
        shapes=[ObservationBufferShape(4, 500)],
    )
    tiles = TileReader(reader, tile_size=2, tile_counts=[2])
    whole, _, _ = reader.read_host(0)
    uids = whole['metadata'].detector_uids

    for start, past_read in [(0, 0), (2, 1)]:
        data, padding, _ = tiles.read(0, start)
        assert_array_equal(data['metadata'].detector_uids, uids[start : start + 2])
        assert padding['metadata'].detector_uids[-1] == past_read


def test_tile_reader_reads_an_observation_once_and_releases_it_after_its_last_tile():
    reader = ObservationReader.from_observations(
        [FakeLazyObservation(n_dets=4, n_samples=500)],
        requested_fields=['sample_data'],
        shapes=[ObservationBufferShape(4, 500)],
    )
    reads = []
    read_host = reader.read_host
    reader.read_host = lambda i: reads.append(i) or read_host(i)  # ty: ignore[invalid-assignment]
    tiles = TileReader(reader, tile_size=1, tile_counts=[4])

    threads = [threading.Thread(target=tiles.read, args=(0, start)) for start in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert reads == [0]
    assert not tiles._held


def test_records_every_item_of_a_repeated_observation():
    observation = FakeLazyObservation(n_dets=2, n_samples=500)
    reader = ObservationReader.from_observations(
        [observation] * 2, requested_fields=['sample_data']
    )
    for item in range(2):
        reader.read_host(item)
    assert sorted(reader.records) == [0, 1]
    reader.reset_failures()
    assert reader.records == {}


def test_record_is_named_as_the_observation_was_listed():
    class Listed(FakeLazyObservation):
        @property
        def name(self) -> str:
            return 'obs__ws0'

    reader = ObservationReader.from_observations(
        [Listed(n_dets=2, n_samples=500)], requested_fields=['sample_data']
    )
    reader.read_host(0)
    assert reader.records[0].name == 'obs__ws0'


def test_failure_to_record_does_not_fail_the_read(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError('cannot record')

    monkeypatch.setattr(ObservationRecord, 'from_fields', fail)
    observations = [FakeLazyObservation(n_dets=2, n_samples=500)]
    reader = ObservationReader.from_observations(observations, requested_fields=['sample_data'])
    _, _, valid = reader.read_host(0)
    assert valid
    assert not reader.errors
    record = reader.records[0]
    assert record.error == 'ValueError: cannot record'
