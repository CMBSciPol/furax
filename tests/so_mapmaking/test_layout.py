"""The `furax-so-layout` planning command."""

import json
import logging

import pytest

from furax.mapmaking import MapMakingConfig, ObservationBufferShape
from furax.mapmaking.config import HealpixConfig, LandscapeConfig
from furax.mapmaking.layout import SlotLayout
from so_mapmaking.layout import probe_shapes, sweep, sweep_layouts
from tests.mapmaking.helpers import FakeLazyObservation


class _UnprobeableObservation(FakeLazyObservation):
    """A lazy observation whose shape probe raises, as an unreadable preproc entry would."""

    def probe_shape(self, intervals: bool = False) -> ObservationBufferShape:
        raise RuntimeError('simulated probe failure')


def test_sweep_matches_the_layout_the_mapmaker_would_choose():
    shapes = [
        ObservationBufferShape(n_det, n_samp)
        for n_det, n_samp in [
            (100, 4900),
            (100, 5000),
            (90, 1200),
            (100, 4800),
            (110, 5000),
            (100, 900),
        ]
    ]
    rows = sweep_layouts(shapes, devices=[1, 4], max_buckets=[1, 2])
    assert [(r.n_devices, r.max_buckets) for r in rows] == [(1, 1), (1, 2), (4, 1), (4, 2)]
    for row in rows:
        layout = SlotLayout.create(shapes, n_devices=row.n_devices, max_buckets=row.max_buckets)
        assert row.n_buckets == len(layout.buckets)
        assert row.n_slots == layout.n_slots
        assert row.padded_volume == layout.padded_volume
        assert len(row.bucket_summaries) == row.n_buckets


@pytest.mark.parametrize(('n_devices', 'n_slots'), [(5, 20), (6, 24)])
def test_sweep_counts_the_empty_slots_of_an_uneven_device_count(n_devices, n_slots):
    # 20 identical observations fit one bucket; it rounds up to a whole number of slots per device
    shapes = [ObservationBufferShape(10, 100)] * 20
    (row,) = sweep_layouts(shapes, devices=[n_devices], max_buckets=[1])
    assert row.n_slots == n_slots
    assert row.slot_overhead == pytest.approx((n_slots - 20) / 20)
    assert row.volume_overhead == pytest.approx((n_slots - 20) / 20)
    # the one bucket is spread evenly, so each device holds a share of its slots
    assert row.peak_device_volume == 10 * 100 * n_slots // n_devices


def test_sweep_reads_the_cache_that_probe_writes(tmp_path, capsys):
    cache = tmp_path / 'shapes.json'
    cache.write_text(json.dumps({'shapes': [[10, 100, 0]] * 20, 'failed': []}))
    sweep(cache=cache, devices='5,6', max_buckets='1')
    out = capsys.readouterr().out
    assert '20 observations' in out
    assert 'detectors 10-10, samples 100-100' in out
    # one table row per device count (dev, budget, buckets, slots, ...): 20 slots on 5 devices,
    # 24 on 6
    rows = [line.split()[:4] for line in out.splitlines() if line.split()[:1] in (['5'], ['6'])]
    assert rows == [['5', '1', '1', '20'], ['6', '1', '1', '24']]


def test_probe_reports_each_observation_shape_and_the_failures():
    # the probe is the mapmaker's own; an observation that cannot be probed is reported and given
    # the largest shape, so the planned layout matches the run's
    config = MapMakingConfig(landscape=LandscapeConfig(healpix=HealpixConfig(nside=8)))
    observations = [
        FakeLazyObservation(n_dets=4, n_samples=512),
        _UnprobeableObservation(n_dets=4, n_samples=512),
        FakeLazyObservation(n_dets=6, n_samples=768),
    ]
    shapes, failed = probe_shapes(observations, config, logging.getLogger('test'))
    assert failed == [1]
    assert [(s.detector_count, s.sample_count) for s in shapes] == [(4, 512), (6, 768), (6, 768)]
