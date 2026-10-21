import jax
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax.mapmaking import ObservationReader
from furax.mapmaking._reader import ObservationRecord
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


def test_failure_to_record_does_not_fail_the_read(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError('cannot record')

    monkeypatch.setattr(ObservationRecord, 'from_fields', fail)
    observations = [FakeLazyObservation(n_dets=2, n_samples=500)]
    reader = ObservationReader.from_observations(observations, requested_fields=['sample_data'])
    _, _, valid = reader.read_host(0)
    assert valid
    assert not reader.failed_indices
    record = reader.records[0]
    assert record.error == 'ValueError: cannot record'
