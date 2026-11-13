import pytest

from furax.mapmaking import ReaderField
from tests.mapmaking.helpers import FakeLazyObservation


class _RecordingLazyObservation(FakeLazyObservation):
    """A fake observation that records the fields its probe requests."""

    requested: list[str] | None = None

    def get_data(self, requested_fields=None):
        self.requested = list(requested_fields or [])
        return super().get_data(requested_fields)


@pytest.mark.parametrize('sample_rate', [False, True])
def test_default_probe_reads_the_timestamps_only_for_the_sample_rate(sample_rate: bool) -> None:
    lazy = _RecordingLazyObservation(n_dets=2, n_samples=50, sample_rate=10.0)
    probe = lazy.probe(sample_rate=sample_rate)
    assert probe.shape == (2, 50, 0)
    assert probe.sample_rate == (pytest.approx(10.0) if sample_rate else None)
    assert lazy.requested is not None
    assert (ReaderField.TIMESTAMPS in lazy.requested) == sample_rate
    assert lazy.probe(good_ranges=True).good_ranges is None  # the default reads no flags
