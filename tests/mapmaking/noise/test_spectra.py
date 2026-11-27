import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.config import NoiseFitConfig
from furax.mapmaking.noise import apodization_window, padding_aware_welch
from furax.mapmaking.noise.spectra import _create_frequency_mask, _create_frequency_mask_from_config


@pytest.mark.parametrize('kind', ['chebwin', 'gaussian'])
def test_apodization_window_shape(kind):
    w = apodization_window(50, kind=kind)
    assert w.shape == (50,)


def test_apodization_window_invalid_kind():
    with pytest.raises(RuntimeError, match='not supported'):
        apodization_window(10, kind='bogus')


@pytest.mark.parametrize(
    ('f_min', 'f_max', 'intervals', 'expected'),
    [
        (2.0, None, None, [0, 0, 1, 1, 1]),  # f_min inclusive
        (None, 2.0, None, [1, 1, 0, 0, 0]),  # f_max exclusive
        (None, None, [[1.0, 3.0]], [1, 0, 0, 1, 1]),  # [start, end)
    ],
)
def test_frequency_mask(f_min, f_max, intervals, expected):
    f = jnp.arange(5.0)
    mask = _create_frequency_mask(
        f,
        f_min=None if f_min is None else jnp.array(f_min),
        f_max=None if f_max is None else jnp.array(f_max),
        f_mask_intervals=None if intervals is None else jnp.array(intervals),
    )
    assert_array_equal(mask, jnp.array(expected, dtype=float))


def test_frequency_mask_hwp_harmonics_masked():
    """HWP 1f harmonic should be masked when mask_hwp_harmonics=True."""
    sample_rate = jnp.array(10.0)
    hwp_frequency = jnp.array(2.0)
    f = jnp.fft.rfftfreq(512, 1 / float(sample_rate))
    config = NoiseFitConfig(mask_hwp_harmonics=True, freq_mask_width=0.1)
    mask = _create_frequency_mask_from_config(
        f, sample_rate=sample_rate, hwp_frequency=hwp_frequency, config=config
    )
    idx = jnp.argmin(jnp.abs(f - 2.0))
    assert mask[idx] == 0.0


@pytest.mark.parametrize(('n_samples', 'padding'), [(1024, 512), (127, 0)], ids=['padded', 'short'])
def test_padding_aware_welch_matches_the_unpadded_signal(n_samples, padding):
    # padding is ignored, and a segment longer than the signal is clipped to it
    signal = jax.random.normal(jax.random.key(0), (2, n_samples), dtype=jnp.float64)
    expected_f, expected_psd = jax.scipy.signal.welch(signal, fs=100.0, nperseg=256)
    padded = jnp.pad(signal, ((0, 0), (0, padding)))
    actual_f, actual_psd = padding_aware_welch(padded, padding, fs=jnp.array(100.0), nperseg=256)
    assert_allclose(actual_f, expected_f)
    assert_allclose(actual_psd, expected_psd, rtol=1e-12, atol=1e-12)


def test_padding_aware_welch_preserves_float32():
    signal = jax.random.normal(jax.random.key(0), (2, 256), dtype=jnp.float32)
    frequencies, psd = padding_aware_welch(
        signal, 0, fs=jnp.array(100.0, dtype=jnp.float32), nperseg=128
    )
    assert frequencies.dtype == jnp.float32
    assert psd.dtype == jnp.float32


@jax.jit
def _welch(signal, padding, valid):
    return padding_aware_welch(signal, padding, fs=jnp.array(100.0), nperseg=256, valid=valid)


def test_padding_aware_welch_skips_flagged_segments():
    nperseg, step = 256, 128
    signal = jax.random.normal(jax.random.key(0), (3, 1024), dtype=jnp.float64)
    glitched = signal.at[0, 300:340].set(1e6)
    valid = jnp.ones_like(signal, bool)
    valid = valid.at[0, 300:340].set(False)  # the segments starting at 128 and 256 hold them
    valid = valid.at[2, ::100].set(False)  # no segment is clean: the detector keeps them all

    _, psd = _welch(glitched.at[2].set(signal[2]), jnp.array(0), valid)  # padding traced

    def segment(s):
        return jax.scipy.signal.welch(signal[0, s : s + nperseg], fs=100.0, nperseg=nperseg)[1]

    starts = [s for s in range(0, 1024 - nperseg + 1, step) if s not in (128, 256)]
    assert_allclose(psd[0], sum(map(segment, starts)) / len(starts), rtol=1e-10, atol=1e-12)
    _, unmasked = padding_aware_welch(signal, 0, fs=jnp.array(100.0), nperseg=nperseg)
    assert_allclose(psd[1:], unmasked[1:], rtol=1e-12, atol=1e-12)
