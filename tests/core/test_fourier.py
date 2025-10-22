"""Tests for FourierOperator and BandpassOperator."""

import jax
import jax.numpy as jnp
import pytest

from furax.core import BandpassOperator, FourierOperator

# Enable float64 for better numerical precision in tests
jax.config.update('jax_enable_x64', True)


class TestFourierOperator:
    """Tests for FourierOperator class."""

    def test_fft_method_basic(self):
        """Test basic FFT method functionality."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5 Hz and 30 Hz components
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        # Low-pass filter kernel (cutoff at 10 Hz)
        freqs = jnp.fft.rfftfreq(n, 1.0 / sample_rate)
        kernel = (freqs < 10.0).astype(jnp.complex128)

        # Create operator
        op = FourierOperator(
            fourier_kernel=kernel,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            method='fft',
        )

        # Apply filter
        filtered = op(signal)

        assert filtered.shape == signal.shape

        # Check that low frequency (5 Hz) is preserved perfectly
        fft_orig = jnp.fft.rfft(signal)
        fft_filt = jnp.fft.rfft(filtered)
        freq_5hz_idx = int(5.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_5hz_idx])

        # Check that high frequency (30 Hz) is removed
        freq_30hz_idx = int(30.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_30hz_idx]) < 0.01 * jnp.abs(fft_orig[freq_30hz_idx])

    def test_overlap_save_method_basic(self):
        """Test basic overlap-save method functionality."""
        n = 2000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5 Hz and 30 Hz components
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        # Low-pass filter kernel for the fft_size
        fft_size = 512
        freqs = jnp.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        kernel = (freqs < 10.0).astype(jnp.complex128)

        # Create operator with overlap-save (apodize is True by default)
        op = FourierOperator(
            fourier_kernel=kernel,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            method='overlap_save',
            fft_size=fft_size,
        )

        # Apply filter
        filtered = op(signal)

        assert filtered.shape == signal.shape

        # Check frequency content
        fft_orig = jnp.fft.rfft(signal)
        fft_filt = jnp.fft.rfft(filtered)
        freq_5hz_idx = int(5.0 * n / sample_rate)
        freq_30hz_idx = int(30.0 * n / sample_rate)

        # 5 Hz should be well preserved (>99% with proper overlap-save)
        assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.99 * jnp.abs(fft_orig[freq_5hz_idx])
        # 30 Hz should be removed
        assert jnp.abs(fft_filt[freq_30hz_idx]) < 0.1 * jnp.abs(fft_orig[freq_30hz_idx])

    def test_fft_vs_overlap_save(self):
        """Test that FFT and overlap-save methods produce similar results."""
        n = 5000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 10 * t) + 0.5 * jnp.sin(2 * jnp.pi * 50 * t)

        # Create kernels - use fft_size for overlap-save kernel
        fft_size = 1024
        freqs_overlap = jnp.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        kernel_overlap = ((freqs_overlap >= 5.0) & (freqs_overlap <= 20.0)).astype(jnp.complex128)

        freqs_fft = jnp.fft.rfftfreq(n, 1.0 / sample_rate)
        kernel_fft = ((freqs_fft >= 5.0) & (freqs_fft <= 20.0)).astype(jnp.complex128)

        # FFT method
        op_fft = FourierOperator(
            fourier_kernel=kernel_fft,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='fft',
        )

        # Overlap-save method
        op_overlap = FourierOperator(
            fourier_kernel=kernel_overlap,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=fft_size,
        )

        filtered_fft = op_fft(signal)
        filtered_overlap = op_overlap(signal)

        # Should produce similar results (not identical due to different frequency resolution
        # and apodization in overlap-save)
        # Check in middle 80% to avoid edge effects
        mid_start = n // 10
        mid_end = 9 * n // 10
        mid_fft = filtered_fft[mid_start:mid_end]
        mid_overlap = filtered_overlap[mid_start:mid_end]
        relative_error = jnp.linalg.norm(mid_fft - mid_overlap) / jnp.linalg.norm(mid_fft)
        # Should have very low error (<0.1%) with proper overlap-save implementation
        assert relative_error < 0.001

    def test_apodization_reduces_edge_artifacts(self):
        """Test that apodization reduces edge artifacts in overlap-save method."""
        # Create a smooth signal for testing
        n = 2000
        sample_rate = 100.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 3 * t)

        fft_size = 512

        # Create low-pass filter
        freqs = jnp.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        kernel = (freqs < 5.0).astype(jnp.complex128)

        # Without apodization
        op_no_apod = FourierOperator(
            fourier_kernel=kernel,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=fft_size,
            apodize=False,
        )

        # With apodization
        op_with_apod = FourierOperator(
            fourier_kernel=kernel,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=fft_size,
            apodize=True,
        )

        filtered_no_apod = op_no_apod(signal)
        filtered_with_apod = op_with_apod(signal)

        # Both should produce valid results
        assert filtered_no_apod.shape == signal.shape
        assert filtered_with_apod.shape == signal.shape

        # Results should be similar but not identical
        assert not jnp.allclose(filtered_no_apod, filtered_with_apod, atol=1e-10)

        # Both should preserve the signal very well since 3 Hz is in passband
        assert jnp.linalg.norm(filtered_no_apod) > 0.999 * jnp.linalg.norm(signal)
        assert jnp.linalg.norm(filtered_with_apod) > 0.999 * jnp.linalg.norm(signal)

    def test_apodization_window_shape(self):
        """Test that apodization window has correct shape."""
        fft_size = 256
        overlap = 64

        window = FourierOperator._create_apodization_window(fft_size, overlap)

        assert window.shape == (fft_size,)
        # Middle should be 1.0
        assert jnp.allclose(window[overlap:-overlap], 1.0)
        # Edges should taper
        assert window[0] < window[overlap // 2] < window[overlap]
        assert window[-1] < window[-overlap // 2] < window[-overlap]

    def test_multidimensional_signal(self):
        """Test FourierOperator with multidimensional signals."""
        n_detectors = 3
        n_samples = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n_samples / sample_rate, n_samples)

        # Multiple detector signals
        signals = jnp.array([jnp.sin(2 * jnp.pi * (5 + i * 5) * t) for i in range(n_detectors)])

        # Bandpass filter
        freqs = jnp.fft.rfftfreq(n_samples, 1.0 / sample_rate)
        kernel = ((freqs >= 3.0) & (freqs <= 12.0)).astype(jnp.complex128)

        op = FourierOperator(
            fourier_kernel=kernel,
            in_structure=jax.ShapeDtypeStruct(signals.shape, signals.dtype),
            method='fft',
        )

        filtered = op(signals)

        assert filtered.shape == signals.shape
        # First detector (5 Hz) should pass through filter very well
        assert jnp.linalg.norm(filtered[0]) > 0.999 * jnp.linalg.norm(signals[0])
        # Third detector (15 Hz) should be attenuated (just outside passband)
        assert jnp.linalg.norm(filtered[2]) < 0.02 * jnp.linalg.norm(signals[2])

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        n = 1000
        freqs = jnp.fft.rfftfreq(n, 1.0 / 100.0)
        kernel = (freqs < 10.0).astype(jnp.complex128)

        with pytest.raises(ValueError, match='Invalid method'):
            FourierOperator(
                fourier_kernel=kernel,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                method='invalid_method',
            )

    def test_fft_method_with_apodization(self):
        """Test FFT method with apodization enabled."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5 Hz and 30 Hz components
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        # Create bandpass filter using BandpassOperator to ensure correct kernel size
        op = BandpassOperator.create(
            f_low=1.0,
            f_high=10.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            method='fft',
            apodize=True,
        )

        # Apply filter
        filtered = op(signal)

        assert filtered.shape == signal.shape

        # Check that low frequency (5 Hz) is preserved well
        fft_orig = jnp.fft.rfft(signal)
        fft_filt = jnp.fft.rfft(filtered)
        freq_5hz_idx = int(5.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.98 * jnp.abs(fft_orig[freq_5hz_idx])

        # Check that high frequency (30 Hz) is removed
        freq_30hz_idx = int(30.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_30hz_idx]) < 0.1 * jnp.abs(fft_orig[freq_30hz_idx])

    def test_fft_apodization_vs_no_apodization(self):
        """Test that apodization works for FFT method."""
        n = 2000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Create a signal
        signal = jnp.sin(2 * jnp.pi * 10 * t)

        # Without apodization
        op_no_apod = BandpassOperator.create(
            f_low=5.0,
            f_high=15.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='fft',
            apodize=False,
        )

        # With apodization
        op_with_apod = BandpassOperator.create(
            f_low=5.0,
            f_high=15.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='fft',
            apodize=True,
        )

        filtered_no_apod = op_no_apod(signal)
        filtered_with_apod = op_with_apod(signal)

        # Both should produce valid results
        assert filtered_no_apod.shape == signal.shape
        assert filtered_with_apod.shape == signal.shape

        # Both should preserve the signal well since 10 Hz is in passband
        fft_orig = jnp.fft.rfft(signal)
        fft_no_apod = jnp.fft.rfft(filtered_no_apod)
        fft_with_apod = jnp.fft.rfft(filtered_with_apod)
        freq_10hz_idx = int(10.0 * n / sample_rate)

        # Both should preserve 10 Hz well
        assert jnp.abs(fft_no_apod[freq_10hz_idx]) > 0.99 * jnp.abs(fft_orig[freq_10hz_idx])
        assert jnp.abs(fft_with_apod[freq_10hz_idx]) > 0.99 * jnp.abs(fft_orig[freq_10hz_idx])

    def test_fft_apodization_custom_padding_width(self):
        """Test FFT method with custom padding width."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        # Test with different padding widths
        for padding_width in [50, 100, 150]:
            op = BandpassOperator.create(
                f_low=1.0,
                f_high=10.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                method='fft',
                apodize=True,
                padding_width=padding_width,
            )

            filtered = op(signal)

            # Should return correct shape
            assert filtered.shape == signal.shape

            # Should preserve low frequency content
            fft_orig = jnp.fft.rfft(signal)
            fft_filt = jnp.fft.rfft(filtered)
            freq_5hz_idx = int(5.0 * n / sample_rate)
            assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.95 * jnp.abs(fft_orig[freq_5hz_idx])

    def test_kernel_size_validation(self):
        """Test that kernel size validation works correctly."""
        n = 1000
        freqs = jnp.fft.rfftfreq(n, 1.0 / 100.0)
        kernel = (freqs < 10.0).astype(jnp.complex128)

        # Should raise error for mismatched kernel size
        with pytest.raises(ValueError, match='Fourier kernel size mismatch'):
            FourierOperator(
                fourier_kernel=kernel,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                method='fft',
                apodize=True,  # This changes expected kernel size
            )


class TestBandpassOperator:
    """Tests for BandpassOperator factory class."""

    def test_basic_bandpass(self):
        """Test basic bandpass filtering."""
        n = 2000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5, 15, 25, and 40 Hz components
        signal = (
            jnp.sin(2 * jnp.pi * 5 * t)
            + jnp.sin(2 * jnp.pi * 15 * t)
            + jnp.sin(2 * jnp.pi * 25 * t)
            + jnp.sin(2 * jnp.pi * 40 * t)
        )

        # Bandpass filter (10-30 Hz)
        op = BandpassOperator.create(
            f_low=10.0,
            f_high=30.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='fft',
        )

        filtered = op(signal)

        # Check frequency content
        fft_filt = jnp.fft.rfft(filtered)
        fft_orig = jnp.fft.rfft(signal)

        # Check that 15 Hz and 25 Hz are preserved (should be essentially perfect)
        freq_15hz_idx = int(15.0 * n / sample_rate)
        freq_25hz_idx = int(25.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_15hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_15hz_idx])
        assert jnp.abs(fft_filt[freq_25hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_25hz_idx])

        # Check that 5 Hz and 40 Hz are removed
        freq_5hz_idx = int(5.0 * n / sample_rate)
        freq_40hz_idx = int(40.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) < 0.01 * jnp.abs(fft_orig[freq_5hz_idx])
        assert jnp.abs(fft_filt[freq_40hz_idx]) < 0.01 * jnp.abs(fft_orig[freq_40hz_idx])

    def test_bandpass_with_apodization(self):
        """Test bandpass filter with apodization."""
        n = 5000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 15 * t)

        op = BandpassOperator.create(
            f_low=10.0,
            f_high=20.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=1024,
            apodize=True,
        )

        filtered = op(signal)

        # Signal should pass through the filter perfectly (15 Hz is in passband)
        # With proper overlap-save implementation, apodization has minimal impact
        fft_filt = jnp.fft.rfft(filtered)
        fft_orig = jnp.fft.rfft(signal)
        freq_15hz_idx = int(15.0 * n / sample_rate)
        # 15 Hz component should be essentially perfectly preserved
        assert jnp.abs(fft_filt[freq_15hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_15hz_idx])

    def test_invalid_frequency_parameters(self):
        """Test that invalid frequency parameters raise errors."""
        n = 1000
        signal = jnp.zeros(n)
        sample_rate = 100.0

        # Negative f_low
        with pytest.raises(ValueError, match='f_low must be non-negative'):
            BandpassOperator.create(
                f_low=-5.0,
                f_high=10.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            )

        # f_high > Nyquist
        with pytest.raises(ValueError, match='f_high must be less than Nyquist frequency'):
            BandpassOperator.create(
                f_low=5.0,
                f_high=60.0,  # Nyquist is 50 Hz
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            )

        # f_low >= f_high
        with pytest.raises(ValueError, match='f_low must be less than f_high'):
            BandpassOperator.create(
                f_low=20.0,
                f_high=10.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            )

    def test_bandpass_fft_vs_overlap_save(self):
        """Test that both methods produce similar results for bandpass."""
        n = 5000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 15 * t) + 0.5 * jnp.sin(2 * jnp.pi * 50 * t)

        op_fft = BandpassOperator.create(
            f_low=10.0,
            f_high=20.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='fft',
        )

        op_overlap = BandpassOperator.create(
            f_low=10.0,
            f_high=20.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
        )

        filtered_fft = op_fft(signal)
        filtered_overlap = op_overlap(signal)

        # Should produce very similar results in middle 80% with proper implementation
        mid_start = n // 10
        mid_end = 9 * n // 10
        assert jnp.allclose(
            filtered_fft[mid_start:mid_end], filtered_overlap[mid_start:mid_end], atol=0.01
        )

    def test_filter_types(self):
        """Test different filter types (square, butter4, cos2)."""
        n = 2000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with frequencies: 5, 15, 25 Hz
        signal = (
            jnp.sin(2 * jnp.pi * 5 * t)
            + jnp.sin(2 * jnp.pi * 15 * t)
            + jnp.sin(2 * jnp.pi * 25 * t)
        )

        f_low, f_high = 10.0, 20.0

        # Test each filter type
        for filter_type in ['square', 'butter4', 'cos2']:
            op = BandpassOperator.create(
                f_low=f_low,
                f_high=f_high,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
                filter_type=filter_type,
                method='fft',
            )

            filtered = op(signal)
            assert filtered.shape == signal.shape

            # Check frequency content
            fft_orig = jnp.fft.rfft(signal)
            fft_filt = jnp.fft.rfft(filtered)

            # 15 Hz (in passband) should be preserved
            freq_15hz_idx = int(15.0 * n / sample_rate)
            passband_ratio = jnp.abs(fft_filt[freq_15hz_idx]) / jnp.abs(fft_orig[freq_15hz_idx])

            if filter_type == 'square':
                # Square filter should have perfect passband
                assert passband_ratio > 0.999
            elif filter_type == 'butter4':
                # Butterworth has some rolloff even in passband
                assert passband_ratio > 0.98
            elif filter_type == 'cos2':
                # Cosine-squared should have good passband preservation
                assert passband_ratio > 0.999

            # 5 Hz and 25 Hz (outside passband) should be attenuated
            freq_5hz_idx = int(5.0 * n / sample_rate)
            freq_25hz_idx = int(25.0 * n / sample_rate)
            stopband_ratio_low = jnp.abs(fft_filt[freq_5hz_idx]) / jnp.abs(fft_orig[freq_5hz_idx])
            stopband_ratio_high = jnp.abs(fft_filt[freq_25hz_idx]) / jnp.abs(
                fft_orig[freq_25hz_idx]
            )

            if filter_type == 'square':
                # Square filter should have sharp cutoff
                assert stopband_ratio_low < 0.01
                assert stopband_ratio_high < 0.01
            elif filter_type == 'butter4':
                # Butterworth has gradual rolloff
                assert stopband_ratio_low < 0.1
                assert stopband_ratio_high < 0.2  # More gradual rolloff on high side
            elif filter_type == 'cos2':
                # Cosine-squared has smooth transition
                assert stopband_ratio_low < 0.01
                assert stopband_ratio_high < 0.01

    def test_invalid_filter_type(self):
        """Test that invalid filter_type raises error."""
        n = 1000
        signal = jnp.zeros(n)
        sample_rate = 100.0

        with pytest.raises(ValueError, match='Invalid filter_type'):
            BandpassOperator.create(
                f_low=10.0,
                f_high=20.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
                filter_type='invalid_filter',
            )

    def test_narrow_bandwidth_validation(self):
        """Test that too-narrow bandwidth raises helpful error."""
        n = 100  # Very small FFT
        sample_rate = 100.0

        # Extremely narrow bandwidth that falls between frequency bins
        # With n=100, sample_rate=100, freq resolution = 1 Hz
        # Bins at 0, 1, 2, ..., 15, 16, ...
        # So [15.3, 15.4] falls between bins and won't reach gain=1
        with pytest.raises(ValueError, match='Filter passband not represented'):
            BandpassOperator.create(
                f_low=15.3,
                f_high=15.4,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                filter_type='square',
                method='fft',
            )

    def test_cos2_passband_unity_gain(self):
        """Test that cos2 filter has unity gain throughout entire passband."""
        n = 2000
        sample_rate = 200.0
        f_low, f_high = 10.0, 20.0

        op = BandpassOperator.create(
            f_low=f_low,
            f_high=f_high,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            filter_type='cos2',
            method='fft',
        )

        # Check that entire passband has unity gain
        freqs = jnp.fft.rfftfreq(n, 1.0 / sample_rate)
        kernel = op.fourier_kernel

        # Test frequencies within passband
        for freq in [f_low, (f_low + f_high) / 2, f_high]:
            idx = jnp.argmin(jnp.abs(freqs - freq))
            gain = jnp.abs(kernel[idx])
            assert gain > 0.999, f'Gain at {freq} Hz should be ~1.0, got {gain}'

        # Test that transitions are outside passband
        bandwidth = f_high - f_low
        transition_width = 0.1 * bandwidth

        # Just outside lower edge (in transition)
        freq_low_trans = f_low - 0.5 * transition_width
        if freq_low_trans > 0:
            idx = jnp.argmin(jnp.abs(freqs - freq_low_trans))
            gain = jnp.abs(kernel[idx])
            assert 0.2 < gain < 0.8, f'Transition gain should be between 0.2-0.8, got {gain}'

        # Just outside upper edge (in transition)
        freq_high_trans = f_high + 0.5 * transition_width
        idx = jnp.argmin(jnp.abs(freqs - freq_high_trans))
        gain = jnp.abs(kernel[idx])
        assert 0.2 < gain < 0.8, f'Transition gain should be between 0.2-0.8, got {gain}'


class TestApodizationDetails:
    """Detailed tests for apodization functionality."""

    def test_apodization_preserves_signal_in_middle(self):
        """Test that apodization preserves signal in the middle region."""
        n = 5000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Smooth signal
        signal = jnp.sin(2 * jnp.pi * 10 * t)

        op_no_apod = BandpassOperator.create(
            f_low=5.0,
            f_high=15.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=1024,
            apodize=False,
        )

        op_with_apod = BandpassOperator.create(
            f_low=5.0,
            f_high=15.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            method='overlap_save',
            fft_size=1024,
            apodize=True,
        )

        filtered_no_apod = op_no_apod(signal)
        filtered_with_apod = op_with_apod(signal)

        # Apodization can change the signal shape, but both should preserve frequency content
        # Check that both preserve the 10 Hz component
        fft_orig = jnp.fft.rfft(signal)
        fft_no_apod = jnp.fft.rfft(filtered_no_apod)
        fft_with_apod = jnp.fft.rfft(filtered_with_apod)
        freq_10hz_idx = int(10.0 * n / sample_rate)

        # Both should preserve 10 Hz very well with proper overlap-save
        assert jnp.abs(fft_no_apod[freq_10hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_10hz_idx])
        assert jnp.abs(fft_with_apod[freq_10hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_10hz_idx])

    def test_apodization_window_symmetry(self):
        """Test that apodization window is symmetric."""
        fft_size = 512
        overlap = 128

        window = FourierOperator._create_apodization_window(fft_size, overlap)

        # Check symmetry
        assert jnp.allclose(window[:overlap], window[-overlap:][::-1])

    def test_apodization_with_zero_overlap(self):
        """Test apodization window with zero overlap."""
        fft_size = 256
        overlap = 0

        window = FourierOperator._create_apodization_window(fft_size, overlap)

        # Should be all ones
        assert jnp.allclose(window, 1.0)

    def test_apodization_window_smooth_transition(self):
        """Test that apodization window has smooth transition."""
        fft_size = 512
        overlap = 128

        window = FourierOperator._create_apodization_window(fft_size, overlap)

        # Check that window increases monotonically in overlap region
        for i in range(overlap - 1):
            assert window[i] <= window[i + 1]

        # Check that window decreases monotonically at the end
        for i in range(fft_size - overlap, fft_size - 1):
            assert window[i] >= window[i + 1]
