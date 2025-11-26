"""Tests for FourierOperator."""

import jax
import jax.numpy as jnp
import pytest

from furax import FourierOperator

# Enable float64 for better numerical precision in tests
jax.config.update('jax_enable_x64', True)


class TestFourierOperator:
    """Tests for FourierOperator class."""

    def test_basic_kernel_function(self):
        """Test FourierOperator with a simple kernel function."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5 Hz and 30 Hz components
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        # Low-pass filter
        op = FourierOperator.create_bandpass_operator(
            f_low=0.0,
            f_high=10.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            apodize=True,
        )

        # Apply filter
        filtered = op(signal)

        assert filtered.shape == signal.shape

        # Check that low frequency (5 Hz) is preserved
        fft_orig = jnp.fft.rfft(signal)
        fft_filt = jnp.fft.rfft(filtered)
        freq_5hz_idx = int(5.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.99 * jnp.abs(fft_orig[freq_5hz_idx])

        # Check that high frequency (30 Hz) is removed
        freq_30hz_idx = int(30.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_30hz_idx]) < 0.1 * jnp.abs(fft_orig[freq_30hz_idx])

    def test_with_apodization(self):
        """Test FourierOperator with apodization enabled."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)

        # Signal with 5 Hz and 30 Hz components
        signal = jnp.sin(2 * jnp.pi * 5 * t) + 0.5 * jnp.sin(2 * jnp.pi * 30 * t)

        # Create operator with apodization
        op = FourierOperator(
            kernel_func=lambda f: ((f >= 1.0) & (f <= 10.0)).astype(jnp.complex128),
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            sample_rate=sample_rate,
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

    def test_custom_padding_width(self):
        """Test FourierOperator with custom padding width."""
        n = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 5 * t)

        # Test with custom padding width
        padding_width = 100
        op = FourierOperator(
            kernel_func=lambda f: ((f >= 1.0) & (f <= 10.0)).astype(jnp.complex128),
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            sample_rate=sample_rate,
            apodize=True,
            padding_width=padding_width,
        )

        filtered = op(signal)

        # Should return correct shape
        assert filtered.shape == signal.shape
        assert op.padding_width == padding_width

        # Should preserve low frequency content
        fft_orig = jnp.fft.rfft(signal)
        fft_filt = jnp.fft.rfft(filtered)
        freq_5hz_idx = int(5.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) > 0.95 * jnp.abs(fft_orig[freq_5hz_idx])

    def test_multidimensional_signal(self):
        """Test FourierOperator with multidimensional signals."""
        n_detectors = 3
        n_samples = 1000
        sample_rate = 200.0
        t = jnp.linspace(0, n_samples / sample_rate, n_samples)

        # Multiple detector signals
        signals = jnp.array([jnp.sin(2 * jnp.pi * (5 + i * 5) * t) for i in range(n_detectors)])

        # Bandpass filter
        op = FourierOperator.create_bandpass_operator(
            f_low=3.0,
            f_high=12.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signals.shape, signals.dtype),
            apodize=True,
        )

        filtered = op(signals)

        assert filtered.shape == signals.shape
        # First detector (5 Hz) should pass through filter well
        assert jnp.linalg.norm(filtered[0]) > 0.99 * jnp.linalg.norm(signals[0])
        # Third detector (15 Hz) should be attenuated (outside passband)
        assert jnp.linalg.norm(filtered[2]) < 0.05 * jnp.linalg.norm(signals[2])

    def test_kernel_validation(self):
        """Test that kernel size validation works correctly."""
        n = 1000

        # Kernel function that returns wrong size should raise error
        def bad_kernel_func(f):
            # Return wrong size array
            return jnp.ones(len(f) + 10, dtype=jnp.complex128)

        with pytest.raises(ValueError, match='Bad kernel shape'):
            FourierOperator(
                kernel_func=bad_kernel_func,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                sample_rate=100.0,
            )

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

    def test_apodization_window_symmetry(self):
        """Test that apodization window is symmetric."""
        fft_size = 512
        overlap = 128

        window = FourierOperator._create_apodization_window(fft_size, overlap)

        # Check symmetry
        assert jnp.allclose(window[:overlap], window[-overlap:][::-1])

    def test_as_matrix(self):
        """Test conversion to dense matrix representation."""
        n = 100
        sample_rate = 100.0

        op = FourierOperator.create_bandpass_operator(
            f_low=5.0,
            f_high=15.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            apodize=True,
        )

        matrix = op.as_matrix()

        # Check matrix shape
        assert matrix.shape == (n, n)

        # Test that matrix multiplication gives same result as operator application
        signal = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, n))
        result_mv = op(signal)
        result_matrix = matrix @ signal

        assert jnp.allclose(result_mv, result_matrix, rtol=1e-5)


class TestBandpassOperator:
    """Tests for FourierOperator.create_bandpass_operator factory method."""

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
        op = FourierOperator.create_bandpass_operator(
            f_low=10.0,
            f_high=30.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
        )

        filtered = op(signal)

        # Check frequency content
        fft_filt = jnp.fft.rfft(filtered)
        fft_orig = jnp.fft.rfft(signal)

        # Check that 15 Hz and 25 Hz are preserved
        freq_15hz_idx = int(15.0 * n / sample_rate)
        freq_25hz_idx = int(25.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_15hz_idx]) > 0.997 * jnp.abs(fft_orig[freq_15hz_idx])
        assert jnp.abs(fft_filt[freq_25hz_idx]) > 0.997 * jnp.abs(fft_orig[freq_25hz_idx])

        # Check that 5 Hz and 40 Hz are removed
        freq_5hz_idx = int(5.0 * n / sample_rate)
        freq_40hz_idx = int(40.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_5hz_idx]) < 0.01 * jnp.abs(fft_orig[freq_5hz_idx])
        assert jnp.abs(fft_filt[freq_40hz_idx]) < 0.01 * jnp.abs(fft_orig[freq_40hz_idx])

    def test_with_apodization(self):
        """Test bandpass filter with apodization."""
        n = 5000
        sample_rate = 200.0
        t = jnp.linspace(0, n / sample_rate, n)
        signal = jnp.sin(2 * jnp.pi * 15 * t)

        op = FourierOperator.create_bandpass_operator(
            f_low=10.0,
            f_high=20.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            apodize=True,
        )

        filtered = op(signal)

        # Signal should pass through the filter well (15 Hz is in passband)
        fft_filt = jnp.fft.rfft(filtered)
        fft_orig = jnp.fft.rfft(signal)
        freq_15hz_idx = int(15.0 * n / sample_rate)
        assert jnp.abs(fft_filt[freq_15hz_idx]) > 0.999 * jnp.abs(fft_orig[freq_15hz_idx])

    def test_invalid_frequency_parameters(self):
        """Test that invalid frequency parameters raise errors."""
        n = 1000
        signal = jnp.zeros(n)
        sample_rate = 100.0

        # Negative f_low
        with pytest.raises(ValueError, match='f_low must be non-negative'):
            FourierOperator.create_bandpass_operator(
                f_low=-5.0,
                f_high=10.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            )

        # f_high > Nyquist
        with pytest.raises(ValueError, match='f_high must be less than Nyquist frequency'):
            FourierOperator.create_bandpass_operator(
                f_low=5.0,
                f_high=60.0,  # Nyquist is 50 Hz
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
            )

        # f_low >= f_high
        with pytest.raises(ValueError, match='f_low must be less than f_high'):
            FourierOperator.create_bandpass_operator(
                f_low=20.0,
                f_high=10.0,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
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
            op = FourierOperator.create_bandpass_operator(
                f_low=f_low,
                f_high=f_high,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct(signal.shape, signal.dtype),
                filter_type=filter_type,
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
                # Square filter should have very good passband preservation
                assert passband_ratio > 0.995
            elif filter_type == 'butter4':
                # Butterworth has some rolloff even in passband
                assert passband_ratio > 0.98
            elif filter_type == 'cos2':
                # Cosine-squared should have good passband preservation
                assert passband_ratio > 0.995

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
                assert stopband_ratio_high < 0.2
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
            FourierOperator.create_bandpass_operator(
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
        with pytest.raises(ValueError, match='Filter passband not represented'):
            FourierOperator.create_bandpass_operator(
                f_low=15.3,
                f_high=15.4,
                sample_rate=sample_rate,
                in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
                filter_type='square',
            )

    def test_cos2_passband_unity_gain(self):
        """Test that cos2 filter has unity gain throughout entire passband."""
        n = 2000
        sample_rate = 200.0
        f_low, f_high = 10.0, 20.0

        op = FourierOperator.create_bandpass_operator(
            f_low=f_low,
            f_high=f_high,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
            filter_type='cos2',
        )

        # Check that entire passband has unity gain
        freqs = jnp.fft.rfftfreq(op.fft_size, 1.0 / sample_rate)
        kernel = op.get_kernel()

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

    def test_multidimensional_arrays(self):
        """Test bandpass operator with multidimensional arrays."""
        # n_detectors = 6
        n_samples = 1500
        sample_rate = 150.0
        t = jnp.linspace(0, n_samples / sample_rate, n_samples)

        # Mixed frequency signals
        signals = jnp.array(
            [
                jnp.sin(2 * jnp.pi * 5 * t) + 0.3 * jnp.sin(2 * jnp.pi * 20 * t),
                jnp.sin(2 * jnp.pi * 8 * t) + 0.3 * jnp.sin(2 * jnp.pi * 25 * t),
                jnp.sin(2 * jnp.pi * 12 * t) + 0.3 * jnp.sin(2 * jnp.pi * 30 * t),
                jnp.sin(2 * jnp.pi * 15 * t) + 0.3 * jnp.sin(2 * jnp.pi * 35 * t),
                jnp.sin(2 * jnp.pi * 18 * t) + 0.3 * jnp.sin(2 * jnp.pi * 40 * t),
                jnp.sin(2 * jnp.pi * 22 * t) + 0.3 * jnp.sin(2 * jnp.pi * 45 * t),
            ]
        )

        # Bandpass filter 10-20 Hz
        op = FourierOperator.create_bandpass_operator(
            f_low=10.0,
            f_high=20.0,
            sample_rate=sample_rate,
            in_structure=jax.ShapeDtypeStruct(signals.shape, signals.dtype),
            apodize=True,
        )

        filtered = op(signals)

        # Check shape
        assert filtered.shape == signals.shape

        # Verify filtering for each detector
        for i, freq in enumerate([5, 8, 12, 15, 18, 22]):
            fft_orig = jnp.fft.rfft(signals[i])
            fft_filt = jnp.fft.rfft(filtered[i])

            # Check at the main frequency
            freq_idx = int(freq * n_samples / sample_rate)

            if 10 <= freq <= 20:
                # Should pass through
                assert jnp.abs(fft_filt[freq_idx]) > 0.95 * jnp.abs(fft_orig[freq_idx])
            else:
                # Should be attenuated
                assert jnp.abs(fft_filt[freq_idx]) < 0.1 * jnp.abs(fft_orig[freq_idx])
