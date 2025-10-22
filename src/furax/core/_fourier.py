from collections.abc import Callable
from typing import ClassVar

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float, Inexact, PyTree

from ._base import AbstractLinearOperator, square

__all__ = [
    'FourierOperator',
    'BandpassOperator',
]


@square
class FourierOperator(AbstractLinearOperator):
    """Apply a kernel in the Fourier domain.

    This operator applies element-wise multiplication with a kernel in the Fourier domain.
    Similar to SymmetricBandToeplitzOperator but allows for general kernels.

    Two methods are available:
        - fft: Apply FFT on the entire input
        - overlap_save: Apply FFT on chunked input for memory efficiency

    Args:
        fourier_kernel: Complex-valued kernel in Fourier domain. Should match the FFT output
            size (for rfft, this is n//2 + 1 elements). Can be multidimensional for
            block-diagonal operators.
        in_structure: Input structure specification.
        method: Computation method ('fft' or 'overlap_save'). Default: 'overlap_save'.
        fft_size: FFT size for overlap_save method. If None, uses default.
        apodize: Apply Hamming window in overlap regions to reduce edge artifacts.
            Only used with overlap_save method. If None, defaults to True for
            overlap_save and False for fft.

    Usage:
        >>> import jax.numpy as jnp
        >>> # Create a low-pass filter kernel
        >>> n = 1000
        >>> freqs = jnp.fft.rfftfreq(n, 1.0/200.0)
        >>> kernel = (freqs < 10.0).astype(jnp.complex128)
        >>> op = FourierOperator(
        ...     fourier_kernel=kernel,
        ...     in_structure=jax.ShapeDtypeStruct((n,), jnp.float64),
        ...     method='fft'
        ... )
        >>> signal = jnp.ones(n)
        >>> filtered = op(signal)
    """

    METHODS: ClassVar[tuple[str, ...]] = ('fft', 'overlap_save')

    fourier_kernel: Float[Array, '...'] | complex
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    method: str = equinox.field(static=True)
    fft_size: int | None
    apodize: bool = equinox.field(static=True)

    def __init__(
        self,
        fourier_kernel: Float[Array, '...'] | complex,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        method: str = 'overlap_save',
        fft_size: int | None = None,
        apodize: bool | None = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f'Invalid method {method}. Choose from: {", ".join(self.METHODS)}')

        self.fourier_kernel = fourier_kernel
        self._in_structure = in_structure
        self.method = method

        # Default apodize to True for overlap_save, False for fft
        if apodize is None:
            apodize = method == 'overlap_save'
        self.apodize = apodize

        if fft_size is not None and method != 'overlap_save':
            raise ValueError('The FFT size is only used by the overlap_save method.')

        if apodize and method != 'overlap_save':
            raise ValueError('Apodization is only used by the overlap_save method.')

        if fft_size is None and method == 'overlap_save':
            # Get the kernel size to determine appropriate FFT size
            kernel_size = self._get_kernel_size()
            fft_size = self._get_default_fft_size(kernel_size)

        self.fft_size = fft_size

    def _get_kernel_size(self) -> int:
        """Get the size of the Fourier kernel."""
        kernel = jnp.asarray(self.fourier_kernel)
        return kernel.shape[-1]

    @staticmethod
    def _get_default_fft_size(kernel_size: int) -> int:
        """Get default FFT size based on kernel size."""
        # Use a power of 2 that's larger than kernel size
        additional_power = 1
        return int(2 ** (additional_power + np.ceil(np.log2(max(kernel_size, 32)))))

    @staticmethod
    def _create_apodization_window(fft_size: int, overlap: int) -> Array:
        """Create a Hamming window for apodization in overlap regions.

        The window is constructed such that:
        - It is 1.0 in the middle (valid output region)
        - It smoothly tapers to 0 in the overlap regions on both sides

        Args:
            fft_size: Size of the FFT block
            overlap: Size of the overlap region on each side

        Returns:
            Window array of shape (fft_size,)
        """
        # Create full Hamming window
        window = jnp.ones(fft_size)

        if overlap > 0:
            # Create Hamming window for the overlap region
            # The Hamming window goes from 0 to 1 over the overlap region
            hamming = 0.54 - 0.46 * jnp.cos(jnp.pi * jnp.arange(overlap) / overlap)

            # Apply taper at the beginning (left overlap)
            window = window.at[:overlap].set(hamming)

            # Apply taper at the end (right overlap)
            window = window.at[-overlap:].set(hamming[::-1])

        return window

    def _get_func(self) -> Callable[[Array], Array]:
        """Get the appropriate computation function based on method."""
        if self.method == 'fft':
            return self._apply_fft
        if self.method == 'overlap_save':
            return self._apply_overlap_save

        raise NotImplementedError

    def _apply_fft(self, x: Array) -> Array:
        """Apply Fourier kernel using FFT on the entire signal.

        This method transforms the entire signal to Fourier domain, applies
        the kernel, and transforms back.
        """
        n = x.shape[-1]

        # Forward FFT (use rfft for real signals)
        X = jnp.fft.rfft(x)

        # Apply Fourier kernel
        # Broadcast kernel to match X if needed
        kernel = jnp.asarray(self.fourier_kernel)
        X_filtered = X * kernel

        # Inverse FFT
        y = jnp.fft.irfft(X_filtered, n=n)

        return y

    def _apply_overlap_save(self, x: Array) -> Array:
        """Apply Fourier kernel using overlap-save method for chunked FFT.

        This method follows the same pattern as SymmetricBandToeplitzOperator._apply_overlap_save.
        It processes the signal in overlapping chunks to reduce memory usage while avoiding
        edge artifacts.
        """
        assert self.fft_size is not None

        l = x.shape[-1]  # Signal length
        kernel = jnp.asarray(self.fourier_kernel)

        # Determine overlap based on kernel characteristics
        # For Fourier filtering, we need overlap on both sides to handle edge effects
        # At each block, discard 25% from each end
        half_overlap = self.fft_size // 4  # 25% on each side
        overlap = 2 * half_overlap  # Total overlap
        step_size = self.fft_size - overlap  # 50% valid output per block

        # Number of blocks needed
        # nblock = int(np.ceil((l + overlap) / step_size))
        nblock = int(np.ceil((l + 3 * half_overlap) / step_size))

        # Total padded length
        total_length = (nblock - 1) * step_size + self.fft_size

        # Padding
        # x_padding_start = half_overlap
        x_padding_start = 2 * half_overlap
        x_padding_end = total_length - half_overlap - l

        # Use edge padding when apodizing to avoid discontinuities
        if self.apodize:
            x_padded = jnp.pad(
                x, (x_padding_start, x_padding_end), mode='reflect', reflect_type='odd'
            )
        else:
            x_padded = jnp.pad(x, (x_padding_start, x_padding_end), mode='constant')

        # Output array
        y = jnp.zeros(l + x_padding_end, dtype=x.dtype)

        # Prepare Fourier kernel for this FFT size
        # The kernel might be sized for the full signal, but we need it for fft_size chunks

        if kernel.size == self.fft_size // 2 + 1:
            # Kernel is already the right size for rfft of fft_size
            # Convert to full FFT by adding conjugate symmetric part
            H_full = jnp.zeros(self.fft_size, dtype=jnp.complex128)
            H_full = H_full.at[: kernel.size].set(kernel)
            # Mirror for negative frequencies (conjugate symmetry for real signals)
            if self.fft_size % 2 == 0:
                H_full = H_full.at[kernel.size :].set(jnp.conj(kernel[-2:0:-1]))
            else:
                H_full = H_full.at[kernel.size :].set(jnp.conj(kernel[-1:0:-1]))
            H = H_full
        elif kernel.size == self.fft_size:
            # Kernel is already for full FFT of the right size
            H = kernel
        else:
            # Kernel is for a different size - need to resample
            # Convert kernel to time domain, resample, and convert back
            # First convert rfft kernel to full fft if needed
            if jnp.iscomplexobj(kernel):
                # Assume it's an rfft kernel - convert to full spectrum
                n_orig = 2 * (kernel.size - 1)  # Original signal length
                kernel_time = jnp.fft.irfft(kernel, n=n_orig)
            else:
                # It's already time domain or full FFT
                kernel_time = jnp.fft.ifft(kernel).real

            # Resample to fft_size
            if kernel_time.size > self.fft_size:
                # Truncate
                kernel_time_resized = kernel_time[: self.fft_size]
            else:
                # Pad with zeros
                kernel_time_resized = jnp.pad(
                    kernel_time, (0, self.fft_size - kernel_time.size), mode='constant'
                )

            # Convert back to frequency domain
            H = jnp.fft.fft(kernel_time_resized)

        # Create apodization window if requested
        # The window tapers only the regions that will be discarded from each end
        if self.apodize:
            window = self._create_apodization_window(self.fft_size, half_overlap)
        else:
            window = None

        def func(iblock, y):  # type: ignore[no-untyped-def]
            """Process one block."""
            position = iblock * step_size
            x_block = lax.dynamic_slice(x_padded, (position,), (self.fft_size,))

            # Apply apodization window if enabled
            if window is not None:
                x_block = x_block * window

            # Forward FFT
            X = jnp.fft.fft(x_block)

            # Apply Fourier kernel
            X_filtered = X * H

            # Inverse FFT
            y_block = jnp.fft.ifft(X_filtered).real

            # Save the valid portion (discard overlap region from beginning)
            # Discard half_overlap samples each from the start and the end
            y = lax.dynamic_update_slice(
                y, lax.dynamic_slice(y_block, (half_overlap,), (step_size,)), (position,)
            )
            return y

        # Process all blocks
        y = lax.fori_loop(0, nblock, func, y)

        # Extract the original signal length
        return y[half_overlap : half_overlap + l]  # type: ignore[no-any-return]

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        """Apply Fourier kernel to input array.

        For multidimensional inputs, the filter is applied along the last axis.
        """
        func = jnp.vectorize(self._get_func(), signature='(n)->(n)')
        return func(x)  # type: ignore[no-any-return]

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def as_matrix(self) -> Inexact[Array, 'a a']:
        """Returns the operator as a dense matrix.

        Warning: This can be memory-intensive for large inputs.
        """
        from functools import partial

        @partial(jnp.vectorize, signature='(n)->(n,n)')
        def func(x: Array) -> Array:
            """Create matrix by applying operator to each basis vector."""
            n = x.size
            matrix = jnp.zeros((n, n), dtype=x.dtype)

            for i in range(n):
                e_i = jnp.zeros(n, dtype=x.dtype)
                e_i = e_i.at[i].set(1.0)
                matrix = matrix.at[:, i].set(self._get_func()(e_i))

            return matrix

        x = jnp.zeros(self.in_structure().shape, self.in_structure().dtype)
        blocks: Array = func(x)

        # Handle multidimensional case
        if blocks.ndim > 2:
            # Return block diagonal matrix
            import jax.scipy.linalg as jsl

            blocks = blocks.reshape(-1, blocks.shape[-1], blocks.shape[-1])
            matrix: Array = jsl.block_diag(*blocks)
            return matrix

        return blocks


class BandpassOperator:
    """Factory for creating bandpass filter operators using FourierOperator.

    This is not a LinearOperator itself, but provides a factory method to create
    a FourierOperator configured for bandpass filtering.

    Usage:
        >>> import jax.numpy as jnp
        >>> tod = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
        >>> op = BandpassOperator.create(
        ...     f_low=5.0,
        ...     f_high=15.0,
        ...     sample_rate=1000.0,
        ...     in_structure=jax.ShapeDtypeStruct(tod.shape, tod.dtype)
        ... )
        >>> filtered = op(tod)
    """

    @classmethod
    def create(
        cls,
        f_low: float,
        f_high: float,
        sample_rate: float,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        filter_type: str = 'square',
        method: str = 'overlap_save',
        fft_size: int | None = None,
        apodize: bool | None = None,
    ) -> FourierOperator:
        """Create a FourierOperator configured for bandpass filtering.

        Args:
            f_low: Lower frequency cutoff (inclusive) [Hz].
            f_high: Upper frequency cutoff (inclusive) [Hz].
            sample_rate: Sampling rate of the input signal [Hz].
            in_structure: Input structure specification.
            filter_type: Filter shape type. Options: 'square', 'butter4', 'cos2'.
                Default: 'square'.
                - 'square': Sharp cutoff (ideal brick-wall filter)
                - 'butter4': 4th-order Butterworth filter (smooth rolloff)
                - 'cos2': Cosine-squared transition (smooth rolloff)
            method: Computation method ('fft' or 'overlap_save'). Default: 'overlap_save'.
            fft_size: FFT size for overlap_save method. If None, uses default.
            apodize: Apply Hamming window in overlap regions to reduce edge artifacts.
                Only used with overlap_save method. If None, defaults to True for
                overlap_save and False for fft.

        Returns:
            FourierOperator configured with bandpass kernel.

        Raises:
            ValueError: If frequency parameters or filter_type are invalid.

        Note:
            For overlap_save method, the kernel is created based on fft_size rather than
            the full signal length to ensure proper chunked processing.
        """
        # Validate filter_type
        valid_filter_types = ('square', 'butter4', 'cos2')
        if filter_type not in valid_filter_types:
            raise ValueError(
                f'Invalid filter_type {filter_type}. Choose from: {", ".join(valid_filter_types)}'
            )

        if f_low < 0:
            raise ValueError(f'f_low must be non-negative, got {f_low}')
        if f_high > sample_rate / 2:
            raise ValueError(
                f'f_high must be less than Nyquist frequency {sample_rate / 2}, got {f_high}'
            )
        if f_low >= f_high:
            raise ValueError(f'f_low must be less than f_high, got f_low={f_low}, f_high={f_high}')

        # Get signal length from in_structure
        shape = in_structure.shape if hasattr(in_structure, 'shape') else in_structure().shape
        n = shape[-1]

        # Determine kernel size based on method
        if method == 'overlap_save':
            # For overlap_save, create kernel for the chunk size
            if fft_size is None:
                # Use default FFT size
                fft_size = FourierOperator._get_default_fft_size(n // 2 + 1)
            kernel_size = fft_size
        else:
            # For 'fft' method, use full signal length
            kernel_size = n

        # Create bandpass kernel in Fourier domain
        # Use rfft frequencies since we're filtering real signals
        freqs = jnp.fft.rfftfreq(kernel_size, 1.0 / sample_rate)

        # Create filter kernel based on filter_type
        if filter_type == 'square':
            # Sharp cutoff (ideal brick-wall filter)
            mask = (freqs >= f_low) & (freqs <= f_high)
            fourier_kernel = mask.astype(jnp.complex128)

        elif filter_type == 'butter4':
            # 4th-order Butterworth bandpass filter using scipy
            import scipy.signal

            # Design Butterworth bandpass filter
            # butter returns (b, a) coefficients for digital filter
            # Use 4th order bandpass (becomes 8th order total - 4th for each band edge)
            sos = scipy.signal.butter(
                4, [f_low, f_high], btype='bandpass', fs=sample_rate, output='sos'
            )

            # Convert to frequency response
            # Use freqs for the actual frequency points we need
            w = 2 * jnp.pi * freqs

            # Compute frequency response from second-order sections
            # H(w) = product of all second-order section responses
            H = jnp.ones(len(freqs), dtype=jnp.complex128)
            for section in sos:
                b0, b1, b2, a0, a1, a2 = section
                # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
                # For frequency response, substitute z = exp(j*w*T) where T = 1/fs
                z = jnp.exp(1j * w / sample_rate)
                z_inv = 1.0 / z
                z_inv2 = z_inv * z_inv

                numerator = b0 + b1 * z_inv + b2 * z_inv2
                denominator = a0 + a1 * z_inv + a2 * z_inv2
                H = H * (numerator / denominator)

            # Take magnitude for real-valued filter
            fourier_kernel = jnp.abs(H).astype(jnp.complex128)

        elif filter_type == 'cos2':
            # Cosine-squared transition outside the passband
            # Define transition width (10% of bandwidth on each side, outside the band)
            bandwidth = f_high - f_low
            transition_width = 0.1 * bandwidth

            # Transition regions are OUTSIDE the passband [f_low, f_high]
            # Lower transition: [max(0, f_low - transition_width), f_low]
            # Upper transition: [f_high, f_high + transition_width]
            f_low_transition_start = max(0.0, f_low - transition_width)

            # Initialize kernel
            kernel = jnp.zeros_like(freqs)

            # Passband (full transmission) - entire [f_low, f_high] range
            passband = (freqs >= f_low) & (freqs <= f_high)
            kernel = jnp.where(passband, 1.0, kernel)

            # Lower transition region (outside passband, below f_low)
            lower_transition = (freqs >= f_low_transition_start) & (freqs < f_low)
            # Cosine-squared from 0 to 1 as frequency increases toward f_low
            phase_low = (
                (freqs - f_low_transition_start) / (f_low - f_low_transition_start) * (jnp.pi / 2)
            )
            kernel = jnp.where(lower_transition, jnp.sin(phase_low) ** 2, kernel)

            # Upper transition region (outside passband, above f_high)
            f_high_transition_end = f_high + transition_width
            upper_transition = (freqs > f_high) & (freqs <= f_high_transition_end)
            # Cosine-squared from 1 to 0 as frequency increases away from f_high
            phase_high = (f_high_transition_end - freqs) / transition_width * (jnp.pi / 2)
            kernel = jnp.where(upper_transition, jnp.sin(phase_high) ** 2, kernel)

            fourier_kernel = kernel.astype(jnp.complex128)

        else:
            raise NotImplementedError(f'Filter type {filter_type} not implemented')

        # Validate that the filter has at least one unity gain point in the passband
        # This ensures the passband is actually represented in the frequency grid
        max_gain = jnp.max(jnp.abs(fourier_kernel))
        if max_gain < 0.99:  # Use 0.99 to account for numerical precision
            raise ValueError(
                f'Filter passband not represented in frequency grid. '
                f'Maximum filter gain is {max_gain:.4f}, expected ~1.0. '
                f'The bandwidth [{f_low}, {f_high}] Hz may be too narrow for the given '
                f'FFT size ({kernel_size}) and sample rate ({sample_rate} Hz). '
                f'Try using a larger FFT size or increasing the bandwidth.'
            )

        return FourierOperator(
            fourier_kernel=fourier_kernel,
            in_structure=in_structure,
            method=method,
            fft_size=fft_size,
            apodize=apodize,
        )
