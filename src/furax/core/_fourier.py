from collections.abc import Callable

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Inexact, PyTree

from ._base import AbstractLinearOperator, square

__all__ = [
    'FourierOperator',
]


@square
class FourierOperator(AbstractLinearOperator):
    """Apply a kernel in the Fourier domain.

    This operator applies element-wise multiplication with a kernel in the Fourier domain.
    The kernel can be complex-valued.

    Usage:
        >>> import jax.numpy as jnp
        >>> n = 1000
        >>> fs = 200.0  # sampling frequency
        >>> cutoff = 10.0  # cutoff frequency
        >>> op = FourierOperator(
        ...     fourier_kernel=lambda f: f < cutoff,  # low-pass filter
        ...     in_structure=jax.ShapeDtypeStruct((n,), float),
        ...     sample_rate=fs,
        ... )
        >>> signal = jnp.ones(n)
        >>> filtered = op(signal)
    """

    kernel_func: Callable[[Array], Array] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    fft_size: int = equinox.field(static=True)
    apodize: bool = equinox.field(static=True)
    padding_width: int = equinox.field(static=True)
    sample_rate: float

    def __init__(
        self,
        kernel_func: Callable[[Float[Array, '...']], Inexact[Array, '...']],
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        sample_rate: float = 1.0,
        apodize: bool = True,
        padding_width: int | None = None,
    ):
        """Create a FourierOperator.

        Args:
            kernel_func: Function that generates the Fourier kernel as a function of frequency.
            in_structure: Input structure of the operator.
            sample_rate: Sample rate of the input signal [Hz].
                Important if the kernel function depends on physical frequency units.
            apodize: Pad and apply Hamming window to both ends to reduce edge artifacts.
            padding_width: Padding width in samples on each end.
                Defaults is 5% of data length (rounded up).
        """
        # Data length
        n = in_structure.shape[-1]

        # Set padding_width if unspecified
        if apodize:
            if padding_width is None:
                padding_width = int(np.ceil(0.05 * n))  # 5% of data length
        else:
            padding_width = 0

        # Use a power-of-2 FFT size for efficiency
        fft_size = _next_power_of_2(n + 2 * padding_width)

        # Compile the kernel function and check its output shape is correct
        jitted_kernel = jax.jit(kernel_func)
        freqs = jnp.fft.rfftfreq(fft_size, d=1 / sample_rate)
        kernel = jitted_kernel(freqs)
        if kernel.shape[-1] != freqs.size:
            raise ValueError('Bad kernel shape')

        self.kernel_func = jitted_kernel
        self._in_structure = in_structure
        self.fft_size = fft_size
        self.apodize = apodize
        self.padding_width = padding_width
        self.sample_rate = sample_rate

    def get_kernel(self) -> Inexact[Array, '...']:
        freqs = jnp.fft.rfftfreq(self.fft_size, d=1 / self.sample_rate)
        return self.kernel_func(freqs)

    @classmethod
    def create_bandpass_operator(
        cls,
        f_low: float,
        f_high: float,
        sample_rate: float,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        filter_type: str = 'square',
        apodize: bool = True,
    ) -> 'FourierOperator':
        """Creates a bandpass filtering operator.

        Example:
            >>> import jax.numpy as jnp
            >>> tod = jnp.sin(2 * jnp.pi * 10 * jnp.linspace(0, 1, 1000))
            >>> op = FourierOperator.create_bandpass_operator(
            ...     f_low=5.0,
            ...     f_high=15.0,
            ...     sample_rate=1000.0,
            ...     in_structure=jax.ShapeDtypeStruct(tod.shape, tod.dtype)
            ... )
            >>> filtered = op(tod)

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
            apodize: Apply Hamming window to reduce edge artifacts.

        Returns:
            FourierOperator configured with bandpass kernel.

        Raises:
            ValueError: If frequency parameters or filter_type are invalid.
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

        # Create filter kernel based on filter_type
        if filter_type == 'square':

            def kernel_func(freqs):  # type: ignore[no-untyped-def]
                # Sharp cutoff (ideal brick-wall filter)
                mask = (freqs >= f_low) & (freqs <= f_high)
                return mask.astype(jnp.complex128)

        elif filter_type == 'butter4':
            import scipy.signal

            def kernel_func(freqs):  # type: ignore[no-untyped-def]
                # 4th-order Butterworth bandpass filter using scipy

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
                H = jnp.ones_like(freqs, dtype=jnp.complex128)
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
                return jnp.abs(H).astype(jnp.complex128)

        elif filter_type == 'cos2':

            def kernel_func(freqs):  # type: ignore[no-untyped-def]
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
                    (freqs - f_low_transition_start)
                    / (f_low - f_low_transition_start)
                    * (jnp.pi / 2)
                )
                kernel = jnp.where(lower_transition, jnp.sin(phase_low) ** 2, kernel)

                # Upper transition region (outside passband, above f_high)
                f_high_transition_end = f_high + transition_width
                upper_transition = (freqs > f_high) & (freqs <= f_high_transition_end)
                # Cosine-squared from 1 to 0 as frequency increases away from f_high
                phase_high = (f_high_transition_end - freqs) / transition_width * (jnp.pi / 2)
                kernel = jnp.where(upper_transition, jnp.sin(phase_high) ** 2, kernel)

                return kernel.astype(jnp.complex128)

        else:
            raise NotImplementedError(f'Filter type {filter_type} not implemented')

        # Validate that the filter has at least one unity gain point in the passband
        # This ensures the passband is actually represented in the frequency grid
        n = in_structure.shape[-1]
        kernel = kernel_func(jnp.fft.rfftfreq(n, d=1.0 / sample_rate))
        max_gain = jnp.max(jnp.abs(kernel))
        if max_gain < 0.99:  # Use 0.99 to account for numerical precision
            msg = (
                f'Filter passband not represented in frequency grid. '
                f'Maximum filter gain is {max_gain:.4f}, expected ~1.0. '
                f'The bandwidth [{f_low}, {f_high}] Hz may be too narrow for the given sample '
                'rate and input size.'
            )
            raise ValueError(msg)

        return cls(
            kernel_func,
            in_structure,
            sample_rate=sample_rate,
            apodize=apodize,
        )

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

    def _apply(self, x: Array, kernel: Array) -> Array:
        """Apply Fourier kernel using FFT on the entire signal.

        This method transforms the entire signal to Fourier domain, applies
        the kernel, and transforms back.

        If apodization is enabled, the signal is padded on both ends and a
        Hamming window is applied to the padded regions to reduce edge artifacts.
        """
        n = x.shape[-1]

        if self.apodize:
            # Compute actual padding needed to match fft_size
            # This may differ from self.padding_width by 1 due to odd/even rounding
            total_padding = self.fft_size - n
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            # Pad the signal
            x_padded = jnp.pad(x, (pad_left, pad_right), mode='edge')

            # Create apodization window
            # The window tapers from 0 to 1 in the padded regions
            # Use the actual padding amounts
            window = self._create_apodization_window(self.fft_size, pad_left)

            # Apply window to padded signal
            x_windowed = x_padded * window

            # Forward FFT
            X = jnp.fft.rfft(x_windowed)

            # Apply Fourier kernel (already validated to match size)
            X_filtered = X * kernel

            # Inverse FFT
            y_padded = jnp.fft.irfft(X_filtered, n=self.fft_size)

            # Extract the original signal region (remove padding)
            y = y_padded[pad_left : pad_left + n]
        else:
            # No apodization
            # Forward FFT (use rfft for real signals)
            X = jnp.fft.rfft(x)

            # Apply Fourier kernel (already validated to match size)
            X_filtered = X * kernel

            # Inverse FFT
            y = jnp.fft.irfft(X_filtered, n=self.fft_size)

        return y

    def mv(self, x: Float[Array, '...']) -> Float[Array, '...']:
        """Apply Fourier kernel to input array.

        For multidimensional inputs, the filter is applied along the last axis.
        """
        kernel = self.get_kernel()
        func = jnp.vectorize(self._apply, signature='(n),(m)->(n)')
        return func(x, kernel)  # type: ignore[no-any-return]

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
                matrix = matrix.at[:, i].set(self.mv(e_i))

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


def _next_power_of_2(n: int) -> int:
    return int(2 ** np.ceil(np.log2(n)))
