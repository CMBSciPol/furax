import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from furax.obs.landscapes import TangentialLandscape
from furax.obs.stokes import StokesI

__all__ = [
    'simulate_kolmogorov_screen',
]


def simulate_kolmogorov_screen(
    landscape: TangentialLandscape,
    key: Key[Array, ''],
    *,
    power: float = -11 / 3,
    amplitude: float = 1.0,
) -> StokesI:
    r"""Generate a 2D Gaussian random field with an isotropic power-law spectrum.

    Models a frozen turbulent atmosphere screen using a Kolmogorov-like power
    spectrum :math:`P(k) \propto k^{\text{power}}`. The default exponent
    ``power = -11/3`` corresponds to the two-dimensional projection of a 3D
    Kolmogorov turbulence field (outer-scale-free regime).

    The field is generated in Fourier space: draw complex Gaussian white noise,
    multiply by :math:`\sqrt{A}\,k^{\text{power}/2}`, then transform back with
    an inverse FFT.  The DC component is excluded (set to zero) to avoid the
    singularity at :math:`k=0`.

    Args:
        landscape: Defines the map shape and dtype.  Only ``landscape.shape``
            and ``landscape.dtype`` are used; height and pixel spacing do not
            affect the generated field.
        key: JAX random key.
        power: Exponent of the power spectrum :math:`P(k) \propto k^p`.
            Default ``-11/3`` (Kolmogorov).
        amplitude: Overall amplitude prefactor :math:`A`.  Scales the variance
            of the output field.

    Returns:
        Stokes I map matching the shape and dtype of ``landscape``.
    """
    ny, nx = landscape.shape
    white_k = jnp.fft.fft2(jax.random.normal(key, (ny, nx), dtype=landscape.dtype))
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    kgrid_x, kgrid_y = jnp.meshgrid(kx, ky)
    k_norm = jnp.sqrt(kgrid_x**2 + kgrid_y**2).at[0, 0].set(jnp.inf)
    field_k = white_k * jnp.sqrt(amplitude * k_norm**power)
    field = jnp.real(jnp.fft.ifft2(field_k)).astype(landscape.dtype)
    return StokesI(i=field)
