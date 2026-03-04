import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

from furax import AbstractLinearOperator
from furax.math.quaternion import to_gamma_angles
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.landscapes import StokesLandscape

from .pointing import PointingOperator

__all__ = [
    'build_acquisition_operator',
]


def build_acquisition_operator(
    landscape: StokesLandscape,
    boresight_quaternions: Array,
    detector_quaternions: Array,
    hwp_angles: Array | None = None,
    *,
    demodulated: bool = False,
    pointing_on_the_fly: bool = True,
    pointing_chunk_size: int = 16,
    dtype: DTypeLike = jnp.float64,
) -> AbstractLinearOperator:
    """Build an acquisition operator for a single observation. Does not include masking."""
    # The TOD shape
    ndet = detector_quaternions.shape[0]
    nsamp = boresight_quaternions.shape[0]
    data_shape = (ndet, nsamp)

    # Rotate into detector frame unless a HWP is present (even if demodulated)
    # NB: for demodulated data we have to rotate the other way...
    has_hwp = hwp_angles is not None or demodulated
    pointing = PointingOperator.create(
        landscape,
        boresight_quaternions,
        detector_quaternions,
        chunk_size=pointing_chunk_size,
        flip=demodulated,
        frame='boresight' if has_hwp else 'detector',
    )
    if not pointing_on_the_fly:
        pointing = pointing.as_expanded_operator()  # type: ignore[assignment]

    # If there is no HWP, we just add a polarizer at the end
    # NB: already in detector frame at this point
    polarizer = LinearPolarizerOperator.create(
        shape=data_shape,
        dtype=dtype,
        stokes=landscape.stokes,
    )
    if not has_hwp:
        return polarizer @ pointing

    # If there is a HWP, we need an additional rotation related to detector angle
    gamma = to_gamma_angles(detector_quaternions)[:, None]
    rot = QURotationOperator.create(data_shape, dtype, landscape.stokes, angles=gamma)

    # In the demodulated case, there is no polarizer
    if demodulated:
        return rot @ pointing

    # In the general case, we include polarizer and HWP
    hwp = HWPOperator.create(
        shape=data_shape,
        dtype=dtype,
        angles=hwp_angles,
        stokes=landscape.stokes,
    )
    return polarizer @ rot @ hwp @ pointing
