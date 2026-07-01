from dataclasses import field

from jaxtyping import Array, Float

from furax import tree
from furax.math.quaternion import qrot_zaxis
from furax.obs.landscapes import TangentialLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import StokesI, StokesPyTreeType

__all__ = [
    'AtmospherePointingOperator',
]


class AtmospherePointingOperator(PointingOperator):
    """Operator that projects a flat atmosphere map to TOD using quaternion pointing.

    Models a "frozen" atmosphere as a 2D intensity pattern on a horizontal plane at
    height ``h``, drifting with wind velocity ``(vx, vy)``. For each detector and time
    sample it:

    1. Computes the gnomonic projection of the pointing direction onto the plane,
       giving physical coordinates ``(x, y)``.
    2. Adds the wind displacement ``(vx * t, vy * t)`` to obtain the atmosphere sample
       position.
    3. Samples the atmosphere map at that position (nearest-neighbour or bilinear).
    4. Optionally weights the sample by the airmass loading modulation ``1 / sin(el)``
       (enabled with ``elevation_modulation``), accounting for the longer line-of-sight
       path through the layer at low elevation.

    The transpose accumulates TOD back into the atmosphere map (binning).

    Attributes:
        landscape: The atmosphere map pixelization.
        qbore: Boresight quaternions in the horizon frame (z-axis = zenith),
            shape ``(n_samples, 4)``.
        qdet: Detector offset quaternions, shape ``(n_detectors, 4)``.
        wind_displacement: Pre-computed wind offset ``(vx * t_k, vy * t_k)`` for each
            sample, shape ``(n_samples, 2)``.
        chunk_size: Number of detectors processed per chunk (memory/speed trade-off).
        interpolate: If ``True``, use bilinear interpolation; otherwise nearest-neighbour.
        elevation_modulation: If ``True``, weight each sample by ``1 / sin(el)`` (airmass).
    """

    landscape: TangentialLandscape  # narrows PointingOperator.landscape
    wind_displacement: Float[Array, 'samp 2']
    elevation_modulation: bool = field(metadata={'static': True})

    @classmethod
    def from_wind(
        cls,
        landscape: TangentialLandscape,
        boresight_quaternions: Float[Array, 'samp 4'],
        detector_quaternions: Float[Array, 'det 4'],
        wind_velocity: Float[Array, '2'],
        times: Float[Array, ' samp'],
        *,
        chunk_size: int = 16,
        interpolate: bool = True,
        elevation_modulation: bool = False,
    ) -> 'AtmospherePointingOperator':
        """Create an AtmospherePointingOperator.

        Args:
            landscape: The atmosphere map pixelization (height stored here).
            boresight_quaternions: Boresight pointing in the horizon frame,
                shape ``(n_samples, 4)``.
            detector_quaternions: Detector offset quaternions, shape ``(n_detectors, 4)``.
            wind_velocity: Wind velocity ``(vx, vy)`` in the same physical units as
                ``landscape.height`` per second.
            times: Elapsed time for each sample, shape ``(n_samples,)``.
            chunk_size: Number of detectors per chunk.
            interpolate: Use bilinear interpolation (default: nearest-neighbour).
            elevation_modulation: Weight each sample by the airmass loading ``1 / sin(el)``.
        """
        ndet = detector_quaternions.shape[0]
        nsamp = boresight_quaternions.shape[0]
        wind_displacement = times[:, None] * wind_velocity[None, :]  # (samp, 2)
        out_structure = StokesI.structure_for((ndet, nsamp), dtype=landscape.dtype)
        return cls(
            landscape,
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            chunk_size=chunk_size,
            interpolate=interpolate,
            _out_structure=out_structure,
            wind_displacement=wind_displacement,
            elevation_modulation=elevation_modulation,
            in_structure=landscape.structure,
        )

    def _wind_xy(
        self, qdet_full: Float[Array, '*dims 4']
    ) -> tuple[Float[Array, ' *dims'], Float[Array, ' *dims']]:
        """Gnomonic projection onto the atmosphere screen, including wind displacement."""
        x, y = self.landscape.quat2xy(qdet_full)
        return (
            x + self.wind_displacement[None, :, 0],
            y + self.wind_displacement[None, :, 1],
        )

    def _modulate(
        self, tod: StokesPyTreeType, qdet_full: Float[Array, '*dims 4']
    ) -> StokesPyTreeType:
        """Weight each sample by the airmass loading ``1 / sin(el)`` when enabled."""
        if not self.elevation_modulation:
            return tod
        sin_el = qrot_zaxis(qdet_full)[..., 2]  # (det, samp)
        return tree.truediv(tod, sin_el)  # type: ignore[no-any-return]

    def _quat2index(self, qdet_full: Float[Array, '*dims 4']) -> Array:
        x, y = self._wind_xy(qdet_full)
        return self.landscape.pixel2index(*self.landscape.xy2pixel(x, y))

    def _quat2interp(self, qdet_full: Float[Array, '*dims 4']) -> tuple[Array, Array]:
        return self.landscape.xy2interp(*self._wind_xy(qdet_full))
