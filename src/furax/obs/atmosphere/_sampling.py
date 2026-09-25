from dataclasses import field
from typing import Self

import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import Array, Float, Int, Integer

from furax.math.coords import ZAXIS
from furax.obs.landscapes import StokesLandscape, TangentialLandscape
from furax.obs.sampling import AbstractSampler, PointingRows, SamplingKernel
from furax.obs.stencil import Interpolation, Stencil

__all__ = [
    'ScreenSampler',
]


class ScreenSampler(AbstractSampler):
    """Detectors reading a frozen atmosphere screen drifting with the wind.

    Models a "frozen" atmosphere as a 2D intensity pattern on a horizontal plane at the height of
    a [`TangentialLandscape`][furax.obs.landscapes.TangentialLandscape], drifting with wind
    velocity `(vx, vy)`. For each detector and time sample it:

    1. Computes the gnomonic projection of the line of sight onto the plane, giving physical
       coordinates `(x, y)`.
    2. Adds the wind displacement `(vx * t, vy * t)` to obtain the atmosphere sample position.
    3. Reads the screen at that position, with the kernel's interpolation.
    4. Optionally weights the sample by the airmass loading modulation `1 / sin(el)` (enabled with
       `elevation_modulation`), accounting for the longer line-of-sight path through the layer at
       low elevation.

    The screen is a projection plane, not the sphere, so it holds intensity only. The samples
    have shape (n_detectors, n_samples). Read it with
    [`PointingOperator.from_sampler`][furax.obs.pointing.PointingOperator.from_sampler].

    Attributes:
        kernel: The interpolation of the screen. Offsets are not supported.
        qbore: Boresight quaternions in the horizon frame (z-axis = zenith), shape (n_samples,).
        qdet: Detector offset quaternions, shape (n_detectors,).
        wind_displacement: Wind offset `(vx * t_k, vy * t_k)` of each sample, shape
            (n_samples, 2).
        elevation_modulation: If True, weight each sample by `1 / sin(el)` (airmass).
    """

    qbore: Quaternion
    qdet: Quaternion
    wind_displacement: Float[Array, 'samp 2']
    elevation_modulation: bool = field(default=False, metadata={'static': True})

    def __post_init__(self) -> None:
        # The wind displacement is added per sample, which only the (det, samp) pointing of an
        # un-offset detector lines up with.
        if self.kernel.offsets is not None:
            raise ValueError(f'{type(self).__name__} does not support offsets')

    @classmethod
    def from_wind(
        cls,
        boresight_quaternions: Quaternion,
        detector_quaternions: Quaternion,
        wind_velocity: Float[Array, '2'],
        times: Float[Array, ' samp'],
        *,
        interpolate: bool = True,
        elevation_modulation: bool = False,
    ) -> Self:
        """Sample a screen drifting at a constant wind velocity.

        Args:
            boresight_quaternions: Boresight pointing in the horizon frame, shape (n_samples,).
            detector_quaternions: Detector offset quaternions, shape (n_detectors,).
            wind_velocity: Wind velocity `(vx, vy)` in the same physical units as the landscape
                height per second.
            times: Elapsed time for each sample, shape (n_samples,).
            interpolate: Use bilinear interpolation, otherwise nearest neighbour.
            elevation_modulation: Weight each sample by the airmass loading `1 / sin(el)`.
        """
        interpolation = Interpolation.BILINEAR if interpolate else Interpolation.NEAREST
        return cls(
            kernel=SamplingKernel(interpolation),
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            wind_displacement=times[:, None] * wind_velocity[None, :],
            elevation_modulation=elevation_modulation,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.qdet.shape[0], self.qbore.shape[0]

    def pointing_rows(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> PointingRows:
        """The screen stencil, which carries no sky positions and no line of sight.

        The screen is a projection plane, not the sphere, so a neighbour has no co-latitude to
        transport a polarisation from. This sampler reads intensity only, which never asks: the
        angles returned alongside are placeholders, and a caller that would read them is refused by
        the stencil's missing positions first.
        """
        screen = _screen(landscape)
        if self.kernel.interpolation is Interpolation.BILINEAR:
            xy = self._wind_xy(screen, index)
            stencil = Stencil.unpositioned(*screen.xy2interp(*xy))
        else:
            indices = self._indices(screen, index)
            weights = jnp.ones((*indices.shape, 1), landscape.dtype)
            stencil = Stencil.unpositioned(indices[..., None], weights)
        nowhere = jnp.zeros(stencil.indices.shape[:-1], landscape.dtype)
        return PointingRows(stencil, nowhere, nowhere)

    def nearest_indices(
        self, landscape: StokesLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch samp'] | None:
        if not self.kernel.reads_one_pixel:
            return None
        return self._indices(_screen(landscape), index)

    def scaling(self, index: Int[Array, ' batch']) -> Float[Array, 'batch samp'] | None:
        """The airmass loading `1 / sin(el)` of each sample, when enabled."""
        if not self.elevation_modulation:
            return None
        sin_el = self._quaternions(index).rotate_vector(ZAXIS)[..., 2]  # (batch, samp)
        return 1 / sin_el

    def _quaternions(self, index: Int[Array, ' batch']) -> Quaternion:
        return self.qbore * self.qdet[index][:, None]

    def _wind_xy(
        self, landscape: TangentialLandscape, index: Int[Array, ' batch']
    ) -> tuple[Float[Array, 'batch samp'], Float[Array, 'batch samp']]:
        """Gnomonic projection onto the atmosphere screen, including wind displacement."""
        x, y = landscape.quat2xy(self._quaternions(index))
        return x + self.wind_displacement[:, 0], y + self.wind_displacement[:, 1]

    def _indices(
        self, landscape: TangentialLandscape, index: Int[Array, ' batch']
    ) -> Integer[Array, 'batch samp']:
        return landscape.pixel2index(*landscape.xy2pixel(*self._wind_xy(landscape, index)))


def _screen(landscape: StokesLandscape) -> TangentialLandscape:
    if not isinstance(landscape, TangentialLandscape):
        raise TypeError(
            f'a screen is read from a TangentialLandscape, not {type(landscape).__name__}'
        )
    return landscape
