import copy
import dataclasses
from dataclasses import field
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import jit, lax
from jaxtyping import Array, Float, Int, PyTree

from furax import AbstractLinearOperator
from furax.core import TransposeOperator
from furax.math.coords import euler, to_gamma_angles
from furax.obs.landscapes import StokesLandscape
from furax.obs.sampling import (
    AbstractSampler,
    PointingRows,
    PrecomputedSampler,
    QuaternionSampler,
    SamplingKernel,
)
from furax.obs.spin2 import rotated_gather, rotated_scatter
from furax.obs.stencil import Interpolation
from furax.obs.stokes import Stokes

__all__ = [
    'PointingOperator',
]

_StokesT = TypeVar('_StokesT', bound=Stokes)


class PointingOperator(AbstractLinearOperator):
    r"""Operator that samples a sky map, e.g. into time-ordered data (TOD).

    Where every sample reads the map is given by a
    [`AbstractSampler`][furax.obs.sampling.AbstractSampler], as one sparse row of the pointing
    matrix per sample: the pixels it reads, their weights, and the rotation of each pixel's
    $(Q, U)$ into the frame the sample is returned in, e.g. that of a detector. The operator loops
    over the samples in batches and, for each batch, gathers the pixels, rotates and weighs them,
    then multiplies the samples by the sampler's scaling, if it has one.

    The transpose accumulates samples into a sky map (binning), and is exact whatever the sampler.

    Build one from quaternions with [`PointingOperator.create`][], or from any sampler with
    [`PointingOperator.from_sampler`][]. [`PointingOperator.precomputed`][] computes the pointing
    once, for a map read many times.

    Attributes:
        landscape: The sky pixelization.
        sampler: Where each sample reads the map.
        batch_size: Number of rows of samples processed per batch (memory/speed tradeoff).
    """

    landscape: StokesLandscape
    sampler: AbstractSampler
    batch_size: int = field(default=32, metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        boresight_quaternions: Quaternion,
        detector_quaternions: Quaternion,
        *,
        batch_size: int = 32,
        frame: Literal['boresight', 'detector'] = 'boresight',
        interpolate: bool = False,
        offsets: Quaternion | None = None,
        offset_weights: Float[Array, ' n_offsets'] | Stokes | None = None,
    ) -> 'PointingOperator':
        r"""Build the operator from the boresight pointing and the detector offsets.

        Args:
            landscape: The sky pixelization.
            boresight_quaternions: Boresight quaternions, shape (n_samples,).
            detector_quaternions: Detector offset quaternions, shape (n_detectors,).
            batch_size: Number of detectors processed per batch.
            frame: Frame the polarization angle is measured in. In the `'boresight'` frame the
                z-rotation of each detector offset is stripped, so the angle is that of the
                boresight.
            interpolate: If True, bilinear interpolation over the four nearest pixels, otherwise
                nearest neighbour.
            offsets: Rotations from the line of sight to each read direction, in the detector
                frame, shape (n_offsets,) for the same offsets on every detector, or
                (n_detectors, n_offsets). Only their direction is used.
                `None` reads the line of sight alone.
            offset_weights: The weight of each offset, shape (n_offsets,), shared by every Stokes
                component, or a [`Stokes`][] of the landscape's components, each of shape
                (n_offsets,), to weigh them differently, e.g. with a beam per component. Required
                with `offsets`. A component's weights act on that component of the output, i.e.
                in the polarization frame chosen by `frame`, not in the sky's meridian basis. The
                weights are normalized to sum to one, per component, over the offsets whose pixels
                are in the map: a sample partly off a partial-sky map is renormalized to the part
                in view, like a partly covered bilinear sample.

        Examples:
            A single detector at the boresight, pointing at random directions given as the ZYZ
            Euler angles $(\phi, \theta, \psi)$ of the boresight:

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from fastquat import Quaternion
            >>> from furax.math.coords import from_iso_angles
            >>> from furax.obs.landscapes import HealpixLandscape
            >>> theta, phi, psi = jax.random.uniform(jax.random.key(0), (3, 1000)) * jnp.array(
            ...     [[jnp.pi], [2 * jnp.pi], [2 * jnp.pi]]
            ... )
            >>> landscape = HealpixLandscape(nside=64, stokes='IQU')
            >>> pointing = PointingOperator.create(
            ...     landscape, from_iso_angles(theta, phi, psi), Quaternion.ones((1,))
            ... )
            >>> pointing.out_structure.shape
            (1, 1000)
        """
        ndet = detector_quaternions.shape[0]
        kernel = SamplingKernel.create(
            landscape,
            ndet,
            interpolation=Interpolation.BILINEAR if interpolate else Interpolation.NEAREST,
            offsets=offsets,
            weights=offset_weights,
        )

        # In boresight frame, strip the z-rotation (gamma) from each detector quaternion.
        # This absorbs the frame correction into qdet so that the polarization angle always
        # works the same way, regardless of frame. Pixel indices are unaffected because
        # a z-rotation does not change the direction of the boresight (z) axis.
        #
        # NB: the xieta parametrization is incomplete and cannot describe all rotations.
        # Thus converting to xieta and back (with gamma=0) may not work in full generality.
        # This approach is more general and just as efficient.
        gamma = jnp.zeros(ndet, dtype=landscape.dtype)
        if frame == 'boresight':
            gamma = to_gamma_angles(detector_quaternions)
            q_z_neg = euler(2, -gamma)  # z-rotation by -gamma
            detector_quaternions = detector_quaternions * q_z_neg

        if kernel.offsets is not None:
            # An offset direction is fixed to the physical detector, so stripping gamma from qdet
            # must not turn it: rotate the offsets by +gamma to compensate, which also gives them
            # their per-detector shape. In the detector frame gamma is zero and this is exact.
            kernel = dataclasses.replace(kernel, offsets=euler(2, gamma)[:, None] * kernel.offsets)

        sampler = QuaternionSampler(
            kernel=kernel, qbore=boresight_quaternions, qdet=detector_quaternions
        )
        return cls.from_sampler(landscape, sampler, batch_size=batch_size)

    @classmethod
    def from_sampler(
        cls, landscape: StokesLandscape, sampler: AbstractSampler, *, batch_size: int = 32
    ) -> 'PointingOperator':
        """Build the operator reading a map where a sampler says.

        Args:
            landscape: The sky pixelization.
            sampler: Where each sample reads the map.
            batch_size: Number of rows of samples (indices into the first axis of the sampler's
                shape) processed per batch.
        """
        return cls(landscape, sampler, batch_size, in_structure=landscape.structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.landscape.structure_for(self.sampler.shape)

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'un-pointing' operation, i.e. map->tod."""
        x_flat = x.ravel()

        # Loop over batches of rows.
        # NB: lax.map was tried here (PR #172) instead of the fori_loop+scatter form
        # It seemed faster on GPU, but there was a 3-4x perf regression on CPU
        shape = self.sampler.shape
        batch_size, n_batches = _batch_plan(self.batch_size, shape[0])

        def body(i: Int[Array, ''], tod: _StokesT) -> _StokesT:
            # interval bounds must be static, so we shift the values afterwards
            # jax indexing semantics automatically clip out-of-bounds indices
            index = jnp.arange(batch_size) + i * batch_size
            tod_batch = self._sample(x_flat, index)
            return type(tod).from_array(tod.data.at[:, index].set(tod_batch.data))

        # Start from an empty timestream: every slot gets overwritten by body.
        tod_out: _StokesT = type(x).empty(shape, dtype=x.dtype)
        tod_out = lax.fori_loop(0, n_batches, body, tod_out)
        return tod_out

    def as_stokes_i(self, *, interpolate: bool | None = None) -> 'PointingOperator':
        """Return a copy of this operator restricted to StokesI.

        The offsets are kept. Offset weights given per Stokes component reduce to those of I, or
        to their mean over the components when the operator has no I component.

        Args:
            interpolate: Override the interpolation: bilinear if True, nearest neighbour if
                False. If `None` (default), the kernel's interpolation is kept.
        """
        effective_interpolate = self._interpolates if interpolate is None else interpolate
        if self.landscape.stokes == 'I' and effective_interpolate == self._interpolates:
            return self
        landscape = copy.copy(self.landscape)
        landscape.stokes = 'I'
        kernel = dataclasses.replace(
            self.sampler.kernel.intensity_only(),
            interpolation=Interpolation.BILINEAR
            if effective_interpolate
            else Interpolation.NEAREST,
        )
        sampler = self.sampler.with_kernel(kernel)
        return PointingOperator.from_sampler(landscape, sampler, batch_size=self.batch_size)

    def precomputed(
        self, store: Literal['rows', 'angles'] | None = None, *, batch_size: int = 0
    ) -> 'PointingOperator':
        """Return the same operator, with its pointing computed once.

        Hoists the quaternion-to-sky computations out of repeated applies, e.g. every iteration of
        an iterative solver, at the cost of storing the pointing. What is stored trades memory for
        speed:

        - `'rows'`: the rows of the pointing matrix, i.e. the pixels, weights and polarization
          rotations, see [`PrecomputedSampler`][furax.obs.sampling.PrecomputedSampler]. The fastest apply;
          the memory grows with the pixels each sample reads, four per direction with bilinear
          interpolation.
        - `'angles'`: the sky angles of each read direction, see
          [`AngleSampler`][furax.obs.sampling.AngleSampler]. The least memory; every apply
          recomputes the rows. Requires a [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler].

        By default, `'rows'` when every sample reads a single pixel, where the rows take about as
        much memory as the angles, and `'angles'` otherwise.

        Args:
            store: What to store, `'rows'` or `'angles'`. `None` (default) chooses as above.
            batch_size: Number of rows of samples processed per batch. The default, 0, processes
                them all at once, which is fastest once the pointing is stored.
        """
        if store is None:
            store = 'rows' if self.sampler.kernel.reads_one_pixel else 'angles'
        if store == 'rows':
            sampler: AbstractSampler = PrecomputedSampler.from_sampler(self.sampler, self.landscape)
        elif isinstance(self.sampler, QuaternionSampler):
            sampler = self.sampler.to_angles(self.landscape)
        else:
            raise TypeError(
                f'only a QuaternionSampler can be stored as angles, not a '
                f'{type(self.sampler).__name__}'
            )
        return PointingOperator.from_sampler(self.landscape, sampler, batch_size=batch_size)

    @property
    def _interpolates(self) -> bool:
        return self.sampler.kernel.interpolation is Interpolation.BILINEAR

    def _sample(self, x_flat: _StokesT, index: Int[Array, ' batch']) -> _StokesT:
        """Sample the flat map for a batch of rows of samples."""
        tod: _StokesT
        pix = (
            None
            if self.landscape.has_spin2
            else self.sampler.nearest_indices(self.landscape, index)
        )
        if pix is not None:
            # fast path for nearest-neighbour: one pixel per sample, so no stencil is needed
            sampled = x_flat[pix]
            # the gather wraps a -1 onto the last pixel, which the sample never observed
            tod = type(x_flat).from_array(jnp.where(pix >= 0, sampled.data, 0))
            return _scaled(tod, self.sampler.scaling(index))
        pointing = self._pointing(index)
        tod = _scaled(rotated_gather(x_flat, *pointing[:2]), self.sampler.scaling(index))
        if pointing.polarization_rotation is None:
            return tod
        return tod.rotate_qu(*pointing.polarization_rotation)

    def _bin(self, tod_batch: _StokesT, index: Int[Array, ' batch']) -> _StokesT:
        """Scatter-add a batch of samples into a sky map."""
        tod_batch = _scaled(tod_batch, self.sampler.scaling(index))
        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        n_stokes = tod_batch.data.shape[0]
        zeros = type(tod_batch).from_array(jnp.zeros((n_stokes, n_pixels), self.landscape.dtype))

        pix = (
            None
            if self.landscape.has_spin2
            else self.sampler.nearest_indices(self.landscape, index)
        )
        if pix is not None:
            # fast path for nearest-neighbour: one pixel per sample, so no stencil is needed
            # the scatter wraps a -1 onto the last pixel, so such a sample must add nothing
            contrib = jnp.where(pix >= 0, tod_batch.data, 0)
            binned = zeros.data.at[:, pix.ravel()].add(contrib.reshape(n_stokes, -1))
        else:
            pointing = self._pointing(index)
            if pointing.polarization_rotation is not None:
                # rotate back with the inverse rotation
                cos_2psi, sin_2psi = pointing.polarization_rotation
                tod_batch = tod_batch.rotate_qu(cos_2psi, -sin_2psi)
            binned = rotated_scatter(zeros, tod_batch, *pointing[:2]).data
        return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

    def _pointing(self, index: Int[Array, ' batch']) -> PointingRows:
        pointing = self.sampler.pointing_rows(self.landscape, index)
        if self.landscape.has_spin2 and pointing.neighbour_rotation is None:
            raise ValueError(
                f'{type(self.sampler).__name__} returns no rotation, so it cannot read a '
                f'polarized map'
            )
        return pointing

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(operator=self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'pointing' operation, i.e. tod->map."""
        # Loop over batches of rows
        n_rows = self.operator.sampler.shape[0]
        batch_size, n_batches = _batch_plan(self.operator.batch_size, n_rows)

        def body(i: Int[Array, ''], sky: _StokesT) -> _StokesT:
            # Past n_rows, indices are out of range; `sky` is never indexed by `index` so we need to
            # use the `unique` indices to mask out redundant/repeated contributions from the last
            # batch
            index = jnp.arange(batch_size) + i * batch_size
            unique = index < n_rows
            xbatch = x[index]
            unique = unique.reshape(-1, *(1,) * (xbatch.data.ndim - 2))
            sky_batch = self.operator._bin(unique * xbatch, index)

            # combine the results of the batches into one sky map
            return sky + sky_batch

        sky_out: _StokesT = self.operator.landscape.zeros()
        sky_out = lax.fori_loop(0, n_batches, body, sky_out)
        return sky_out


def _scaled[S: Stokes](tod: S, factor: Float[Array, '...'] | None) -> S:
    """The samples multiplied by a factor per sample, if any."""
    return tod if factor is None else type(tod).from_array(tod.data * factor)


def _batch_plan(batch_size: int, n: int) -> tuple[int, int]:
    """Resolve `(batch_size, n_batches)` for looping over `n` items in batches."""
    batch_size = min(batch_size, n) if batch_size > 0 else n
    n_batches = (n + batch_size - 1) // batch_size
    return batch_size, n_batches
