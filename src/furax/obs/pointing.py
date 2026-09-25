import copy
import dataclasses
from dataclasses import field
from typing import Literal, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import jit, lax
from jaxtyping import Array, Float, Int, Integer, PyTree

from furax import AbstractLinearOperator
from furax.core import IndexOperator, MaskOperator, RavelOperator, TransposeOperator
from furax.math.coords import euler, to_gamma_angles, to_polarization_angle
from furax.obs.landscapes import StokesLandscape
from furax.obs.operators._qu_rotations import QURotationOperator, rotate_qu_cs
from furax.obs.sampling import AbstractSampler, QuaternionSampler, SamplingKernel
from furax.obs.spin2 import transported_gather, transported_scatter
from furax.obs.stencil import Interpolation
from furax.obs.stokes import Stokes, StokesI

__all__ = [
    'PointingOperator',
    'XSamplingOperator',
]

_StokesT = TypeVar('_StokesT', bound=Stokes)
_CosSin = tuple[Float[Array, '...'], Float[Array, '...']]


class PointingOperator(AbstractLinearOperator):
    """Operator that samples a sky map, e.g. into time-ordered data (TOD).

    Where every sample reads the map is given by a [`AbstractSampler`][furax.obs.sampling.AbstractSampler]:
    the pixels it reads with their weights, and the polarization frame it returns. The operator
    loops over the samples in batches and, for each batch:

    1. Reads the stencil of every sample, parallel-transporting the Q and U of every pixel read
       into the frame of the line of sight when the map is polarized
    2. Multiplies the sampled values by the sampler's scaling, if it has one
    3. Rotates Q and U into the sampler's polarization frame, e.g. that of a detector

    The transpose accumulates samples into a sky map (binning), and is exact whatever the
    sampler. With offsets in the sampler's kernel, the rotation of step 3 happens before
    the offsets are summed, so the offsets never mix polarization bases and weights given per
    Stokes component act on the rotated Q and U.

    Build one from quaternions with [`PointingOperator.create`][], or from any sampler with
    [`PointingOperator.from_sampler`][].

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
        sampler = dataclasses.replace(self.sampler, kernel=kernel)
        return PointingOperator.from_sampler(landscape, sampler, batch_size=self.batch_size)

    def as_expanded_operator(self) -> AbstractLinearOperator:
        """Return the equivalent QURotation @ (Index or XSampling) @ Ravel composition.

        Materialises the pointing once (pixel indices / coordinates and polarisation angles) so the
        expensive quaternion-to-sky transcendentals are hoisted out of repeated applies (e.g. every
        CG iteration). The polarisation rotation stays a [`QURotationOperator`][] so it still fuses
        with the acquisition chain via operator algebra, except with offsets: the rotation must then
        come before the offset weights, so the [`XSamplingOperator`][] applies it and the
        composition is only XSampling @ Ravel.

        Only a [`QuaternionSampler`][furax.obs.sampling.QuaternionSampler] can be expanded.
        """
        sampler = self.sampler
        if not isinstance(sampler, QuaternionSampler):
            raise NotImplementedError(
                f'only a QuaternionSampler can be expanded, not a {type(sampler).__name__}'
            )
        all_rows = jnp.arange(sampler.shape[0])
        qdet_full = sampler.quaternions(all_rows)
        # Ravel the spatial axes only; the Stokes container's backing array carries a leading
        # Stokes axis (axis 0) that must survive, so ravel axes 1..-1 and index the pixel axis last.
        ravel_op = RavelOperator(1, -1, in_structure=self.landscape.structure)
        kernel = sampler.kernel
        if not kernel.reads_one_pixel:
            sampler: AbstractLinearOperator = XSamplingOperator.create(
                self.landscape,
                qdet_full,
                interpolation=kernel.interpolation,
                offsets=kernel.offsets,
                offset_weights=kernel.weights,
            )
        else:
            sampler = self._nearest_sampler(all_rows, self.landscape.raveled_structure)
        if kernel.offsets is not None:
            return sampler @ ravel_op
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=sampler.out_structure)
        return qu_rot_op @ sampler @ ravel_op

    def _nearest_sampler(
        self, index: Int[Array, ' rows'], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        if not self.landscape.has_spin2:
            pix = self.sampler.nearest_indices(self.landscape, index)
            assert pix is not None  # (ndet, nsamp), -1 for out-of-bounds samples
            gather = self._index_operator(pix, in_structure)
            # A -1 index would wrap onto the last pixel, which the sample never observed.
            mask = MaskOperator.from_boolean_mask(pix >= 0, in_structure=gather.out_structure)
            return mask @ gather

        pointing = self.sampler.pointing_rows(self.landscape, index)
        stencil = pointing.stencil
        gather = self._index_operator(stencil.indices[..., 0], in_structure)
        # A nearest stencil weighs one or zero, so we can use a boolean mask
        weight_op = MaskOperator.from_boolean_mask(
            stencil.weights[..., 0] > 0, in_structure=gather.out_structure
        )
        transport_op = QURotationOperator(
            angles=pointing.transport_angles(), in_structure=gather.out_structure
        )
        return transport_op @ weight_op @ gather

    def _index_operator(
        self, pix: Integer[Array, 'det samp'], in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        """The gather of one pixel per sample, over the raveled map."""
        # Index the (leading) Stokes axis and the (trailing) pixel axis with broadcast arrays,
        # rather than the ergonomic `(..., pix)`. An Ellipsis (or slice) index element is a
        # non-array pytree leaf and is not a valid JAX type, so it would break the operator as a
        # multi-observation scan leaf; an all-array index tuple keeps it scan-safe.
        stokes_idx = jnp.arange(len(self.landscape.stokes))[:, None, None]
        return IndexOperator((stokes_idx, pix[None]), in_structure=in_structure)

    @property
    def _interpolates(self) -> bool:
        return self.sampler.kernel.interpolation is Interpolation.BILINEAR

    def _rotations(
        self, index: Int[Array, ' batch'], stokes_i: bool
    ) -> tuple[_CosSin | None, _CosSin | None]:
        r"""The rotation by the polarization angle, split by where it applies.

        Returns `(inside, after)`: the $(\cos 2\psi, \sin 2\psi)$ applied inside the
        transported gather and scatter, between the transport and the stencil weights, and the
        $(\cos \psi, \sin \psi)$ applied to the sampled values. With offsets the rotation must
        come before the offsets are summed, so that weights differing between Q and U act on the
        rotated Q and U rather than on the sky's; without offsets it applies afterwards.
        """
        if stokes_i:
            return None, None
        cos_sin = self.sampler.polarization_rotation(index)
        if cos_sin is None:
            return None, None
        if self.sampler.kernel.offsets is None:
            return None, cos_sin
        cos_psi, sin_psi = cos_sin
        return (cos_psi**2 - sin_psi**2, 2 * cos_psi * sin_psi), None

    def _sample(self, x_flat: _StokesT, index: Int[Array, ' batch']) -> _StokesT:
        """Sample the flat map for a batch of rows of samples."""
        inside, after = self._rotations(index, isinstance(x_flat, StokesI))
        tod: _StokesT
        pix = self.sampler.nearest_indices(self.landscape, index)
        if self.landscape.has_spin2:
            pointing = self.sampler.pointing_rows(self.landscape, index)
            tod = transported_gather(x_flat, *pointing, rotation=inside)
        elif pix is not None:
            # fast path for nearest-neighbour: one pixel per sample, so no stencil is needed
            sampled = x_flat[pix]
            # the gather wraps a -1 onto the last pixel, which the sample never observed
            tod = type(x_flat).from_array(jnp.where(pix >= 0, sampled.data, 0))
        else:
            stencil = self.sampler.pointing_rows(self.landscape, index).stencil
            # leading Stokes axis: index the (trailing) pixel axis and sum over the neighbour axis;
            # the weights broadcast over the leading Stokes axis for free.
            sampled = jnp.sum(x_flat.data[:, stencil.indices] * stencil.weights, axis=-1)
            tod = type(x_flat).from_array(sampled)

        tod = _scaled(tod, self.sampler.scaling(index))
        if after is None:
            return tod
        return rotate_qu_cs(tod, *after)

    def _bin(self, tod_batch: _StokesT, index: Int[Array, ' batch']) -> _StokesT:
        """Scatter-add a batch of samples into a sky map."""
        inside, after = self._rotations(index, isinstance(tod_batch, StokesI))
        tod_batch = _scaled(tod_batch, self.sampler.scaling(index))
        if after is not None:
            # Rotate back to the celestial frame with the inverse rotation
            cos_angles, sin_angles = after
            tod_batch = rotate_qu_cs(tod_batch, cos_angles, -sin_angles)

        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        arr = tod_batch.data  # (n_stokes, *batch_sample)
        n_stokes = arr.shape[0]
        zeros = jnp.zeros((n_stokes, n_pixels), self.landscape.dtype)

        pix = self.sampler.nearest_indices(self.landscape, index)
        if self.landscape.has_spin2:
            flat_sky = type(tod_batch).from_array(zeros)
            pointing = self.sampler.pointing_rows(self.landscape, index)
            binned_sky = transported_scatter(flat_sky, tod_batch, *pointing, rotation=inside)
            return type(tod_batch).from_array(binned_sky.data.reshape(n_stokes, *sky_shape))

        if pix is not None:
            # fast path for nearest-neighbour: one pixel per sample, so no stencil is needed
            # the scatter wraps a -1 onto the last pixel, so such a sample must add nothing
            contrib = jnp.where(pix >= 0, arr, 0)
            binned = zeros.at[:, pix.ravel()].add(contrib.reshape(n_stokes, -1))
            return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

        stencil = self.sampler.pointing_rows(self.landscape, index).stencil
        # (n_stokes, *batch_sample, n_nb): spread each sample over its neighbours (weights
        # broadcast over the leading Stokes axis for free).
        contrib = arr[..., None] * stencil.weights
        binned = zeros.at[:, stencil.indices.ravel()].add(contrib.reshape(n_stokes, -1))
        return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

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


class XSamplingOperator(AbstractLinearOperator):
    r"""Precomputed sky-sampling operator from cached world angles.

    The "expanded pointing" sampler. It stores the per-sample world angles `(theta, phi)`
    (computed once from the quaternion pointing, so the expensive quaternion-to-angle
    transcendentals are hoisted out of repeated applies) and on every apply gathers a raveled
    sky map at the pixels around each of them, four for bilinear interpolation and one for nearest
    neighbour.

    A sample may integrate over offsets around its direction, as in [`PointingOperator`][]: the
    world angles of every read direction are cached too, each is interpolated the same way, and
    the polarization of every pixel read is transported into the frame of the line of sight
    and rotated by the polarization angle before the weighted sum, so that weights given per Stokes
    component act on the detector's Q and U. The output is then in the detector frame, unlike the
    output without offsets, which is in the frame of the line of sight. The cache then holds
    `n_offsets` angle pairs per sample instead of one, so with many offsets, a beam sampled at
    thousands of nodes say, it can exceed the memory of the on-the-fly [`PointingOperator`][],
    which recomputes the directions on every apply.

    Works for any landscape supplying a [`Stencil`][furax.obs.stencil.Stencil] (HEALPix and WCS/CAR).

    Attributes:
        landscape: The sky pixelization supplying the stencil.
        theta: Cached spherical co-latitude angles, shape ``(ndet, nsamp)``.
        phi: Cached spherical longitude angles, shape ``(ndet, nsamp)``.
        kernel: What each sample integrates over: the interpolation, and the offsets and their
            weights if any.
        offset_theta: Co-latitude of every read direction, shape ``(ndet, nsamp, n_offsets)``,
            or `None` to read the line of sight alone.
        offset_phi: Longitude of every read direction, of the same shape, or `None`.
        polarization_angles: Cached polarization angles, shape ``(ndet, nsamp)``, applied between
            the transport and the offset weights. `None` if there are no offsets.
    """

    landscape: StokesLandscape
    theta: Float[Array, 'det samp']
    phi: Float[Array, 'det samp']
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})
    kernel: SamplingKernel = field(
        default_factory=lambda: SamplingKernel(Interpolation.BILINEAR), kw_only=True
    )
    offset_theta: Float[Array, 'det samp n_offsets'] | None = field(default=None, kw_only=True)
    offset_phi: Float[Array, 'det samp n_offsets'] | None = field(default=None, kw_only=True)
    polarization_angles: Float[Array, 'det samp'] | None = field(default=None, kw_only=True)

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        quaternions: Quaternion,
        *,
        interpolation: Interpolation = Interpolation.BILINEAR,
        offsets: Quaternion | None = None,
        offset_weights: Float[Array, ' n_offsets'] | Stokes | None = None,
    ) -> Self:
        """Cache the world angles of the given pointing.

        Args:
            landscape: The sky pixelization.
            quaternions: The pointing of every sample, shape (ndet, nsamp).
            interpolation: How each direction is read from the map.
            offsets: Rotations from the pointing to each direction a detector integrates over,
                in the frame of `quaternions`, shape (ndet, n_offsets), or `None`. With offsets,
                the output is rotated by the polarization angle of `quaternions`.
            offset_weights: The weight of each offset, shape (n_offsets,), or a [`Stokes`][] of
                the landscape's components, each of shape (n_offsets,). Required with `offsets`.
                Normalized to sum to one over the offsets in the map, as in
                [`PointingOperator.create`][].
        """
        kernel = SamplingKernel.create(
            landscape,
            quaternions.shape[0],
            interpolation=interpolation,
            offsets=offsets,
            weights=offset_weights,
        )
        theta, phi = landscape.quat2world(quaternions)
        offset_theta = offset_phi = polarization_angles = None
        if offsets is not None:
            # (ndet, nsamp, 1) x (ndet, 1, n_offsets) -> (ndet, nsamp, n_offsets)
            offset_theta, offset_phi = landscape.quat2world(
                quaternions[:, :, None] * offsets[:, None, :]
            )
            polarization_angles = to_polarization_angle(quaternions)
        # The map reaching this operator is raveled by PointingOperator.as_expanded_operator.
        return cls(
            landscape,
            theta=theta,
            phi=phi,
            in_structure=landscape.raveled_structure,
            _out_structure=landscape.structure_for(theta.shape),
            kernel=kernel,
            offset_theta=offset_theta,
            offset_phi=offset_phi,
            polarization_angles=polarization_angles,
        )

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def mv(self, x: _StokesT) -> _StokesT:
        # `x` is a raveled sky map: its single backing array is (n_stokes, n_pixels). Index the pixel
        # (last) axis with the cached pointing to produce the (n_stokes, ndet, nsamp) TOD.
        if self.offset_theta is None:
            stencil = self.landscape.world2stencil(self.theta, self.phi, self.kernel.interpolation)
        else:
            assert self.offset_phi is not None
            stencil = self.kernel.integrate(
                self.landscape.world2stencil(
                    self.offset_theta, self.offset_phi, self.kernel.interpolation
                )
            )
        if self.landscape.has_spin2:
            rotation = (
                None
                if self.polarization_angles is None
                else (jnp.cos(2 * self.polarization_angles), jnp.sin(2 * self.polarization_angles))
            )
            return transported_gather(x, stencil, self.theta, self.phi, rotation=rotation)
        return type(x).from_array(jnp.sum(x.data[..., stencil.indices] * stencil.weights, axis=-1))
