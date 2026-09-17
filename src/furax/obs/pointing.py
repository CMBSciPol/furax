import copy
from dataclasses import field
from typing import Literal, NamedTuple, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import jit, lax
from jaxtyping import Array, Float, Int, Integer, PyTree

from furax import AbstractLinearOperator
from furax.core import IndexOperator, MaskOperator, RavelOperator, TransposeOperator
from furax.math.coords import (
    euler,
    to_gamma_angles,
    to_polarization_angle,
    to_polarization_angle_cos_sin,
)
from furax.obs.landscapes import StokesLandscape
from furax.obs.operators._qu_rotations import QURotationOperator, rotate_qu_cs
from furax.obs.spin2 import spin2_cos_sin_zs, transported_gather, transported_scatter
from furax.obs.stencil import Interpolation, Stencil
from furax.obs.stokes import Stokes, StokesI

__all__ = [
    'PointingOperator',
    'XSamplingOperator',
]

_StokesT = TypeVar('_StokesT', bound=Stokes)


class SampledPointing(NamedTuple):
    stencil: Stencil
    theta: Float[Array, 'det samp']
    phi: Float[Array, 'det samp']

    def transport_angles(self) -> Float[Array, 'det samp']:
        assert self.stencil.positions is not None  # the caller transports, so it has positions
        cos_2delta, sin_2delta = spin2_cos_sin_zs(
            *self.stencil.positions,
            jnp.cos(self.theta)[..., None],
            jnp.sin(self.theta)[..., None],
            self.phi[..., None],
        )
        return 0.5 * jnp.arctan2(sin_2delta[..., 0], cos_2delta[..., 0])


class PointingOperator(AbstractLinearOperator):
    """Operator that projects sky maps to time-ordered data (TOD) using quaternion pointing.

    Equivalent to: QURotation @ Index @ Ravel, but computed on-the-fly to save memory.
    For each detector and time sample, it:
    1. Computes the sky pixel from boresight and detector quaternions
    2. Samples the sky map at that pixel, parallel-transporting the Q and U of every pixel it
       reads into the sampled direction's frame when the map is polarized
    3. Rotates Stokes QU by the polarization angle

    The transpose accumulates TOD into a sky map (binning).

    A detector may read the sky at several offsets around its pointing direction, each with a
    weight, and return their weighted sum. The offsets model an integration within a sample: a
    finite time integration, a pixel window, or a beam. Every offset reads the map with the same
    interpolation, and the polarization of every pixel read is transported into the frame of the
    un-offset direction before the sum, so the offsets never mix polarization bases.

    Attributes:
        landscape: The sky pixelization (HEALPix landscape).
        qbore: Boresight quaternions, shape (n_samples,).
        qdet: Detector quaternions, shape (n_detectors,).
        batch_size: Number of detectors processed per batch (memory/speed tradeoff).
        interpolate: If True, bilinear interpolation over the four nearest pixels; else nearest.
        offsets: Directions a sample integrates over, one row per detector, expressed in the frame
            `qdet` is in, shape (n_detectors, n_offsets). `None` to read the pointing direction
            alone.
        offset_weights: The weight of each offset, shape (n_offsets,), or one row per Stokes
            component, shape (n_stokes, n_offsets). `None` if `offsets` is `None`.
    """

    landscape: StokesLandscape
    qbore: Quaternion
    qdet: Quaternion
    batch_size: int = field(metadata={'static': True})
    interpolate: bool = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})
    # keyword-only so that subclasses can still declare required fields
    offsets: Quaternion | None = field(default=None, kw_only=True)
    offset_weights: Float[Array, ' n_offsets'] | Float[Array, 'n_stokes n_offsets'] | None = field(
        default=None, kw_only=True
    )

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
        offset_weights: Float[Array, ' n_offsets']
        | Float[Array, 'n_stokes n_offsets']
        | None = None,
    ) -> 'PointingOperator':
        """Build the operator from the boresight pointing and the detector offsets.

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
            offsets: Directions a sample integrates over, as detector-frame quaternions composed
                like a detector offset, shape (n_offsets,) for the same directions on every
                detector, or (n_detectors, n_offsets). Only their direction is used. `None` reads
                the pointing direction alone.
            offset_weights: The weight of each offset, shape (n_offsets,), or one row per Stokes
                component of the landscape, shape (n_stokes, n_offsets). Required with `offsets`.
        """
        # Explicitly determine the output structure
        ndet = detector_quaternions.shape[0]
        nsamp = boresight_quaternions.shape[0]
        out_structure = landscape.structure_for((ndet, nsamp))

        if (offsets is None) != (offset_weights is None):
            raise ValueError('offsets and offset_weights must be given together')
        if offsets is not None:
            assert offset_weights is not None
            offset_weights = jnp.asarray(offset_weights, dtype=landscape.dtype)
            n_offsets = offsets.shape[-1]
            n_stokes = len(landscape.stokes)
            if offsets.shape not in {(n_offsets,), (ndet, n_offsets)}:
                raise ValueError(
                    f'offsets has shape {offsets.shape}, expected ({n_offsets},) or '
                    f'({ndet}, {n_offsets}) for {ndet} detectors'
                )
            if offset_weights.shape not in {(n_offsets,), (n_stokes, n_offsets)}:
                raise ValueError(
                    f'offset_weights has shape {offset_weights.shape}, expected ({n_offsets},) or '
                    f'({n_stokes}, {n_offsets}) for {n_offsets} offsets and a {landscape.stokes} '
                    'landscape'
                )

        # In boresight frame, strip the z-rotation (gamma) from each detector quaternion.
        # This absorbs the frame correction into qdet so that _get_cos_sin_angles always
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

        if offsets is not None:
            # An offset direction is fixed to the physical detector, so stripping gamma from qdet
            # must not turn it: rotate the offsets by +gamma to compensate, which also gives them
            # their per-detector shape. In the detector frame gamma is zero and this is exact.
            offsets = euler(2, gamma)[:, None] * offsets

        return cls(
            landscape,
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            batch_size=batch_size,
            interpolate=interpolate,
            in_structure=landscape.structure,
            _out_structure=out_structure,
            offsets=offsets,
            offset_weights=offset_weights,
        )

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'un-pointing' operation, i.e. map->tod."""
        x_flat = x.ravel()

        def mv_inner(qdet: Quaternion, offsets: Quaternion | None) -> _StokesT:
            # Expand detector quaternions from boresight and offsets: (samp) x (det, 1) -> (det, samp)
            qdet_full = self.qbore * qdet[:, None]

            tod = self._sample(x_flat, qdet_full, offsets)
            tod = self._modulate(tod, qdet_full)

            if isinstance(tod, StokesI):
                # no rotation needed
                return tod

            # Return the rotated Stokes parameters
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            return rotate_qu_cs(tod, cos_angles, sin_angles)

        # Loop over batches of detectors.
        # NB: lax.map was tried here (PR #172) instead of the fori_loop+scatter form
        # It seemed faster on GPU, but there was a 3-4x perf regression on CPU
        ndet, nsamp = self.out_structure.shape
        batch_size, n_batches = _batch_plan(self.batch_size, ndet)

        def body(i: Int[Array, ''], tod: _StokesT) -> _StokesT:
            # interval bounds must be static, so we shift the values afterwards
            # jax indexing semantics automatically clip out-of-bounds indices
            idet = jnp.arange(batch_size) + i * batch_size
            tod_batch = mv_inner(self.qdet[idet], self._offsets_batch(idet))
            return type(tod).from_array(tod.data.at[:, idet].set(tod_batch.data))

        # Start from an empty timestream: every slot gets overwritten by body.
        tod_out: _StokesT = type(x).empty((ndet, nsamp), dtype=x.dtype)
        tod_out = lax.fori_loop(0, n_batches, body, tod_out)
        return tod_out

    def _offsets_batch(self, idet: Int[Array, ' batch']) -> Quaternion | None:
        """The offsets of a batch of detectors, or `None` when the operator has none."""
        return None if self.offsets is None else self.offsets[idet]

    def as_stokes_i(self, *, interpolate: bool | None = None) -> 'PointingOperator':
        """Return a copy of this operator restricted to StokesI.

        Args:
            interpolate: Override the interpolation flag.  If ``None`` (default),
                the flag is inherited from ``self.interpolate``.
        """
        effective_interpolate = self.interpolate if interpolate is None else interpolate
        if self.landscape.stokes == 'I' and effective_interpolate == self.interpolate:
            return self
        landscape = copy.copy(self.landscape)
        landscape.stokes = 'I'
        ndet, nsamp = self.qdet.shape[0], self.qbore.shape[0]
        out_structure = StokesI.structure_for((ndet, nsamp), dtype=landscape.dtype)
        return PointingOperator(
            landscape,
            qbore=self.qbore,
            qdet=self.qdet,
            batch_size=self.batch_size,
            interpolate=effective_interpolate,
            in_structure=landscape.structure,
            _out_structure=out_structure,
        )

    def as_expanded_operator(self) -> AbstractLinearOperator:
        """Return the equivalent QURotation @ (Index or XSampling) @ Ravel composition.

        Materialises the pointing once (pixel indices / coordinates and polarisation angles) so the
        expensive quaternion-to-sky transcendentals are hoisted out of repeated applies (e.g. every
        CG iteration). The polarisation rotation stays a [`QURotationOperator`][] so it still fuses
        with the acquisition chain via operator algebra.
        """
        qdet_full = self.qbore * self.qdet[:, None]
        # Ravel the spatial axes only; the Stokes container's backing array carries a leading
        # Stokes axis (axis 0) that must survive, so ravel axes 1..-1 and index the pixel axis last.
        ravel_op = RavelOperator(1, -1, in_structure=self.landscape.structure)
        sampler = (
            XSamplingOperator.create(self.landscape, qdet_full)
            if self.interpolate
            else self._nearest_sampler(qdet_full, self.landscape.raveled_structure)
        )
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=sampler.out_structure)
        return qu_rot_op @ sampler @ ravel_op

    def _nearest_sampler(
        self, qdet_full: Quaternion, in_structure: PyTree[jax.ShapeDtypeStruct]
    ) -> AbstractLinearOperator:
        if not self.landscape.has_spin2:
            pix = self._quat2index(qdet_full)  # (ndet, nsamp), -1 for out-of-bounds samples
            gather = self._index_operator(pix, in_structure)
            # A -1 index would wrap onto the last pixel, which the sample never observed.
            mask = MaskOperator.from_boolean_mask(pix >= 0, in_structure=gather.out_structure)
            return mask @ gather

        pointing = self._quat2pointing(qdet_full)
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
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    @property
    def _interpolation(self) -> Interpolation:
        return Interpolation.BILINEAR if self.interpolate else Interpolation.NEAREST

    def _quat2index(self, qdet_full: Quaternion) -> Array:
        """Convert full detector quaternions to flat pixel indices.

        Override in subclasses to change the pointing-to-index mapping.
        """
        return self.landscape.quat2index(qdet_full)

    def _quat2pointing(self, qdet_full: Quaternion) -> SampledPointing:
        """Convert quaternions to the [`SampledPointing`][] of every sample.

        This method *must be* overriden in any subclass that changes [`_quat2index`][] (pointing
        to index mapping).
        """
        # rewriting `_quat2index` but not `_quat2pointing` is very likely a bug
        self._check_index_hook_not_overridden()
        world = self.landscape.quat2world(qdet_full)
        stencil = (
            self.landscape.world2stencil(*world, self._interpolation)
            if self.interpolate
            # Nearest case: index through `_quat2index` to stay consistent with hitmap etc.
            else self.landscape.index2stencil(self._quat2index(qdet_full))
        )
        return SampledPointing(stencil, *world)

    def _pointing(self, qdet_full: Quaternion, offsets: Quaternion | None) -> SampledPointing:
        """The [`SampledPointing`][] of every sample, integrated over the offsets if any.

        Each offset reads the map around its own direction through [`_quat2pointing`][], and the
        parts are merged into one stencil per sample. The transport target stays the un-offset
        direction, so every pixel read, whichever offset reads it, is carried into the same frame
        before the sum.

        Args:
            qdet_full: The pointing of every sample, shape (det, samp).
            offsets: The offsets of those detectors, shape (det, n_offsets), or `None`.
        """
        pointing = self._quat2pointing(qdet_full)
        if offsets is None:
            return pointing
        assert self.offset_weights is not None
        n_offsets = offsets.shape[-1]
        parts = [self._quat2pointing(qdet_full * offsets[:, k, None]) for k in range(n_offsets)]
        stencil = Stencil.concatenate([part.stencil for part in parts], self.offset_weights)
        return SampledPointing(stencil, pointing.theta, pointing.phi)

    def _check_index_hook_not_overridden(self) -> None:
        if type(self)._quat2index is not PointingOperator._quat2index:
            msg = (
                f'{type(self).__name__} overrides _quat2index, so it must also override '
                f'_quat2pointing to sample a polarized map'
            )
            raise NotImplementedError(msg)

    def _modulate(self, tod: _StokesT, qdet_full: Quaternion) -> _StokesT:
        """Hook applied to the sampled TOD (identity in the base class).

        Subclasses override this to inject a per-sample diagonal weighting. Because the
        weighting is a symmetric diagonal, the same hook is applied in mv (after sampling)
        and in the transpose (before binning), keeping the adjoint exact.
        """
        return tod

    def _sample(
        self, x_flat: _StokesT, qdet_full: Quaternion, offsets: Quaternion | None = None
    ) -> _StokesT:
        """Sample the flat map at positions given by qdet_full."""
        if self.landscape.has_spin2:
            return transported_gather(x_flat, *self._pointing(qdet_full, offsets))

        if not self.interpolate:
            # fast path for nearest-neighbour
            pix = self._quat2index(qdet_full)  # (ndet, nsamp), -1 for out-of-bounds samples
            sampled = x_flat[pix]
            # the gather wraps a -1 onto the last pixel, which the sample never observed
            return type(x_flat).from_array(jnp.where(pix >= 0, sampled.data, 0))

        stencil = self._pointing(qdet_full, offsets).stencil
        # leading Stokes axis: index the (trailing) pixel axis and sum over the neighbour axis (-1);
        # the weights broadcast over the leading Stokes axis for free.
        sampled = jnp.sum(x_flat.data[:, stencil.indices] * stencil.weights, axis=-1)
        return type(x_flat).from_array(sampled)

    def _bin(
        self, tod_batch: _StokesT, qdet_full: Quaternion, offsets: Quaternion | None = None
    ) -> _StokesT:
        """Scatter-add a batch of TOD into a sky map."""
        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        arr = tod_batch.data  # (n_stokes, *det_sample)
        n_stokes = arr.shape[0]
        zeros = jnp.zeros((n_stokes, n_pixels), self.landscape.dtype)

        if self.landscape.has_spin2:
            flat_sky = type(tod_batch).from_array(zeros)
            binned_sky = transported_scatter(
                flat_sky, tod_batch, *self._pointing(qdet_full, offsets)
            )
            return type(tod_batch).from_array(binned_sky.data.reshape(n_stokes, *sky_shape))

        if not self.interpolate:
            # fast path for nearest-neighbour
            pix = self._quat2index(qdet_full)  # (ndet, nsamp), -1 for out-of-bounds samples
            # the scatter wraps a -1 onto the last pixel, so such a sample must add nothing
            contrib = jnp.where(pix >= 0, arr, 0)
            binned = zeros.at[:, pix.ravel()].add(contrib.reshape(n_stokes, -1))
            return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

        stencil = self._pointing(qdet_full, offsets).stencil
        # (n_stokes, *det_sample, n_nb): spread each sample over its neighbours (weights broadcast
        # over the leading Stokes axis for free).
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

        def mv_inner(xbatch: _StokesT, qdet: Quaternion, offsets: Quaternion | None) -> _StokesT:
            # Expand detector quaternions from boresight and offsets
            qdet_full = self.operator.qbore * qdet[:, None]
            xbatch = self.operator._modulate(xbatch, qdet_full)

            if isinstance(xbatch, StokesI):
                # no rotation needed
                return self.operator._bin(xbatch, qdet_full, offsets)

            # Rotate back to the celestial frame with the inverse rotation
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            rotated: _StokesT = rotate_qu_cs(xbatch, cos_angles, -sin_angles)
            return self.operator._bin(rotated, qdet_full, offsets)

        # Loop over batches of detectors
        ndet, _ = self.in_structure.shape
        batch_size, n_batches = _batch_plan(self.operator.batch_size, ndet)

        def body(i: Int[Array, ''], sky: _StokesT) -> _StokesT:
            # Past ndet, indices are out of range; `sky` is never indexed by `idet` so we need to use
            # the `unique` indices to mask out redundant/repeated contributions from the last batch
            idet = jnp.arange(batch_size) + i * batch_size
            unique = idet < ndet

            # process batch
            sky_batch = mv_inner(
                unique[:, None] * x[idet],
                self.operator.qdet[idet],
                self.operator._offsets_batch(idet),
            )

            # combine the results of the batches into one sky map
            return sky + sky_batch

        sky_out: _StokesT = self.operator.landscape.zeros()
        sky_out = lax.fori_loop(0, n_batches, body, sky_out)
        return sky_out


def _batch_plan(batch_size: int, n: int) -> tuple[int, int]:
    """Resolve `(batch_size, n_batches)` for looping over `n` items in batches."""
    batch_size = min(batch_size, n) if batch_size > 0 else n
    n_batches = (n + batch_size - 1) // batch_size
    return batch_size, n_batches


class XSamplingOperator(AbstractLinearOperator):
    r"""Precomputed bilinear sky-sampling operator from cached world angles.

    The "expanded pointing" sampler. It stores the per-sample world angles `(theta, phi)`
    (computed once from the quaternion pointing, so the expensive quaternion-to-angle
    transcendentals are hoisted out of repeated applies) and on every apply gathers a raveled
    sky map at the four pixels around each of them.

    Works for any landscape supplying a bilinear [`Stencil`][] (HEALPix and WCS/CAR).

    Attributes:
        landscape: The sky pixelization supplying the bilinear stencil.
        theta: Cached spherical co-latitude angles, shape ``(ndet, nsamp)``.
        phi: Cached spherical longitude angles, shape ``(ndet, nsamp)``.
    """

    landscape: StokesLandscape
    theta: Float[Array, 'det samp']
    phi: Float[Array, 'det samp']
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    @classmethod
    def create(cls, landscape: StokesLandscape, quaternions: Quaternion) -> Self:
        theta, phi = landscape.quat2world(quaternions)
        # The map reaching this operator is raveled by PointingOperator.as_expanded_operator.
        return cls(
            landscape,
            theta=theta,
            phi=phi,
            in_structure=landscape.raveled_structure,
            _out_structure=landscape.structure_for(theta.shape),
        )

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def mv(self, x: _StokesT) -> _StokesT:
        # `x` is a raveled sky map: its single backing array is (n_stokes, n_pixels). Index the pixel
        # (last) axis with the cached pointing to produce the (n_stokes, ndet, nsamp) TOD.
        stencil = self.landscape.world2stencil(self.theta, self.phi, Interpolation.BILINEAR)
        if self.landscape.has_spin2:
            return transported_gather(x, stencil, self.theta, self.phi)
        return type(x).from_array(jnp.sum(x.data[..., stencil.indices] * stencil.weights, axis=-1))
