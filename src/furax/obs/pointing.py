import copy
from dataclasses import field
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jaxtyping import Array, Float, Int, PyTree

from furax import AbstractLinearOperator
from furax.core import IndexOperator, RavelOperator, TransposeOperator
from furax.math.quaternion import (
    euler,
    qmul,
    to_gamma_angles,
    to_polarization_angle,
    to_polarization_angle_cos_sin,
)
from furax.obs.landscapes import StokesLandscape, WCSLandscape
from furax.obs.operators._qu_rotations import QURotationOperator, rotate_qu_cs
from furax.obs.stokes import Stokes, StokesI

__all__ = [
    'PointingOperator',
    'XSamplingOperator',
]

_StokesT = TypeVar('_StokesT', bound=Stokes)


class PointingOperator(AbstractLinearOperator):
    """Operator that projects sky maps to time-ordered data (TOD) using quaternion pointing.

    Equivalent to: QURotation @ Index @ Ravel, but computed on-the-fly to save memory.
    For each detector and time sample, it:
    1. Computes the sky pixel from boresight and detector quaternions
    2. Samples the sky map at that pixel
    3. Rotates Stokes QU by the polarization angle

    The transpose accumulates TOD into a sky map (binning).

    Attributes:
        landscape: The sky pixelization (HEALPix landscape).
        qbore: Boresight quaternions, shape (n_samples, 4).
        qdet: Detector quaternions, shape (n_detectors, 4).
        batch_size: Number of detectors processed per batch (memory/speed tradeoff).
    """

    landscape: StokesLandscape
    qbore: Float[Array, 'samp 4']
    qdet: Float[Array, 'det 4']
    batch_size: int = field(metadata={'static': True})
    interpolate: bool = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        boresight_quaternions: Float[Array, 'samp 4'],
        detector_quaternions: Float[Array, 'det 4'],
        *,
        batch_size: int = 32,
        frame: Literal['boresight', 'detector'] = 'boresight',
        interpolate: bool = False,
    ) -> 'PointingOperator':
        # Explicitly determine the output structure
        ndet = detector_quaternions.shape[0]
        nsamp = boresight_quaternions.shape[0]
        out_structure = Stokes.class_for(landscape.stokes).structure_for(
            (ndet, nsamp), dtype=landscape.dtype
        )

        # In boresight frame, strip the z-rotation (gamma) from each detector quaternion.
        # This absorbs the frame correction into qdet so that _get_cos_sin_angles always
        # works the same way, regardless of frame. Pixel indices are unaffected because
        # a z-rotation does not change the direction of the boresight (z) axis.
        #
        # NB: the xieta parametrization is incomplete and cannot describe all rotations.
        # Thus converting to xieta and back (with gamma=0) may not work in full generality.
        # This approach is more general and just as efficient.
        if frame == 'boresight':
            gamma = to_gamma_angles(detector_quaternions)
            q_z_neg = euler(2, -gamma)  # z-rotation by -gamma
            detector_quaternions = qmul(detector_quaternions, q_z_neg)

        return cls(
            landscape,
            qbore=boresight_quaternions,
            qdet=detector_quaternions,
            batch_size=batch_size,
            interpolate=interpolate,
            in_structure=landscape.structure,
            _out_structure=out_structure,
        )

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'un-pointing' operation, i.e. map->tod."""
        x_flat = x.ravel()

        def mv_inner(qdet: Float[Array, 'det 4']) -> _StokesT:
            # Expand detector quaternions from boresight and offsets
            # (samples, 4) x (det, 1, 4) -> (det, samples, 4)
            qdet_full = qmul(self.qbore, qdet[:, None, :])

            tod = self._sample(x_flat, qdet_full)
            tod = self._modulate(tod, qdet_full)

            if isinstance(tod, StokesI):
                # no rotation needed
                return tod

            # Return the rotated Stokes parameters
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            return rotate_qu_cs(tod, cos_angles, sin_angles)  # type: ignore[no-any-return]

        # Loop over batches of detectors.
        # NB: lax.map was tried here (PR #172) instead of the fori_loop+scatter form
        # It seemed faster on GPU, but there was a 3-4x perf regression on CPU
        ndet, nsamp = self.out_structure.shape
        batch_size, n_batches = _batch_plan(self.batch_size, ndet)

        def body(i: Int[Array, ''], tod: _StokesT) -> _StokesT:
            # interval bounds must be static, so we shift the values afterwards
            # jax indexing semantics automatically clip out-of-bounds indices
            idet = jnp.arange(batch_size) + i * batch_size
            tod_batch = mv_inner(self.qdet[idet])
            return type(tod).from_array(tod.data.at[:, idet].set(tod_batch.data))

        # Start from an empty timestream: every slot gets overwritten by body.
        tod_out: _StokesT = type(x).empty((ndet, nsamp), dtype=x.dtype)
        tod_out = lax.fori_loop(0, n_batches, body, tod_out)
        return tod_out

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

        Nearest-neighbour uses a precomputed [`IndexOperator`][]. Bilinear interpolation uses an
        [`XSamplingOperator`][] that caches the float pixel coordinates and recovers the four
        interpolation weights cheaply on each apply (requires a WCS/CAR landscape).
        """
        qdet_full = qmul(self.qbore, self.qdet[:, None, :])
        # Ravel the spatial axes only; the Stokes container's backing array carries a leading
        # Stokes axis (axis 0) that must survive, so ravel axes 1..-1 and index the pixel axis last.
        ravel_op = RavelOperator(1, -1, in_structure=self.landscape.structure)
        sampler: AbstractLinearOperator
        if self.interpolate:
            sampler = XSamplingOperator.create(self.landscape, qdet_full, interpolate=True)
        else:
            # Index the (leading) Stokes axis and the (trailing) pixel axis with broadcast arrays,
            # rather than the ergonomic `(..., pix)`. An Ellipsis (or slice) index element is a
            # non-array pytree leaf and is not a valid JAX type, so it would break the operator as a
            # multi-observation scan leaf; an all-array index tuple keeps it scan-safe.
            pix = self._quat2index(qdet_full)  # (ndet, nsamp), -1 for out-of-bounds samples
            n_stokes = len(self.landscape.stokes)
            stokes_idx = jnp.arange(n_stokes)[:, None, None]
            sampler = IndexOperator((stokes_idx, pix[None]), in_structure=ravel_op.out_structure)
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=sampler.out_structure)
        return qu_rot_op @ sampler @ ravel_op

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def _quat2index(self, qdet_full: Float[Array, '*dims 4']) -> Array:
        """Convert full detector quaternions to flat pixel indices.

        Override in subclasses to change the pointing-to-index mapping.
        """
        return self.landscape.quat2index(qdet_full)

    def _quat2interp(self, qdet_full: Float[Array, '*dims 4']) -> tuple[Array, Array]:
        """Convert full detector quaternions to (indices, weights) for interpolation.

        Override in subclasses to change the pointing-to-index mapping.
        """
        return self.landscape.quat2interp(qdet_full)

    def _modulate(self, tod: _StokesT, qdet_full: Float[Array, '*dims 4']) -> _StokesT:
        """Hook applied to the sampled TOD (identity in the base class).

        Subclasses override this to inject a per-sample diagonal weighting. Because the
        weighting is a symmetric diagonal, the same hook is applied in mv (after sampling)
        and in the transpose (before binning), keeping the adjoint exact.
        """
        return tod

    def _sample(self, x_flat: _StokesT, qdet_full: Float[Array, '*dims 4']) -> _StokesT:
        """Sample the flat map at positions given by qdet_full."""
        if not self.interpolate:
            return x_flat[self._quat2index(qdet_full)]

        indices, weights = self._quat2interp(qdet_full)
        # Zero out contributions from out-of-bounds pixels (index == -1)
        # pixel index 0 is guaranteed to exist, and weight is zeroed simultaneously
        valid = indices >= 0
        indices = jnp.where(valid, indices, 0)
        weights = jnp.where(valid, weights, 0.0)
        weight_sum = weights.sum(axis=-1, keepdims=True)
        unit_weights = weights / jnp.where(weight_sum > 0, weight_sum, 1.0)
        # leading Stokes axis: index the (trailing) pixel axis and sum over the neighbour axis (-1);
        # the weights broadcast over the leading Stokes axis for free.
        sampled = jnp.sum(x_flat.data[:, indices] * unit_weights, axis=-1)
        return type(x_flat).from_array(sampled)

    def _bin(self, tod_batch: _StokesT, qdet_full: Float[Array, '*dims 4']) -> _StokesT:
        """Scatter-add a batch of TOD into a sky map."""
        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        arr = tod_batch.data  # (n_stokes, *det_sample)
        n_stokes = arr.shape[0]
        zeros = jnp.zeros((n_stokes, n_pixels), self.landscape.dtype)

        if not self.interpolate:
            flat_pixels = self._quat2index(qdet_full).ravel()
            binned = zeros.at[:, flat_pixels].add(arr.reshape(n_stokes, -1))
            return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

        indices, weights = self._quat2interp(qdet_full)
        valid = indices >= 0
        safe_indices = jnp.where(valid, indices, 0)
        valid_weights = jnp.where(valid, weights, jnp.zeros_like(weights))
        weight_sum = valid_weights.sum(axis=-1, keepdims=True)
        valid_weights = valid_weights / jnp.where(weight_sum > 0, weight_sum, 1.0)
        flat_indices = safe_indices.ravel()
        # (n_stokes, *det_sample, n_nb): spread each sample over its neighbours (weights broadcast
        # over the leading Stokes axis for free).
        contrib = arr[..., None] * valid_weights
        binned = zeros.at[:, flat_indices].add(contrib.reshape(n_stokes, -1))
        return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(operator=self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: _StokesT) -> _StokesT:
        """Performs the 'pointing' operation, i.e. tod->map."""

        def mv_inner(xbatch: _StokesT, qdet: Float[Array, 'det 4']) -> _StokesT:
            # Expand detector quaternions from boresight and offsets
            qdet_full = qmul(self.operator.qbore, qdet[:, None, :])
            xbatch = self.operator._modulate(xbatch, qdet_full)

            if isinstance(xbatch, StokesI):
                # no rotation needed
                return self.operator._bin(xbatch, qdet_full)

            # Rotate back to the celestial frame with the inverse rotation
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            rotated: _StokesT = rotate_qu_cs(xbatch, cos_angles, -sin_angles)
            return self.operator._bin(rotated, qdet_full)

        # Loop over batches of detectors
        ndet, _ = self.in_structure.shape
        batch_size, n_batches = _batch_plan(self.operator.batch_size, ndet)

        def body(i: Int[Array, ''], sky: _StokesT) -> _StokesT:
            # Past ndet, indices are out of range; `sky` is never indexed by `idet` so we need to use
            # the `unique` indices to mask out redundant/repeated contributions from the last batch
            idet = jnp.arange(batch_size) + i * batch_size
            unique = idet < ndet

            # process batch
            sky_batch = mv_inner(unique[:, None] * x[idet], self.operator.qdet[idet])

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
    r"""Precomputed pixel-sampling operator from cached float pixel coordinates.

    The "expanded pointing" sampler. It stores the per-sample projected pixel coordinates
    `(pix_x, pix_y)` (computed once from the quaternion pointing) and on every apply gathers a
    raveled sky map at those coordinates, nearest-neighbour or bilinear.

    Requires a landscape exposing `pixel2index` / `pixel2interp` (WCS/CAR).
    HEALPix has no 2-D pixel coordinates and is not supported.

    Attributes:
        landscape: The WCS/CAR sky pixelization providing `pixel2index` / `pixel2interp`.
        pix_x: Cached pixel x-coordinates, shape ``(ndet, nsamp)``.
        pix_y: Cached pixel y-coordinates, shape ``(ndet, nsamp)``.
        interpolate: If True, bilinear interpolation over the four nearest pixels; else nearest.
    """

    landscape: StokesLandscape
    pix_x: Float[Array, 'det samp']
    pix_y: Float[Array, 'det samp']
    interpolate: bool = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        quaternions: Float[Array, 'det samp 4'],
        *,
        interpolate: bool = False,
    ) -> 'XSamplingOperator':
        if interpolate and not isinstance(landscape, WCSLandscape):
            raise NotImplementedError(
                f'{type(landscape).__name__} does not support cached bilinear interpolation; '
                'a WCSLandscape is required.'
            )
        pix_x, pix_y = landscape.quat2pixel(quaternions)
        # The map is raveled along its spatial axes (see PointingOperator.as_expanded_operator),
        # leaving a single pixel axis that this operator indexes.
        ravel_op = RavelOperator(1, -1, in_structure=landscape.structure)
        out_structure = Stokes.class_for(landscape.stokes).structure_for(
            pix_x.shape, dtype=landscape.dtype
        )
        return cls(
            landscape,
            pix_x=pix_x,
            pix_y=pix_y,
            interpolate=interpolate,
            in_structure=ravel_op.out_structure,
            _out_structure=out_structure,
        )

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def mv(self, x: _StokesT) -> _StokesT:
        # `x` is a raveled sky map: its single backing array is (n_stokes, n_pixels). Index the pixel
        # (last) axis with the cached per-sample coordinates to produce the (n_stokes, ndet, nsamp) TOD.
        if not self.interpolate:
            indices = self.landscape.pixel2index(self.pix_x, self.pix_y)
            return type(x).from_array(x.data[..., indices])

        indices, weights = self.landscape.pixel2interp(self.pix_x, self.pix_y)
        # Zero contributions from out-of-bounds pixels (index == -1 -> pixel 0, weight 0) and
        # renormalise so partially-covered samples stay unbiased -- matches PointingOperator._sample.
        valid = indices >= 0
        indices = jnp.where(valid, indices, 0)
        weights = jnp.where(valid, weights, 0.0)
        weight_sum = weights.sum(axis=-1, keepdims=True)
        unit_weights = weights / jnp.where(weight_sum > 0, weight_sum, 1.0)
        return type(x).from_array(jnp.sum(x.data[..., indices] * unit_weights, axis=-1))
