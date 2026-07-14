import copy
from dataclasses import field
from typing import Literal

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
from furax.obs.landscapes import StokesLandscape
from furax.obs.operators._qu_rotations import QURotationOperator, rotate_qu_cs
from furax.obs.stokes import Stokes, StokesI, StokesType

__all__ = [
    'PointingOperator',
]


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
        batch_size: Detector batch size (memory/speed tradeoff; see jax.lax.map documentation).
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
    def mv(self, x: StokesType) -> StokesType:
        """Performs the 'un-pointing' operation, i.e. map->tod."""
        x_flat = x.ravel()

        def mv_inner(qdet: Float[Array, ' 4']) -> StokesType:
            # Expand one detector's quaternion from boresight and offset: (samp, 4)
            qdet_full = qmul(self.qbore, qdet)

            tod = self._sample(x_flat, qdet_full)
            tod = self._modulate(tod, qdet_full)

            if isinstance(tod, StokesI):
                # no rotation needed
                return tod

            # Return the rotated Stokes parameters
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            return rotate_qu_cs(tod, cos_angles, sin_angles)  # type: ignore[no-any-return]

        tod_out: StokesType = lax.map(mv_inner, self.qdet, batch_size=self.batch_size)
        # lax.map stacks the new detector axis at position 0
        # so move it back: (det, n_stokes, samp) -> (n_stokes, det, samp).
        return type(tod_out).from_array(jnp.moveaxis(tod_out.data, 0, 1))

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
        """Return the equivalent QURotationOperator @ IndexOperator @ RavelOperator.

        Equivalent to mv() but as an explicit composition, useful for testing.
        """
        if self.interpolate:
            raise NotImplementedError('as_expanded_operator does not support interpolate=True')
        qdet_full = qmul(self.qbore, self.qdet[:, None, :])
        indices = self._quat2index(qdet_full)
        # Ravel the spatial axes only; the Stokes container's backing array carries a leading
        # Stokes axis (axis 0) that must survive, so ravel axes 1..-1 and index the pixel axis last.
        ravel_op = RavelOperator(1, -1, in_structure=self.landscape.structure)
        index_op = IndexOperator((..., indices), in_structure=ravel_op.out_structure)
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=index_op.out_structure)
        return qu_rot_op @ index_op @ ravel_op

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

    def _modulate(self, tod: StokesType, qdet_full: Float[Array, '*dims 4']) -> StokesType:
        """Hook applied to the sampled TOD (identity in the base class).

        Subclasses override this to inject a per-sample diagonal weighting. Because the
        weighting is a symmetric diagonal, the same hook is applied in mv (after sampling)
        and in the transpose (before binning), keeping the adjoint exact.
        """
        return tod

    def _sample(self, x_flat: StokesType, qdet_full: Float[Array, '*dims 4']) -> StokesType:
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

    def _bin(self, tod_chunk: StokesType, qdet_full: Float[Array, '*dims 4']) -> StokesType:
        """Scatter-add a TOD chunk into a sky map."""
        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        arr = tod_chunk.data  # (n_stokes, *det_sample)
        n_stokes = arr.shape[0]
        zeros = jnp.zeros((n_stokes, n_pixels), self.landscape.dtype)

        if not self.interpolate:
            flat_pixels = self._quat2index(qdet_full).ravel()
            binned = zeros.at[:, flat_pixels].add(arr.reshape(n_stokes, -1))
            return type(tod_chunk).from_array(binned.reshape(n_stokes, *sky_shape))

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
        return type(tod_chunk).from_array(binned.reshape(n_stokes, *sky_shape))

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(operator=self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: StokesType) -> StokesType:
        """Performs the 'pointing' operation, i.e. tod->map."""

        def mv_inner(xchunk: StokesType, qdet: Float[Array, 'det 4']) -> StokesType:
            # Expand detector quaternions from boresight and offsets
            qdet_full = qmul(self.operator.qbore, qdet[:, None, :])
            xchunk = self.operator._modulate(xchunk, qdet_full)

            if isinstance(xchunk, StokesI):
                # no rotation needed
                return self.operator._bin(xchunk, qdet_full)

            # Rotate back to the celestial frame with the inverse rotation
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            rotated = rotate_qu_cs(xchunk, cos_angles, -sin_angles)
            return self.operator._bin(rotated, qdet_full)

        # Loop over chunks of detectors
        ndet, _ = self.in_structure.shape
        batch_size = min(self.operator.batch_size, ndet)
        if batch_size > 0:
            n_chunks = (ndet + batch_size - 1) // batch_size
        else:
            n_chunks = 1
            batch_size = ndet

        def body(i: Int[Array, ''], sky: StokesType) -> StokesType:
            idet = jnp.arange(batch_size) + i * batch_size

            # clip, but avoid multiple contributions in the last chunk
            unique = idet < ndet
            idet = jnp.clip(idet, max=ndet - 1)

            # process chunk
            sky_chunk = mv_inner(unique[:, None] * x[idet], self.operator.qdet[idet])

            # combine the results of the chunks into one sky map
            return sky + sky_chunk

        sky_out: StokesType = self.operator.landscape.zeros()
        sky_out = lax.fori_loop(0, n_chunks, body, sky_out)
        return sky_out
