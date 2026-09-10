import copy
from dataclasses import field
from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from fastquat import Quaternion
from jax import jit, lax
from jaxtyping import Array, Float, Int, Integer, PyTree

from furax import AbstractLinearOperator
from furax.core import DiagonalOperator, IndexOperator, RavelOperator, TransposeOperator
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


def _transports_spin2(landscape: StokesLandscape) -> bool:
    """Whether a sampler on this landscape must transport Q and U from the pixels it reads.

    True for any map holding Q and U. Each pixel expresses them in its own meridian basis, which is
    not the basis of the direction being sampled: bilinear sampling would otherwise sum four
    different bases, and nearest neighbour would return the pixel center's basis for a sample that
    sits off center. An intensity-only map has nothing to rotate.
    """
    return 'Q' in landscape.stokes


def _transport_angles(
    stencil: Stencil, theta: Float[Array, ' *dims'], phi: Float[Array, ' *dims']
) -> Float[Array, ' *dims']:
    """The rotation angle carrying a one-neighbour stencil's Q and U to the sampled direction.

    [`QURotationOperator`][] takes an angle, while the transport is naturally a (cos, sin) pair, so
    the pair is turned back into an angle here. The precision of the pair survives the round trip:
    it is the pair that is delicate to compute at sub-pixel separations, not the arc tangent of it.
    """
    assert stencil.positions is not None  # mypy: the caller transports, so it has positions
    cos_2delta, sin_2delta = spin2_cos_sin_zs(
        *stencil.positions, jnp.cos(theta)[..., None], jnp.sin(theta)[..., None], phi[..., None]
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

    Attributes:
        landscape: The sky pixelization (HEALPix landscape).
        qbore: Boresight quaternions, shape (n_samples,).
        qdet: Detector quaternions, shape (n_detectors,).
        batch_size: Number of detectors processed per batch (memory/speed tradeoff).
        interpolate: If True, bilinear interpolation over the four nearest pixels; else nearest.
    """

    landscape: StokesLandscape
    qbore: Quaternion
    qdet: Quaternion
    batch_size: int = field(metadata={'static': True})
    interpolate: bool = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

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
    ) -> 'PointingOperator':
        # Explicitly determine the output structure
        ndet = detector_quaternions.shape[0]
        nsamp = boresight_quaternions.shape[0]
        out_structure = landscape.structure_for((ndet, nsamp))

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
            detector_quaternions = detector_quaternions * q_z_neg

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

        def mv_inner(qdet: Quaternion) -> _StokesT:
            # Expand detector quaternions from boresight and offsets: (samp) x (det, 1) -> (det, samp)
            qdet_full = self.qbore * qdet[:, None]

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

        Nearest-neighbour sampling uses a precomputed [`IndexOperator`][]. On a polarized map it
        carries a second [`QURotationOperator`][] for the transport from the pixel center to the
        sampled direction, which the composition rules fuse with the polarisation rotation, and a
        [`DiagonalOperator`][] holding the stencil weight, which is zero for a sample outside the
        map. Bilinear interpolation uses an [`XSamplingOperator`][] that caches the world angles
        ``(theta, phi)`` and recovers the stencil on each apply (works for HEALPix and WCS/CAR
        landscapes), because its four weights depend on where in the pixel the sample falls.
        """
        qdet_full = self.qbore * self.qdet[:, None]
        # Ravel the spatial axes only; the Stokes container's backing array carries a leading
        # Stokes axis (axis 0) that must survive, so ravel axes 1..-1 and index the pixel axis last.
        ravel_op = RavelOperator(1, -1, in_structure=self.landscape.structure)
        sampler: AbstractLinearOperator
        if self.interpolate:
            sampler = XSamplingOperator.create(self.landscape, qdet_full, interpolate=True)
        elif self._transports:
            stencil, theta, phi = self._quat2stencil(qdet_full)
            sampler = self._index_operator(stencil.indices[..., 0], ravel_op.out_structure)
            # The index alone cannot express a sample outside the map, which the stencil gives a
            # zero weight; the weight rides along as a diagonal so that this equals `_sample`.
            weight_op = DiagonalOperator(
                stencil.weights[..., 0], in_structure=sampler.out_structure
            )
            transport_op = QURotationOperator(
                angles=_transport_angles(stencil, theta, phi), in_structure=sampler.out_structure
            )
            sampler = transport_op @ weight_op @ sampler
        else:
            pix = self._quat2index(qdet_full)  # (ndet, nsamp), -1 for out-of-bounds samples
            sampler = self._index_operator(pix, ravel_op.out_structure)
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=sampler.out_structure)
        return qu_rot_op @ sampler @ ravel_op

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
    def _transports(self) -> bool:
        return _transports_spin2(self.landscape)

    @property
    def _interpolation(self) -> Interpolation:
        return Interpolation.BILINEAR if self.interpolate else Interpolation.NEAREST

    def _quat2index(self, qdet_full: Quaternion) -> Array:
        """Convert full detector quaternions to flat pixel indices.

        Override in subclasses to change the pointing-to-index mapping.
        """
        return self.landscape.quat2index(qdet_full)

    def _quat2stencil(self, qdet_full: Quaternion) -> tuple[Stencil, Array, Array]:
        """Convert quaternions to the sampling stencil and the sampled direction ``(theta, phi)``.

        The single hook for stencil sampling, at whichever interpolation [`interpolate`][]
        selects. Override it in a subclass that changes the pointing-to-index mapping;
        [`_quat2index`][] is its scalar shortcut, for a nearest-neighbour sample of a map with
        nothing to transport.
        """
        # A subclass redefining the nearest pointing in `_quat2index` alone would be sampled at the
        # base class's directions here instead. Refuse rather than return the wrong operator. An
        # intensity-only map never reaches here, so such a subclass still works.
        if not self.interpolate and type(self)._quat2index is not PointingOperator._quat2index:
            raise NotImplementedError(
                f'{type(self).__name__} overrides _quat2index, so it must also override '
                f'_quat2stencil to sample a polarized map'
            )
        theta, phi = self.landscape.quat2world(qdet_full)
        if self.interpolate:
            return self.landscape.world2stencil(theta, phi, Interpolation.BILINEAR), theta, phi
        # Index through `_quat2index`, not through `theta, phi`: HEALPix reads the pointing axis
        # with `vec2pix` and the angles with `ang2pix`, and in float32 the two disagree often enough
        # that a sample would bin into a pixel the hit map never counted, which drops it.
        return self.landscape.index2stencil(self._quat2index(qdet_full)), theta, phi

    def _modulate(self, tod: _StokesT, qdet_full: Quaternion) -> _StokesT:
        """Hook applied to the sampled TOD (identity in the base class).

        Subclasses override this to inject a per-sample diagonal weighting. Because the
        weighting is a symmetric diagonal, the same hook is applied in mv (after sampling)
        and in the transpose (before binning), keeping the adjoint exact.
        """
        return tod

    def _sample(self, x_flat: _StokesT, qdet_full: Quaternion) -> _StokesT:
        """Sample the flat map at positions given by qdet_full."""
        if self._transports:
            stencil, theta, phi = self._quat2stencil(qdet_full)
            return transported_gather(x_flat, stencil, theta, phi)

        if not self.interpolate:
            return x_flat[self._quat2index(qdet_full)]

        stencil, _, _ = self._quat2stencil(qdet_full)
        # leading Stokes axis: index the (trailing) pixel axis and sum over the neighbour axis (-1);
        # the weights broadcast over the leading Stokes axis for free.
        sampled = jnp.sum(x_flat.data[:, stencil.indices] * stencil.weights, axis=-1)
        return type(x_flat).from_array(sampled)

    def _bin(self, tod_batch: _StokesT, qdet_full: Quaternion) -> _StokesT:
        """Scatter-add a batch of TOD into a sky map."""
        sky_shape = self.landscape.shape
        n_pixels = int(np.prod(sky_shape))
        # scatter-add per pixel while keeping the leading Stokes axis of the backing array.
        arr = tod_batch.data  # (n_stokes, *det_sample)
        n_stokes = arr.shape[0]
        zeros = jnp.zeros((n_stokes, n_pixels), self.landscape.dtype)

        if self._transports:
            stencil, theta, phi = self._quat2stencil(qdet_full)
            flat_sky = type(tod_batch).from_array(zeros)
            binned_sky = transported_scatter(flat_sky, tod_batch, stencil, theta, phi)
            return type(tod_batch).from_array(binned_sky.data.reshape(n_stokes, *sky_shape))

        if not self.interpolate:
            flat_pixels = self._quat2index(qdet_full).ravel()
            binned = zeros.at[:, flat_pixels].add(arr.reshape(n_stokes, -1))
            return type(tod_batch).from_array(binned.reshape(n_stokes, *sky_shape))

        stencil, _, _ = self._quat2stencil(qdet_full)
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

        def mv_inner(xbatch: _StokesT, qdet: Quaternion) -> _StokesT:
            # Expand detector quaternions from boresight and offsets
            qdet_full = self.operator.qbore * qdet[:, None]
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
    r"""Precomputed sky-sampling operator from cached world angles.

    The "expanded pointing" sampler. It stores the per-sample world angles `(theta, phi)`
    (computed once from the quaternion pointing, so the expensive quaternion-to-angle
    transcendentals are hoisted out of repeated applies) and on every apply gathers a raveled
    sky map at those angles, nearest-neighbour or bilinear.

    Nearest-neighbour sampling caches the pixel indices too, from `quat2index`, so that it reads
    the pixels a hit map built the same way counts rather than the ones the angles fall in.

    Works for any landscape exposing `world2index` / `world2interp` (HEALPix and WCS/CAR).

    Attributes:
        landscape: The sky pixelization providing `world2index` / `world2interp`.
        theta: Cached spherical co-latitude angles, shape ``(ndet, nsamp)``.
        phi: Cached spherical longitude angles, shape ``(ndet, nsamp)``.
        indices: Cached nearest-pixel indices, shape ``(ndet, nsamp)``, `None` when interpolating.
        interpolate: If True, bilinear interpolation over the four nearest pixels; else nearest.
    """

    landscape: StokesLandscape
    theta: Float[Array, 'det samp']
    phi: Float[Array, 'det samp']
    indices: Integer[Array, 'det samp'] | None
    interpolate: bool = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        quaternions: Quaternion,
        *,
        interpolate: bool = False,
    ) -> 'XSamplingOperator':
        theta, phi = landscape.quat2world(quaternions)
        indices = None if interpolate else landscape.quat2index(quaternions)
        # The map is raveled along its spatial axes (see PointingOperator.as_expanded_operator),
        # leaving a single pixel axis that this operator indexes.
        ravel_op = RavelOperator(1, -1, in_structure=landscape.structure)
        out_structure = landscape.structure_for(theta.shape)
        return cls(
            landscape,
            theta=theta,
            phi=phi,
            indices=indices,
            interpolate=interpolate,
            in_structure=ravel_op.out_structure,
            _out_structure=out_structure,
        )

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    @property
    def _transports(self) -> bool:
        return _transports_spin2(self.landscape)

    def mv(self, x: _StokesT) -> _StokesT:
        # `x` is a raveled sky map: its single backing array is (n_stokes, n_pixels). Index the pixel
        # (last) axis with the cached pointing to produce the (n_stokes, ndet, nsamp) TOD.
        if self._transports:
            return transported_gather(x, self._stencil(), self.theta, self.phi)

        if not self.interpolate:
            return type(x).from_array(x.data[..., self.indices])

        stencil = self._stencil()
        return type(x).from_array(jnp.sum(x.data[..., stencil.indices] * stencil.weights, axis=-1))

    def _stencil(self) -> Stencil:
        """The stencil the cached pointing reads, recovered on every apply."""
        if self.interpolate:
            return self.landscape.world2stencil(self.theta, self.phi, Interpolation.BILINEAR)
        assert self.indices is not None  # mypy assert: `create` caches them when not interpolating
        return self.landscape.index2stencil(self.indices)
