from dataclasses import field
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jaxtyping import Array, Float, PyTree

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
from furax.obs.stokes import Stokes, StokesI, StokesPyTreeType

__all__ = [
    'PointingOperator',
]


class PointingOperator(AbstractLinearOperator):
    """Pointing operator that expands pointing on the fly to save memory.

    It is equivalent to the composition of (i) an IndexOperator to sample the sky pixels (ii) a
    QURotationOperator to rotate into the telescope frame.

    The performance/memory tradeoff is controlled by the ``chunk_size`` parameter.
    A chunk size of 0 means no chunking, i.e. the entire operation is done in one go.
    Changing this parameter will trigger recompilation of the operator.
    """

    landscape: StokesLandscape
    qbore: Float[Array, 'samp 4']
    qdet: Float[Array, 'det 4']
    chunk_size: int = field(metadata={'static': True})
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        landscape: StokesLandscape,
        boresight_quaternions: Float[Array, 'samp 4'],
        detector_quaternions: Float[Array, 'det 4'],
        *,
        chunk_size: int = 16,
        frame: Literal['boresight', 'detector'] = 'boresight',
    ) -> 'PointingOperator':
        # Explicitly determine the output structure
        ndet = detector_quaternions.shape[0]
        nsamp = boresight_quaternions.shape[0]

        stokes_cls = Stokes.class_for(landscape.stokes)
        out_structure = stokes_cls.structure_for((ndet, nsamp), dtype=landscape.dtype)

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
            chunk_size=chunk_size,
            in_structure=landscape.structure,
            _out_structure=out_structure,
        )

    @jit
    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        """Performs the 'un-pointing' operation, i.e. map->tod."""

        def mv_inner(qdet: Float[Array, 'det 4']) -> StokesPyTreeType:
            # Expand detector quaternions from boresight and offsets
            # (samples, 4) x (det, 1, 4) -> (det, samples, 4)
            qdet_full = qmul(self.qbore, qdet[:, None, :])

            # Get pixel indices and sample the pixels
            indices = self.landscape.quat2index(qdet_full)
            tod = x.ravel()[indices]

            if isinstance(tod, StokesI):
                # no rotation needed
                return tod

            # Return the rotated Stokes parameters
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)

            # Do all the arithmetic in double precision but cast to the required dtype at the end
            return rotate_qu_cs(tod, cos_angles, sin_angles)  # type: ignore[no-any-return]

        # Loop over chunks of detectors
        ndet, nsamp = self.out_structure.shape
        chunk_size = min(self.chunk_size, ndet)
        if chunk_size > 0:
            n_chunks = (ndet + chunk_size - 1) // chunk_size
        else:
            n_chunks = 1
            chunk_size = ndet

        def body(i, tod):  # type: ignore[no-untyped-def]
            # interval bounds must be static, so we shift the values afterwards
            idet = jnp.arange(chunk_size) + i * chunk_size

            # clip array to avoid out of bounds
            # this means that the last chunk may do redundant work
            # but avoids recompilation
            idet = jnp.clip(idet, max=ndet - 1)

            # process chunk
            tod_chunk = mv_inner(self.qdet[idet])

            # update the output pytree
            return jax.tree.map(
                lambda tod_leaf, chunk_leaf: tod_leaf.at[idet, :].set(chunk_leaf), tod, tod_chunk
            )

        # Start from empty timestream
        tod_out = jax.tree.map(lambda leaf: jnp.empty_like(leaf, leaf.dtype), self.out_structure)
        return lax.fori_loop(0, n_chunks, body, tod_out)  # type: ignore[no-any-return]

    def as_expanded_operator(self) -> AbstractLinearOperator:
        """Return the equivalent QURotationOperator @ IndexOperator @ RavelOperator.

        Equivalent to mv() but as an explicit composition, useful for testing.
        """
        qdet_full = qmul(self.qbore, self.qdet[:, None, :])
        indices = self.landscape.quat2index(qdet_full)
        # this takes care of multi-dimensional landscapes
        ravel_op = RavelOperator(in_structure=self.landscape.structure)
        index_op = IndexOperator(indices, in_structure=ravel_op.out_structure)
        pa = to_polarization_angle(qdet_full)
        qu_rot_op = QURotationOperator(angles=pa, in_structure=index_op.out_structure)
        return qu_rot_op @ index_op @ ravel_op

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(operator=self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        """Performs the 'pointing' operation, i.e. tod->map."""

        def mv_inner(xchunk: StokesPyTreeType, qdet: Float[Array, 'det 4']) -> StokesPyTreeType:
            # Expand detector quaternions from boresight and offsets
            qdet_full = qmul(self.operator.qbore, qdet[:, None, :])

            # Get pixel indices
            indices = self.operator.landscape.quat2index(qdet_full)

            if isinstance(xchunk, StokesI):
                # no rotation needed
                return self._point(xchunk, indices)

            # Rotate back to the celestial frame with the inverse rotation
            cos_angles, sin_angles = to_polarization_angle_cos_sin(qdet_full)
            rotated = rotate_qu_cs(xchunk, cos_angles, -sin_angles)
            return self._point(rotated, indices)

        # Loop over chunks of detectors
        ndet, _ = self.in_structure.shape
        chunk_size = min(self.operator.chunk_size, ndet)
        if chunk_size > 0:
            n_chunks = (ndet + chunk_size - 1) // chunk_size
        else:
            n_chunks = 1
            chunk_size = ndet

        def body(i, sky):  # type: ignore[no-untyped-def]
            idet = jnp.arange(chunk_size) + i * chunk_size

            # clip, but avoid multiple contributions in the last chunk
            unique = idet < ndet
            idet = jnp.clip(idet, max=ndet - 1)

            # process chunk
            sky_chunk = mv_inner(unique[:, None] * x[idet], self.operator.qdet[idet])

            # combine the results of the chunks into one sky map
            return jax.tree.map(
                lambda sky_leaf, chunk_leaf: sky_leaf.at[:].add(chunk_leaf),
                sky,
                sky_chunk,
            )

        sky_out: StokesPyTreeType = self.operator.landscape.zeros()
        sky_out = lax.fori_loop(0, n_chunks, body, sky_out)
        return sky_out

    def _point(self, tod: StokesPyTreeType, pixels: Array) -> StokesPyTreeType:
        flat_pixels = pixels.ravel()
        sky_shape = self.operator.landscape.shape
        zeros = jnp.zeros(np.prod(sky_shape), self.operator.landscape.dtype)
        sky: StokesPyTreeType = jax.tree.map(
            lambda x: zeros.at[flat_pixels].add(x.ravel()).reshape(sky_shape),
            tod,
        )
        return sky
