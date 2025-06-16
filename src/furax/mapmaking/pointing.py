from functools import partial

import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from jaxtyping import Array, Float, PyTree

from furax import AbstractLinearOperator
from furax.core import TransposeOperator
from furax.math.quaternion import qmul, qrot_xaxis, qrot_zaxis
from furax.obs.landscapes import HealpixLandscape, HorizonLandscape
from furax.obs.stokes import StokesI, StokesIQU, StokesIQUV, StokesPyTreeType, StokesQU

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

    For now, only HEALPix landscapes are supported.
    """

    landscape: HealpixLandscape | HorizonLandscape = equinox.field(static=True)
    qbore: Float[Array, 'samp 4']
    qdet: Float[Array, 'det 4']
    det_gamma: Float[Array, ' det']
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _out_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    chunk_size: int = equinox.field(static=True, default=16)

    @jit
    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        """Performs the 'un-pointing' operation, i.e. map->tod."""

        def mv_inner(qdet: Float[Array, 'det 4'], gamma: Float[Array, ' det']) -> StokesPyTreeType:
            # Expand detector quaternions from boresight and offsets
            # (samples, 4) x (det, 1, 4) -> (det, samples, 4)
            qdet_full = qmul(self.qbore, qdet[:, None, :])

            # Get pixel indices and sample the pixels
            indices = self.landscape.quat2index(qdet_full)
            if isinstance(self.landscape, HorizonLandscape):
                indices = self.landscape.combined_indices(indices[0], indices[1])
            tod = x.ravel()[indices]

            if isinstance(tod, StokesI):
                # no rotation needed
                return tod

            # Get the angles to rotate into the telescope frame
            angles = get_local_meridian_angle(qdet_full)
            angles -= gamma[:, None]

            cos_2angles = jnp.cos(2 * angles)
            sin_2angles = jnp.sin(2 * angles)
            q = tod.q * cos_2angles - tod.u * sin_2angles
            u = tod.q * sin_2angles + tod.u * cos_2angles

            # Return the rotated Stokes parameters
            if isinstance(tod, StokesQU):
                return StokesQU(q, u)
            if isinstance(tod, StokesIQU):
                return StokesIQU(tod.i, q, u)
            if isinstance(tod, StokesIQUV):
                return StokesIQUV(tod.i, q, u, tod.v)
            raise NotImplementedError

        # Loop over chunks of detectors
        ndet, nsamp = self.out_structure().shape
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
            tod_chunk = mv_inner(self.qdet[idet], self.det_gamma[idet])

            # update the output pytree
            return jax.tree.map(
                lambda tod_leaf, chunk_leaf: tod_leaf.at[idet, :].set(chunk_leaf), tod, tod_chunk
            )

        # Start from empty timestream
        tod_out: StokesPyTreeType = jax.tree.map(
            lambda leaf: jnp.empty_like(leaf), x.structure_for(shape=(ndet, nsamp), dtype=x.dtype)
        )
        tod_out = lax.fori_loop(0, n_chunks, body, tod_out)
        return tod_out

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def transpose(self) -> AbstractLinearOperator:
        return PointingTransposeOperator(self)


class PointingTransposeOperator(TransposeOperator):
    operator: PointingOperator

    @jit
    def mv(self, x: StokesPyTreeType) -> StokesPyTreeType:
        """Performs the 'pointing' operation, i.e. tod->map."""

        def mv_inner(
            xchunk: StokesPyTreeType, qdet: Float[Array, 'det 4'], gamma: Float[Array, ' det']
        ) -> StokesPyTreeType:
            # Expand detector quaternions from boresight and offsets
            qdet_full = qmul(self.operator.qbore, qdet[:, None, :])

            # Get pixel indices
            indices = self.operator.landscape.quat2index(qdet_full)
            if isinstance(self.operator.landscape, HorizonLandscape):
                indices = self.operator.landscape.combined_indices(indices[0], indices[1])

            if isinstance(xchunk, StokesI):
                # no rotation needed
                return self._point(xchunk, indices)

            # Get the angles to rotate back to the celestial frame
            angles = get_local_meridian_angle(qdet_full)
            angles -= gamma[:, None]

            cos_2angles = jnp.cos(2 * angles)
            sin_2angles = jnp.sin(2 * angles)

            # opposite sign for the transpose
            q = xchunk.q * cos_2angles + xchunk.u * sin_2angles
            u = -xchunk.q * sin_2angles + xchunk.u * cos_2angles

            tod: StokesPyTreeType
            if isinstance(xchunk, StokesQU):
                tod = StokesQU(q, u)
            elif isinstance(xchunk, StokesIQU):
                tod = StokesIQU(xchunk.i, q, u)
            elif isinstance(xchunk, StokesIQUV):
                tod = StokesIQUV(xchunk.i, q, u, xchunk.v)
            else:
                raise NotImplementedError
            return self._point(tod, indices)

        # Loop over chunks of detectors
        ndet, _ = self.in_structure().shape
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
            sky_chunk = mv_inner(
                unique[:, None] * x[idet], self.operator.qdet[idet], self.operator.det_gamma[idet]
            )

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


@jit
@partial(jnp.vectorize, signature='(4)->()')
def get_local_meridian_angle(q: Float[Array, '*dims 4']) -> Float[Array, ' *dims']:
    """
    Compute angle between local meridian and orientation vector from quaternions.

    Assumes that the quaternions encode the rotation between the celestial frame
    and some other frame (e.g. detector or boresight frame). The "orientation vector"
    is the unit vector of the latter frame obtained by rotating the X axis of the
    celestial frame. For a detector this will be the polarization sensitive direction.
    The local meridian vector is obtained by projecting the -Z axis of the celestial
    frame onto the plane orthogonal to the pointing direction.
    The angle is then measured clockwise from the orientation vector.

    partially taken from
    https://github.com/hpc4cmb/toast/blob/toast3/src/toast/ops/stokes_weights/kernels_jax.py#L19
    """
    vd = qrot_zaxis(q)
    vo = qrot_xaxis(q)

    # The vector orthogonal to the line of sight that is parallel
    # to the local meridian.
    dir_ang = jnp.arctan2(vd[1], vd[0])
    dir_r = jnp.sqrt(1.0 - vd[2] * vd[2])
    vm_z = -dir_r
    vm_x = vd[2] * jnp.cos(dir_ang)
    vm_y = vd[2] * jnp.sin(dir_ang)

    # Compute the rotation angle from the meridian vector to the
    # orientation vector.  The direction vector is normal to the plane
    # containing these two vectors, so the rotation angle is:
    #
    # angle = atan2((v_m x v_o) . v_d, v_m . v_o)
    #
    alpha_y = (
        vd[0] * (vm_y * vo[2] - vm_z * vo[1])
        - vd[1] * (vm_x * vo[2] - vm_z * vo[0])
        + vd[2] * (vm_x * vo[1] - vm_y * vo[0])
    )
    alpha_x = vm_x * vo[0] + vm_y * vo[1] + vm_z * vo[2]

    return jnp.arctan2(alpha_y, alpha_x)
