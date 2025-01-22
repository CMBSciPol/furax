import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from furax import AbstractLinearOperator, IndexOperator, RavelOperator
from furax.obs import QURotationOperator
from furax.obs._detectors import DetectorArray
from furax.obs._samplings import Sampling
from furax.obs.landscapes import HealpixLandscape
from furax.obs.stokes import Stokes


def create_projection_operator(
    landscape: HealpixLandscape, samplings: Sampling, detector_dirs: DetectorArray
) -> AbstractLinearOperator:
    rot = get_rotation_matrix(samplings)

    # i, j: rotation (3x3 xyz)
    # k: number of samplings
    # l: number of detectors
    # m: number of directions per detector

    # (3, ndet, ndir, nsampling)
    rotated_coords = jnp.einsum('ijk, jlm -> ilmk', rot, detector_dirs.coords)
    theta, phi = vec2dir(*rotated_coords)

    # (ndet, ndir, nsampling)
    indices = landscape.world2index(theta, phi)
    if indices.shape[1] == 1:
        # remove the number of directions per pixels if there is only one.
        indices = indices.reshape(indices.shape[0], indices.shape[2])

    tod_structure = Stokes.class_for(landscape.stokes).structure_for(indices.shape, landscape.dtype)

    rotation = QURotationOperator(samplings.pa, tod_structure)
    reshape = RavelOperator(in_structure=landscape.structure)
    sampling = IndexOperator(indices, in_structure=reshape.out_structure())
    projection = rotation @ sampling @ reshape
    return projection


def get_rotation_matrix(samplings: Sampling) -> Float[Array, '...']:
    """Returns the rotation matrices associtated to the samplings.

    See: https://en.wikipedia.org/wiki/Euler_angles Convention Z1-Y2-Z3.
    Rotations along Z1 (alpha=phi), Y2 (beta=theta) and Z3 (gamma=pa).
    """
    alpha, beta, gamma = samplings.phi, samplings.theta, samplings.pa
    s1, c1 = jnp.sin(alpha), jnp.cos(alpha)
    s2, c2 = jnp.sin(beta), jnp.cos(beta)
    s3, c3 = jnp.sin(gamma), jnp.cos(gamma)
    r = jnp.array(
        [
            [-s1 * s3 + c1 * c2 * c3, -s1 * c3 - c1 * c2 * s3, c1 * s2],
            [c1 * s3 + s1 * c2 * c3, c1 * c3 - s1 * c2 * s3, s1 * s2],
            [-s2 * c3, s2 * s3, c2],
        ],
        dtype=jnp.float64,
    )
    return r


@jax.jit
@jax.vmap
def vec2dir(
    x: Float[Array, '*#dims'], y: Float[Array, '*#dims'], z: Float[Array, '*#dims']
) -> tuple[Float[Array, '*#dims'], Float[Array, '*#dims']]:
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)
    return theta, phi
