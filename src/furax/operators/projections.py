import math

import equinox
import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, PyTree

from furax.detectors import DetectorArray
from furax.landscapes import HealpixLandscape, StokesLandscape, StokesPyTree
from furax.operators import AbstractLinearOperator, DiagonalOperator, TransposeOperator
from furax.operators.qu_rotations import QURotationOperator
from furax.samplings import Sampling


class SamplingOperator(AbstractLinearOperator):
    landscape: StokesLandscape = equinox.field(static=True)
    indices: Integer[Array, '...'] = equinox.field(static=True)

    def __init__(self, landscape: StokesLandscape, indices: Array):
        self.landscape = landscape
        self.indices = indices  # (ndet, nsampling)

    def mv(self, sky: StokesPyTree) -> StokesPyTree:
        return sky.ravel()[self.indices]

    def transpose(self) -> AbstractLinearOperator:
        return SamplingTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.class_for(self.landscape.stokes).structure_for(
            self.landscape.shape, self.landscape.dtype
        )

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return StokesPyTree.class_for(self.landscape.stokes).structure_for(
            self.indices.shape, self.landscape.dtype
        )


class SamplingTransposeOperator(TransposeOperator):
    operator: SamplingOperator

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        flat_pixels = self.operator.indices.ravel()
        arrays_out = []
        zeros = jnp.zeros(np.prod(self.operator.landscape.shape), self.operator.landscape.dtype)
        for stoke in self.operator.landscape.stokes:
            arrays_out.append(
                zeros.at[flat_pixels]
                .add(getattr(x, stoke.lower()).ravel())
                .reshape(self.operator.landscape.shape)
            )
        return StokesPyTree.from_stokes(*arrays_out)

    def __matmul__(self, other: AbstractLinearOperator) -> AbstractLinearOperator:
        if other is not self.operator:
            return super().__matmul__(other)

        dtype = jax.tree.leaves(other.in_structure())[0].dtype

        # P.T @ P
        size_max = math.prod(self.operator.landscape.shape)
        unique_indices, counts = jnp.unique(
            self.operator.indices, return_counts=True, size=size_max, fill_value=-1
        )
        coverage = jnp.zeros(np.prod(self.operator.landscape.shape), dtype=dtype)
        coverage = (
            coverage.at[unique_indices]
            .add(counts, indices_are_sorted=True, unique_indices=True)
            .reshape(self.operator.landscape.shape)
        )
        arrays_out = len(self.operator.landscape.stokes) * [coverage]
        return DiagonalOperator(StokesPyTree.from_stokes(*arrays_out))


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

    tod_structure = StokesPyTree.class_for(landscape.stokes).structure_for(
        indices.shape, landscape.dtype
    )

    rotation = QURotationOperator(samplings.pa, tod_structure)
    sampling = SamplingOperator(landscape, indices)
    return rotation @ sampling


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
