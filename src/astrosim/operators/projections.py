import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer, PyTree

from astrosim.operators import AbstractLinearOperator, diagonal
from astrosim.operators.qu_rotations import QURotationOperator

from ..detectors import DetectorArray
from ..landscapes import HealpixLandscape, StokesPyTree, stokes_pytree_cls
from ..samplings import Sampling


class SamplingOperator(AbstractLinearOperator):  # type: ignore[misc]
    landscape: HealpixLandscape
    indices: Integer[Array, '...']

    def __init__(self, landscape: HealpixLandscape, indices: Array):
        self.landscape = landscape
        self.indices = indices  # (ndet, nsampling)

    def mv(self, sky: StokesPyTree) -> StokesPyTree:
        return sky[self.indices]

    def transpose(self) -> AbstractLinearOperator:
        return SamplingOperatorT(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.landscape.stokes)
        return cls.shape_pytree((self.landscape.npixel,), self.landscape.dtype)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.landscape.stokes)
        return cls.shape_pytree(self.indices.shape, self.landscape.dtype)


class SamplingOperatorT(AbstractLinearOperator):  # type: ignore[misc]
    operator: SamplingOperator

    def __init__(self, operator: SamplingOperator):
        self.operator = operator

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        flat_pixels = self.operator.indices.ravel()
        arrays_out = []
        zeros = jnp.zeros(self.operator.landscape.npixel, self.operator.landscape.dtype)
        for stoke in self.operator.landscape.stokes:
            arrays_out.append(zeros.at[flat_pixels].add(getattr(x, stoke).ravel()))
        return stokes_pytree_cls(self.operator.landscape.stokes)(*arrays_out)

    def transpose(self) -> AbstractLinearOperator:
        return self.operator

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


@diagonal
class PytreeDiagonalOperator(AbstractLinearOperator):  # type: ignore[misc]
    diagonal_values: PyTree[Float[Array, '...']]

    def __init__(self, diagonal: PyTree[Float[Array, '...']]):
        self.diagonal_values = diagonal

    def mv(self, sky: PyTree[Float[Array, '...']]) -> PyTree[Float[Array, '...']]:
        return jax.tree_map((lambda a, b: a * b), sky, self.diagonal_values)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(lambda: self.diagonal_values)


def create_projection_operator(
    landscape: HealpixLandscape, samplings: Sampling, detector_dirs: DetectorArray
) -> SamplingOperator:
    rot = get_rotation_matrix(samplings)

    # i, j: rotation (3x3 xyz)
    # k: number of samplings
    # l: number of detectors
    # m: number of directions per detector

    # (3, ndet, ndir, nsampling)
    rotated_coords = jnp.einsum('ijk, jlm -> ilmk', rot, detector_dirs.coords)
    theta, phi = vec2dir(*rotated_coords)

    # (ndet, ndir, nsampling)
    pixels = landscape.ang2pix(theta, phi)
    if pixels.shape[1] == 1:
        # remove the number of directions per pixels if there is only one.
        pixels = pixels.reshape(pixels.shape[0], pixels.shape[2])

    rotation = QURotationOperator.create(pixels.shape, landscape.stokes, samplings.pa)
    sampling = SamplingOperator(landscape, pixels)
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
