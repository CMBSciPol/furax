import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Inexact, Integer, PyTree

from ..detectors import DetectorArray
from ..landscapes import HealpixLandscape, StokesPyTree, stokes_pytree_cls
from ..samplings import Sampling


class ProjectionOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    landscape: HealpixLandscape
    indices: Integer[Array, '...']

    def __init__(self, landscape: HealpixLandscape, indices: Array):
        self.landscape = landscape
        self.indices = indices  # (ndet, nsampling)

    def mv(self, sky: StokesPyTree) -> StokesPyTree:
        return sky[self.indices]

    def transpose(self) -> lx.AbstractLinearOperator:
        return ProjectionOperatorT(self)

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.landscape.stokes)
        return cls.shape_pytree((self.landscape.npixel,), self.landscape.dtype)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.landscape.stokes)
        return cls.shape_pytree(self.indices.shape, self.landscape.dtype)


class ProjectionOperatorT(lx.AbstractLinearOperator):  # type: ignore[misc]
    operator: ProjectionOperator

    def __init__(self, operator: ProjectionOperator):
        self.operator = operator

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        flat_pixels = self.operator.indices.ravel()
        arrays_out = []
        zeros = jnp.zeros(self.operator.landscape.npixel, self.operator.landscape.dtype)
        for stoke in self.operator.landscape.stokes:
            arrays_out.append(zeros.at[flat_pixels].add(getattr(x, stoke).ravel()))
        return stokes_pytree_cls(self.operator.landscape.stokes)(*arrays_out)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self.operator

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure()

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure()


class PytreeDiagonalOperator(lx.AbstractLinearOperator):  # type: ignore[misc]
    diagonal: PyTree[Float[Array, '...']]

    def __init__(self, diagonal: PyTree[Float[Array, '...']]):
        self.diagonal = diagonal

    def mv(self, sky: PyTree[Float[Array, '...']]) -> PyTree[Float[Array, '...']]:
        return jax.tree_map((lambda a, b: a * b), sky, self.diagonal)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self

    def as_matrix(self) -> Inexact[Array, 'a b']:
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(lambda: self.diagonal)

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.eval_shape(lambda: self.diagonal)


@lx.is_symmetric.register(ProjectionOperator)
@lx.is_symmetric.register(ProjectionOperatorT)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_symmetric.register(PytreeDiagonalOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return True


@lx.is_positive_semidefinite.register(ProjectionOperator)
@lx.is_positive_semidefinite.register(ProjectionOperatorT)
@lx.is_positive_semidefinite.register(PytreeDiagonalOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.is_negative_semidefinite.register(ProjectionOperator)
@lx.is_negative_semidefinite.register(ProjectionOperatorT)
@lx.is_negative_semidefinite.register(PytreeDiagonalOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return False


@lx.linearise.register(ProjectionOperator)
@lx.linearise.register(ProjectionOperatorT)
@lx.linearise.register(PytreeDiagonalOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


@lx.conj.register(ProjectionOperator)
@lx.conj.register(ProjectionOperatorT)
@lx.conj.register(PytreeDiagonalOperator)
def _(operator):  # type: ignore[no-untyped-def]
    return operator


def create_projection_operator(
    landscape: HealpixLandscape, samplings: Sampling, detector_dirs: DetectorArray
) -> ProjectionOperator:
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

    p = ProjectionOperator(landscape, pixels)
    return p


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
