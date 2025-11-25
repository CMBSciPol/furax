import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from jaxtyping import Float, Inexact, PyTree

from furax import AbstractLinearOperator, diagonal
from furax.obs.operators._qu_rotations import QURotationOperator
from furax.obs.stokes import (
    Stokes,
    StokesIQU,
    StokesIQUV,
    StokesPyTreeType,
    StokesQU,
    ValidStokesType,
)

mm = 1e-3
c = 299792458.0  # m/s
GHz = 1e9
deg = jnp.pi / 180.0
angleIncidence = 5.0

# Material definitions (unchanged)
thicknesses = [
    0.394 * mm, 0.04 * mm, 0.212 * mm, 3.75 * mm, 3.75 * mm, 3.75 * mm,
    0.212 * mm, 0.04 * mm, 0.394 * mm
]
thicknesses_HF = [
    0.183 * mm, 0.04 * mm, 0.097 * mm, 1.60 * mm, 1.60 * mm, 1.60 * mm,
    0.097 * mm, 0.04 * mm, 0.183 * mm
]
angles_MF = [0.0, 0.0, 0.0, 0.0, 54.0 * deg, 0.0, 0.0, 0.0, 0.0]
angles_HF = [0.0, 0.0, 0.0, 0.0, 57.0 * deg, 0.0, 0.0, 0.0, 0.0]


def get_delta(nu, theta, n, nO=3.05):
    """Compute phase difference with better precision"""
    return 2 * jnp.pi * nu * (nO - n) * theta / c


def compute_effective_index(angleIncidence, chi, nE=3.38, nO=3.05):
    """Compute effective refractive index"""
    sin_inc_sq = jnp.sin(angleIncidence) ** 2
    cos_chi_sq = jnp.cos(chi) ** 2
    return nE * jnp.sqrt(
        1 + (nE ** -2 - nO ** -2) * sin_inc_sq * cos_chi_sq
    )


def HWP_Stack(nu, theta, angleIncidence, chi, nE=3.38, nO=3.05):
    n = compute_effective_index(angleIncidence, chi, nE, nO)
    d = get_delta(nu, theta, n, nO)
    cos_d = jnp.cos(d)
    sin_d = jnp.sin(d)

    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, cos_d, -sin_d],
        [0.0, 0.0, sin_d, cos_d]
    ])


def rotation_matrix_mueller(angle):
    """4x4 Mueller matrix for rotation"""
    cos_2a = jnp.cos(2 * angle)
    sin_2a = jnp.sin(2 * angle)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos_2a, -sin_2a, 0.0],
        [0.0, sin_2a, cos_2a, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def m11m22(alpha, theta, freq):
    return (m11(alpha, theta, freq) - m22(alpha, theta, freq))


def m12m21(alpha, theta, freq):
    return (m12(alpha, theta, freq) + m21(alpha, theta, freq))


def m12m21b(alpha, theta, freq):
    return (-(m12(alpha, theta, freq) + m21(alpha, theta, freq)))


def m11(alpha, theta, nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    return (
        jnp.cos(2 * alpha) ** 2 + jnp.cos(delta) * jnp.sin(2 * alpha) ** 2
    )


def m22(alpha, theta, nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    return (
        jnp.cos(delta) ** 2 * jnp.sin(2 * alpha) ** 2
        + jnp.cos(delta) ** 3 * jnp.cos(2 * alpha) ** 2
        - 2 * jnp.sin(delta) ** 2 * jnp.cos(delta) * jnp.cos(2 * alpha)
        - jnp.sin(delta) ** 2 * jnp.cos(delta)
    )


def m12(alpha, theta, nu):
    delta = get_delta(nu, theta, n=3.38, nO=3.05)
    return (
        jnp.sin(2 * alpha) * jnp.cos(2 * alpha)
        * (jnp.cos(delta) ** 2 - jnp.cos(delta))
        - jnp.sin(delta) ** 2 * jnp.sin(2 * alpha)
    )


def m21(alpha, theta, freq):
    return m12(alpha, theta, freq)


@diagonal
class HWPStackOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(
        static=True
    )
    frequency: float
    _Mueller_qu: jax.Array
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float,
        angleIncidence: float,
        epsilon: float,
        phi: float,
        thickness: float,
        alpha_2: float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        dtype = np.float64
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)

        HWP1 = HWP_Stack(
            frequency * GHz, thickness, angleIncidence * deg, chi=0.
        )
        HWP2_base = HWP_Stack(
            frequency * GHz, thickness, angleIncidence * deg, chi=alpha_2
        )

        Mueller_full = (
            HWP1
            @ rotation_matrix_mueller(alpha_2).T
            @ HWP2_base
            @ rotation_matrix_mueller(alpha_2)
            @ HWP1
        )

        Mueller_qu = Mueller_full.at[2, :].multiply(-1)
        Mueller_qu = Mueller_qu.at[:, 2].multiply(-1)
        Mueller_qu = Mueller_qu[:-1, :-1]

        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _thickness=thickness,
            _alpha_2=alpha_2,
            _epsilon=epsilon,
            _phi=phi,
            _Mueller_qu=Mueller_qu
        )

        if angles is None:
            return hwp

        rot = QURotationOperator(angles + phi, in_structure)
        return rot.T @ hwp @ rot

    def mv(self, x: StokesPyTreeType) -> Stokes:
        Mueller_qu = self._Mueller_qu

        i_new = (
            Mueller_qu[0, 0] * x.i + Mueller_qu[0, 1] * x.q
            + Mueller_qu[0, 2] * x.u
        )
        q_new = (
            Mueller_qu[1, 0] * x.i + Mueller_qu[1, 1] * x.q
            + Mueller_qu[1, 2] * x.u
        )
        u_new = (
            Mueller_qu[2, 0] * x.i + Mueller_qu[2, 1] * x.q
            + Mueller_qu[2, 2] * x.u
        )

        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            return StokesIQU(i_new, self._epsilon * q_new,
                             self._epsilon * u_new)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(i_new, q_new, u_new, x.v)
        else:
            raise NotImplementedError(
                f"Stokes type {type(x)} not supported"
            )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


@diagonal
class MixedStokesOperator(AbstractLinearOperator):
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(
        static=True
    )
    frequency: float
    _thickness: float
    _alpha_2: float
    _epsilon: float
    _phi: float
    _m11_m22: jax.Array
    _m12_m21: jax.Array

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        frequency: float,
        angleIncidence: float,
        epsilon: float,
        phi: float,
        thickness: float,
        alpha_2: float,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        dtype = np.float64
        in_structure = Stokes.class_for(stokes).structure_for(shape, dtype)

        m11_m22 = m11m22(alpha_2, thickness, frequency * GHz)
        m12_m21 = m12m21(alpha_2, thickness, frequency * GHz)

        hwp = cls(
            _in_structure=in_structure,
            frequency=frequency,
            _thickness=thickness,
            _alpha_2=alpha_2,
            _epsilon=epsilon,
            _phi=phi,
            _m11_m22=m11_m22,
            _m12_m21=m12_m21,
        )

        if angles is None:
            return hwp

        rot = QURotationOperator(angles + phi, in_structure)
        return rot.T @ hwp @ rot

    def mv(self, x: StokesPyTreeType) -> Stokes:
        q_new = 0.5 * (self._m11_m22 * x.q - self._m12_m21 * x.u)
        u_new = 0.5 * (-self._m12_m21 * x.q - self._m11_m22 * x.u)

        if isinstance(x, StokesQU):
            return StokesQU(q_new, u_new)
        elif isinstance(x, StokesIQU):
            return StokesIQU(x.i, self._epsilon * q_new,
                             self._epsilon * u_new)
        elif isinstance(x, StokesIQUV):
            return StokesIQUV(x.i, q_new, u_new, x.v)
        else:
            raise NotImplementedError(
                f"Stokes type {type(x)} not supported"
            )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure


class StokesToListOperator(AbstractLinearOperator):
    """Convert StokesIQU[Array[n,m]] to [StokesIQU[Array[m]], ...].
    Index notation: x[stokes,(freq,pix)] -> x[freq,stokes,(pix)]

    Attributes:
        axis: The axis along which the leaves were originally stacked.
        in_structure: The in_structure of the pytree to be unstacked.
        num_elements: Number of elements along the splitting axis.
    """
    axis: int = equinox.field(static=True)
    num_elements: int = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(
        static=True
    )

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
        num_elements: int | None = None,
    ):
        self.axis = axis
        self._in_structure = in_structure

        if num_elements is None:
            if hasattr(in_structure, 'i') and hasattr(
                in_structure.i, 'shape'
            ):
                num_elements = in_structure.i.shape[axis]
            else:
                raise ValueError(
                    "num_elements must be provided if it cannot be "
                    "inferred from in_structure"
                )

        self.num_elements = num_elements

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: Inexact[Array, '...']) -> PyTree[Inexact[Array, '...']]:
        return [
            StokesIQU(
                i=jnp.take(x.i, f, axis=self.axis),
                q=jnp.take(x.q, f, axis=self.axis),
                u=jnp.take(x.u, f, axis=self.axis),
            )
            for f in range(self.num_elements)
        ]


class ListToStokesOperator(AbstractLinearOperator):
    """Convert [StokesIQU[Array[m]], ...] to StokesIQU[Array[n,m]].
    Index notation: x[freq,stokes,(pix)] -> x[stokes,(freq,pix)]

    Attributes:
        axis: The axis along which the leaves are stacked.
        in_structure: The in_structure of the pytree to be stacked.
    """
    axis: int = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(
        static=True
    )

    def __init__(
        self,
        axis: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.axis = axis
        self._in_structure = in_structure
        assert isinstance(in_structure, list) and isinstance(
            in_structure[0], StokesIQU
        ), (
            'Wrong format: the input should be a list of StokesIQU trees'
        )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> Inexact[Array, '...']:
        i_arrays = [x_i.i for x_i in x]
        q_arrays = [x_i.q for x_i in x]
        u_arrays = [x_i.u for x_i in x]

        return StokesIQU(
            i=jnp.stack(i_arrays, axis=self.axis),
            q=jnp.stack(q_arrays, axis=self.axis),
            u=jnp.stack(u_arrays, axis=self.axis),
        )