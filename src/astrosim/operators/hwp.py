import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Float, PyTree

from astrosim.landscapes import (
    StokesIPyTree,
    StokesIQUPyTree,
    StokesIQUVPyTree,
    StokesPyTree,
    StokesQUPyTree,
    ValidStokesType,
    stokes_pytree_cls,
)
from astrosim.operators import AbstractLinearOperator, symmetric


@symmetric
class HWPOperator(AbstractLinearOperator):
    """Operator for an ideal Half-wave plate."""

    shape: tuple[int, ...]
    dtype: DTypeLike = eqx.field(static=True)
    stokes: str = eqx.field(static=True)
    cos_4angles: Float[Array, '...']
    sin_4angles: Float[Array, '...']

    def __init__(
        self,
        shape: tuple[int, ...],
        stokes: ValidStokesType,
        cos_angles: Float[Array, '...'],
        sin_angles: Float[Array, '...'],
        dtype: DTypeLike = float,
    ):
        self.shape = shape
        self.stokes = stokes
        self.dtype = np.dtype(dtype)
        self.cos_4angles = jnp.asarray(cos_angles, dtype=dtype)
        self.sin_4angles = jnp.asarray(sin_angles, dtype=dtype)

    @classmethod
    def create(cls, shape: tuple[int, ...], stokes: ValidStokesType, angles: Float[Array, '...']):
        cos_4angles = jnp.cos(4 * angles)
        sin_4angles = jnp.sin(4 * angles)
        return cls(shape, stokes, cos_4angles, sin_4angles)

    def mv(self, x: StokesPyTree) -> StokesPyTree:
        if self.stokes != x.stokes:
            raise TypeError('Invalid input')

        # we should try using cls(x).from_iquv
        if isinstance(x, StokesIPyTree):
            return x
        Q = x.Q * self.cos_4angles - x.U * self.sin_4angles
        U = -(x.Q * self.sin_4angles + x.U * self.cos_4angles)

        if isinstance(x, StokesQUPyTree):
            return StokesQUPyTree(Q, U)
        if isinstance(x, StokesIQUPyTree):
            return StokesIQUPyTree(x.I, Q, U)
        if isinstance(x, StokesIQUVPyTree):
            V = -x.V
            return StokesIQUVPyTree(x.I, Q, U, V)
        raise NotImplementedError

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        cls = stokes_pytree_cls(self.stokes)
        return cls.shape_pytree(self.shape, self.dtype)
