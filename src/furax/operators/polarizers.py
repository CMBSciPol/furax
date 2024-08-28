import equinox
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree

from furax._base.rules import AbstractBinaryRule
from furax.landscapes import (
    DTypeLike,
    StokesIPyTree,
    StokesPyTree,
    StokesPyTreeType,
    StokesQUPyTree,
    ValidStokesType,
)
from furax.operators import AbstractLazyTransposeOperator, AbstractLinearOperator
from furax.operators.hwp import HWPOperator
from furax.operators.qu_rotations import QURotationOperator


class LinearPolarizerOperator(AbstractLinearOperator):
    """Class that integrates the input Stokes parameters assuming a linear polarizer."""

    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    @classmethod
    def create(
        cls,
        shape: tuple[int, ...],
        dtype: DTypeLike = np.float64,
        stokes: ValidStokesType = 'IQU',
        *,
        angles: Float[Array, '...'] | None = None,
    ) -> AbstractLinearOperator:
        in_structure = StokesPyTree.structure_for(shape, dtype, stokes)
        polarizer = cls(in_structure)
        if angles is None:
            return polarizer
        rot = QURotationOperator(angles, in_structure)
        rotated_polarizer: AbstractLinearOperator = polarizer @ rot
        return rotated_polarizer

    def mv(self, x: StokesPyTreeType) -> Float[Array, '...']:
        if isinstance(x, StokesIPyTree):
            return 0.5 * x.i
        if isinstance(x, StokesQUPyTree):
            return 0.5 * x.q
        return 0.5 * (x.i + x.q)

    def transpose(self) -> AbstractLinearOperator:
        return LinearPolarizerTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> jax.ShapeDtypeStruct:
        leaf = jax.tree.leaves(self.in_structure())[0]
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)


class LinearPolarizerTransposeOperator(AbstractLazyTransposeOperator):
    operator: LinearPolarizerOperator

    def mv(self, x: Float[Array, '...']) -> StokesPyTree:
        i = q = 0.5 * x
        u = v = jnp.array(0)
        cls: type[StokesPyTree] = type(self.operator.in_structure())
        return cls.from_iquv(i, q, u, v)


class LinearPolarizerHWPRule(AbstractBinaryRule):
    """Binary rule for R(theta) @ HWP = HWP @ R(-theta)`."""

    left_operator_class = LinearPolarizerOperator
    right_operator_class = HWPOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        return [left]
