import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Bool, PyTree

from .core import AbstractLazyTransposeOperator, AbstractLinearOperator
from .rules import AbstractBinaryRule, NoReduction


class PackOperator(AbstractLinearOperator):
    """Class for packing the leaves of a PyTree according to a common mask.

    The operation is conceptually the same as:
        y = x[mask]
    """

    mask: Bool[Array, '...'] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)
    _out_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(self, mask: Bool[Array, '...'], in_structure: PyTree[jax.ShapeDtypeStruct]):
        self.mask = mask
        self._in_structure = in_structure
        # out_structure is stored in the constructor, because the mask may be traced later on
        self._out_structure = self._get_out_structure()

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        return x[self.mask]

    def transpose(self) -> AbstractLinearOperator:
        return PackTransposeOperator(self)

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def _get_out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        out_shape = (int(jnp.sum(self.mask)),)

        def pack(leaf: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
            return jax.ShapeDtypeStruct(out_shape, leaf.dtype)

        return jax.tree.map(pack, self.in_structure())


class PackTransposeOperator(AbstractLazyTransposeOperator):
    """Class for unpacking the leaves of a PyTree according to a common mask.

    The operation is conceptually the same as:
        y = jnp.zeros(out_structure)
        y[mask] = x
    """

    operator: PackOperator

    def mv(self, x: PyTree[Array, '...']) -> PyTree[Array]:
        y = jax.tree.map(
            lambda leaf: jnp.zeros(self.operator.mask.shape, leaf.dtype)
            .at[self.operator.mask]
            .set(leaf),
            x,
        )
        return y


class PackUnpackRule(AbstractBinaryRule):
    operator_class = PackOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        if (
            isinstance(left, PackOperator)
            and isinstance(right, PackTransposeOperator)
            and left is right.operator
        ):
            return []
        raise NoReduction
