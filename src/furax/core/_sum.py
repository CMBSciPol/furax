import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jaxtyping import Inexact, PyTree

from ._base import AbstractLinearOperator


class SumOperator(AbstractLinearOperator):
    """Class to perform summation over pytree leaves.

    The axes along which summation is performed follow the NumPy conventions:

    - `None`: all dimensions are reduced.
    - `int`: a single axis is reduced.
    - `()`: no reduction is applied.
    - `tuple of int`: multiple axes are reduced.

    As with `jax.vmap`, the structure of the specified axes must match the structure of the operator
    input. Specifically, they must form a tree prefix of the input, meaning that their structure
    must align with the leading part of the input's PyTree structure.

    Example:
        To sum along every dimension of all leaves:
        >>> from furax.tree import as_structure
        >>> x = {'a': jnp.array([[0, 0, 0], [1, 1, 1]]),
        ...      'b': jnp.array([[1, 1, 1], [2, 2, 2]])}
        >>> op = SumOperator(axis=None, in_structure=as_structure(x))
        >>> op(x)
        {'a': Array(3, dtype=int32), 'b': Array(9, dtype=int32)}

        To sum along every dimension of only one leaf:
        >>> op = SumOperator(axis={'a': None, 'b': ()}, in_structure=as_structure(x))
        >>> op(x)
        {'a': Array(3, dtype=int32), 'b': Array([[1, 1, 1], [2, 2, 2]], dtype=int32)}

        To sum the leaves along different axes:
        >>> op = SumOperator(axis={'a': 0, 'b': 1}, in_structure=as_structure(x))
        >>> op(x)
        {'a': Array([1, 1, 1], dtype=int32), 'b': Array([3, 6], dtype=int32)}
    """

    axis: PyTree[tuple[int, ...] | None] = equinox.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = equinox.field(static=True)

    def __init__(
        self,
        axis: PyTree[int | tuple[int, ...] | None] = None,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ):
        self.axis = axis
        self._in_structure = in_structure

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._in_structure

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(
            self._sum_leaf,
            self.axis,
            x,
            is_leaf=lambda leaf: leaf is None
            or isinstance(leaf, tuple)
            and all(isinstance(element, int) for element in leaf),
        )

    @staticmethod
    def _sum_leaf(
        axes: tuple[int, ...] | None, leaf: PyTree[Inexact[Array, ' _a']]
    ) -> PyTree[Inexact[Array, ' _b']]:
        if isinstance(axes, tuple) and len(axes) == 0:
            return leaf
        return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=axes), leaf)
