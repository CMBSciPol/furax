"""An operator built from its data each time it is applied."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from jaxtyping import PyTree

from furax.tree import zeros_like

from ._base import AbstractLinearOperator


class DeferredOperator(AbstractLinearOperator):
    """The operator `build(member)`, built each time it is applied.

    `member` holds the data, such as arrays and operators, and `build` turns it into an operator.
    Lifted over an ensemble, `lift(lambda m: DeferredOperator(build, m), ensemble)` reuses the
    arrays of the ensemble with their strides, however much `build` computes from them: the
    engine builds the operator of each member as it evaluates it, instead of `lift` vectorizing
    `build` over copies of the leaves shared by several members.

    `build` is part of the static structure of the operator: `jax.jit` compiles one program per
    `build`, as compared by equality. A module-level function, or a frozen dataclass of static
    parameters, is the same `build` from one construction to the next. A closure created anew, or
    a `functools.partial`, is a different one each time, and an array it closes over is compiled
    in as a constant: arrays belong in `member`.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> d = jnp.array([1.0, 2.0, 3.0], jnp.float32)
        >>> op = DeferredOperator(lambda d: DiagonalOperator(2 * d), d)
        >>> op(jnp.ones(3, jnp.float32))
        Array([2., 4., 6.], dtype=float32)
    """

    build: Callable[[Any], AbstractLinearOperator] = field(metadata={'static': True})
    """From `member` to the operator."""
    member: PyTree[Any]
    """The data of the operator."""

    def __init__(self, build: Callable[[Any], AbstractLinearOperator], member: PyTree[Any]) -> None:
        try:
            hash(build)
        except TypeError:
            raise ValueError('build must be hashable, as static data of the operator') from None
        object.__setattr__(self, 'build', build)
        object.__setattr__(self, 'member', member)
        super().__init__(in_structure=self._eval_shape(lambda op: zeros_like(op.in_structure)))

    def mv(self, x: PyTree[jax.Array]) -> PyTree[jax.Array]:
        return self.build(self.member)(x)

    def transpose(self) -> AbstractLinearOperator:
        if isinstance(self.build, _Transposed):
            return DeferredOperator(self.build.build, self.member)
        return DeferredOperator(_Transposed(self.build), self.member)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._eval_shape(lambda op, x: op(x), self.in_structure)

    def _eval_shape(self, f: Callable[..., Any], *args: Any) -> Any:
        """`f(build(member), *args)`, evaluated abstractly.

        The member may itself be abstract, such as the template of an ensemble, whose leaves are
        `jax.ShapeDtypeStruct`s: its arrays are inputs of the evaluation, not constants.
        """
        leaves, treedef = jax.tree.flatten(self.member)
        is_array = [_is_array_like(leaf) for leaf in leaves]

        def evaluate(arrays: list[Any], *args: Any) -> Any:
            remaining = iter(arrays)
            rebuilt = [next(remaining) if a else leaf for leaf, a in zip(leaves, is_array)]
            return f(self.build(jax.tree.unflatten(treedef, rebuilt)), *args)

        arrays = [leaf for leaf, a in zip(leaves, is_array) if a]
        return jax.eval_shape(evaluate, arrays, *args)


@dataclass(frozen=True)
class _Transposed:
    """The transpose of what `build` builds, equal for equal `build`s."""

    build: Callable[[Any], AbstractLinearOperator]

    def __call__(self, member: PyTree[Any]) -> AbstractLinearOperator:
        return self.build(member).T


def _is_array_like(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array | np.ndarray | jax.ShapeDtypeStruct)
