"""An operator built from its data each time it is applied."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from jaxtyping import PyTree

from ._base import AbstractLinearOperator, OperatorTag
from ._ensembles import _is_abstract, _is_array, _rebuild


class DeferredOperator(AbstractLinearOperator):
    """The operator `build(*members)`, built each time it is applied.

    `members` hold the data, such as arrays and operators, and `build` turns them into an
    operator. Lifted over an ensemble, `lift(lambda m: DeferredOperator(build, m), ensemble)`
    reuses the arrays of the ensemble with their strides, however much `build` computes from them:
    the engine builds the operator of each member as it evaluates it, instead of `lift`
    vectorizing `build` over copies of the leaves shared by several members. Its structures and
    tags are those of `build(*members)`, and its transpose and inverse build those of
    `build(*members)`.

    `build` is part of the static structure of the operator: `jax.jit` compiles one program per
    `build`, as compared by equality. A module-level function, or a frozen dataclass of static
    parameters, is the same `build` from one construction to the next. A closure created anew, or
    a `functools.partial`, is a different one each time. An array it closes over is not a leaf of
    the operator: `lift` cannot reuse it, and `jax.jit` treats it as a constant. Arrays belong in
    `members`. An unbound method, `Class.method`, is the same `build` too, with the instance as
    its first member.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator
        >>> def doubled(d):
        ...     return DiagonalOperator(2 * d)
        >>> d = jnp.array([1.0, 2.0, 3.0], jnp.float32)
        >>> op = DeferredOperator(doubled, d)
        >>> op(jnp.ones(3, jnp.float32))
        Array([2., 4., 6.], dtype=float32)
    """

    build: Callable[..., AbstractLinearOperator] = field(metadata={'static': True})
    """From `members` to the operator."""
    members: tuple[PyTree[Any], ...]
    """The data of the operator, the arguments of `build`."""
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(
        kw_only=True, metadata={'static': True}, default=None
    )
    _tags: OperatorTag = field(kw_only=True, metadata={'static': True}, default=OperatorTag.NONE)

    def __init__(self, build: Callable[..., AbstractLinearOperator], *members: PyTree[Any]) -> None:
        try:
            hash(build)
        except TypeError:
            raise ValueError('build must be hashable, as static data of the operator') from None
        object.__setattr__(self, 'build', build)
        object.__setattr__(self, 'members', members)
        # one abstract build, as rebuilding the operator traces its constructor
        in_structure, out_structure, tags = self._static_attributes()
        object.__setattr__(self, '_out_structure', out_structure)
        object.__setattr__(self, '_tags', tags)
        super().__init__(in_structure=in_structure)

    def mv(self, x: PyTree[jax.Array]) -> PyTree[jax.Array]:
        return self.build(*self.members)(x)

    def transpose(self) -> AbstractLinearOperator:
        build = self.build.build if isinstance(self.build, _Transposed) else _Transposed(self.build)
        return DeferredOperator(build, *self.members)

    def inverse(self) -> AbstractLinearOperator:
        build = self.build.build if isinstance(self.build, _Inverted) else _Inverted(self.build)
        return DeferredOperator(build, *self.members)

    @property
    def tags(self) -> OperatorTag:
        return self._tags

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def _static_attributes(self) -> tuple[Any, Any, OperatorTag]:
        """The structures and tags of `build(*members)`, built under an abstract evaluation.

        The members may themselves be abstract, such as the template of an ensemble, whose leaves are
        `jax.ShapeDtypeStruct`s: its arrays are inputs of the evaluation, not constants.
        """
        leaves, treedef = jax.tree.flatten(self.members)
        is_array = [_is_array(leaf) or _is_abstract(leaf) for leaf in leaves]
        attributes = []

        def evaluate(arrays: list[Any]) -> None:
            op = self.build(*_rebuild(treedef, leaves, is_array, arrays))
            # static, so they outlive the evaluation, unlike the operator's traced arrays
            attributes.extend([op.in_structure, op.out_structure, op.tags])

        jax.eval_shape(evaluate, [leaf for leaf, a in zip(leaves, is_array) if a])
        return tuple(attributes)


@dataclass(frozen=True)
class _Transposed:
    """The transpose of what `build` builds, equal for equal `build`s."""

    build: Callable[..., AbstractLinearOperator]

    def __call__(self, *members: PyTree[Any]) -> AbstractLinearOperator:
        return self.build(*members).T


@dataclass(frozen=True)
class _Inverted:
    """The inverse of what `build` builds, equal for equal `build`s."""

    build: Callable[..., AbstractLinearOperator]

    def __call__(self, *members: PyTree[Any]) -> AbstractLinearOperator:
        return self.build(*members).I
