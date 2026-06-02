from abc import ABC, abstractmethod
from typing import Self

import jax
from jax import Array
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, tree
from furax.core.rules import AbstractBinaryRule

# jax.eval_shape inside a jax.set_mesh context annotates outputs with sharding information,
# which breaks operator compatibility checks. Wrapping inner transpose/reduction calls with
# use_abstract_mesh(_EMPTY_MESH) clears the active mesh so those structs stay sharding-free.
_EMPTY_MESH = jax.sharding.AbstractMesh((), ())


def _get_mesh() -> AbstractMesh:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        raise RuntimeError('active mesh context required')
    return mesh


class AbstractScanBlockOperator(AbstractLinearOperator, ABC):
    """Base class for operators that apply a batched pytree operator slice-by-slice via scan.

    ``operator`` is an operator whose leaves carry a leading axis of size ``N`` obtained by
    stacking N individual operators into a single pytree beforehand.  ``operator`` itself is
    not expected to handle that leading axis: ``mv`` uses ``jax.lax.scan`` to peel off one
    slice at a time and feed it to the underlying operator logic, inside a ``shard_map`` kernel
    that distributes the work across devices along the first mesh axis.

    The "block" denomination refers to how each slice appears in the global operator matrix:
    subclasses arrange the N per-slice operators as blocks of a larger matrix (diagonal,
    column, or row layout), giving rise to block-diagonal, block-column, and block-row
    structures respectively.

    An active mesh context is required when calling ``mv``; use ``jax.set_mesh`` beforehand.
    """

    operator: AbstractLinearOperator

    @classmethod
    def create(cls, operator: AbstractLinearOperator) -> Self:
        if len(jax.tree.leaves(operator)) == 0:
            msg = 'unable to infer structures from operator with no leaf'
            raise RuntimeError(msg)
        in_structure = cls._infer_in_structure(operator)
        return cls(operator, in_structure=in_structure)

    @classmethod
    @abstractmethod
    def _infer_in_structure(cls, operator: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns sharding-aware in_structure"""

    def reduce(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            reduced_operator = self.operator.reduce()
        return type(self)(reduced_operator, in_structure=self.in_structure)


def _augment_structure(
    structure: PyTree[jax.ShapeDtypeStruct],
    *,
    axis_size: int | None = None,
) -> PyTree[jax.ShapeDtypeStruct]:
    mesh = jax.sharding.get_abstract_mesh()
    axis_name = mesh.axis_names[0] if not mesh.empty and axis_size is not None else None

    def transform(s: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        shape = (axis_size, *s.shape) if axis_size is not None else s.shape
        if axis_name is None:
            return jax.ShapeDtypeStruct(shape, s.dtype)
        spec = P(axis_name, *([None] * len(s.shape)))
        return jax.ShapeDtypeStruct(shape, s.dtype, sharding=NamedSharding(mesh, spec))

    return jax.tree.map(transform, structure)


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    """Block-diagonal operator: each block acts independently on its own slice of the input.

    Given ``operator: (*in,) -> (*out,)`` with ``N`` slices, maps ``(N, *in) -> (N, *out)``.

    Example — per-observation noise weighting (square blocks, ``*in == *out``):

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     W = ScanBlockDiagonalOperator.create(noise_op)  # leaves: (N, *in)
        ...     weighted = W(samples)                           # (N, *in) -> (N, *out)
    """

    @classmethod
    def _infer_in_structure(cls, operator: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.tree.leaves(operator)[0].shape[0]
        return _augment_structure(operator.in_structure, axis_size=axis_size)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        n = jax.tree.leaves(self.in_structure)[0].shape[0]
        return _augment_structure(self.operator.out_structure, axis_size=n)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(operator, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return None, op(x_i)

            _, out = jax.lax.scan(step, None, (operator, x))
            return out

        return kernel(self.operator, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            operator_T = self.operator.T
        return ScanBlockDiagonalOperator(operator_T, in_structure=self.out_structure)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    """Column operator: applies all blocks to the same input and stacks the results.

    Given ``operator: (*in,) -> (*out,)`` with ``N`` slices, maps ``(*in,) -> (N, *out)``.

    Example — pointing matrix from pixel map to time-ordered data:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     H = ScanBlockColumnOperator.create(pointing_op)  # leaves: (N, *out)
        ...     tod = H(pixel_map)                               # (*in,) -> (N, *out)
    """

    @classmethod
    def _infer_in_structure(cls, operator: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return operator.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        n = jax.tree.leaves(self.operator)[0].shape[0]
        return _augment_structure(self.operator.out_structure, axis_size=n)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(operator, x):  # type: ignore[no-untyped-def]
            def step(_, op):  # type: ignore[no-untyped-def]
                return None, op(x)

            _, out = jax.lax.scan(step, None, operator)
            return out

        return kernel(self.operator, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            operator_T = self.operator.T
        return ScanBlockRowOperator(operator_T, in_structure=self.out_structure)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    """Row operator: applies each block to its own input slice and sums the results.

    Given ``operator: (*in,) -> (*out,)`` with ``N`` slices, maps ``(N, *in) -> (*out,)``.

    Example — co-addition of time-ordered data back to a pixel map:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     HT = ScanBlockRowOperator.create(pointing_op_T)  # leaves: (N, *in)
        ...     pixel_map = HT(tod)                              # (N, *in) -> (*out,)
    """

    @classmethod
    def _infer_in_structure(cls, operator: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.tree.leaves(operator)[0].shape[0]
        return _augment_structure(operator.in_structure, axis_size=axis_size)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.operator.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(operator, x):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return tree.add(carry, op(x_i)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, (operator, x))
            return jax.lax.psum(out, axis_name=axis)

        return kernel(self.operator, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            operator_T = self.operator.T
        return ScanBlockColumnOperator(operator_T, in_structure=self.out_structure)


class ScanAdditionOperator(AbstractScanBlockOperator):
    """Addition operator: applies all blocks to the same input and sums the results.

    Given ``operator: (*in,) -> (*out,)`` with ``N`` slices, maps ``(*in,) -> (*out,)``.

    Arises naturally as the reduction of ``ScanBlockRowOperator @ ScanBlockColumnOperator``,
    e.g. the normal equations operator ``H.T @ W @ H`` in mapmaking.

    Example — normal equations from a pointing and weighting operator:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     A = (H.T @ W @ H).reduce()  # reduces to ScanAdditionOperator
        ...     rhs = A(pixel_map)          # (*in,) -> (*out,)
    """

    @classmethod
    def _infer_in_structure(cls, operator: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return operator.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.operator.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(operator, x):  # type: ignore[no-untyped-def]
            def step(carry, op):  # type: ignore[no-untyped-def]
                return tree.add(carry, op(x)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, operator)
            return jax.lax.psum(out, axis_name=axis)

        return kernel(self.operator, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            operator_T = self.operator.T
        return ScanAdditionOperator(operator_T, in_structure=self.out_structure)


class AbstractScanFusionRule(AbstractBinaryRule):
    reduced_class: type[AbstractScanBlockOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            composed = (left.operator @ right.operator).reduce()
        return [self.reduced_class.create(composed)]


class ScanBlockDiagonalScanBlockDiagonalRule(AbstractScanFusionRule):
    """``ScanBlockDiagonal @ ScanBlockDiagonal = ScanBlockDiagonal``."""

    left_operator_class = ScanBlockDiagonalOperator
    right_operator_class = ScanBlockDiagonalOperator
    reduced_class = ScanBlockDiagonalOperator


class ScanBlockDiagonalScanBlockColumnRule(AbstractScanFusionRule):
    """``ScanBlockDiagonal @ ScanBlockColumn = ScanBlockColumn``."""

    left_operator_class = ScanBlockDiagonalOperator
    right_operator_class = ScanBlockColumnOperator
    reduced_class = ScanBlockColumnOperator


class ScanBlockRowScanBlockDiagonalRule(AbstractScanFusionRule):
    """``ScanBlockRow @ ScanBlockDiagonal = ScanBlockRow``."""

    left_operator_class = ScanBlockRowOperator
    right_operator_class = ScanBlockDiagonalOperator
    reduced_class = ScanBlockRowOperator


class ScanBlockRowScanBlockColumnRule(AbstractScanFusionRule):
    """``ScanBlockRow @ ScanBlockColumn = ScanAddition``."""

    left_operator_class = ScanBlockRowOperator
    right_operator_class = ScanBlockColumnOperator
    reduced_class = ScanAdditionOperator
