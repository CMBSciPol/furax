from abc import ABC, abstractmethod
from typing import Self

import jax
from jax import Array
from jax.sharding import AbstractMesh
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
    blocks: AbstractLinearOperator

    @classmethod
    def create(cls, blocks: AbstractLinearOperator) -> Self:
        if len(jax.tree.leaves(blocks)) == 0:
            msg = 'unable to infer structures from blocks with no leaf'
            raise RuntimeError(msg)
        in_structure = cls._infer_in_structure(blocks)
        return cls(blocks, in_structure=in_structure)

    @classmethod
    @abstractmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns sharding-aware in_structure"""

    def reduce(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            reduced_blocks = self.blocks.reduce()
        return type(self)(reduced_blocks, in_structure=self.in_structure)


def _prepend_axis(
    structure: PyTree[jax.ShapeDtypeStruct],
    *,
    axis_size: int | None = None,
) -> PyTree[jax.ShapeDtypeStruct]:
    def transform(s: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        shape = (axis_size, *s.shape) if axis_size is not None else s.shape
        return jax.ShapeDtypeStruct(shape, s.dtype)

    return jax.tree.map(transform, structure)


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0]).shape[0]
        return _prepend_axis(blocks.in_structure, axis_size=axis_size)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        n = jax.tree.leaves(self.in_structure)[0].shape[0]
        return _prepend_axis(self.blocks.out_structure, axis_size=n)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return None, op.mv(x_i)

            _, out = jax.lax.scan(step, None, (blocks, x))
            return out

        # shard_map validates in_specs against actual array sharding; reshard sets it
        # explicitly so the check passes even when sharding is lost inside a JIT trace
        blocks = jax.tree.map(lambda a: jax.reshard(a, P(axis)), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, P(axis)), x)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            blocks_T = self.blocks.T
        return ScanBlockDiagonalOperator(blocks_T, in_structure=self.out_structure)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return blocks.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        n = jax.tree.leaves(self.blocks)[0].shape[0]
        return _prepend_axis(self.blocks.out_structure, axis_size=n)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, op):  # type: ignore[no-untyped-def]
                return None, op.mv(x)

            _, out = jax.lax.scan(step, None, blocks)
            return out

        blocks = jax.tree.map(lambda leaf: jax.reshard(leaf, P(axis)), self.blocks)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            blocks_T = self.blocks.T
        return ScanBlockRowOperator(blocks_T, in_structure=self.out_structure)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0]).shape[0]
        return _prepend_axis(blocks.in_structure, axis_size=axis_size)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.blocks.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.blocks.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return tree.add(carry, op.mv(x_i)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, (blocks, x))
            return jax.lax.psum(out, axis_name=axis)

        blocks = jax.tree.map(lambda a: jax.reshard(a, P(axis)), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, P(axis)), x)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            blocks_T = self.blocks.T
        return ScanBlockColumnOperator(blocks_T, in_structure=self.out_structure)


class ScanAdditionOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return blocks.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.blocks.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.blocks.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(carry, op):  # type: ignore[no-untyped-def]
                return tree.add(carry, op.mv(x)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, blocks)
            return jax.lax.psum(out, axis_name=axis)

        blocks = jax.tree.map(lambda leaf: jax.reshard(leaf, P(axis)), self.blocks)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            blocks_T = self.blocks.T
        return ScanAdditionOperator(blocks_T, in_structure=self.out_structure)


class AbstractScanFusionRule(AbstractBinaryRule):
    reduced_class: type[AbstractScanBlockOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            composed = (left.blocks @ right.blocks).reduce()
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
