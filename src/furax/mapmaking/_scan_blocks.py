from abc import ABC, abstractmethod
from typing import Self

import jax
from jax import Array
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, tree
from furax.core.rules import AbstractBinaryRule


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
        return type(self)(self.blocks.reduce(), in_structure=self.in_structure)


def _augment_structure(
    structure: PyTree[jax.ShapeDtypeStruct],
    *,
    spec: P,
    axis_size: int | None = None,
) -> PyTree[jax.ShapeDtypeStruct]:
    mesh = _get_mesh()

    def transform(s: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        shape = (axis_size, *s.shape) if axis_size is not None else s.shape
        padded = P(*spec, *((None,) * (len(shape) - len(spec))))
        return jax.ShapeDtypeStruct(shape, s.dtype, sharding=NamedSharding(mesh, padded))

    return jax.tree.map(transform, structure)


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0]).shape[0]
        spec = P(_get_mesh().axis_names[0])
        return _augment_structure(blocks.in_structure, axis_size=axis_size, spec=spec)

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
        return ScanBlockDiagonalOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return _augment_structure(blocks.in_structure, spec=P())

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
        return ScanBlockRowOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        axis_size = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0]).shape[0]
        spec = P(_get_mesh().axis_names[0])
        return _augment_structure(blocks.in_structure, axis_size=axis_size, spec=spec)

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
        return ScanBlockColumnOperator(self.blocks.T, in_structure=self.out_structure)


class ScanAdditionOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        return _augment_structure(blocks.in_structure, spec=P())

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
        return ScanAdditionOperator(self.blocks.T, in_structure=self.out_structure)


class AbstractScanFusionRule(AbstractBinaryRule):
    reduced_class: type[AbstractScanBlockOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
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
