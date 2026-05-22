from abc import ABC, abstractmethod
from typing import Self

import jax
from jax import Array
from jax.sharding import AbstractMesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, tree
from furax.core.rules import AbstractBinaryRule


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

    def _get_mesh(self) -> AbstractMesh:
        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty:
            raise RuntimeError('active mesh context required')
        return mesh

    def reduce(self) -> AbstractLinearOperator:
        return type(self)(self.blocks.reduce(), in_structure=self.in_structure)


def _match_sharding_rank(sharding: NamedSharding | None, rank: int) -> NamedSharding | None:
    if sharding is None or len(sharding.spec) >= rank:
        return sharding
    padding = (None,) * (rank - len(sharding.spec))
    return NamedSharding(sharding.mesh, P(*sharding.spec, *padding))


def _augment_structure(
    structure: PyTree[jax.ShapeDtypeStruct],
    *,
    axis_size: int | None = None,
    sharding: NamedSharding | None = None,
) -> PyTree[jax.ShapeDtypeStruct]:
    def transform(s: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
        shape = (axis_size, *s.shape) if axis_size is not None else s.shape
        new_sharding = _match_sharding_rank(sharding, len(shape))
        return jax.ShapeDtypeStruct(shape, s.dtype, sharding=new_sharding)

    return jax.tree.map(transform, structure)


def _leaf_named_sharding(blocks: AbstractLinearOperator) -> NamedSharding | None:
    leaf_shape = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0])
    s = leaf_shape.sharding
    return s if isinstance(s, NamedSharding) else None


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        leaf_shape = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0])
        leaf_sharding = leaf_shape.sharding
        if isinstance(leaf_sharding, NamedSharding):
            sharding = NamedSharding(leaf_sharding.mesh, P(leaf_sharding.spec[0]))
        else:
            sharding = None
        return _augment_structure(
            blocks.in_structure, axis_size=leaf_shape.shape[0], sharding=sharding
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = self._get_mesh()
        axis = mesh.axis_names[0]

        @jax.shard_map(mesh=mesh, out_specs=P(axis), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return None, op.mv(x_i)

            _, out = jax.lax.scan(step, None, (blocks, x))
            return out

        # shard_map validates in_specs against actual array sharding; reshard sets it
        # explicitly so the check passes even when sharding is lost inside a JIT trace
        sharding = NamedSharding(mesh, P(axis))
        blocks = jax.tree.map(lambda a: jax.reshard(a, sharding), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, sharding), x)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockDiagonalOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        sharding = _leaf_named_sharding(blocks)
        replicated = NamedSharding(sharding.mesh, P()) if sharding is not None else None
        return _augment_structure(blocks.in_structure, sharding=replicated)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = self._get_mesh()
        axis = mesh.axis_names[0]

        @jax.shard_map(mesh=mesh, out_specs=P(axis), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, op):  # type: ignore[no-untyped-def]
                return None, op.mv(x)

            _, out = jax.lax.scan(step, None, blocks)
            return out

        sharding = NamedSharding(mesh, P(axis))
        blocks = jax.tree.map(lambda leaf: jax.reshard(leaf, sharding), self.blocks)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockRowOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        leaf_shape = jax.eval_shape(lambda: jax.tree.leaves(blocks)[0])
        if isinstance(leaf_shape.sharding, NamedSharding):
            sharding = NamedSharding(leaf_shape.sharding.mesh, P(leaf_shape.sharding.spec[0]))
        else:
            sharding = None
        return _augment_structure(
            blocks.in_structure, axis_size=leaf_shape.shape[0], sharding=sharding
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = self._get_mesh()
        axis = mesh.axis_names[0]
        out_structure = self.blocks.out_structure

        @jax.shard_map(mesh=mesh, out_specs=P(), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return tree.add(carry, op.mv(x_i)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, (blocks, x))
            return jax.lax.psum(out, axis_name=axis)

        sharding = NamedSharding(mesh, P(axis))
        blocks = jax.tree.map(lambda a: jax.reshard(a, sharding), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, sharding), x)
        return kernel(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockColumnOperator(self.blocks.T, in_structure=self.out_structure)


class ScanAdditionOperator(AbstractScanBlockOperator):
    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        sharding = _leaf_named_sharding(blocks)
        replicated = NamedSharding(sharding.mesh, P()) if sharding is not None else None
        return _augment_structure(blocks.in_structure, sharding=replicated)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = self._get_mesh()
        axis = mesh.axis_names[0]
        out_structure = self.blocks.out_structure

        @jax.shard_map(mesh=mesh, out_specs=P(), check_vma=False)
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(carry, op):  # type: ignore[no-untyped-def]
                return tree.add(carry, op.mv(x)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), mesh.axis_names, to='varying')
            out, _ = jax.lax.scan(step, init, blocks)
            return jax.lax.psum(out, axis_name=mesh.axis_names)

        sharding = NamedSharding(mesh, P(axis))
        blocks = jax.tree.map(lambda leaf: jax.reshard(leaf, sharding), self.blocks)
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
