from abc import ABC, abstractmethod
from typing import Self

import jax
from jax import Array
from jax.sharding import AbstractMesh, Mesh, NamedSharding
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

    @property
    def _mesh(self) -> Mesh | AbstractMesh | None:
        s = jax.tree.leaves(self.in_structure)[0].sharding
        return s.mesh if isinstance(s, NamedSharding) else None

    @classmethod
    @abstractmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        """Returns sharding-aware in_structure"""

    @property
    def _in_specs(self) -> P:
        """PartitionSpec for obs-sharded blocks. Only valid when sharded."""
        return P(self._mesh.axis_names)  # type: ignore[union-attr]

    @property
    @abstractmethod
    def _out_specs(self) -> P:
        """PartitionSpec for shard_map out_specs. Only valid when sharded."""

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
    @property
    def _out_specs(self) -> P:
        return P(self._mesh.axis_names)  # type: ignore[union-attr]

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
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return None, op.mv(x_i)

            _, out = jax.lax.scan(step, None, (blocks, x))
            return out

        mesh = self._mesh
        if mesh is None:
            return kernel(self.blocks, x)

        in_sharding = NamedSharding(mesh, self._in_specs)
        # shard_map validates in_specs against actual array sharding; reshard sets it
        # explicitly so the check passes even when sharding is lost inside a JIT trace
        blocks = jax.tree.map(lambda a: jax.reshard(a, in_sharding), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, in_sharding), x)
        return jax.shard_map(kernel, mesh=mesh, out_specs=self._out_specs)(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockDiagonalOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    @property
    def _out_specs(self) -> P:
        return P(self._mesh.axis_names)  # type: ignore[union-attr]

    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        sharding = _leaf_named_sharding(blocks)
        replicated = NamedSharding(sharding.mesh, P()) if sharding is not None else None
        return _augment_structure(blocks.in_structure, sharding=replicated)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        def kernel(blocks, x):  # type: ignore[no-untyped-def]
            def step(_, op):  # type: ignore[no-untyped-def]
                return None, op.mv(x)

            _, out = jax.lax.scan(step, None, blocks)
            return out

        mesh = self._mesh
        if mesh is None:
            return kernel(self.blocks, x)

        # see Diagonal.mv
        blocks = jax.tree.map(
            lambda a: jax.reshard(a, NamedSharding(mesh, self._in_specs)), self.blocks
        )
        return jax.shard_map(kernel, mesh=mesh, out_specs=self._out_specs)(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockRowOperator(self.blocks.T, in_structure=self.out_structure)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    @property
    def _out_specs(self) -> P:
        return P()

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
        out_structure = self.blocks.out_structure
        zeros = tree.zeros_like(out_structure)

        def kernel(blocks, x, init):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                op, x_i = args
                return tree.add(carry, op.mv(x_i)), None

            out, _ = jax.lax.scan(step, init, (blocks, x))
            return out

        mesh = self._mesh
        if mesh is None:
            return kernel(self.blocks, x, zeros)

        in_sharding = NamedSharding(mesh, self._in_specs)
        # see Diagonal.mv
        blocks = jax.tree.map(lambda a: jax.reshard(a, in_sharding), self.blocks)
        x = jax.tree.map(lambda a: jax.reshard(a, in_sharding), x)

        def sharded_kernel(blocks, x):  # type: ignore[no-untyped-def]
            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(zeros, mesh.axis_names, to='varying')
            return jax.lax.psum(kernel(blocks, x, init), axis_name=mesh.axis_names)

        return jax.shard_map(sharded_kernel, mesh=mesh, out_specs=self._out_specs)(blocks, x)

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockColumnOperator(self.blocks.T, in_structure=self.out_structure)


class ScanAdditionOperator(AbstractScanBlockOperator):
    @property
    def _out_specs(self) -> P:
        return P()

    @classmethod
    def _infer_in_structure(cls, blocks: AbstractLinearOperator) -> PyTree[jax.ShapeDtypeStruct]:
        sharding = _leaf_named_sharding(blocks)
        replicated = NamedSharding(sharding.mesh, P()) if sharding is not None else None
        return _augment_structure(blocks.in_structure, sharding=replicated)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        out_structure = self.blocks.out_structure
        zeros = tree.zeros_like(out_structure)

        def kernel(blocks, x, init):  # type: ignore[no-untyped-def]
            def step(carry, op):  # type: ignore[no-untyped-def]
                return tree.add(carry, op.mv(x)), None

            out, _ = jax.lax.scan(step, init, blocks)
            return out

        mesh = self._mesh
        if mesh is None:
            return kernel(self.blocks, x, zeros)

        # see Diagonal.mv
        blocks = jax.tree.map(
            lambda a: jax.reshard(a, NamedSharding(mesh, self._in_specs)), self.blocks
        )

        def sharded_kernel(blocks, x):  # type: ignore[no-untyped-def]
            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(zeros, mesh.axis_names, to='varying')
            return jax.lax.psum(kernel(blocks, x, init), axis_name=mesh.axis_names)

        return jax.shard_map(sharded_kernel, mesh=mesh, out_specs=self._out_specs)(blocks, x)

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
