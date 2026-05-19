from abc import ABC
from dataclasses import field

import jax
from jax import Array
from jax.sharding import NamedSharding
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, tree
from furax.core.rules import AbstractBinaryRule


def _prepend_axis(structure: PyTree[jax.ShapeDtypeStruct], n: int) -> PyTree[jax.ShapeDtypeStruct]:
    return jax.tree.map(lambda s: jax.ShapeDtypeStruct((n, *s.shape), s.dtype), structure)


class AbstractScanBlockOperator(AbstractLinearOperator, ABC):
    blocks: AbstractLinearOperator
    n_blocks: int = field(metadata={'static': True})

    def __init__(
        self,
        blocks: AbstractLinearOperator,
        n_blocks: int,
        *,
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'blocks', blocks)
        object.__setattr__(self, 'n_blocks', n_blocks)
        super().__init__(in_structure=in_structure)

    def reduce(self) -> AbstractLinearOperator:
        return type(self)(self.blocks.reduce(), self.n_blocks)  # type: ignore[call-arg]


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    def __init__(self, blocks: AbstractLinearOperator, n_blocks: int) -> None:
        in_structure = _prepend_axis(blocks.in_structure, n_blocks)
        super().__init__(blocks, n_blocks, in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return _prepend_axis(self.blocks.out_structure, self.n_blocks)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        def step(_, args):  # type: ignore[no-untyped-def]
            op, x_i = args
            return None, op.mv(x_i)

        _, out = jax.lax.scan(step, None, (self.blocks, x))
        return out

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockDiagonalOperator(self.blocks.T, self.n_blocks)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    def __init__(self, blocks: AbstractLinearOperator, n_blocks: int) -> None:
        # no leading dim because input is broadcast
        super().__init__(blocks, n_blocks, in_structure=blocks.in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return _prepend_axis(self.blocks.out_structure, self.n_blocks)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        def step(_, op):  # type: ignore[no-untyped-def]
            return None, op.mv(x)

        _, out = jax.lax.scan(step, None, self.blocks)
        return out

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockRowOperator(self.blocks.T, self.n_blocks)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    """Block-row across the batch axis: stacked → single, sum over slices.

    Mirrors :class:`BlockRowOperator` but iterates via ``jax.lax.scan``.

    An optional ``output_sharding`` triggers a ``with_sharding_constraint`` on
    the (post-reduction) output — useful under distributed execution where the
    summed-axis output should be replicated across devices.
    """

    output_sharding: NamedSharding | None = field(metadata={'static': True}, default=None)

    def __init__(
        self,
        blocks: AbstractLinearOperator,
        n_blocks: int,
        output_sharding: NamedSharding | None = None,
    ) -> None:
        in_structure = _prepend_axis(blocks.in_structure, n_blocks)
        object.__setattr__(self, 'output_sharding', output_sharding)
        super().__init__(blocks, n_blocks, in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        # no leading dim because sum over batch dimension
        return self.blocks.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        init = tree.zeros_like(self.blocks.out_structure)

        def step(carry, args):  # type: ignore[no-untyped-def]
            op, x_i = args
            return tree.add(carry, op.mv(x_i)), None

        out, _ = jax.lax.scan(step, init, (self.blocks, x))
        if self.output_sharding is not None:
            out = jax.lax.with_sharding_constraint(out, self.output_sharding)
        return out

    def transpose(self) -> AbstractLinearOperator:
        return ScanBlockColumnOperator(self.blocks.T, self.n_blocks)

    def reduce(self) -> AbstractLinearOperator:
        return ScanBlockRowOperator(self.blocks.reduce(), self.n_blocks, self.output_sharding)


class ScanAdditionOperator(AbstractScanBlockOperator):
    """Sums per-block outputs into a single output (no leading batch axis).

    An optional ``output_sharding`` triggers a ``with_sharding_constraint`` on
    the summed result — useful under distributed execution where the reduction
    output should be replicated across devices.
    """

    output_sharding: NamedSharding | None = field(metadata={'static': True}, default=None)

    def __init__(
        self,
        blocks: AbstractLinearOperator,
        n_blocks: int,
        output_sharding: NamedSharding | None = None,
    ) -> None:
        object.__setattr__(self, 'output_sharding', output_sharding)
        super().__init__(blocks, n_blocks, in_structure=blocks.in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.blocks.out_structure

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        init = tree.zeros_like(self.blocks.out_structure)

        def step(carry, op):  # type: ignore[no-untyped-def]
            return tree.add(carry, op.mv(x)), None

        out, _ = jax.lax.scan(step, init, self.blocks)
        if self.output_sharding is not None:
            out = jax.lax.with_sharding_constraint(out, self.output_sharding)
        return out

    def transpose(self) -> AbstractLinearOperator:
        return ScanAdditionOperator(self.blocks.T, self.n_blocks, self.output_sharding)

    def reduce(self) -> AbstractLinearOperator:
        return ScanAdditionOperator(self.blocks.reduce(), self.n_blocks, self.output_sharding)


class AbstractScanFusionRule(AbstractBinaryRule):
    reduced_class: type[AbstractScanBlockOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        composed = (left.blocks @ right.blocks).reduce()
        return [self.reduced_class(composed, left.n_blocks)]  # type: ignore[call-arg]


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

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, ScanBlockRowOperator)  # mypy
        assert isinstance(right, ScanBlockDiagonalOperator)  # mypy
        composed = (left.blocks @ right.blocks).reduce()
        return [ScanBlockRowOperator(composed, left.n_blocks, left.output_sharding)]


class ScanBlockRowScanBlockColumnRule(AbstractScanFusionRule):
    """``ScanBlockRow @ ScanBlockColumn = ScanAddition``."""

    left_operator_class = ScanBlockRowOperator
    right_operator_class = ScanBlockColumnOperator
    reduced_class = ScanAdditionOperator

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, ScanBlockRowOperator)  # mypy
        assert isinstance(right, ScanBlockColumnOperator)  # mypy
        composed = (left.blocks @ right.blocks).reduce()
        return [ScanAdditionOperator(composed, left.n_blocks, left.output_sharding)]
