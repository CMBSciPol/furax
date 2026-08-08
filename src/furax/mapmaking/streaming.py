"""Operators that stream a batched operator slice-by-slice across a sharded leading axis."""

import functools
from collections.abc import Sequence
from dataclasses import field
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import AbstractMesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator, tree
from furax.core import (
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockRowOperator,
    HomothetyOperator,
    IdentityOperator,
)
from furax.core._base import structure_equal
from furax.core.rules import AbstractAdditionRule, AbstractCompositionRule, NoReduction
from furax.mapmaking._distributed import cross_process_sum, stream_is_split

__all__ = [
    'StackSpec',
    'StreamOperator',
    'StreamSegment',
]

type StackSpec = bool | PyTree[bool]
"""Per-component stackedness of a structure: a bare bool (uniform) or a prefix pytree of bools."""


class StreamSegment(eqx.Module):
    """One link of a stream body, and where its data goes in the scan.

    A *sliced* segment owns data carrying the batch axis, so the scan takes a slice of it each
    step. A *constant* one owns data that does not, so it is held fixed and applied identically at
    every step. Being sliced is a claim about the data, checked at construction: every array leaf
    must lead with `n_lead`.

    A third case cuts across those two. A *pure* segment ([`is_pure`][]) owns no data at all --
    an identity, or a block operator assembled only to route components around. It has nothing to
    slice or hold, so it belongs to neither camp and can join whichever neighbour it sits next to.
    That is what lets the block constructors and the slot alignment emit structural operators
    freely: `_normalize` folds them away again.
    """

    operator: AbstractLinearOperator
    sliced: bool = eqx.field(static=True)

    @property
    def is_identity(self) -> bool:
        """A constant identity: contributes nothing and can be dropped outright, not just folded."""
        return not self.sliced and isinstance(self.operator, IdentityOperator)

    @property
    def is_pure(self) -> bool:
        """Owns no data, so the sliced/constant distinction does not apply to it.

        Any pytree leaf counts as data, not just arrays: a Python float is not an array until
        something traces it, and a segment's role must not change under `jit`.
        """
        return not jax.tree.leaves(self.operator)

    @property
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure

    def transpose(self) -> 'StreamSegment':
        return StreamSegment(self.operator.T, self.sliced)

    def reduce(self) -> 'StreamSegment':
        return StreamSegment(self.operator.reduce(), self.sliced)


class StreamOperator(AbstractLinearOperator):
    """Applies a batched pytree operator slice-by-slice via scan, over a sharded leading axis.

    Two independent things carry the batch axis, and they are described separately:

    - The *interior*: [`StreamSegment`][] links, held in composition order (segments[0] applied
      last, segments[-1] first), each *sliced* or *constant* according to whether its own data
      carries the batch axis.
    - The *boundary*: `in_stacked`/`out_stacked` ([`StackSpec`][]) say which input/output
      *components* carry it. A bare bool covers the whole structure, a prefix pytree resolves per
      component. A stacked component is sliced on input and stacked on output; a shared one is
      broadcast on input and sum-reduced (scan carry, then `psum`) on output.

    Neither constrains the other: a sliced segment may sit between shared boundary components, as
    in a stream that sums over the batch axis. The boundary is what gives a stream its block
    layout, and the four uniform combinations have constructors of their own:

    | in_stacked | out_stacked | constructor  | signature         |
    |------------|-------------|--------------|-------------------|
    | True       | True        | `diagonal`   | (N,in) -> (N,out) |
    | False      | True        | `column`     | (in,)  -> (N,out) |
    | True       | False       | `row`        | (N,in) -> (out,)  |
    | False      | False       | `addition`   | (in,)  -> (out,)  |

    Those four lay a single operator out over the batch axis. `block_row` / `block_column` do
    something different despite the shared vocabulary: they lay several *streams* out over the
    components of a block structure, giving a mixed boundary -- some components stacked, some
    shared -- which is this same class with prefix pytree specs. The reduction rules produce mixed
    streams too.

    An active mesh context is required when calling `mv`; use `jax.set_mesh` beforehand.
    """

    segments: tuple[StreamSegment, ...]
    n_lead: int = field(kw_only=True, metadata={'static': True})
    in_stacked: StackSpec = field(kw_only=True, metadata={'static': True})
    out_stacked: StackSpec = field(kw_only=True, metadata={'static': True})

    @classmethod
    def diagonal(
        cls, operator: AbstractLinearOperator, *, n_lead: int | None = None
    ) -> 'StreamOperator':
        """Block-diagonal stream: each block acts independently on its own slice of the input.

        Given a per-slice operator `(*in,) -> (*out,)` with `N` slices, maps
        `(N, *in) -> (N, *out)`.

        Args:
            operator: The per-slice operator, stacked along a leading (batch) axis.
            n_lead: The batch-axis size. Inferred from the operator leaves if omitted; required if
                the operator has no array leaves.

        Examples:
            Per-slice noise weighting (square blocks, `*in == *out`):

            >>> with jax.set_mesh(jax.make_mesh((4,), ('batch',))):
            ...     W = StreamOperator.diagonal(noise_op)  # leaves: (N, *in)
            ...     weighted = W(samples)                  # (N, *in) -> (N, *out)
        """
        return cls._single_segment(operator, n_lead, in_stacked=True, out_stacked=True)

    @classmethod
    def column(
        cls, operator: AbstractLinearOperator, *, n_lead: int | None = None
    ) -> 'StreamOperator':
        """Column stream: applies all blocks to the same input and stacks the results.

        Given a per-slice operator `(*in,) -> (*out,)` with `N` slices, maps `(*in,) -> (N, *out)`.

        Args:
            operator: The per-slice operator, stacked along a leading (batch) axis.
            n_lead: The batch-axis size. Inferred from the operator leaves if omitted; required if
                the operator has no array leaves.

        Examples:
            Pointing matrix from pixel map to time-ordered data:

            >>> with jax.set_mesh(jax.make_mesh((4,), ('batch',))):
            ...     H = StreamOperator.column(pointing_op)  # leaves: (N, *out)
            ...     tod = H(pixel_map)                      # (*in,) -> (N, *out)
        """
        return cls._single_segment(operator, n_lead, in_stacked=False, out_stacked=True)

    @classmethod
    def row(
        cls, operator: AbstractLinearOperator, *, n_lead: int | None = None
    ) -> 'StreamOperator':
        """Row stream: applies each block to its own input slice and sums the results.

        Given a per-slice operator `(*in,) -> (*out,)` with `N` slices, maps `(N, *in) -> (*out,)`.

        Args:
            operator: The per-slice operator, stacked along a leading (batch) axis.
            n_lead: The batch-axis size. Inferred from the operator leaves if omitted; required if
                the operator has no array leaves.

        Examples:
            Co-addition of time-ordered data back to a pixel map:

            >>> with jax.set_mesh(jax.make_mesh((4,), ('batch',))):
            ...     HT = StreamOperator.row(pointing_op_T)  # leaves: (N, *in)
            ...     pixel_map = HT(tod)                     # (N, *in) -> (*out,)
        """
        return cls._single_segment(operator, n_lead, in_stacked=True, out_stacked=False)

    @classmethod
    def addition(
        cls, operator: AbstractLinearOperator, *, n_lead: int | None = None
    ) -> 'StreamOperator':
        """Addition stream: applies all blocks to the same input and sums the results.

        Given a per-slice operator `(*in,) -> (*out,)` with `N` slices, maps `(*in,) -> (*out,)`.

        Arises naturally as the reduction of a row stream composed with a column one, e.g. the
        normal equations operator `H.T @ W @ H` in mapmaking.

        Args:
            operator: The per-slice operator, stacked along a leading (batch) axis.
            n_lead: The batch-axis size. Inferred from the operator leaves if omitted; required if
                the operator has no array leaves.

        Examples:
            Normal equations from a pointing and weighting operator:

            >>> with jax.set_mesh(jax.make_mesh((4,), ('batch',))):
            ...     A = (H.T @ W @ H).reduce()  # in_stacked and out_stacked both False
            ...     rhs = A(pixel_map)          # (*in,) -> (*out,)
        """
        return cls._single_segment(operator, n_lead, in_stacked=False, out_stacked=False)

    @classmethod
    def block_row(cls, operands: Sequence[AbstractLinearOperator]) -> 'StreamOperator':
        """Fuse parallel streams ``[S₁ | S₂ | ...]`` sharing one batch axis into one stream.

        Where [`column`][furax.mapmaking.streaming.StreamOperator.column] and friends lay *one*
        operator out over the batch axis, this lays several *streams* out over the components of a
        block structure: ``H([u₁, ...]) = Σᵢ Sᵢ(uᵢ)``.

        Writing each operand as a chain ``Sᵢ = Aᵢ ∘ Bᵢ ∘ … ∘ Zᵢ``, the identity
        ``BlockRow([Sᵢ]) = BlockRow([Aᵢ]) @ BlockDiagonal([Bᵢ]) @ … @ BlockDiagonal([Zᵢ])`` lays
        the fused stream out slot by slot: the leftmost sums the blocks, the rest act
        componentwise, and every slot keeps its own sliced/constant kind. Chains of different shapes
        are padded to agree first (see `_aligned_segments`). The result carries a per-block
        ``in_stacked`` list and the operands' shared ``out_stacked``.

        This is an explicit constructor, not a deferring reduction: a non-conforming operand raises.

        Args:
            operands: The streams to lay out side by side, at least one.

        Raises:
            TypeError: If an operand is not a stream operator.
            ValueError: If an operand disagrees with the first on ``n_lead``, per-slice output
                structure or ``out_stacked``, or if no operand has a sliced segment.
        """
        if not operands:
            raise ValueError('stream block requires at least one operand')
        ops: list[StreamOperator] = []
        for op in operands:
            if not isinstance(op, StreamOperator):
                raise TypeError('stream block operands must be stream operators')
            ops.append(op)
        n_lead = ops[0].n_lead
        per_slice_out = ops[0].per_slice_out_structure
        ref_mask = jax.tree.leaves(jax.tree.broadcast(ops[0].out_stacked, per_slice_out))
        for op in ops[1:]:
            if op.n_lead != n_lead:
                raise ValueError('stream block operands must share n_lead')
            if not structure_equal(op.per_slice_out_structure, per_slice_out):
                raise ValueError(
                    'stream block operands must share their per-slice junction structure'
                )
            if jax.tree.leaves(jax.tree.broadcast(op.out_stacked, per_slice_out)) != ref_mask:
                raise ValueError('stream block operands must share their junction stack spec')
        n_sliced = max(op.sliced_count for op in ops)
        if n_sliced == 0:
            raise ValueError('stream block operands must have at least one sliced segment')
        # Align the bodies slot for slot, then block them up position-wise: the leftmost slot sums
        # the blocks, the rest act componentwise. `_normalize` folds away the slots that end up
        # all-identity, so a body of plain streams still collapses to a single sliced segment.
        aligned = [op._aligned_segments(n_sliced) for op in ops]
        segments = []
        for position, slot in enumerate(zip(*aligned, strict=True)):
            operators = [seg.operator for seg in slot]
            block = (
                BlockRowOperator(operators) if position == 0 else BlockDiagonalOperator(operators)
            )
            segments.append(StreamSegment(block, position % 2 == 1))
        return cls.create(
            tuple(segments),
            n_lead=n_lead,
            in_stacked=[op.in_stacked for op in ops],
            out_stacked=ops[0].out_stacked,
        )

    @classmethod
    def block_column(cls, operands: Sequence[AbstractLinearOperator]) -> 'StreamOperator':
        """Fuse parallel streams ``[S₁; S₂; ...]`` sharing one batch axis into one stream.

        The transpose of [`block_row`][furax.mapmaking.streaming.StreamOperator.block_row]: fans a
        single shared input across the blocks and collects their outputs into a list,
        ``H(u) = [S₁(u), S₂(u), ...]``. Built as ``BlockRow([Sᵢᵀ])ᵀ``, so it fuses into the same
        one-stacked-core layout, with the per-block spec landing on ``out_stacked`` instead of
        ``in_stacked``.

        Args:
            operands: The streams to stack, at least one.

        Raises:
            TypeError: If an operand is not a stream operator.
            ValueError: As for [`block_row`][furax.mapmaking.streaming.StreamOperator.block_row].
        """
        result = cls.block_row([op.T for op in operands]).T
        assert isinstance(result, StreamOperator)  # mypy
        return result

    @classmethod
    def create(
        cls,
        segments: tuple[StreamSegment, ...],
        *,
        n_lead: int,
        in_stacked: StackSpec,
        out_stacked: StackSpec,
    ) -> 'StreamOperator':
        """Build a stream from an explicit segment chain; the general constructor.

        Args:
            segments: The per-slice operator chain, in composition order; at least one.
            n_lead: The batch-axis size.
            in_stacked: Which input components carry the batch axis.
            out_stacked: Which output components carry the batch axis.

        Raises:
            ValueError: If ``segments`` is empty, if a sliced segment has an array leaf that does
                not lead with ``n_lead``, or a spec is not a prefix of the structure it applies to.
        """
        if not segments:
            raise ValueError('a stream needs at least one segment')
        segments = _normalize(segments, n_lead)
        for seg in segments:
            if seg.sliced:
                _check_sliceable(seg.operator, n_lead)
        per_slice_in = segments[-1].in_structure  # rightmost segment is applied first
        in_stacked = _canonical_spec(in_stacked, per_slice_in)
        out_stacked = _canonical_spec(out_stacked, segments[0].out_structure)
        return cls(
            segments,
            n_lead=n_lead,
            in_stacked=in_stacked,
            out_stacked=out_stacked,
            in_structure=_expand_structure(per_slice_in, in_stacked, n_lead),
        )

    @classmethod
    def _single_segment(
        cls,
        operator: AbstractLinearOperator,
        n_lead: int | None,
        *,
        in_stacked: bool,
        out_stacked: bool,
    ) -> 'StreamOperator':
        """Wrap a freshly stacked operator as a stream with a single sliced segment."""
        if n_lead is None:
            n_lead = _leading_size(operator)
        return cls.create(
            (StreamSegment(operator, True),),
            n_lead=n_lead,
            in_stacked=in_stacked,
            out_stacked=out_stacked,
        )

    @property
    def per_slice_in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """Input structure of one slice, before `in_stacked` adds the batch axis.

        Segments are in composition order, so the *last* one is applied first and owns the input.
        """
        return self.segments[-1].in_structure

    @property
    def per_slice_out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        """Output structure of one slice, before `out_stacked` adds the batch axis.

        Segments are in composition order, so the *first* one is applied last and owns the output.
        """
        return self.segments[0].out_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return _expand_structure(self.per_slice_out_structure, self.out_stacked, self.n_lead)

    @property
    def operator(self) -> AbstractLinearOperator:
        """Effective per-slice operator (composition of the segments; for introspection)."""
        # segments is never empty (see _normalize), so _compose needs no fallback structure
        return _compose(tuple(seg.operator for seg in self.segments))

    def reduce(self) -> AbstractLinearOperator:
        return type(self).create(
            tuple(seg.reduce() for seg in self.segments),
            n_lead=self.n_lead,
            in_stacked=self.in_stacked,
            out_stacked=self.out_stacked,
        )

    def transpose(self) -> AbstractLinearOperator:
        # Swapping the specs transposes the layout: diagonal and addition map to themselves,
        # column and row to each other.
        return type(self).create(
            segments=tuple(seg.transpose() for seg in reversed(self.segments)),
            n_lead=self.n_lead,
            in_stacked=self.out_stacked,
            out_stacked=self.in_stacked,
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = _get_mesh()
        axis = mesh.axis_names[0]
        # The scan length is declared rather than inferred, so it also asserts the per-shard
        # leading axis of every sliced leaf. Nothing else would catch an indivisible batch axis: a
        # body whose sliced segments own no arrays gives `scan` no leaf to disagree with.
        length, remainder = divmod(self.n_lead, mesh.shape[axis])  # slices per shard
        if remainder:
            raise ValueError(
                f'batch axis {self.n_lead} is not divisible by the {mesh.shape[axis]} shards '
                f'of mesh axis {axis!r}'
            )

        # Stacked inputs ride the scan, shared ones are closed over. `eqx.partition` broadcasts a
        # prefix spec itself, so the input side needs no per-leaf mask.
        x_stacked, x_shared = eqx.partition(x, self.in_stacked)
        dyn, static = self._partition()
        self._check_shared_replicated(x_shared, static, axis)

        # Stacked outputs are emitted per step; shared ones accumulate in the carry, then psum.
        per_slice_out = self.per_slice_out_structure
        out_mask = jax.tree.broadcast(self.out_stacked, per_slice_out)
        out_pspecs = jax.tree.map(lambda stacked: P(axis) if stacked else P(), out_mask)
        _, shared_out_structure = eqx.partition(per_slice_out, out_mask)

        # One spec per argument, broadcast over that argument's subtree: the sliced segments and
        # the stacked input components carry the batch axis, the constant segments and the shared
        # components are replicated. Given rather than inferred, since the mesh may be `Auto`.
        in_pspecs = (P(axis), P(), P(axis), P())

        @jax.shard_map(in_specs=in_pspecs, out_specs=out_pspecs, check_vma=False)
        def kernel(dyn, static, x_stacked, x_shared):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                dyn_i, xs_i = args
                y = _apply_chain(dyn_i, static, eqx.combine(xs_i, x_shared))
                ys_i, y_shared = eqx.partition(y, out_mask)
                return tree.add(carry, y_shared), ys_i

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(shared_out_structure), axis, to='varying')
            carry, ys = jax.lax.scan(step, init, (dyn, x_stacked), length=length)
            return eqx.combine(ys, jax.lax.psum(carry, axis_name=axis))

        out = kernel(dyn, static, x_stacked, x_shared)
        if stream_is_split():
            # The psum reached this process's devices only; the rest of the stream lives on the
            # other processes. Stacked outputs stay process-local, shared ones are partials.
            stacked_out, shared_out = eqx.partition(out, out_mask)
            out = eqx.combine(stacked_out, cross_process_sum(shared_out))
        return out

    def _check_shared_replicated(
        self, x_shared: PyTree[Any], static: tuple[AbstractLinearOperator, ...], axis: str
    ) -> None:
        """Reject shared data sharded over the stream axis, by either route it arrives."""
        _reject_axis_sharded(x_shared, axis)  # caller-supplied components
        _reject_axis_sharded(static, axis)  # the operator's own shared-segment arrays

    @property
    def sliced_count(self) -> int:
        """How many segments carry the batch axis, i.e. how many sliced passes the body makes."""
        return sum(seg.sliced for seg in self.segments)

    def _aligned_segments(self, n_sliced: int) -> tuple[StreamSegment, ...]:
        """Pad this body onto the canonical slot pattern ``[constant, sliced, ..., constant]``.

        Laying several bodies side by side needs them to agree slot for slot, which they do not:
        one may be ``[sliced]`` and another ``[sliced, constant, sliced]``. Padding with identities
        makes them agree without changing what any of them computes.

        Alignment is cheap because `_normalize` merges adjacent segments of the same kind, so every
        body's kinds strictly alternate and the pattern follows from the sliced count alone: send
        the j-th sliced segment to slot ``2j + 1`` and the constant segments to the even slots
        around them. A sliced identity in an unused odd slot is legal -- `_check_sliceable` is
        vacuous on an operator with no array leaves -- and a slot left identity across *all* the
        bodies folds back into its neighbour, since the block operator built from it is pure.
        """
        slots: list[StreamSegment | None] = [None] * (2 * n_sliced + 1)
        seen = 0
        for seg in self.segments:
            if seg.sliced:
                slots[2 * seen + 1] = seg
                seen += 1
            else:
                slots[2 * seen] = seg  # the constant slot just left of the next sliced one
        # walk right to left, the direction values flow, so each identity gets its slot's structure
        filled: list[StreamSegment] = []
        structure = self.per_slice_in_structure
        for position, slot in reversed(list(enumerate(slots))):
            if slot is None:
                filled.append(
                    StreamSegment(IdentityOperator(in_structure=structure), position % 2 == 1)
                )
            else:
                structure = slot.out_structure
                filled.append(slot)
        return tuple(reversed(filled))

    def _partition(
        self,
    ) -> tuple[tuple[AbstractLinearOperator, ...], tuple[AbstractLinearOperator, ...]]:
        """Split each segment's operator into (dynamic, static).

        A sliced segment exposes its arrays to the scan, which takes a slice each step; a constant
        one keeps everything static, so the same operator is applied at every step.
        """
        dyn: list[AbstractLinearOperator] = []
        stat: list[AbstractLinearOperator] = []
        for seg in self.segments:
            if seg.sliced:
                dyn_i, stat_i = eqx.partition(seg.operator, eqx.is_array)
            else:
                dyn_i, stat_i = eqx.partition(seg.operator, lambda _: False)
            dyn.append(dyn_i)
            stat.append(stat_i)
        return tuple(dyn), tuple(stat)


class StreamStreamFusionRule(AbstractCompositionRule):
    """Fuse `left @ right` streams when the junction is entirely stacked.

    The *junction* is the intermediate structure the two streams meet at: `right`'s output, which
    `left` consumes as input (in mapmaking, the per-slice TOD between e.g. `Hᵀ` and `H`).

    In composition order the segment lists concatenate; the fused specs are the outer ones
    (`in = right.in_stacked`, `out = left.out_stacked`). Fusion is only valid when every junction
    component is stacked on both sides: a *shared* junction component is a psum-reduction only
    available after the full scan, so threading it through a single fused scan would be wrong
    (e.g. addition @ addition: `(Σᵢaᵢ)(Σⱼbⱼ) ≠ Σᵢ aᵢbᵢ`). Such compositions stay unreduced. Every
    composition among the four uniform layouts meets at a stacked junction, so this single rule
    handles them all without per-layout special cases.
    """

    left_operator_class = StreamOperator
    right_operator_class = StreamOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        assert isinstance(left, StreamOperator)  # mypy
        assert isinstance(right, StreamOperator)  # mypy
        # n_lead must be checked explicitly: the all-stacked test below is vacuous on a leafless
        # junction (no leaves to disagree), so it cannot catch a slot-count mismatch on its own.
        if left.n_lead != right.n_lead:
            raise NoReduction
        junction = right.per_slice_out_structure  # == left.per_slice_in_structure if it fuses
        if not structure_equal(left.per_slice_in_structure, junction):
            raise NoReduction
        left_in = jax.tree.broadcast(left.in_stacked, junction)
        right_out = jax.tree.broadcast(right.out_stacked, junction)
        if not jax.tree.all(left_in) or not jax.tree.all(right_out):
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, StreamOperator)  # mypy
        assert isinstance(right, StreamOperator)  # mypy
        segments = left.segments + right.segments
        return [
            StreamOperator.create(
                segments,
                n_lead=left.n_lead,
                in_stacked=right.in_stacked,
                out_stacked=left.out_stacked,
            )
        ]


class HomothetyStreamRule(AbstractCompositionRule):
    """`Homothety @ Stream = Stream` with the scalar attached as a constant segment.

    The scalar becomes a constant segment -- never sliced -- leading for ``Homothety @ block`` and
    trailing for ``block @ Homothety``.

    Which of those the rule actually sees is not up to the caller: a scalar commutes through a
    linear operator, and the algebra relocates it to whichever side has the smaller structure
    before this rule fires. So ``c * block`` can land on either end depending only on the block
    shape. Both spellings compute the same thing, and either way the scalar stays out of the sliced
    body, so the surrounding sum still collapses to a single fused stream via the addition rule.
    """

    operator_class = HomothetyOperator

    @staticmethod
    def _split(
        left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> tuple[HomothetyOperator, StreamOperator, bool] | None:
        """Returns `(homothety, block, on_output_side)` or `None` if the rule does not apply."""
        if isinstance(left, HomothetyOperator) and isinstance(right, StreamOperator):
            return left, right, True
        if isinstance(right, HomothetyOperator) and isinstance(left, StreamOperator):
            return right, left, False
        return None

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        if self._split(left, right) is None:
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        split = self._split(left, right)
        assert split is not None  # mypy
        homo, block, on_output_side = split
        if on_output_side:  # homo @ block: leading constant segment
            # we need the per-block structure here, not the public one with the leading axis
            scalar = HomothetyOperator(homo.value, in_structure=block.per_slice_out_structure)
            segments = (StreamSegment(scalar, False),) + block.segments
        else:  # block @ homo: trailing constant segment
            scalar = HomothetyOperator(homo.value, in_structure=block.per_slice_in_structure)
            segments = block.segments + (StreamSegment(scalar, False),)
        return [
            StreamOperator.create(
                segments,
                n_lead=block.n_lead,
                in_stacked=block.in_stacked,
                out_stacked=block.out_stacked,
            )
        ]


class StreamStreamAdditionRule(AbstractAdditionRule):
    """Fuse a sum of two matching stream operators into one stream.

    The two bodies are padded onto a common slot pattern (see `_aligned_segments`) and blocked
    up slot by slot, each slot keeping its own tag: the rightmost fans the shared input across the
    two legs through a ``BlockColumnOperator``, the leftmost sums them back through a
    ``BlockRowOperator``, and everything between acts componentwise on a ``BlockDiagonalOperator``.
    Slots that stay identity across both operands fold away in `_normalize`, so a sum of two plain
    streams still comes out as a single sliced segment.

    The operands must share n_lead, per-slice in/out structure, and both stack specs (a sum only
    fuses if both sides map the same layout); otherwise, or if neither operand has a sliced
    segment, the rule defers to a plain ``AdditionOperator``.
    """

    left_operator_class = StreamOperator
    right_operator_class = StreamOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        assert isinstance(left, StreamOperator)  # mypy
        assert isinstance(right, StreamOperator)  # mypy
        # An addition stream's structures are per-slice, so `__add__`'s structure check does not
        # force equal n; a mismatched-n sum is legal algebra that must stay unreduced. Mixed specs
        # (previously guaranteed equal by same-class dispatch) must now be checked explicitly too.
        if left.n_lead != right.n_lead:
            raise NoReduction
        per_slice_in = left.per_slice_in_structure
        per_slice_out = left.per_slice_out_structure
        if not structure_equal(per_slice_in, right.per_slice_in_structure):
            raise NoReduction
        if not structure_equal(per_slice_out, right.per_slice_out_structure):
            raise NoReduction
        if not _specs_equal(left.in_stacked, right.in_stacked, per_slice_in):
            raise NoReduction
        if not _specs_equal(left.out_stacked, right.out_stacked, per_slice_out):
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, StreamOperator)  # mypy
        assert isinstance(right, StreamOperator)  # mypy
        n_sliced = max(left.sliced_count, right.sliced_count)
        if n_sliced == 0:
            raise NoReduction  # nothing to stream: defer to a plain AdditionOperator
        # Align the two bodies slot for slot, then block them up position-wise. The rightmost slot
        # fans the shared input out over the two legs, the leftmost sums them back, and everything
        # between acts componentwise. `_normalize` folds away the slots that stay identity.
        aligned = [op._aligned_segments(n_sliced) for op in (left, right)]
        last = 2 * n_sliced
        segments = []
        for position, slot in enumerate(zip(*aligned, strict=True)):
            operators = [seg.operator for seg in slot]
            if position == 0:
                block: AbstractLinearOperator = BlockRowOperator(operators)
            elif position == last:
                block = BlockColumnOperator(operators)
            else:
                block = BlockDiagonalOperator(operators)
            segments.append(StreamSegment(block, position % 2 == 1))
        return [
            StreamOperator.create(
                tuple(segments),
                n_lead=left.n_lead,
                in_stacked=left.in_stacked,
                out_stacked=left.out_stacked,
            )
        ]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_mesh() -> AbstractMesh:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        raise RuntimeError('active mesh context required')
    return mesh


def _reject_axis_sharded(pytree: PyTree[Any], axis: str) -> None:
    """Raise if any array leaf of ``pytree`` is sharded along ``axis``."""
    # a PartitionSpec is a leaf itself; `tuple(...)` exposes its entries, and `jax.tree.leaves`
    # then flattens tuple entries and drops the None (unsharded) ones
    bad = [
        jax.tree_util.keystr(path) or '<root>'
        for path, leaf in jax.tree.leaves_with_path(pytree)
        if eqx.is_array(leaf) and axis in jax.tree.leaves(tuple(jax.typeof(leaf).sharding.spec))
    ]
    if bad:
        msg = (
            f'Found arrays sharded over {axis!r}: {", ".join(bad)}. '
            'All shared components must be replicated along the stream axis.'
        )
        raise ValueError(msg)


def _leading_size(operator: AbstractLinearOperator) -> int:
    """Batch-axis size of a *sliceable* operator: all (array) leaves share a leading axis.

    This only returns the leading axis size of the first leaf encountered. It does not check that
    all leaves have consistent dimensions (see [`_check_sliceable`][] for that).
    """
    for leaf in jax.tree.leaves(operator):
        if eqx.is_array(leaf) and jnp.ndim(leaf) >= 1:
            return jnp.shape(leaf)[0]
    raise RuntimeError('cannot infer leading axis size: no non-scalar array leaf')


def _canonical_spec(spec: StackSpec, structure: PyTree[Any]) -> StackSpec:
    """Collapse a spec that is uniform over ``structure`` to a bare bool; pass a mixed one through.

    A spec has redundant spellings -- over a two-component structure, ``True`` and ``[True, True]``
    say the same thing -- so equivalence is only decidable against the structure it applies to.

    Worth normalising because specs are static fields, hence pytree aux data, hence part of treedef
    equality: the same operator spelled two ways would otherwise land in two `jax.jit` cache
    entries. `block_column` of two column streams really does produce ``[True, True]`` where
    `column` produces ``True``.
    """
    leaves = jax.tree.leaves(jax.tree.broadcast(spec, structure))
    if not leaves:
        return spec  # no components, so no spelling to prefer
    if all(leaves):
        return True
    if not any(leaves):
        return False
    return spec


def _expand_structure(
    structure: PyTree[jax.ShapeDtypeStruct], spec: StackSpec, n_lead: int
) -> PyTree[jax.ShapeDtypeStruct]:
    """Prepend the batch axis on the stacked leaves of ``structure`` only."""
    return jax.tree.map(
        lambda stacked, s: jax.ShapeDtypeStruct((n_lead, *s.shape), s.dtype) if stacked else s,
        jax.tree.broadcast(spec, structure),
        structure,
    )


def _specs_equal(a: StackSpec, b: StackSpec, structure: PyTree[Any]) -> bool:
    """Whether two specs mark the same leaves of ``structure`` as stacked.

    Compares the expanded leaves rather than the specs themselves: `_canonical_spec` collapses
    only *uniform* specs, so two mixed specs can agree leafwise while being spelled differently.
    """
    ea = jax.tree.leaves(jax.tree.broadcast(a, structure))
    eb = jax.tree.leaves(jax.tree.broadcast(b, structure))
    return ea == eb


def _unsliceable_leaf_shape(
    operator: AbstractLinearOperator, n_lead: int
) -> tuple[int, ...] | None:
    """Shape of the first array leaf that does not lead with the batch axis, else None."""
    for leaf in jax.tree.leaves(operator):
        if eqx.is_array(leaf) and (jnp.ndim(leaf) < 1 or jnp.shape(leaf)[0] != n_lead):
            return jnp.shape(leaf)
    return None


def _is_sliceable(operator: AbstractLinearOperator, n_lead: int) -> bool:
    """Whether every array leaf leads with the batch axis, i.e. the operator can be sliced."""
    return _unsliceable_leaf_shape(operator, n_lead) is None


def _check_sliceable(operator: AbstractLinearOperator, n_lead: int) -> None:
    """Assert an operator may be sliced: every array leaf leads with the batch axis."""
    # `is not None`, not truthiness: a 0-d leaf has shape `()`, which is falsy but is a failure
    if (shape := _unsliceable_leaf_shape(operator, n_lead)) is not None:
        raise ValueError(f'expected leading axis size {n_lead=}, got shape {shape}')


def _compose(
    operators: tuple[AbstractLinearOperator, ...],
    in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
) -> AbstractLinearOperator:
    """Effective operator of a composition-ordered operator list (``operators[-1]`` applied first).

    ``in_structure`` is only needed for the empty list, where it gives the identity its structure.
    """
    if not operators:
        if in_structure is None:
            raise ValueError('_compose of an empty operator list requires in_structure')
        return IdentityOperator(in_structure=in_structure)
    return functools.reduce(lambda acc, operator: acc @ operator, operators).reduce()


def _try_merge(left: StreamSegment, right: StreamSegment, n_lead: int) -> StreamSegment | None:
    """Compose two adjacent segments into one, or return None if that would change what they do.

    Segments of the same kind always compose. Across kinds, only a *pure* segment may join its
    neighbour: it owns nothing, so being re-labelled sliced costs it nothing. A constant segment
    that owns data must stay put -- that data is applied whole to every slice, and slicing it
    instead would silently compute something else, including when its leading axis happens to
    equal `n_lead` and the sliceability check below would wave it through.

    The composed operator is checked too, since reduction can materialise data that was not there
    before: relocating a `HomothetyOperator` promotes its scalar to a 0-d array, which `scan`
    cannot slice. A scalar therefore never joins a sliced segment. That costs one constant segment,
    applied once per step and leaving the sliced count unchanged, so nothing downstream is blocked.
    """
    # composition order: left is applied later, right first
    if left.sliced != right.sliced and not (left.is_pure or right.is_pure):
        return None
    sliced = left.sliced or right.sliced
    operator = (left.operator @ right.operator).reduce()
    if sliced and not _is_sliceable(operator, n_lead):
        return None
    return StreamSegment(operator, sliced)


def _normalize(segments: tuple[StreamSegment, ...], n_lead: int) -> tuple[StreamSegment, ...]:
    """Drop constant identities and merge adjacent segments wherever that is legal."""
    merged: list[StreamSegment] = []
    for seg in segments:
        if seg.is_identity:
            continue  # contributes nothing at all
        candidate = _try_merge(merged[-1], seg, n_lead) if merged else None
        if candidate is not None:
            merged[-1] = candidate
        else:
            merged.append(seg)
    # keep at least one segment, even if all were trivial (`segments` is non-empty by contract)
    return tuple(merged) or (segments[0],)


def _apply_chain(
    dyn: tuple[AbstractLinearOperator, ...],
    static: tuple[AbstractLinearOperator, ...],
    x: PyTree[Inexact[Array, '...']],
) -> PyTree[Inexact[Array, '...']]:
    """Apply one slice's segment chain, recombining each segment from its dyn/static split.

    Segments are in composition order, so the last one is applied first (innermost).
    """
    y = x
    for dyn_i, static_i in zip(reversed(dyn), reversed(static), strict=True):
        y = eqx.combine(dyn_i, static_i)(y)
    return y
