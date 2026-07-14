import functools
from abc import ABC
from dataclasses import field
from typing import ClassVar, Self

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
from furax.core.rules import AbstractAdditionRule, AbstractCompositionRule, NoReduction


class StreamSegment(eqx.Module):
    """One segment of a stream body."""

    operator: AbstractLinearOperator
    stacked: bool = eqx.field(static=True)

    @property
    def trivial(self) -> bool:
        """A trivial segment is a shared IdentityOperator."""
        return not self.stacked and isinstance(self.operator, IdentityOperator)

    @property
    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.in_structure

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.operator.out_structure

    def transpose(self) -> 'StreamSegment':
        return StreamSegment(self.operator.T, self.stacked)

    def reduce(self) -> 'StreamSegment':
        return StreamSegment(self.operator.reduce(), self.stacked)


class AbstractStreamOperator(AbstractLinearOperator, ABC):
    """Base class for operators that apply a batched pytree operator slice-by-slice via scan.

    The effective per-observation operator is stored as a list of "segments" ([`StreamSegment`][]).
    Segments are in composition (left-to-right) order: segments[0] is applied last (output side),
    segments[-1] first (input side). Each segment is tagged obs-*stacked* (sliced per observation)
    or *shared* (broadcast across observations).

    The "block" denomination refers to how each slice appears in the global operator matrix:
    subclasses arrange the N per-slice operators as blocks of a larger matrix (diagonal, column, or
    row layout). The `_prepend_in`/`_prepend_out` class flags say whether the block prepends the
    observation axis to its input/output structure.

    An active mesh context is required when calling `mv`; use `jax.set_mesh` beforehand.
    """

    segments: tuple[StreamSegment, ...]
    n_lead: int = field(kw_only=True, metadata={'static': True})

    _prepend_in: ClassVar[bool]
    _prepend_out: ClassVar[bool]

    @classmethod
    def create(cls, operator: AbstractLinearOperator, *, n_lead: int | None = None) -> Self:
        """Wrap a freshly stacked operator as a block with a single stacked segment.

        Args:
            operator: The per-observation operator, stacked along a leading (observation) axis.
            n_lead: The observation-axis size. If not explicitly provided, is inferred from the
                operator leaves. Required if the operator has no array leaves.
        """
        if n_lead is None:
            n_lead = _leading_size(operator)
        return cls._build((StreamSegment(operator, True),), n_lead=n_lead)

    @classmethod
    def _build(cls, segments: tuple[StreamSegment, ...], *, n_lead: int) -> Self:
        segments = _normalize(segments)
        for seg in segments:
            if seg.stacked:
                _check_stacked(seg.operator, n_lead)
        per_obs_in = segments[-1].in_structure  # rightmost segment is applied first
        if cls._prepend_in:
            in_structure = _prepend_axis(per_obs_in, axis_size=n_lead)
        else:
            in_structure = per_obs_in
        return cls(segments, n_lead=n_lead, in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        per_obs_out = self.segments[0].out_structure  # leftmost segment is applied last
        if self._prepend_out:
            return _prepend_axis(per_obs_out, axis_size=self.n_lead)
        return per_obs_out

    @property
    def operator(self) -> AbstractLinearOperator:
        """Effective per-observation operator (composition of the segments; for introspection)."""
        # segments is never empty (see _normalize), so _compose needs no fallback structure
        return _compose(tuple(seg.operator for seg in self.segments))

    def reduce(self) -> AbstractLinearOperator:
        segments = tuple(seg.reduce() for seg in self.segments)
        return type(self)._build(segments, n_lead=self.n_lead)

    def _partition(
        self,
    ) -> tuple[tuple[AbstractLinearOperator, ...], tuple[AbstractLinearOperator, ...]]:
        """Split each segment's operator into (dynamic, static).

        Stacked segments expose their arrays to the scan, shared segments keep everything static
        (broadcast across observations).
        """
        dyn: list[AbstractLinearOperator] = []
        stat: list[AbstractLinearOperator] = []
        for seg in self.segments:
            if seg.stacked:
                dyn_i, stat_i = eqx.partition(seg.operator, eqx.is_array)
            else:
                dyn_i, stat_i = eqx.partition(seg.operator, lambda _: False)
            dyn.append(dyn_i)
            stat.append(stat_i)
        return tuple(dyn), tuple(stat)


class StreamDiagonalOperator(AbstractStreamOperator):
    """Block-diagonal operator: each block acts independently on its own slice of the input.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(N, *in) -> (N, *out)`.

    Example — per-observation noise weighting (square blocks, `*in == *out`):

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     W = StreamDiagonalOperator.create(noise_op)  # leaves: (N, *in)
        ...     weighted = W(samples)                           # (N, *in) -> (N, *out)
    """

    _prepend_in = True
    _prepend_out = True

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        dyn, static = self._partition()

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(dyn, static, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                dyn_i, x_i = args
                return None, _apply_chain(dyn_i, static, x_i)

            _, out = jax.lax.scan(step, None, (dyn, x))
            return out

        return kernel(dyn, static, x)

    def transpose(self) -> AbstractLinearOperator:
        segments = tuple(seg.transpose() for seg in reversed(self.segments))
        return StreamDiagonalOperator._build(segments, n_lead=self.n_lead)


class StreamColumnOperator(AbstractStreamOperator):
    """Column operator: applies all blocks to the same input and stacks the results.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(*in,) -> (N, *out)`.

    Example — pointing matrix from pixel map to time-ordered data:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     H = StreamColumnOperator.create(pointing_op)  # leaves: (N, *out)
        ...     tod = H(pixel_map)                               # (*in,) -> (N, *out)
    """

    _prepend_in = False
    _prepend_out = True

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        dyn, static = self._partition()

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(dyn, static, x):  # type: ignore[no-untyped-def]
            def step(_, dyn_i):  # type: ignore[no-untyped-def]
                return None, _apply_chain(dyn_i, static, x)

            _, out = jax.lax.scan(step, None, dyn)
            return out

        return kernel(dyn, static, x)

    def transpose(self) -> AbstractLinearOperator:
        segments = tuple(seg.transpose() for seg in reversed(self.segments))
        return StreamRowOperator._build(segments, n_lead=self.n_lead)


class StreamRowOperator(AbstractStreamOperator):
    """Row operator: applies each block to its own input slice and sums the results.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(N, *in) -> (*out,)`.

    Example — co-addition of time-ordered data back to a pixel map:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     HT = StreamRowOperator.create(pointing_op_T)  # leaves: (N, *in)
        ...     pixel_map = HT(tod)                              # (N, *in) -> (*out,)
    """

    _prepend_in = True
    _prepend_out = False

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.segments[0].out_structure
        dyn, static = self._partition()

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(dyn, static, x):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                dyn_i, x_i = args
                return tree.add(carry, _apply_chain(dyn_i, static, x_i)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, (dyn, x))
            return jax.lax.psum(out, axis_name=axis)

        return kernel(dyn, static, x)

    def transpose(self) -> AbstractLinearOperator:
        segments = tuple(seg.transpose() for seg in reversed(self.segments))
        return StreamColumnOperator._build(segments, n_lead=self.n_lead)


class StreamAdditionOperator(AbstractStreamOperator):
    """Addition operator: applies all blocks to the same input and sums the results.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(*in,) -> (*out,)`.

    Arises naturally as the reduction of `StreamRowOperator @ StreamColumnOperator`,
    e.g. the normal equations operator `H.T @ W @ H` in mapmaking.

    Example — normal equations from a pointing and weighting operator:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     A = (H.T @ W @ H).reduce()  # reduces to StreamAdditionOperator
        ...     rhs = A(pixel_map)          # (*in,) -> (*out,)
    """

    _prepend_in = False
    _prepend_out = False

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.segments[0].out_structure
        dyn, static = self._partition()

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(dyn, static, x):  # type: ignore[no-untyped-def]
            def step(carry, dyn_i):  # type: ignore[no-untyped-def]
                return tree.add(carry, _apply_chain(dyn_i, static, x)), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, dyn)
            return jax.lax.psum(out, axis_name=axis)

        return kernel(dyn, static, x)

    def transpose(self) -> AbstractLinearOperator:
        segments = tuple(seg.transpose() for seg in reversed(self.segments))
        return StreamAdditionOperator._build(segments, n_lead=self.n_lead)


class AbstractStreamFusionRule(AbstractCompositionRule):
    """Fuse a composition of two stream operators into one stream.

    In composition order the segment lists simply concatenate. The obs-independent maps that meet
    at the junction ride along as shared segments; a non-scalar junction fuses just like a scalar
    one. Whatever adjacent segments can genuinely merge do so in the reduced class constructor.
    """

    reduced_class: type[AbstractStreamOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        segments = left.segments + right.segments
        return [self.reduced_class._build(segments, n_lead=left.n_lead)]


class StreamDiagonalStreamDiagonalRule(AbstractStreamFusionRule):
    """`StreamDiagonal @ StreamDiagonal = StreamDiagonal`."""

    left_operator_class = StreamDiagonalOperator
    right_operator_class = StreamDiagonalOperator
    reduced_class = StreamDiagonalOperator


class StreamDiagonalStreamColumnRule(AbstractStreamFusionRule):
    """`StreamDiagonal @ StreamColumn = StreamColumn`."""

    left_operator_class = StreamDiagonalOperator
    right_operator_class = StreamColumnOperator
    reduced_class = StreamColumnOperator


class StreamRowStreamDiagonalRule(AbstractStreamFusionRule):
    """`StreamRow @ StreamDiagonal = StreamRow`."""

    left_operator_class = StreamRowOperator
    right_operator_class = StreamDiagonalOperator
    reduced_class = StreamRowOperator


class StreamRowStreamColumnRule(AbstractStreamFusionRule):
    """`StreamRow @ StreamColumn = StreamAddition`."""

    left_operator_class = StreamRowOperator
    right_operator_class = StreamColumnOperator
    reduced_class = StreamAdditionOperator


class HomothetyStreamRule(AbstractCompositionRule):
    """`Homothety @ Stream = Stream` with the scalar attached as a shared segment.

    The scalar becomes a shared (obs-independent) segment on the output side (``Homothety @ block``)
    or input side (``block @ Homothety``); it is not sliced. A scalar commutes through a linear
    operator, so the surrounding sum still collapses to a single fused stream via the addition-
    fusion rules.
    """

    operator_class = HomothetyOperator

    @staticmethod
    def _split(
        left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> tuple[HomothetyOperator, AbstractStreamOperator, bool] | None:
        """Returns `(homothety, block, on_output_side)` or `None` if the rule does not apply."""
        if isinstance(left, HomothetyOperator) and isinstance(right, AbstractStreamOperator):
            return left, right, True
        if isinstance(right, HomothetyOperator) and isinstance(left, AbstractStreamOperator):
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
        if on_output_side:  # homo @ block: leading shared segment
            # we need the per-block structure here, not the public one with the leading axis
            scalar = HomothetyOperator(homo.value, in_structure=block.segments[0].out_structure)
            segments = (StreamSegment(scalar, False),) + block.segments
        else:  # block @ homo: trailing shared segment
            scalar = HomothetyOperator(homo.value, in_structure=block.segments[-1].in_structure)
            segments = block.segments + (StreamSegment(scalar, False),)
        return [type(block)._build(segments, n_lead=block.n_lead)]


class AbstractStreamAdditionFusionRule(AbstractAdditionRule):
    """Fuse a sum of two stream operators of the same kind into one stream.

    Each operand is split into ``(pre, core, post)`` around its single stacked segment. When both
    have trivial (identity) shared maps the cores add directly. Otherwise the shared maps are kept
    out of the sliced body: the ``pre`` maps fan the input into a ``BlockColumnOperator``, the two
    cores sit on a ``BlockDiagonalOperator`` (the one stacked segment), and the ``post`` maps
    recombine through a ``BlockRowOperator``.

    An operand without exactly one stacked segment cannot be laid out this way, so the rule defers.
    """

    reduced_class: type[AbstractStreamOperator]

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        # Diagonal/row/column blocks carry the obs axis in their structures, so `__add__` already
        # forces equal n there; StreamAddition structures are per-observation, making a mismatched-n
        # sum legal algebra that must stay unreduced rather than crash in `_build`.
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        if left.n_lead != right.n_lead:
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        split_left = _split_core(left)
        split_right = _split_core(right)
        if split_left is None or split_right is None:
            raise NoReduction  # not a single-stacked-core body: defer to a plain AdditionOperator
        pre_l, core_l, post_l = split_left
        pre_r, core_r, post_r = split_right
        trivial = all(isinstance(op, IdentityOperator) for op in (pre_l, post_l, pre_r, post_r))
        if trivial:
            core = (core_l + core_r).reduce()
            return [self.reduced_class._build((StreamSegment(core, True),), n_lead=left.n_lead)]
        pre = BlockColumnOperator([pre_l, pre_r])
        core = BlockDiagonalOperator([core_l, core_r])
        post = BlockRowOperator([post_l, post_r])
        # composition order: post @ core @ pre
        segments = (
            StreamSegment(post, False),
            StreamSegment(core, True),
            StreamSegment(pre, False),
        )
        return [self.reduced_class._build(segments, n_lead=left.n_lead)]


class StreamDiagonalAdditionRule(AbstractStreamAdditionFusionRule):
    """`StreamDiagonal + StreamDiagonal = StreamDiagonal`."""

    left_operator_class = StreamDiagonalOperator
    right_operator_class = StreamDiagonalOperator
    reduced_class = StreamDiagonalOperator


class StreamColumnAdditionRule(AbstractStreamAdditionFusionRule):
    """`StreamColumn + StreamColumn = StreamColumn`."""

    left_operator_class = StreamColumnOperator
    right_operator_class = StreamColumnOperator
    reduced_class = StreamColumnOperator


class StreamRowAdditionRule(AbstractStreamAdditionFusionRule):
    """`StreamRow + StreamRow = StreamRow`."""

    left_operator_class = StreamRowOperator
    right_operator_class = StreamRowOperator
    reduced_class = StreamRowOperator


class StreamAdditionAdditionRule(AbstractStreamAdditionFusionRule):
    """`StreamAddition + StreamAddition = StreamAddition`."""

    left_operator_class = StreamAdditionOperator
    right_operator_class = StreamAdditionOperator
    reduced_class = StreamAdditionOperator


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_mesh() -> AbstractMesh:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        raise RuntimeError('active mesh context required')
    return mesh


def _leading_size(operator: AbstractLinearOperator) -> int:
    """Observation-axis size of a *stacked* operator: all (array) leaves share a leading axis.

    This only returns the leading axis size of the first leaf encountered. It does not check that
    all leaves have consistent dimensions (see [`_check_stacked`][] for that).
    """
    for leaf in jax.tree.leaves(operator):
        if eqx.is_array(leaf) and jnp.ndim(leaf) >= 1:
            return jnp.shape(leaf)[0]
    raise RuntimeError('cannot infer leading axis size: no non-scalar array leaf')


def _prepend_axis(
    structure: PyTree[jax.ShapeDtypeStruct], axis_size: int
) -> PyTree[jax.ShapeDtypeStruct]:
    return jax.tree.map(lambda s: jax.ShapeDtypeStruct((axis_size, *s.shape), s.dtype), structure)


def _check_stacked(operator: AbstractLinearOperator, n_lead: int) -> None:
    """Assert every array leaf of an *stacked* segment leads with the same axis size."""
    for leaf in jax.tree.leaves(operator):
        if eqx.is_array(leaf) and (jnp.ndim(leaf) < 1 or jnp.shape(leaf)[0] != n_lead):
            msg = f'expected leading axis size {n_lead=}, got shape {jnp.shape(leaf)}'
            raise ValueError(msg)


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


def _normalize(segments: tuple[StreamSegment, ...]) -> tuple[StreamSegment, ...]:
    """Drop trivial shared identities and merge consecutive same-tag segments."""
    merged: list[StreamSegment] = []
    for seg in segments:
        if seg.trivial:
            continue  # drop shared identities
        if merged and merged[-1].stacked == seg.stacked:
            # composition order: merged[-1] is to the left (applied later), seg to the right
            merged[-1] = StreamSegment((merged[-1].operator @ seg.operator).reduce(), seg.stacked)
        else:
            merged.append(seg)
    # keep at least one segment, even if all were trivial
    return tuple(merged) or (segments[0],)


def _split_core(
    op: AbstractStreamOperator,
) -> tuple[AbstractLinearOperator, AbstractLinearOperator, AbstractLinearOperator] | None:
    """Split a body into ``(pre, core, post)`` around its single stacked segment, or ``None``."""
    stacked = [i for i, seg in enumerate(op.segments) if seg.stacked]
    if len(stacked) != 1:
        return None
    i = stacked[0]
    core = op.segments[i].operator
    post = _compose(tuple(seg.operator for seg in op.segments[:i]), core.out_structure)
    pre = _compose(tuple(seg.operator for seg in op.segments[i + 1 :]), core.in_structure)
    return pre, core, post


def _apply_chain(
    dyn: tuple[AbstractLinearOperator, ...],
    static: tuple[AbstractLinearOperator, ...],
    x: PyTree[Inexact[Array, '...']],
) -> PyTree[Inexact[Array, '...']]:
    """Apply one observation's segment chain, recombining each segment from its dyn/static split.

    Segments are in composition order, so the last one is applied first (innermost).
    """
    y = x
    for dyn_i, static_i in zip(reversed(dyn), reversed(static), strict=True):
        y = eqx.combine(dyn_i, static_i)(y)
    return y
