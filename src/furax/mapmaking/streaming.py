"""Operators that stream a per-observation operator across a sharded observation axis."""

import functools
from abc import ABC
from collections.abc import Sequence
from dataclasses import field
from typing import Any, ClassVar

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

__all__ = [
    'StreamAdditionOperator',
    'StreamColumnOperator',
    'StreamDiagonalOperator',
    'StreamOperator',
    'StreamRowOperator',
    'stream_block_column',
    'stream_block_row',
]

type StackSpec = bool | PyTree[bool]
"""Obs-stackedness of a structure: a bare bool (uniform) or a prefix pytree of bools."""


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

    `in_stacked`/`out_stacked` say which *components* of the input/output carry the observation
    axis, as a [`StackSpec`][]: a bare bool applies to the whole structure, a prefix pytree of
    bools resolves per component. A stacked component is sliced on input and stacked on output; a
    shared component is broadcast on input and sum-reduced (scan carry, then `psum`) on output.
    The four uniform combinations are the named subclasses:

    | in_stacked | out_stacked | class                    | signature       |
    |------------|-------------|--------------------------|-----------------|
    | True       | True        | [`StreamDiagonalOperator`][] | (N,in) -> (N,out) |
    | False      | True        | [`StreamColumnOperator`][]   | (in,)  -> (N,out) |
    | True       | False       | [`StreamRowOperator`][]      | (N,in) -> (out,)  |
    | False      | False       | [`StreamAdditionOperator`][] | (in,)  -> (out,)  |

    Anything mixed is a [`StreamOperator`][], e.g. a joint sky (shared) + per-observation
    amplitude (stacked) system, whose single scan computes each observation's data once and
    feeds both legs.

    An active mesh context is required when calling `mv`; use `jax.set_mesh` beforehand.
    """

    segments: tuple[StreamSegment, ...]
    n_lead: int = field(kw_only=True, metadata={'static': True})
    # Static like `in_structure`, and holding pytrees just like it: never hashed (operators are
    # closed over by the solvers, not passed as jit arguments).
    in_stacked: StackSpec = field(kw_only=True, metadata={'static': True})
    out_stacked: StackSpec = field(kw_only=True, metadata={'static': True})

    # Class-canonical uniform specs; None on StreamOperator, which requires explicit specs.
    _uniform_in: ClassVar[bool | None] = None
    _uniform_out: ClassVar[bool | None] = None

    @classmethod
    def create(
        cls, operator: AbstractLinearOperator, *, n_lead: int | None = None
    ) -> 'AbstractStreamOperator':
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
    def _build(
        cls,
        segments: tuple[StreamSegment, ...],
        *,
        n_lead: int,
        in_stacked: StackSpec | None = None,
        out_stacked: StackSpec | None = None,
    ) -> 'AbstractStreamOperator':
        """Build from segments, defaulting the specs to the class's uniform ones."""
        in_stacked = cls._uniform_in if in_stacked is None else in_stacked
        out_stacked = cls._uniform_out if out_stacked is None else out_stacked
        if in_stacked is None or out_stacked is None:
            raise TypeError(f'{cls.__name__} requires explicit in_stacked/out_stacked')
        return _build_stream(
            segments, n_lead=n_lead, in_stacked=in_stacked, out_stacked=out_stacked
        )

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        per_obs_out = self.segments[0].out_structure  # leftmost segment is applied last
        return _expand_structure(per_obs_out, self.out_stacked, self.n_lead)

    @property
    def operator(self) -> AbstractLinearOperator:
        """Effective per-observation operator (composition of the segments; for introspection)."""
        # segments is never empty (see _normalize), so _compose needs no fallback structure
        return _compose(tuple(seg.operator for seg in self.segments))

    def reduce(self) -> AbstractLinearOperator:
        segments = tuple(seg.reduce() for seg in self.segments)
        return _build_stream(
            segments, n_lead=self.n_lead, in_stacked=self.in_stacked, out_stacked=self.out_stacked
        )

    def transpose(self) -> AbstractLinearOperator:
        segments = tuple(seg.transpose() for seg in reversed(self.segments))
        # Swapping the specs maps each named class to its transpose: Diagonal -> Diagonal,
        # Column <-> Row, Addition -> Addition.
        return _build_stream(
            segments, n_lead=self.n_lead, in_stacked=self.out_stacked, out_stacked=self.in_stacked
        )

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        mesh = _get_mesh()
        axis = mesh.axis_names[0]
        per_obs_in = self.segments[-1].in_structure
        per_obs_out = self.segments[0].out_structure
        in_spec = _expand_spec(self.in_stacked, per_obs_in)
        out_spec = _expand_spec(self.out_stacked, per_obs_out)
        # Stacked inputs ride the scan; shared inputs are closed over and broadcast to every step.
        x_stacked, x_shared = eqx.partition(x, in_spec)
        # Shared outputs reduce into the scan carry; stacked outputs are emitted as scan outputs.
        _, shared_out = eqx.partition(per_obs_out, out_spec)
        out_specs = jax.tree.map(lambda stacked: P(axis) if stacked else P(), out_spec)
        dyn, static = self._partition()
        # scan infers its length from the scanned leaves' (per-shard) leading axis. Only when the
        # body has no array leaves at all (e.g. a trivial F) is an explicit length needed; the
        # per-shard slot count is n_lead over the number of obs shards.
        has_scanned_leaves = bool(jax.tree.leaves((dyn, x_stacked)))
        length = None if has_scanned_leaves else self.n_lead // mesh.shape[axis]

        @jax.shard_map(out_specs=out_specs, check_vma=False)
        def kernel(dyn, static, x_stacked, x_shared):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                dyn_i, xs_i = args
                y = _apply_chain(dyn_i, static, eqx.combine(xs_i, x_shared))
                ys_i, y_shared = eqx.partition(y, out_spec)
                return tree.add(carry, y_shared), ys_i

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(shared_out), axis, to='varying')
            carry, ys = jax.lax.scan(step, init, (dyn, x_stacked), length=length)
            return eqx.combine(ys, jax.lax.psum(carry, axis_name=axis))

        return kernel(dyn, static, x_stacked, x_shared)

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

    _uniform_in = True
    _uniform_out = True


class StreamColumnOperator(AbstractStreamOperator):
    """Column operator: applies all blocks to the same input and stacks the results.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(*in,) -> (N, *out)`.

    Example — pointing matrix from pixel map to time-ordered data:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     H = StreamColumnOperator.create(pointing_op)  # leaves: (N, *out)
        ...     tod = H(pixel_map)                               # (*in,) -> (N, *out)
    """

    _uniform_in = False
    _uniform_out = True


class StreamRowOperator(AbstractStreamOperator):
    """Row operator: applies each block to its own input slice and sums the results.

    Given a per-observation operator `(*in,) -> (*out,)` with `N` slices, maps
    `(N, *in) -> (*out,)`.

    Example — co-addition of time-ordered data back to a pixel map:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     HT = StreamRowOperator.create(pointing_op_T)  # leaves: (N, *in)
        ...     pixel_map = HT(tod)                              # (N, *in) -> (*out,)
    """

    _uniform_in = True
    _uniform_out = False


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

    _uniform_in = False
    _uniform_out = False


class StreamOperator(AbstractStreamOperator):
    """General stream with per-component obs-stackedness (mixed reduce+stack scan).

    `in_stacked`/`out_stacked` are prefix pytrees of bools over the per-observation input/output
    structures: True components carry the leading observation axis (sliced on input, stacked on
    output), False components are shared (broadcast on input, sum-reduced with a `psum` on output).
    Arises from fusing a system whose unknowns mix shared and per-observation parts, e.g. a joint
    sky map (shared) and per-observation template amplitudes (stacked): the single scan computes
    each observation's data once and feeds both legs.
    """

    # _uniform_in/_uniform_out stay None: the specs must be provided explicitly.


class StreamStreamFusionRule(AbstractCompositionRule):
    """Fuse `left @ right` streams when the junction is entirely obs-stacked.

    In composition order the segment lists concatenate; the fused specs are the outer ones
    (`in = right.in_stacked`, `out = left.out_stacked`). Fusion is only valid when every junction
    component is stacked on both sides: a *shared* junction component is a psum-reduction only
    available after the full scan, so threading it through a single fused scan would be wrong
    (e.g. `StreamAddition @ StreamAddition`: `(Σᵢaᵢ)(Σⱼbⱼ) ≠ Σᵢ aᵢbᵢ`). Such compositions stay
    unreduced. The four uniform stream compositions all have stacked TOD junctions, so this one
    rule subsumes them.
    """

    left_operator_class = AbstractStreamOperator
    right_operator_class = AbstractStreamOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        # n_lead must be checked explicitly: the all-stacked test below is vacuous on a leafless
        # junction (no leaves to disagree), so it cannot catch a slot-count mismatch on its own.
        if left.n_lead != right.n_lead:
            raise NoReduction
        junction = right.segments[0].out_structure  # == left.segments[-1].in_structure if it fuses
        if not structure_equal(left.segments[-1].in_structure, junction):
            raise NoReduction
        left_in = _expand_spec(left.in_stacked, junction)
        right_out = _expand_spec(right.out_stacked, junction)
        if not all(jax.tree.leaves(left_in)) or not all(jax.tree.leaves(right_out)):
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        segments = left.segments + right.segments
        return [
            _build_stream(
                segments,
                n_lead=left.n_lead,
                in_stacked=right.in_stacked,
                out_stacked=left.out_stacked,
            )
        ]


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
        return [
            _build_stream(
                segments,
                n_lead=block.n_lead,
                in_stacked=block.in_stacked,
                out_stacked=block.out_stacked,
            )
        ]


class StreamStreamAdditionRule(AbstractAdditionRule):
    """Fuse a sum of two matching stream operators into one stream.

    Each operand is split into ``(pre, core, post)`` around its single stacked segment. When both
    have trivial (identity) shared maps the cores add directly. Otherwise the shared maps are kept
    out of the sliced body: the ``pre`` maps fan the input into a ``BlockColumnOperator``, the two
    cores sit on a ``BlockDiagonalOperator`` (the one stacked segment), and the ``post`` maps
    recombine through a ``BlockRowOperator``.

    The operands must share n_lead, per-observation in/out structure, and both stack specs (a sum
    only fuses if both sides map the same layout); otherwise, or if an operand has no single
    stacked segment, the rule defers to a plain ``AdditionOperator``.
    """

    left_operator_class = AbstractStreamOperator
    right_operator_class = AbstractStreamOperator

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        assert isinstance(left, AbstractStreamOperator)  # mypy
        assert isinstance(right, AbstractStreamOperator)  # mypy
        # StreamAddition structures are per-observation, so `__add__`'s structure check does not
        # force equal n; a mismatched-n sum is legal algebra that must stay unreduced. Mixed specs
        # (previously guaranteed equal by same-class dispatch) must now be checked explicitly too.
        if left.n_lead != right.n_lead:
            raise NoReduction
        per_obs_in = left.segments[-1].in_structure
        per_obs_out = left.segments[0].out_structure
        if not structure_equal(per_obs_in, right.segments[-1].in_structure):
            raise NoReduction
        if not structure_equal(per_obs_out, right.segments[0].out_structure):
            raise NoReduction
        if not _specs_equal(left.in_stacked, right.in_stacked, per_obs_in):
            raise NoReduction
        if not _specs_equal(left.out_stacked, right.out_stacked, per_obs_out):
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
        kw = {'in_stacked': left.in_stacked, 'out_stacked': left.out_stacked}
        trivial = all(isinstance(op, IdentityOperator) for op in (pre_l, post_l, pre_r, post_r))
        if trivial:
            core = (core_l + core_r).reduce()
            segments: tuple[StreamSegment, ...] = (StreamSegment(core, True),)
            return [_build_stream(segments, n_lead=left.n_lead, **kw)]
        pre = BlockColumnOperator([pre_l, pre_r])
        core = BlockDiagonalOperator([core_l, core_r])
        post = BlockRowOperator([post_l, post_r])
        # composition order: post @ core @ pre
        segments = (
            StreamSegment(post, False),
            StreamSegment(core, True),
            StreamSegment(pre, False),
        )
        return [_build_stream(segments, n_lead=left.n_lead, **kw)]


def stream_block_row(operands: Sequence[AbstractLinearOperator]) -> AbstractStreamOperator:
    """Fuse parallel streams ``[S₁ | S₂ | ...]`` sharing one observation axis into one stream.

    The block-row ``H`` maps a list of per-block inputs to a single shared output:
    ``H([u₁, ...]) = Σᵢ Sᵢ(uᵢ)``. Splitting each operand as ``Sᵢ = postᵢ @ coreᵢ @ preᵢ`` around
    its single stacked segment, the identity
    ``BlockRow([Sᵢ]) = BlockRow([postᵢ]) @ BlockDiagonal([coreᵢ]) @ BlockDiagonal([preᵢ])`` lays the
    fused stream out with one stacked core segment (the ``BlockDiagonal`` of cores) and the shared
    pre/post maps as shared segments. The result carries a per-block ``in_stacked`` list and the
    operands' shared ``out_stacked``.

    All operands must be stream operators sharing ``n_lead``, per-observation output structure and
    output stack spec, and have a single stacked segment. This is an explicit constructor (not a
    deferring reduction): a non-conforming operand raises ``ValueError``.
    """
    if not operands:
        raise ValueError('stream_block_row requires at least one operand')
    ops: list[AbstractStreamOperator] = []
    for op in operands:
        if not isinstance(op, AbstractStreamOperator):
            raise ValueError('stream_block_row operands must be stream operators')
        ops.append(op)
    n_lead = ops[0].n_lead
    per_obs_out = ops[0].segments[0].out_structure
    for op in ops[1:]:
        if op.n_lead != n_lead:
            raise ValueError('stream_block_row operands must share n_lead')
        if not structure_equal(op.segments[0].out_structure, per_obs_out):
            raise ValueError(
                'stream_block_row operands must share the per-observation out structure'
            )
        if not _specs_equal(op.out_stacked, ops[0].out_stacked, per_obs_out):
            raise ValueError('stream_block_row operands must share out_stacked')
    splits = [_split_core(op) for op in ops]
    conforming = [s for s in splits if s is not None]
    if len(conforming) != len(ops):
        raise ValueError('stream_block_row requires operands with a single stacked segment')
    pres, cores, posts = zip(*conforming, strict=True)
    segments: list[StreamSegment] = []
    if all(isinstance(p, IdentityOperator) for p in posts):
        # Shared posts would not merge into the stacked core (_normalize joins same-tag segments
        # only), so collapse them into a single stacked BlockRow of cores.
        segments.append(StreamSegment(BlockRowOperator(list(cores)), True))
    else:
        segments.append(StreamSegment(BlockRowOperator(list(posts)), False))
        segments.append(StreamSegment(BlockDiagonalOperator(list(cores)), True))
    if not all(isinstance(p, IdentityOperator) for p in pres):
        segments.append(StreamSegment(BlockDiagonalOperator(list(pres)), False))
    return _build_stream(
        tuple(segments),
        n_lead=n_lead,
        in_stacked=[op.in_stacked for op in ops],
        out_stacked=ops[0].out_stacked,
    )


def stream_block_column(operands: Sequence[AbstractLinearOperator]) -> AbstractStreamOperator:
    """Fuse parallel streams into one column block; the transpose of [`stream_block_row`][]."""
    transposed = stream_block_row([op.T for op in operands])
    result = transposed.T
    assert isinstance(result, AbstractStreamOperator)  # mypy
    return result


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


def _expand_spec(spec: StackSpec, structure: PyTree[Any]) -> PyTree[bool]:
    """Broadcast a bare bool or prefix pytree of bools to a per-leaf bool tree over ``structure``."""
    if isinstance(spec, bool):
        return jax.tree.map(lambda _: spec, structure)
    treedef = jax.tree.structure(spec)
    subtrees = treedef.flatten_up_to(structure)  # type: ignore[attr-defined]  # raises if not a prefix
    leaves = jax.tree.leaves(spec)
    return jax.tree.unflatten(
        treedef,
        [jax.tree.map(lambda _, s=s: s, sub) for s, sub in zip(leaves, subtrees, strict=True)],
    )


def _uniformity(spec: StackSpec, structure: PyTree[Any]) -> bool | None:
    """True/False if the expanded spec is uniformly stacked/shared; None if mixed or leafless."""
    if isinstance(spec, bool):
        return spec
    leaves = jax.tree.leaves(_expand_spec(spec, structure))
    if leaves and all(leaves):
        return True
    if leaves and not any(leaves):
        return False
    return None


def _expand_structure(
    structure: PyTree[jax.ShapeDtypeStruct], spec: StackSpec, n_lead: int
) -> PyTree[jax.ShapeDtypeStruct]:
    """Prepend the observation axis on the stacked leaves of ``structure`` only."""
    return jax.tree.map(
        lambda stacked, s: jax.ShapeDtypeStruct((n_lead, *s.shape), s.dtype) if stacked else s,
        _expand_spec(spec, structure),
        structure,
    )


def _specs_equal(a: StackSpec, b: StackSpec, structure: PyTree[Any]) -> bool:
    ea = jax.tree.leaves(_expand_spec(a, structure))
    eb = jax.tree.leaves(_expand_spec(b, structure))
    return ea == eb


def _build_stream(
    segments: tuple[StreamSegment, ...],
    *,
    n_lead: int,
    in_stacked: StackSpec,
    out_stacked: StackSpec,
) -> AbstractStreamOperator:
    """Normalize segments, pick the concrete class from the specs, and construct the operator."""
    segments = _normalize(segments)
    for seg in segments:
        if seg.stacked:
            _check_stacked(seg.operator, n_lead)
    per_obs_in = segments[-1].in_structure  # rightmost segment is applied first
    per_obs_out = segments[0].out_structure  # leftmost segment is applied last
    cls, in_stacked, out_stacked = _class_for(in_stacked, out_stacked, per_obs_in, per_obs_out)
    in_structure = _expand_structure(per_obs_in, in_stacked, n_lead)
    return cls(
        segments,
        n_lead=n_lead,
        in_stacked=in_stacked,
        out_stacked=out_stacked,
        in_structure=in_structure,
    )


# Uniform (in_stacked, out_stacked) -> named class. Anything mixed uses StreamOperator.
_UNIFORM_CLASS: dict[tuple[bool, bool], type[AbstractStreamOperator]] = {
    (True, True): StreamDiagonalOperator,
    (False, True): StreamColumnOperator,
    (True, False): StreamRowOperator,
    (False, False): StreamAdditionOperator,
}


def _class_for(
    in_stacked: StackSpec,
    out_stacked: StackSpec,
    per_obs_in: PyTree[jax.ShapeDtypeStruct],
    per_obs_out: PyTree[jax.ShapeDtypeStruct],
) -> tuple[type[AbstractStreamOperator], StackSpec, StackSpec]:
    """Pick the concrete class; normalize uniform specs to bare bools so the named classes match."""
    ui = _uniformity(in_stacked, per_obs_in)
    uo = _uniformity(out_stacked, per_obs_out)
    if ui is not None and uo is not None:
        return _UNIFORM_CLASS[(ui, uo)], ui, uo
    return StreamOperator, in_stacked, out_stacked


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
