from abc import ABC
from dataclasses import field
from typing import ClassVar, Self

import jax
from jax import Array
from jax.sharding import AbstractMesh, NamedSharding
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

# jax.eval_shape inside a jax.set_mesh context annotates outputs with sharding information,
# which breaks operator compatibility checks. Wrapping inner transpose/reduction calls with
# use_abstract_mesh(_EMPTY_MESH) clears the active mesh so those structs stay sharding-free.
_EMPTY_MESH = jax.sharding.AbstractMesh((), ())


def _get_mesh() -> AbstractMesh:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        raise RuntimeError('active mesh context required')
    return mesh


def _leading_size(operator: AbstractLinearOperator) -> int:
    """Observation-axis size of a freshly *stacked* operator: every leaf shares its leading axis.

    Safe only at the stacking boundary — :meth:`AbstractScanBlockOperator.create` called on an
    operator built by stacking ``N`` per-observation operators along a new axis 0. There the
    caller's contract guarantees every leaf carries that axis, so the first array leaf's leading
    dimension is unambiguous. Downstream the size is carried explicitly (``n``), never re-inferred.
    """
    for leaf in jax.tree.leaves(operator):
        if getattr(leaf, 'ndim', 0) >= 1:
            return int(leaf.shape[0])
    raise RuntimeError('cannot infer observation-axis size: operator has no array leaf')


def _check_scanned(scanned: AbstractLinearOperator, n: int) -> None:
    """Assert the scanned body is strictly obs-stacked: every leaf leads with the obs axis ``n``.

    This is an invariant check, not a classifier. A leaf that does not lead with ``n`` means a
    non-observation operator leaked into the scanned body (a bug), and it raises loudly. Closed-over
    maps (``pre``/``post``) are deliberately *not* checked — they may carry leaves of any shape
    (scalars, shared-across-observation matrices), which is precisely the lifted limitation.
    """
    for leaf in jax.tree.leaves(scanned):
        if getattr(leaf, 'ndim', 0) < 1 or leaf.shape[0] != n:
            raise ValueError(
                f'scanned body leaf has shape {getattr(leaf, "shape", None)}, but every leaf must '
                f'lead with the observation axis of size n={n}; closed-over maps belong in pre/post'
            )


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


class AbstractScanBlockOperator(AbstractLinearOperator, ABC):
    """Base class for operators that apply a batched pytree operator slice-by-slice via scan.

    The per-observation operator is stored as three explicit pieces whose composition is the
    effective ``i``-th block ``post @ scanned_i @ pre``:

    - ``scanned``: the body, built by stacking ``N`` per-observation operators along a new axis 0.
      It is *strictly* obs-stacked — every leaf leads with that axis — and is sliced one observation
      at a time by ``jax.lax.scan``.
    - ``pre`` / ``post``: closed-over input- and output-side maps broadcast across observations (a
      scalar ``−1`` left by reduction of ``W − …``, a shared basis matrix, ...). They are *not*
      sliced and carry no observation axis; their leaves may be any shape. Two maps (rather than one)
      keep the representation closed under transpose: ``(post @ scanned)ᵀ = scannedᵀ @ postᵀ`` turns
      an output-side map into an input-side one.

    The axis size ``n`` is carried explicitly rather than re-inferred from leaf shapes, so the
    scanned/closed-over distinction never depends on a shape coincidence.

    The "block" denomination refers to how each slice appears in the global operator matrix:
    subclasses arrange the N per-slice operators as blocks of a larger matrix (diagonal, column, or
    row layout). The ``_augment_in`` / ``_augment_out`` class flags say whether the block prepends the
    observation axis to its input / output structure.

    An active mesh context is required when calling ``mv``; use ``jax.set_mesh`` beforehand.
    """

    scanned: AbstractLinearOperator
    pre: AbstractLinearOperator
    post: AbstractLinearOperator
    n: int = field(kw_only=True, metadata={'static': True})

    _augment_in: ClassVar[bool]
    _augment_out: ClassVar[bool]

    @classmethod
    def create(cls, operator: AbstractLinearOperator, *, n: int | None = None) -> Self:
        """Wrap a freshly stacked operator as a block with trivial (identity) closed-over maps.

        ``n`` is inferred from the leaves only here, at the stacking boundary (see
        :func:`_leading_size`); operator-algebra rules carry it explicitly instead.
        """
        if len(jax.tree.leaves(operator)) == 0:
            raise RuntimeError('unable to infer structures from operator with no leaf')
        if n is None:
            n = _leading_size(operator)
        pre = IdentityOperator(in_structure=operator.in_structure)
        post = IdentityOperator(in_structure=operator.out_structure)
        return cls._build(operator, pre, post, n=n)

    @classmethod
    def _build(
        cls,
        scanned: AbstractLinearOperator,
        pre: AbstractLinearOperator,
        post: AbstractLinearOperator,
        *,
        n: int,
    ) -> Self:
        _check_scanned(scanned, n)
        per_obs_in = pre.in_structure
        in_structure = (
            _augment_structure(per_obs_in, axis_size=n) if cls._augment_in else per_obs_in
        )
        return cls(scanned, pre, post, n=n, in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        per_obs_out = self.post.out_structure
        if self._augment_out:
            return _augment_structure(per_obs_out, axis_size=self.n)
        return per_obs_out

    @property
    def operator(self) -> AbstractLinearOperator:
        """Effective per-observation operator ``post @ scanned @ pre`` (for introspection)."""
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            return (self.post @ self.scanned @ self.pre).reduce()

    def reduce(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned = self.scanned.reduce()
            pre = self.pre.reduce()
            post = self.post.reduce()
        return type(self)._build(scanned, pre, post, n=self.n)


class ScanBlockDiagonalOperator(AbstractScanBlockOperator):
    """Block-diagonal operator: each block acts independently on its own slice of the input.

    Given a per-observation operator ``(*in,) -> (*out,)`` with ``N`` slices, maps
    ``(N, *in) -> (N, *out)``.

    Example — per-observation noise weighting (square blocks, ``*in == *out``):

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     W = ScanBlockDiagonalOperator.create(noise_op)  # leaves: (N, *in)
        ...     weighted = W(samples)                           # (N, *in) -> (N, *out)
    """

    _augment_in = True
    _augment_out = True

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(scanned, pre, post, x):  # type: ignore[no-untyped-def]
            def step(_, args):  # type: ignore[no-untyped-def]
                scanned_i, x_i = args
                return None, post(scanned_i(pre(x_i)))

            _, out = jax.lax.scan(step, None, (scanned, x))
            return out

        return kernel(self.scanned, self.pre, self.post, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned_t, pre_t, post_t = self.scanned.T, self.post.T, self.pre.T
        return ScanBlockDiagonalOperator._build(scanned_t, pre_t, post_t, n=self.n)


class ScanBlockColumnOperator(AbstractScanBlockOperator):
    """Column operator: applies all blocks to the same input and stacks the results.

    Given a per-observation operator ``(*in,) -> (*out,)`` with ``N`` slices, maps
    ``(*in,) -> (N, *out)``.

    Example — pointing matrix from pixel map to time-ordered data:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     H = ScanBlockColumnOperator.create(pointing_op)  # leaves: (N, *out)
        ...     tod = H(pixel_map)                               # (*in,) -> (N, *out)
    """

    _augment_in = False
    _augment_out = True

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]

        @jax.shard_map(out_specs=P(axis), check_vma=False)
        def kernel(scanned, pre, post, x):  # type: ignore[no-untyped-def]
            def step(_, scanned_i):  # type: ignore[no-untyped-def]
                return None, post(scanned_i(pre(x)))

            _, out = jax.lax.scan(step, None, scanned)
            return out

        return kernel(self.scanned, self.pre, self.post, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned_t, pre_t, post_t = self.scanned.T, self.post.T, self.pre.T
        return ScanBlockRowOperator._build(scanned_t, pre_t, post_t, n=self.n)


class ScanBlockRowOperator(AbstractScanBlockOperator):
    """Row operator: applies each block to its own input slice and sums the results.

    Given a per-observation operator ``(*in,) -> (*out,)`` with ``N`` slices, maps
    ``(N, *in) -> (*out,)``.

    Example — co-addition of time-ordered data back to a pixel map:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     HT = ScanBlockRowOperator.create(pointing_op_T)  # leaves: (N, *in)
        ...     pixel_map = HT(tod)                              # (N, *in) -> (*out,)
    """

    _augment_in = True
    _augment_out = False

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.post.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(scanned, pre, post, x):  # type: ignore[no-untyped-def]
            def step(carry, args):  # type: ignore[no-untyped-def]
                scanned_i, x_i = args
                return tree.add(carry, post(scanned_i(pre(x_i)))), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, (scanned, x))
            return jax.lax.psum(out, axis_name=axis)

        return kernel(self.scanned, self.pre, self.post, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned_t, pre_t, post_t = self.scanned.T, self.post.T, self.pre.T
        return ScanBlockColumnOperator._build(scanned_t, pre_t, post_t, n=self.n)


class ScanAdditionOperator(AbstractScanBlockOperator):
    """Addition operator: applies all blocks to the same input and sums the results.

    Given a per-observation operator ``(*in,) -> (*out,)`` with ``N`` slices, maps
    ``(*in,) -> (*out,)``.

    Arises naturally as the reduction of ``ScanBlockRowOperator @ ScanBlockColumnOperator``,
    e.g. the normal equations operator ``H.T @ W @ H`` in mapmaking.

    Example — normal equations from a pointing and weighting operator:

        >>> with jax.set_mesh(jax.make_mesh((4,), ('obs',))):
        ...     A = (H.T @ W @ H).reduce()  # reduces to ScanAdditionOperator
        ...     rhs = A(pixel_map)          # (*in,) -> (*out,)
    """

    _augment_in = False
    _augment_out = False

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        axis = _get_mesh().axis_names[0]
        out_structure = self.post.out_structure

        @jax.shard_map(out_specs=P(), check_vma=False)
        def kernel(scanned, pre, post, x):  # type: ignore[no-untyped-def]
            def step(carry, scanned_i):  # type: ignore[no-untyped-def]
                return tree.add(carry, post(scanned_i(pre(x)))), None

            # pcast makes the replicated zeros match the varying carry type inside shard_map
            init = jax.lax.pcast(tree.zeros_like(out_structure), axis, to='varying')
            out, _ = jax.lax.scan(step, init, scanned)
            return jax.lax.psum(out, axis_name=axis)

        return kernel(self.scanned, self.pre, self.post, x)

    def transpose(self) -> AbstractLinearOperator:
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned_t, pre_t, post_t = self.scanned.T, self.post.T, self.pre.T
        return ScanAdditionOperator._build(scanned_t, pre_t, post_t, n=self.n)


class AbstractScanFusionRule(AbstractCompositionRule):
    """Fuse a composition of two scan-block operators into one scan block.

    Per observation the product is ``post1 s1 pre1 @ post2 s2 pre2``. The bodies ``s1 @ s2`` fuse;
    the inner closed-over maps meet at the junction ``mid = left.pre @ right.post`` (between the two
    bodies). The rule fires only when ``mid`` commutes past a body — i.e. ``mid`` is an identity or a
    scalar :class:`~furax.core.HomothetyOperator` — in which case it slides to the output edge and
    multiplies ``post1``. The outer maps ``left.post`` / ``right.pre`` ride along as the fused block's
    ``post`` / ``pre``. A non-commuting junction defers (``NoReduction``) to an outer composition.
    """

    reduced_class: type[AbstractScanBlockOperator]

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        # The junction test lives here rather than in ``check`` so the reduction of
        # ``left.pre @ right.post`` is computed once; the registry catches a ``NoReduction``
        # raised from ``apply`` just as it does from ``check``.
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            mid = (left.pre @ right.post).reduce()
        if not isinstance(mid, (IdentityOperator, HomothetyOperator)):
            raise NoReduction
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            scanned = (left.scanned @ right.scanned).reduce()
            if isinstance(mid, HomothetyOperator):
                scalar = HomothetyOperator(mid.value, in_structure=left.post.out_structure)
                post = (scalar @ left.post).reduce()
            else:
                post = left.post
        return [self.reduced_class._build(scanned, right.pre, post, n=left.n)]


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


class HomothetyScanBlockRule(AbstractCompositionRule):
    """``Homothety @ ScanBlock = ScanBlock`` with the scalar folded into a closed-over map.

    The scalar attaches to the closed-over ``post`` (output side) or ``pre`` (input side) map
    depending on which side it sits — it is *not* folded into the scanned body, so the body stays
    strictly obs-stacked. A scalar commutes through a linear operator, so the surrounding sum still
    collapses to a single fused scan block via the addition-fusion rules.
    """

    operator_class = HomothetyOperator

    @staticmethod
    def _split(
        left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> tuple[HomothetyOperator, AbstractScanBlockOperator, bool] | None:
        """Returns ``(homothety, block, on_output_side)`` or ``None`` if the rule does not apply."""
        if isinstance(left, HomothetyOperator) and isinstance(right, AbstractScanBlockOperator):
            return left, right, True
        if isinstance(right, HomothetyOperator) and isinstance(left, AbstractScanBlockOperator):
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
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            if on_output_side:
                scalar = HomothetyOperator(homo.value, in_structure=block.post.out_structure)
                post = (scalar @ block.post).reduce()
                return [type(block)._build(block.scanned, block.pre, post, n=block.n)]
            scalar = HomothetyOperator(homo.value, in_structure=block.pre.in_structure)
            pre = (block.pre @ scalar).reduce()
            return [type(block)._build(block.scanned, pre, block.post, n=block.n)]


class AbstractScanAdditionFusionRule(AbstractAdditionRule):
    """Fuse a sum of two scan-block operators of the same kind into one scan block.

    When both operands have trivial closed-over maps the bodies add directly. Otherwise the per-branch
    maps are kept out of the strictly obs-stacked body: ``pre`` maps fan the input into a
    :class:`~furax.core.BlockColumnOperator`, the two bodies sit on a
    :class:`~furax.core.BlockDiagonalOperator`, and the ``post`` maps recombine through a
    :class:`~furax.core.BlockRowOperator`. The composite per observation is
    ``post1 s1 pre1 x + post2 s2 pre2 x`` — keeping ``W − W T G⁻¹ Tᵀ W`` a single fused block while
    the scalar/shared maps stay closed over rather than folded into the body.
    """

    reduced_class: type[AbstractScanBlockOperator]

    def check(self, left: AbstractLinearOperator, right: AbstractLinearOperator) -> None:
        super().check(left, right)
        # Diagonal/row/column blocks carry the obs axis in their structures, so ``__add__`` already
        # forces equal n there; ScanAddition structures are per-observation, making a mismatched-n
        # sum legal algebra that must stay unreduced rather than crash in ``_build``.
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        if left.n != right.n:
            raise NoReduction

    def apply(
        self, left: AbstractLinearOperator, right: AbstractLinearOperator
    ) -> list[AbstractLinearOperator]:
        assert isinstance(left, AbstractScanBlockOperator)  # mypy
        assert isinstance(right, AbstractScanBlockOperator)  # mypy
        trivial = all(
            isinstance(op, IdentityOperator) for op in (left.pre, left.post, right.pre, right.post)
        )
        with jax.sharding.use_abstract_mesh(_EMPTY_MESH):
            if trivial:
                bodies = (left.scanned + right.scanned).reduce()
                return [self.reduced_class.create(bodies, n=left.n)]
            pre = BlockColumnOperator([left.pre, right.pre])
            scanned = BlockDiagonalOperator([left.scanned, right.scanned])
            post = BlockRowOperator([left.post, right.post])
            return [self.reduced_class._build(scanned, pre, post, n=left.n)]


class ScanBlockDiagonalAdditionRule(AbstractScanAdditionFusionRule):
    """``ScanBlockDiagonal + ScanBlockDiagonal = ScanBlockDiagonal``."""

    left_operator_class = ScanBlockDiagonalOperator
    right_operator_class = ScanBlockDiagonalOperator
    reduced_class = ScanBlockDiagonalOperator


class ScanBlockColumnAdditionRule(AbstractScanAdditionFusionRule):
    """``ScanBlockColumn + ScanBlockColumn = ScanBlockColumn``."""

    left_operator_class = ScanBlockColumnOperator
    right_operator_class = ScanBlockColumnOperator
    reduced_class = ScanBlockColumnOperator


class ScanBlockRowAdditionRule(AbstractScanAdditionFusionRule):
    """``ScanBlockRow + ScanBlockRow = ScanBlockRow``."""

    left_operator_class = ScanBlockRowOperator
    right_operator_class = ScanBlockRowOperator
    reduced_class = ScanBlockRowOperator


class ScanAdditionAdditionRule(AbstractScanAdditionFusionRule):
    """``ScanAddition + ScanAddition = ScanAddition``."""

    left_operator_class = ScanAdditionOperator
    right_operator_class = ScanAdditionOperator
    reduced_class = ScanAdditionOperator
