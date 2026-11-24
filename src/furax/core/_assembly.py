"""The assembly of operators that each act on their own parts of the vectors."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import EllipsisType
from typing import Any

import jax
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Inexact, PyTree

from furax.tree import add as tree_add
from furax.tree import zeros_like

from ._base import AbstractLinearOperator, structure_equal
from ._ensembles import Ensemble, Stride, _check_strides, _per_leaf, _stack_length, lift, map_reduce


class _AbstractAssembly(AbstractLinearOperator):
    """The tags common to the assemblies: those of their operands, under the same strides."""

    @property
    def is_square(self) -> bool:
        return super().is_square or self._witnessed('is_square')

    @property
    def is_symmetric(self) -> bool:
        return super().is_symmetric or self._witnessed('is_symmetric')

    @property
    def is_diagonal(self) -> bool:
        return super().is_diagonal or self._witnessed('is_diagonal')

    @property
    def is_positive_semidefinite(self) -> bool:
        return super().is_positive_semidefinite or self._witnessed('is_positive_semidefinite')

    @property
    def is_negative_semidefinite(self) -> bool:
        return super().is_negative_semidefinite or self._witnessed('is_negative_semidefinite')

    def _witnessed(self, tag: str) -> bool:
        """Whether the operands have `tag`, which then holds for the assembly."""
        witnesses = self._witnesses()
        return bool(witnesses) and all(getattr(op, tag) for op in witnesses)

    @abstractmethod
    def _witnesses(self) -> Sequence[AbstractLinearOperator]:
        r"""The operands whose tags hold for the assembly, none if the strides differ.

        $P_i^\top A_i Q_i$ inherits the tags of $A_i$ only with $P_i = Q_i$: with other strides on
        each side, the operands' tags say nothing of the assembly.
        """


class AssemblyOperator(_AbstractAssembly):
    r"""The assembly $\sum_i P_i^\top A_i Q_i$ of the members of an ensemble, without forming it.

    Each member $A_i$ reads its part $Q_i x$ of the input, and its output is added into its part
    of the output through $P_i^\top$, as the global matrix of finite elements is assembled from
    the element matrices. The assembly is applied, not formed: the engine evaluates the members
    with the ensemble's policy. The operands of an [`AdditionOperator`][furax.core.AdditionOperator]
    all act on the same vectors; those of an assembly act on spaces of their own, which the
    assembly lays out side by side, next to the parts they share.

    The parts are set per leaf of a member's input and output by `in_strides` and `out_strides`, a
    single value or a pytree, possibly a prefix of the member's structure (see
    [`Stride`][furax.core.Stride]):

    - None: the leaf is read whole by every member, and their outputs to it are summed;
    - 1: the leaf has a leading axis with one entry per member, which reads and writes its own;
    - `'segment'`: the leaf has one entry per segment of the ensemble, which its members read,
      and to which their outputs are summed.

    Without `out_strides`, the members write their output as they read their input. With every
    stride None, the assembly is the sum of the members. Operators of different shapes, which
    cannot be stacked into an ensemble, are assembled by
    [`RaggedAssemblyOperator`][furax.core.RaggedAssemblyOperator].

    Examples:
        Three members, each adding the map, which they all read, to its own entry of `'own'`:

        >>> import jax.numpy as jnp
        >>> from furax.core import BlockRowOperator, IdentityOperator, StackedEnsemble
        >>> I = IdentityOperator(in_structure=jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> members = StackedEnsemble(3, BlockRowOperator({'map': I, 'own': I}))
        >>> op = AssemblyOperator(members, in_strides={'map': None, 'own': 1}, out_strides=None)
        >>> op({'map': jnp.ones(2), 'own': jnp.arange(6.0).reshape(3, 2)})
        Array([ 9., 12.], dtype=float32)
    """

    operands: Ensemble
    in_strides: tuple[Stride, ...] = field(default=(), metadata={'static': True})
    """How the members read the input, one stride per leaf of a member's input."""
    out_strides: tuple[Stride, ...] = field(default=(), metadata={'static': True})
    """How the outputs of the members combine, one stride per leaf of a member's output."""
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(
        default=None, repr=False, metadata={'static': True}
    )
    """The output structure, computed once at construction."""

    def __init__(
        self,
        operands: Ensemble,
        in_strides: PyTree[Stride] = None,
        out_strides: PyTree[Stride] | EllipsisType = ...,
    ) -> None:
        if not isinstance(operands, Ensemble):
            raise TypeError(
                'the operands of an assembly are an ensemble: assemble a plain list of operators '
                'with RaggedAssemblyOperator'
            )
        # ellipsis used as a sentinel to distinguish from None, which sums the outputs
        if out_strides is ...:
            out_strides = in_strides

        def strided(structure, strides):
            # one stride per leaf of a member's structure, and the assembly's vector: an entry
            # per member or per segment on the strided leaves
            leaves, treedef = jax.tree.flatten(structure)
            flat = tuple(_per_leaf(treedef, strides, 'strides'))
            _check_strides(flat, grouped=operands.segments is not None)
            stacked = [
                leaf
                if s is None
                else jax.ShapeDtypeStruct(
                    (_stack_length(s, operands.size, operands.segments), *leaf.shape), leaf.dtype
                )
                for leaf, s in zip(leaves, flat)
            ]
            return flat, treedef.unflatten(stacked)

        member = operands.template
        ins, in_structure = strided(member.in_structure, in_strides)
        outs, out_structure = strided(member.out_structure, out_strides)
        object.__setattr__(self, 'operands', operands)
        object.__setattr__(self, 'in_strides', ins)
        object.__setattr__(self, 'out_strides', outs)
        object.__setattr__(self, '_out_structure', out_structure)
        super().__init__(in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        # how the engine combines the outputs of the members on each leaf
        combined = {None: 'sum', 1: 1, 'segment': 'segment_sum'}
        out_treedef = jax.tree.structure(self.out_structure)
        out_spec = out_treedef.unflatten([combined[s] for s in self.out_strides])
        if all(s is None for s in self.in_strides):
            # every member reads x whole: it is handed to them as is, with nothing to zip
            y, _ = map_reduce(lambda op, v: op(v), self.operands, out_spec, x)
            return y
        # each member paired with its part of x, which the engine slices out by the strides
        pairs = self.operands.zip(x, strides=self.in_strides)
        y, _ = map_reduce(lambda pair: pair[0](pair[1]), pairs, out_spec)
        return y

    def transpose(self) -> AbstractLinearOperator:
        transposed = lift(lambda op: op.T, self.operands)
        return AssemblyOperator(transposed, self.out_strides, self.in_strides)

    def reduce(self) -> AbstractLinearOperator:
        reduced = lift(lambda op: op.reduce(), self.operands)
        return AssemblyOperator(reduced, self.in_strides, self.out_strides)

    def _witnesses(self) -> Sequence[AbstractLinearOperator]:
        # the members share their class and static fields, so one answers for all of them
        return [self.operands.template] if self.in_strides == self.out_strides else []


class RaggedAssemblyOperator(_AbstractAssembly):
    r"""The assembly $\sum_g P_g^\top A_g Q_g$ of a list of operators, of different shapes.

    It is the [`AssemblyOperator`][furax.core.AssemblyOperator] of operators that cannot be stacked
    into an ensemble, applied one after the other. Their strides are None or 1, given as a prefix
    of the operands' input or output structure, and apply at the node where they are given, the
    subtrees below it differing from one operand to the next:

    - None: the operands read the same subtree whole, and their outputs to it are summed;
    - 1: the vector holds a list with one subtree per operand, as the input of a
      [`BlockDiagonalOperator`][furax.core.BlockDiagonalOperator] of the operands does, and
      operand $g$ reads and writes its entry $g$.

    Without `out_strides`, the operands write their output as they read their input. They are
    applied one at a time, so that the peak memory of an application is set by the largest
    operand rather than by all of them together; set `sequential` to False to let the compiler
    schedule them together. Operands of the same shapes are better stacked in an ensemble, which
    the engine evaluates in batches.

    Examples:
        Two operands, each adding the map to its own entry of `'own'`, a list:

        >>> import jax.numpy as jnp
        >>> from furax.core import BlockRowOperator, IdentityOperator
        >>> I = IdentityOperator(in_structure=jax.ShapeDtypeStruct((2,), jnp.float32))
        >>> rows = [
        ...     BlockRowOperator({'map': I, 'own': I}),
        ...     BlockRowOperator({'map': I, 'own': 2 * I}),
        ... ]
        >>> op = RaggedAssemblyOperator(rows, in_strides={'map': None, 'own': 1}, out_strides=None)
        >>> op({'map': jnp.ones(2), 'own': [jnp.zeros(2), jnp.ones(2)]})
        Array([4., 4.], dtype=float32)
    """

    operands: list[AbstractLinearOperator]
    in_strides: _PrefixStrides = field(metadata={'static': True})
    """How the operands read the input."""
    out_strides: _PrefixStrides = field(metadata={'static': True})
    """How the outputs of the operands combine."""
    sequential: bool = field(default=True, metadata={'static': True})
    """Whether the operands are applied one at a time."""
    _out_structure: PyTree[jax.ShapeDtypeStruct] = field(
        default=None, repr=False, metadata={'static': True}
    )
    """The output structure, computed once at construction."""

    def __init__(
        self,
        operands: Sequence[AbstractLinearOperator],
        in_strides: PyTree[Stride] = None,
        out_strides: PyTree[Stride] | EllipsisType = ...,
        sequential: bool = True,
    ) -> None:
        if isinstance(operands, Ensemble) or not isinstance(operands, Sequence) or not operands:
            raise TypeError('the operands of a ragged assembly are a list of operators')
        if out_strides is ...:
            # not given, unlike None, which sums the outputs of the operands
            out_strides = in_strides
        operands = list(operands)

        def strided(structures, strides):
            # the strides kept as given, and the assembly's vector: a list of the operands'
            # subtrees where the stride is 1, the subtree they share where it is None
            def node(s, *parts):
                if s is not None:
                    return list(parts)
                if not all(structure_equal(part, parts[0]) for part in parts[1:]):
                    raise ValueError(
                        'the operands differ on a part of stride None, which they share'
                    )
                return parts[0]

            leaves, treedef = jax.tree.flatten(strides, is_leaf=_is_none)
            for s in leaves:
                if s not in (None, 1) or isinstance(s, bool):
                    raise ValueError(
                        f'stride {s!r}: the strides of a list of operands are None or 1'
                    )
            try:
                structure = jax.tree.map(node, strides, *structures, is_leaf=_is_none)
            except (TypeError, ValueError) as error:
                raise ValueError(f'the strides do not fit the operands: {error}') from error
            return _PrefixStrides(treedef, tuple(leaves)), structure

        ins, in_structure = strided([op.in_structure for op in operands], in_strides)
        outs, out_structure = strided([op.out_structure for op in operands], out_strides)
        object.__setattr__(self, 'operands', operands)
        object.__setattr__(self, 'in_strides', ins)
        object.__setattr__(self, 'out_strides', outs)
        object.__setattr__(self, 'sequential', sequential)
        object.__setattr__(self, '_out_structure', out_structure)
        super().__init__(in_structure=in_structure)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self._out_structure

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        own_in, own_out = self.in_strides.owned(), self.out_strides.owned()

        def read(x, g):
            # what operand g reads: its own entry where it has one, the shared part elsewhere
            return jax.tree.map(lambda own, v: v[g] if own else v, own_in, x)

        def add(total, y):
            # an output, appended to the own entries and summed into the shared parts
            def added(own, t, v):
                return [*t, v] if own else tree_add(t, v)

            return jax.tree.map(added, own_out, total, y)

        # no own entries yet, and zero on the shared parts
        total = jax.tree.map(
            lambda own, s: [] if own else zeros_like(s), own_out, self.out_structure
        )
        for g, operand in enumerate(self.operands):
            if self.sequential:
                # wait for the total, so that the temporaries do not overlap
                total, x = jax.lax.optimization_barrier((total, x))
            total = add(total, operand(read(x, g)))
        return total

    def transpose(self) -> AbstractLinearOperator:
        transposed = [op.T for op in self.operands]
        ins, outs = self.in_strides.as_tree(), self.out_strides.as_tree()
        return RaggedAssemblyOperator(transposed, outs, ins, self.sequential)

    def reduce(self) -> AbstractLinearOperator:
        reduced = [op.reduce() for op in self.operands]
        ins, outs = self.in_strides.as_tree(), self.out_strides.as_tree()
        return RaggedAssemblyOperator(reduced, ins, outs, self.sequential)

    def _witnesses(self) -> Sequence[AbstractLinearOperator]:
        return self.operands if self.in_strides == self.out_strides else []


@dataclass(frozen=True)
class _PrefixStrides:
    """The strides of a ragged assembly, a prefix of its operands' structures, flattened.

    A static field of an operator is hashed by `jax.jit` to look up its compiled code, which a
    dictionary or a list of strides cannot be: they are kept as their treedef and leaves.
    """

    treedef: PyTreeDef
    leaves: tuple[Stride, ...]

    def as_tree(self) -> PyTree[Stride]:
        """The strides as given, a pytree."""
        return self.treedef.unflatten(self.leaves)

    def owned(self) -> PyTree[bool]:
        """Where each operand has its own entry, stride 1, rather than sharing the part."""
        return self.treedef.unflatten([s == 1 for s in self.leaves])


def _is_none(x: Any) -> bool:
    return x is None
