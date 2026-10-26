"""Evaluate one computation over many members, one member at a time.

The mapmaker applies the same computation to every detector batch of an observation: weighting its
TOD, assembling the Grams of its templates, and so on. Doing it for all the detectors at once would
hold everything computed from the TOD in memory together, so it runs on one batch at a time, and
only the outputs of each batch are kept.

A `Family` is N members, each a pytree of the same structure: `DetectorFamily` the detector batches
of an observation, `StackedFamily` members stacked along an axis, `ReplicatedFamily` the same value
for every member, `ZippedFamily` the members of several families taken together. `family_apply`
runs the computation on every member in a single `jax.lax.scan`, and combines its outputs as
declared for each leaf, in rounds of members a `Policy` sets:

- `'sum'`: added over the members, such as the right-hand side of the map, or the Gram of amplitudes
  the detectors share;
- `'stack'`: kept per member, along a new leading axis, such as the Gram of each detector's own
  amplitudes;
- `'shared'`: the same for every member, such as the boresight pointing, and returned once;
- `'auto'`: `'stack'` if the leaf depends on the member, `'shared'` otherwise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal as JaxprLiteral
from jax.extend.core import jaxpr_as_fun
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int, PyTree

from furax import tree

from ._observation import ReaderField
from ._reader import slice_detectors

OutKind = Literal['sum', 'stack', 'shared']
"""How the members combine for one output leaf."""

type OutSpec = PyTree[OutKind | Literal['auto']]
"""How the members combine, per output leaf; a prefix of the output tree, such as a single kind."""


class Family(ABC):
    """N members, each a pytree of the same structure."""

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of members."""

    @abstractmethod
    def fetch(self, i: Int[Array, '']) -> PyTree[Any]:
        """Member `i`."""

    def split(self) -> Split | None:
        """The family as arrays whose entries its members are, if it is one."""
        return None


class Split(NamedTuple):
    """A family whose members are entries of arrays, along a member axis of each.

    The arrays its members take entries of, and those every member holds whole, are the arrays of
    the family; `member` holds its other leaves, such as an `Ellipsis` in an index.
    """

    stacked: list[Array]
    """The arrays the members take an entry of."""
    axes: list[int]
    """The member axis of each of `stacked`."""
    whole: list[Array]
    """The arrays every member holds whole."""
    member: Callable[[Sequence[Array], Sequence[Array]], PyTree[Any]]
    """A member, from its entries of `stacked` and from `whole`."""


@dataclass(frozen=True)
class DetectorFamily(Family):
    """The detector batches of one observation, from the fields read for it.

    Member `i` holds the per-detector fields of detectors `i · batch_size` to
    `(i + 1) · batch_size`, such as their TOD, and the fields the detectors share, such as the
    timestamps, whole.
    """

    data: dict[str, Any]
    batch_size: int
    """The number of detectors of a batch, which must divide those of the observation."""

    def __post_init__(self) -> None:
        n_dets = self._n_dets
        if self.batch_size < 1 or n_dets % self.batch_size:
            raise ValueError(f'batches of {self.batch_size} do not divide {n_dets} detectors')

    @property
    def size(self) -> int:
        return self._n_dets // self.batch_size

    def fetch(self, i: Int[Array, '']) -> dict[str, Any]:
        return slice_detectors(self.data, i * self.batch_size, self.batch_size)

    @property
    def _n_dets(self) -> int:
        per_detector = [
            value for name, value in self.data.items() if ReaderField(name).per_detector
        ]
        if not per_detector:
            raise ValueError('the observation has no per-detector field')
        return jax.tree.leaves(per_detector)[0].shape[-2]


@dataclass(frozen=True)
class StackedFamily(Family):
    """Members stacked along one axis of every leaf of a pytree, by default the leading one.

    Member `i` is entry `i` of the axis, which it drops, or with `width`, the `width` entries from
    `i · width` on, keeping the axis: the detector batches of a per-detector array, for instance.
    With `shared`, the members also hold the leaves that pytree has in place of the `None` leaves
    of `stacked`, the same for every member, such as the `'stack'` and `'shared'` outputs of
    [`family_apply`][furax.mapmaking._engine.family_apply] put back together.
    """

    stacked: PyTree[Array]
    axis: int = 0
    """The axis of the members; negative counts from the last, such as -2 for a TOD's detectors."""
    width: int | None = None
    """The number of entries of the axis per member; by default, one, and the axis is dropped."""
    shared: PyTree[Array] | None = None
    """The leaves every member holds whole, `None` where `stacked` has its own."""

    def __post_init__(self) -> None:
        entries = self._entries
        if self.width is not None and (self.width < 1 or entries % self.width):
            raise ValueError(f'members of width {self.width} do not divide {entries} entries')

    @property
    def size(self) -> int:
        return self._entries // (self.width or 1)

    def fetch(self, i: Int[Array, '']) -> PyTree[Array]:
        def member(x: Array) -> Array:
            axis = self.axis % x.ndim
            if self.width is None:
                return jax.lax.dynamic_index_in_dim(x, i, axis, keepdims=False)
            return jax.lax.dynamic_slice_in_dim(x, i * self.width, self.width, axis)

        fetched = jax.tree.map(member, self.stacked)
        if self.shared is None:
            return fetched
        return combine(fetched, self.shared)

    def split(self) -> Split | None:
        if self.width is not None:
            return None
        (stacked, whole), statics = _split_arrays((self.stacked, self.shared))
        stacked_leaves, stacked_def = jax.tree.flatten(stacked)
        whole_leaves, whole_def = jax.tree.flatten(whole)

        def member(entries, whole):
            arrays = (
                jax.tree.unflatten(stacked_def, entries),
                jax.tree.unflatten(whole_def, whole),
            )
            fetched, shared = combine(arrays, statics)
            return fetched if self.shared is None else combine(fetched, shared)

        axes = [self.axis] * len(stacked_leaves)
        return Split(stacked_leaves, axes, whole_leaves, member)

    @property
    def _entries(self) -> int:
        return jax.tree.leaves(self.stacked)[0].shape[self.axis]


@dataclass(frozen=True)
class ReplicatedFamily(Family):
    """`count` members that are all the same pytree, such as the input of every observation."""

    value: PyTree[Array]
    count: int

    @property
    def size(self) -> int:
        return self.count

    def fetch(self, i: Int[Array, '']) -> PyTree[Array]:
        return self.value

    def split(self) -> Split:
        arrays, statics = _split_arrays(self.value)
        leaves, treedef = jax.tree.flatten(arrays)
        return Split(
            [], [], leaves, lambda _, whole: combine(jax.tree.unflatten(treedef, whole), statics)
        )


@dataclass(frozen=True)
class ZippedFamily(Family):
    """Families of the same size, whose members are taken together: a member is a tuple."""

    families: tuple[Family, ...]

    def __post_init__(self) -> None:
        if len({family.size for family in self.families}) > 1:
            sizes = [family.size for family in self.families]
            raise ValueError(f'zipped families have different sizes {sizes}')

    @property
    def size(self) -> int:
        return self.families[0].size

    def fetch(self, i: Int[Array, '']) -> tuple[PyTree[Any], ...]:
        return tuple(family.fetch(i) for family in self.families)

    def split(self) -> Split | None:
        splits = [family.split() for family in self.families]
        if any(split is None for split in splits):
            return None
        parts = [split for split in splits if split is not None]
        stacked_ends = np.cumsum([0] + [len(part.stacked) for part in parts])
        whole_ends = np.cumsum([0] + [len(part.whole) for part in parts])

        def member(entries, whole):
            return tuple(
                part.member(entries[a:b], whole[c:d])
                for part, a, b, c, d in zip(
                    parts,
                    stacked_ends[:-1],
                    stacked_ends[1:],
                    whole_ends[:-1],
                    whole_ends[1:],
                    strict=True,
                )
            )

        return Split(
            [x for part in parts for x in part.stacked],
            [axis for part in parts for axis in part.axes],
            [x for part in parts for x in part.whole],
            member,
        )


@dataclass(frozen=True)
class Policy:
    """How a family is evaluated: in rounds of `batch` members, over `devices`."""

    batch: int = 1
    """The number of members of a round, which must divide the family: evaluated together,
    vectorized, rather than one after the other."""
    devices: str | None = None
    """The mesh axis to split the members over, each device evaluating its own, if any."""

    def __post_init__(self) -> None:
        if self.batch < 1:
            raise ValueError(f'batch must be >= 1, got {self.batch}')


def family_apply(
    family: Family,
    apply: Callable[[PyTree[Any]], PyTree[Array]],
    out_spec: OutSpec,
    policy: Policy | None = None,
) -> tuple[PyTree[Array], PyTree[OutKind]]:
    """`apply` on every member of a family, its outputs combined over the members.

    The members are evaluated in rounds of `policy.batch`, one after the other by one
    `jax.lax.scan`, so `apply` is compiled once. The members of a round are evaluated together,
    vectorized by `jax.vmap`, which holds the intermediate values of all of them at once: rounds of
    a single member suit computations on a TOD, a single round of every member the small ones,
    which a scan would serialize. A round of one member is `apply` itself, not vectorized.

    With `policy.devices`, the members are first split over the devices of that axis of the
    current mesh, each device evaluating its own: the family must then be a `StackedFamily` whose
    stacked leaves are sharded alike, along their leading axis, or a `ZippedFamily` of those and
    of `ReplicatedFamily`s, whose value every device holds. The arrays `apply` needs are best
    handed to it so, rather than closed over: closed-over arrays must be held whole by every
    device, as on an `Auto` mesh axis, and keep the sharding of that mesh. The `'sum'` outputs are
    summed over the devices, and held by all of them; the `'stack'` ones stay with the devices
    that computed them.

    Args:
        family: The members.
        apply: The computation, from one member to a pytree of arrays.
        out_spec: How each output leaf combines over the members: `'sum'`, `'stack'`, `'shared'`
            or, without `policy.devices`, `'auto'`, as a prefix of the output tree. A `'shared'`
            leaf must be the same for every member: the last round's is returned.
        policy: How the members are evaluated; by default, one at a time.

    Returns:
        The outputs of `apply`, the `'stack'` leaves with a leading axis of members, and the kind
        of each leaf, `'auto'` resolved.

    Raises:
        ValueError: If `out_spec` names an unknown kind or is not a prefix of the output, if the
            rounds do not divide the family, or if the members of a family that is not split
            along a leading axis are sharded.
    """
    policy = policy or Policy()
    if policy.devices is not None:
        return _sharded_apply(family, apply, out_spec, policy)
    batch = policy.batch
    n_rounds, remainder = divmod(family.size, batch)
    if remainder:
        raise ValueError(f'rounds of {batch} members do not divide a family of {family.size}')

    # A single round of a family of stacks is vectorized over the stacks themselves, rather than
    # over members fetched by index: indexing an axis sharded over devices would gather it.
    stacks = family.split() if batch == family.size > 1 else None
    if stacks is not None and not stacks.stacked:  # nothing to vectorize over
        stacks = None
    leaves = [] if stacks is None else stacks.stacked

    def body(r: Int[Array, ''], *leaves: Array) -> PyTree[Array]:
        if batch == 1:
            return apply(family.fetch(r))
        if stacks is not None:

            def member(*entries):
                return apply(stacks.member(entries, stacks.whole))

            return jax.vmap(member, in_axes=tuple(stacks.axes))(*leaves)
        members = r * batch + jnp.arange(batch, dtype=jnp.int32)
        return jax.vmap(lambda i: apply(family.fetch(i)))(members)

    # the stacked leaves are inputs, so that an output depends on the member where it does
    closed, out_shape = jax.make_jaxpr(body, return_shape=True)(jnp.zeros((), jnp.int32), *leaves)
    treedef = jax.tree.structure(out_shape)
    declared = jax.tree.leaves(jax.tree.broadcast(out_spec, out_shape))
    if unknown := set(declared) - {*get_args(OutKind), 'auto'}:
        raise ValueError(f'unknown output kinds {sorted(unknown)}')
    dependent = _depend_on_inputs(closed.jaxpr)
    kinds: list[OutKind] = [
        ('stack' if depends else 'shared') if kind == 'auto' else kind
        for kind, depends in zip(declared, dependent, strict=True)
    ]

    def split(leaves: list[Any]) -> dict[OutKind, list[Any]]:
        """The output leaves, by kind."""
        return {kind: [x for x, k in zip(leaves, kinds) if k == kind] for kind in get_args(OutKind)}

    def reduced(out: dict[OutKind, list[Any]]) -> dict[OutKind, list[Any]]:
        """The outputs of a round, its members summed or, the same for all, taken once."""
        if batch == 1:
            return out
        return {
            'sum': [x.sum(axis=0) for x in out['sum']],
            'stack': out['stack'],
            'shared': [x[0] for x in out['shared']],
        }

    # The carry starts from zeros typed like the outputs, varying over the same mesh axes, so that
    # the scan accepts it inside `shard_map`.
    types = split(closed.out_avals)
    lead = 0 if batch == 1 else 1  # the member axis of a round's outputs
    init = (
        [_zeros(aval, lead) for aval in types['sum']],
        [_zeros(aval, lead) for aval in types['shared']],
    )

    # The scan evaluates the trace already made rather than tracing `apply` again, which nested
    # families would otherwise multiply.
    traced = jaxpr_as_fun(closed)

    def step(carry, r):
        total, _ = carry
        out = reduced(split(traced(r, *leaves)))
        return (tree.add(total, out['sum']), out['shared']), out['stack']

    rounds = jnp.arange(n_rounds, dtype=jnp.int32)
    (total, shared), stacked = jax.lax.scan(step, init, rounds)
    if batch > 1:  # (round, member, ...) -> (member, ...)
        stacked = [x.reshape(family.size, *x.shape[2:]) for x in stacked]
    parts = {'sum': iter(total), 'stack': iter(stacked), 'shared': iter(shared)}
    outputs = jax.tree.unflatten(treedef, [next(parts[kind]) for kind in kinds])
    return outputs, jax.tree.unflatten(treedef, kinds)


def partition(
    outputs: PyTree[Array], kinds: PyTree[OutKind]
) -> tuple[PyTree[Array | None], PyTree[Array | None]]:
    """The `'stack'` outputs of `family_apply`, and the others.

    Each is `None` where the other has a leaf; `combine` puts them back together.
    """
    stacked = jax.tree.map(lambda x, kind: x if kind == 'stack' else None, outputs, kinds)
    others = jax.tree.map(lambda x, kind: None if kind == 'stack' else x, outputs, kinds)
    return stacked, others


def combine(*trees: PyTree[Any]) -> PyTree[Any]:
    """Pytrees of the same structure, `None` where another has a leaf, merged into one."""
    return jax.tree.map(
        lambda *leaves: next((x for x in leaves if x is not None), None),
        *trees,
        is_leaf=lambda x: x is None,
    )


def _sharded_apply(
    family: Family,
    apply: Callable[[PyTree[Any]], PyTree[Array]],
    out_spec: OutSpec,
    policy: Policy,
) -> tuple[PyTree[Array], PyTree[OutKind]]:
    """[`family_apply`][furax.mapmaking._engine.family_apply], the members split over devices.

    The family is a `StackedFamily` along a leading axis, or a `ZippedFamily` of such families and
    of `ReplicatedFamily`s, whose value every device holds whole.
    """
    axis = policy.devices
    assert axis is not None
    split = family.split()
    if split is None or not split.stacked or any(a != 0 for a in split.axes):
        raise ValueError('only families stacked along a leading axis are sharded')
    if 'auto' in jax.tree.leaves(out_spec):
        raise ValueError("the outputs of a sharded family are declared, not 'auto'")
    # a prefix of the outputs, as `out_spec` is
    out_specs = jax.tree.map(lambda kind: P(axis) if kind == 'stack' else P(), out_spec)

    # The arrays of the family, those the members take entries of split over the devices, the
    # others whole, are arguments of the shard rather than values it closes over.
    n_stacked = len(split.stacked)

    def shard(*flat: Array) -> PyTree[Array]:
        local = _SplitFamily(
            split._replace(stacked=list(flat[:n_stacked]), whole=list(flat[n_stacked:]))
        )
        out, kinds = family_apply(local, apply, out_spec, Policy(batch=policy.batch))
        return jax.tree.map(lambda x, k: jax.lax.psum(x, axis) if k == 'sum' else x, out, kinds)

    in_specs = (P(axis),) * n_stacked + (P(),) * len(split.whole)
    run = jax.shard_map(shard, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    out = run(*split.stacked, *split.whole)
    declared = jax.tree.leaves(jax.tree.broadcast(out_spec, out))
    return out, jax.tree.unflatten(jax.tree.structure(out), declared)


@dataclass(frozen=True)
class _SplitFamily(Family):
    """The family of a [`Split`][furax.mapmaking._engine.Split]: a device's members, in a shard."""

    parts: Split

    @property
    def size(self) -> int:
        return self.parts.stacked[0].shape[self.parts.axes[0]]

    def fetch(self, i: Int[Array, '']) -> PyTree[Any]:
        entries = [
            jax.lax.dynamic_index_in_dim(x, i, axis % x.ndim, keepdims=False)
            for x, axis in zip(self.parts.stacked, self.parts.axes, strict=True)
        ]
        return self.parts.member(entries, self.parts.whole)

    def split(self) -> Split:
        return self.parts


def _split_arrays(tree: PyTree[Any]) -> tuple[PyTree[Array | None], PyTree[Any]]:
    """The array leaves of a pytree, and the others, each `None` where the other has a leaf."""
    arrays = jax.tree.map(lambda x: x if _is_array(x) else None, tree)
    statics = jax.tree.map(lambda x: None if _is_array(x) else x, tree)
    return arrays, statics


def _is_array(x: Any) -> bool:
    return isinstance(x, jax.Array | np.ndarray)


def _depend_on_inputs(jaxpr: Any) -> list[bool]:
    """Which outputs of `jaxpr` depend on its inputs, rather than only on its constants.

    It is conservative: an operation with any dependent input makes all its outputs dependent.
    """
    dependent = set(jaxpr.invars)
    for eqn in jaxpr.eqns:
        if any(not isinstance(v, JaxprLiteral) and v in dependent for v in eqn.invars):
            dependent.update(eqn.outvars)
    return [not isinstance(v, JaxprLiteral) and v in dependent for v in jaxpr.outvars]


def _zeros(aval: Any, drop: int = 0) -> Array:
    """Zeros of the type of `aval`, its first `drop` axes dropped, over the same mesh axes."""
    zeros = jnp.zeros(aval.shape[drop:], aval.dtype)
    if axes := tuple(aval.manual_axis_type.varying):
        zeros = jax.lax.pcast(zeros, axes, to='varying')
    return zeros
