"""One computation over every member of an ensemble, as a memory-bounded map-reduce.

An ensemble is N members sharing one structure: the same code, different array values. A member is
a pytree -- an operator, a parameter set, an operator paired with its input -- and two members only
differ by the values of their array leaves. The engine, [`map_reduce`][furax.core.map_reduce],
applies a function to every member and combines the outputs as declared per leaf: summed over the
members, or kept at the stride it varies with. The members are evaluated in rounds of
`Policy.batch` members, vectorized by `jax.vmap`, one round after the other by a single loop, and
split over the devices of a mesh axis by `jax.shard_map` when `Policy.axis` is set, with the
function traced once whatever the number of members.

The ensemble's data can be held in three ways, which decide what the engine may do with it:

- [`ListEnsemble`][furax.core.ListEnsemble]: N member pytrees, selected one at a time by a
  multiplexing `lax.switch`;
- [`StackedEnsemble`][furax.core.StackedEnsemble]: one pytree whose array leaves carry a leading
  member axis, the only representation that can be batched and sharded;
- [`HostEnsemble`][furax.core.HostEnsemble]: members read from the host, one round at a time,
  while the devices evaluate the current round.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal as JaxprLiteral
from jax.extend.core import jaxpr_as_fun
from jax.sharding import PartitionSpec as P
from jax.tree_util import GetAttrKey, register_pytree_with_keys
from jaxtyping import Array, Bool, Int, PyTree

__all__ = [
    'Ensemble',
    'HostEnsemble',
    'ListEnsemble',
    'OutSpec',
    'Policy',
    'StackedEnsemble',
    'Stride',
    'lift',
    'map_reduce',
]

type Stride = int | None
"""Members per entry on a leaf's leading axis: 1, s, or None for a leaf shared by the ensemble."""

type OutSpec = PyTree[Literal['sum', 'auto'] | Stride]
"""How the members combine, per output leaf, as a prefix of the output tree: `'sum'` reduces the
leaf over the ensemble; a stride keeps it, with one entry per member (1), per run of s members (s)
or for the whole ensemble (None); `'auto'` resolves to 1 or None from the traced computation."""


@dataclass(frozen=True)
class Policy:
    """How an ensemble is evaluated: in rounds of `batch` members, split over `axis`."""

    axis: str | None = None
    """The mesh axis the members are split over, each device evaluating its own, if any."""
    batch: int = 1
    """The number of members of a round, which must divide the members of a device: evaluated
    together, vectorized, rather than one after the other."""

    def __post_init__(self) -> None:
        if self.batch < 1:
            raise ValueError(f'batch must be >= 1, got {self.batch}')

    @classmethod
    def from_budget(
        cls,
        ensemble: 'Ensemble',
        apply: Callable[..., PyTree[Array]],
        *args: Any,
        budget: int | None = None,
        axis: str | None = None,
        margin: float = 0.8,
    ) -> 'Policy':
        """The batch size from a memory budget: the members that fit together run in parallel.

        `apply` is compiled on one abstract member and the batch size is set to
        `margin × (budget − resident) / temporaries`, clipped to the divisors of the per-device
        member count, where `resident` is the byte size of the ensemble's arrays held by one
        device and `temporaries` the temporary bytes of one member's evaluation.

        Args:
            ensemble: The members.
            apply: The computation, from one member (and `args`) to a pytree of arrays.
            *args: The shared inputs of `apply`, as `map_reduce` receives them.
            budget: The memory budget in bytes per device; by default, the device's
                `memory_stats()['bytes_limit']`, which must be given explicitly when the backend
                does not report it.
            axis: The mesh axis of the returned policy, if any.
            margin: The fraction of the free budget the batch may use.

        Raises:
            ValueError: If the backend reports no memory limit and `budget` is not given, or if
                one member does not fit in the budget.
        """
        if budget is None:
            stats = jax.devices()[0].memory_stats()
            if stats is None or 'bytes_limit' not in stats:
                raise ValueError('the backend does not report its memory: pass budget explicitly')
            budget = stats['bytes_limit']
        abstract = jax.tree.map(_abstract, ensemble.fetch(0))
        abstract_args = jax.tree.map(_abstract, args)
        compiled = jax.jit(apply).lower(abstract, *abstract_args).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            raise ValueError('the backend reports no memory analysis: pass the policy explicitly')
        temporaries = analysis.temp_size_in_bytes
        devices = 1 if axis is None else _axis_size(axis)
        per_device = ensemble.size // devices
        resident = ensemble._nbytes(devices)
        if temporaries == 0:
            return cls(axis=axis, batch=per_device)
        batch = int(margin * (budget - resident) / temporaries)
        if batch < 1:
            raise ValueError(
                f'one member does not fit: temporaries {temporaries} B, resident {resident} B, '
                f'budget {budget} B'
            )
        batch = max(b for b in range(1, min(batch, per_device) + 1) if per_device % b == 0)
        return cls(axis=axis, batch=batch)


class Ensemble(ABC):
    """N members sharing one structure: the same code, different array values.

    A member is a pytree; two members of an ensemble share their treedef, their static fields and
    their non-array leaves, and differ only by the values of their array leaves.
    """

    size: int
    """The number of members."""
    valid: Any
    """Booleans, one per member, False on padded members, or None when all members count."""
    policy: Policy
    """How [`map_reduce`][furax.core.map_reduce] walks this ensemble."""

    @abstractmethod
    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        """Member `i`."""

    @abstractmethod
    def _arrays(self) -> tuple[list[Array], list[Array]]:
        """The ensemble's arrays: those that carry the members, and those every member holds."""

    @abstractmethod
    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int
    ) -> Callable[..., PyTree[Array]]:
        """The round body `(g, *member_arrays, *fixed_arrays, *args_leaves) -> outputs`.

        The outputs carry a leading axis of `batch` members when `batch > 1`. The member-carrying
        arrays are arguments rather than closed-over values, so that an output depends on the
        member exactly where the traced computation does.
        """

    def _nbytes(self, devices: int) -> int:
        """The byte size of the arrays one device holds."""
        member, fixed = self._arrays()
        return sum(x.nbytes for x in member) // devices + sum(x.nbytes for x in fixed)


@dataclass(frozen=True)
class ListEnsemble(Ensemble):
    """N members held as N pytrees.

    The engine selects one member at a time with a multiplexing `lax.switch`: like the electronic
    component, it routes one of N inputs to a single output according to a selector. The selector
    is the loop index, the inputs are the array leaves of each member, and the output feeds the one
    compute body, which appears once in the compiled program and reuses its buffers from one round
    to the next. Nothing is copied or stacked, but the selection grows the program by a branch per
    member: fine for tens of members, not thousands. A list ensemble is evaluated one member per
    round, on a single device.
    """

    members: Sequence[PyTree[Any]]
    valid: Bool[Array, ' _n'] | None = None
    policy: Policy = Policy()

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError('an ensemble needs at least one member')
        treedef = jax.tree.structure(self.members[0])
        if any(jax.tree.structure(member) != treedef for member in self.members[1:]):
            raise ValueError('the members of an ensemble share their treedef')

    @property
    def size(self) -> int:
        return len(self.members)

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, arrays, is_array, treedef = self._parts()
        if not arrays[0]:
            return self.members[0]
        selected = jax.lax.switch(i, [lambda a=a: list(a) for a in arrays])
        return _rebuild(treedef, leaves, is_array, selected)

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        _, arrays, _, _ = self._parts()
        return [leaf for member in arrays for leaf in member], []

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int
    ) -> Callable[..., PyTree[Array]]:
        leaves, arrays, is_array, treedef = self._parts()
        n_per = len(arrays[0])
        n_members = self.size

        def body(g: Int[Array, ''], *flat: Array) -> PyTree[Array]:
            args = jax.tree.unflatten(args_def, flat[n_per * n_members :])
            if n_per == 0:
                return apply(self.members[0], *args)
            branches = [flat[j * n_per : (j + 1) * n_per] for j in range(n_members)]
            selected = jax.lax.switch(g, [lambda b=b: list(b) for b in branches])
            return apply(_rebuild(treedef, leaves, is_array, selected), *args)

        return body

    def _parts(self) -> tuple[list[Any], list[list[Array]], list[bool], Any]:
        """The leaves of member 0, each member's arrays, the array flags and the treedef."""
        leaves, treedef = jax.tree.flatten(self.members[0])
        is_array = [_is_array(leaf) for leaf in leaves]
        arrays = [
            [leaf for leaf, flag in zip(jax.tree.leaves(member), is_array) if flag]
            for member in self.members
        ]
        return leaves, arrays, is_array, treedef


@dataclass(frozen=True)
class StackedEnsemble(Ensemble):
    """N members held as one pytree whose array leaves carry a leading member axis.

    A leaf with stride s has a leading axis of `size // s` entries, one per run of s consecutive
    members; a leaf with stride None, and any non-array leaf, is shared by the whole ensemble.
    This is the only representation the engine can batch (`Policy.batch`) and shard
    (`Policy.axis`), and the one the stacked outputs of `map_reduce` come back as.
    """

    size: int
    """The number of members."""
    stacked: PyTree[Any]
    """The members' pytree; an array leaf of stride s has shape `(size // s, ...)`."""
    strides: PyTree[Stride] = 1
    """The stride of each leaf of `stacked`, or one stride for all."""
    valid: Bool[Array, ' _n'] | None = None
    policy: Policy = Policy()

    def __post_init__(self) -> None:
        for leaf, stride in zip(*self._leaves_and_strides(), strict=True):
            if stride is None:
                continue
            if self.size % stride:
                raise ValueError(f'stride {stride} does not divide {self.size} members')
            if leaf.shape[0] != self.size // stride:
                raise ValueError(
                    f'a stride-{stride} leaf of {self.size} members has '
                    f'{self.size // stride} entries, got shape {leaf.shape}'
                )

    @property
    def size_per_device(self) -> int:
        """The members of one device."""
        return self.size // (1 if self.policy.axis is None else _axis_size(self.policy.axis))

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, strides = self._leaves_and_strides()
        picked = [
            leaf if s is None else jax.lax.dynamic_index_in_dim(leaf, i // s, keepdims=False)
            for leaf, s in zip(leaves, strides)
        ]
        return jax.tree.unflatten(jax.tree.structure(self.stacked), picked)

    def shard(self, axis: str) -> 'StackedEnsemble':
        """The ensemble laid out over a mesh axis: consecutive members on each device.

        The strided leaves are split over their leading axis, the shared leaves and `valid` are
        replicated, and the returned ensemble's policy records the axis.
        """
        devices = _axis_size(axis)
        leaves, strides = self._leaves_and_strides()
        for leaf, s in zip(leaves, strides):
            if s is not None and leaf.shape[0] % devices:
                raise ValueError(f'{leaf.shape[0]} entries do not split over {devices} devices')
        placed = [
            leaf if s is None else jax.device_put(leaf, P(axis)) for leaf, s in zip(leaves, strides)
        ]
        valid = None if self.valid is None else jax.device_put(self.valid, P(axis))
        return dataclasses.replace(
            self,
            stacked=jax.tree.unflatten(jax.tree.structure(self.stacked), placed),
            valid=valid,
            policy=dataclasses.replace(self.policy, axis=axis),
        )

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        leaves, strides = self._leaves_and_strides()
        member = [leaf for leaf, s in zip(leaves, strides) if s is not None]
        fixed = [leaf for leaf, s in zip(leaves, strides) if s is None and _is_array(leaf)]
        return member, fixed

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int
    ) -> Callable[..., PyTree[Array]]:
        leaves, strides = self._leaves_and_strides()
        treedef = jax.tree.structure(self.stacked)
        member_at = [i for i, s in enumerate(strides) if s is not None]
        fixed_at = [i for i, s in enumerate(strides) if s is None and _is_array(leaves[i])]
        n_member, n_fixed = len(member_at), len(fixed_at)

        def merged(flat: tuple[Array, ...]) -> list[Any]:
            full = list(leaves)
            for i, leaf in zip(member_at + fixed_at, flat[: n_member + n_fixed]):
                full[i] = leaf
            return full

        def body(g: Int[Array, ''], *flat: Array) -> PyTree[Array]:
            args = jax.tree.unflatten(args_def, flat[n_member + n_fixed :])
            full = merged(flat)
            start = g * batch
            if batch == 1:
                picked = [
                    leaf
                    if s is None
                    else jax.lax.dynamic_index_in_dim(leaf, start // s, keepdims=False)
                    for leaf, s in zip(full, strides)
                ]
                return apply(jax.tree.unflatten(treedef, picked), *args)
            batched: list[Array | None] = []
            shared: list[Any] = []
            for leaf, s in zip(full, strides):
                if s is None:
                    batched.append(None)
                    shared.append(leaf)
                elif s >= batch:
                    batched.append(None)
                    shared.append(jax.lax.dynamic_index_in_dim(leaf, start // s, keepdims=False))
                else:
                    sliced = jax.lax.dynamic_slice_in_dim(leaf, start // s, batch // s)
                    if s > 1:
                        sliced = jnp.repeat(sliced, s, axis=0, total_repeat_length=batch)
                    batched.append(sliced)
                    shared.append(None)
            picked = [(i, leaf) for i, leaf in enumerate(batched) if leaf is not None]

            def member(*entries: Array) -> PyTree[Array]:
                full = list(shared)
                for (i, _), leaf in zip(picked, entries):
                    full[i] = leaf
                return apply(jax.tree.unflatten(treedef, full), *args)

            return jax.vmap(member)(*[leaf for _, leaf in picked])

        return body

    def _leaves_and_strides(self) -> tuple[list[Any], list[Stride]]:
        """The leaves of `stacked` and their strides, non-array leaves forced to None."""
        leaves = jax.tree.leaves(self.stacked)
        if self.strides is None or isinstance(self.strides, int):
            strides: list[Stride] = [self.strides] * len(leaves)
        else:
            strides = jax.tree.flatten(self.strides, is_leaf=lambda x: x is None)[0]
            if len(strides) != len(leaves):
                raise ValueError(
                    f'{len(strides)} strides for {len(leaves)} leaves: strides must be a single '
                    f'value or a pytree aligned with the stacked leaves'
                )
        return leaves, [s if _is_array(leaf) else None for leaf, s in zip(leaves, strides)]


@dataclass(frozen=True)
class HostEnsemble(Ensemble):
    """N members read from the host, one round at a time.

    `read(i)` returns the per-member leaves of member i as numpy arrays, in the order the
    `jax.ShapeDtypeStruct` leaves appear in `template`; the template's other leaves are shared by
    every member. The engine runs a Python loop: each round is one compiled launch on
    `Policy.batch` members per device while the host reads the next round in a thread, so the pass
    runs at the speed of the slower of the disk and the devices, not their sum. The summed
    accumulators ride the launches as donated buffers and the stacked outputs land sharded over
    the mesh axis, in position order: device d holds the outputs of members
    `d * size_per_device + [0, size_per_device)`, ready to be used as the leaves of a resident
    [`StackedEnsemble`][furax.core.StackedEnsemble] with the same layout.
    """

    size: int
    """The number of members."""
    read: Callable[[int], tuple[np.ndarray, ...]]
    """The per-member leaves of member i, called from host threads."""
    template: PyTree[Any]
    """The member structure: `jax.ShapeDtypeStruct` on the per-member leaves, values on the
    shared ones."""
    valid: np.ndarray | None = None
    policy: Policy = Policy()

    @property
    def size_per_device(self) -> int:
        """The members of one device."""
        return self.size // (1 if self.policy.axis is None else _axis_size(self.policy.axis))

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, is_member, treedef = self._parts()
        arrays = [jnp.asarray(x) for x in self.read(int(i))]
        return _rebuild(treedef, leaves, is_member, arrays)

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        leaves, is_member, _ = self._parts()
        return [], [leaf for leaf, flag in zip(leaves, is_member) if not flag and _is_array(leaf)]

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int
    ) -> Callable[..., PyTree[Array]]:
        leaves, is_member, treedef = self._parts()
        n_member = sum(is_member)
        fixed_at = [i for i, flag in enumerate(is_member) if not flag and _is_array(leaves[i])]
        n_fixed = len(fixed_at)

        def body(*flat: Array) -> PyTree[Array]:
            args = jax.tree.unflatten(args_def, flat[n_member + n_fixed :])
            full = list(leaves)
            for i, leaf in zip(fixed_at, flat[n_member : n_member + n_fixed]):
                full[i] = leaf

            def member(*arrays: Array) -> PyTree[Array]:
                return apply(_rebuild(treedef, full, is_member, list(arrays)), *args)

            if batch == 1:
                return member(*(x[0] for x in flat[:n_member]))
            return jax.vmap(member)(*flat[:n_member])

        return body

    def _parts(self) -> tuple[list[Any], list[bool], Any]:
        leaves, treedef = jax.tree.flatten(self.template)
        is_member = [isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves]
        return leaves, is_member, treedef


def map_reduce(
    apply: Callable[..., PyTree[Array]],
    ensemble: Ensemble,
    out_spec: OutSpec,
    *args: Any,
) -> tuple[PyTree[Array], PyTree[Literal['sum'] | Stride]]:
    """`apply` on every member of an ensemble, its outputs combined over the members.

    The members are evaluated in rounds of `ensemble.policy.batch`, one round after the other by a
    single loop, so `apply` is traced once whatever the ensemble size. The members of a round are
    evaluated together, vectorized by `jax.vmap`, which holds the temporaries of all of them at
    once: rounds of one member suit heavy computations, a single round of every member the small
    ones, which a loop would serialize. A round of one member is `apply` itself, not vectorized.

    With `ensemble.policy.axis`, the members are first split over the devices of that axis of the
    current mesh -- the ensemble must then be a [`StackedEnsemble`][furax.core.StackedEnsemble]
    laid out over it (see [`StackedEnsemble.shard`][furax.core.StackedEnsemble.shard]) -- and each
    device loops over its own members. The summed outputs are summed across the devices and held
    by all of them; the stacked outputs stay with the devices that computed them, in position
    order.

    Args:
        apply: The computation, from one member and `args` to a pytree of arrays.
        ensemble: The members.
        out_spec: How each output leaf combines over the members, as a prefix of the output tree:
            `'sum'` reduces it (padded members masked); stride 1 stacks one entry per member along
            a new leading axis; stride s keeps one entry per run of s members sharing it; stride
            None keeps one entry, the same for every member. `'auto'` resolves to 1 or None by
            whether the leaf depends on the member in the traced computation. A declared stride is
            a promise the engine does not check.
        *args: The shared inputs, which every member receives whole.

    Returns:
        The outputs of `apply` combined over the members, and the output specification with
        `'auto'` resolved.

    Raises:
        ValueError: If `out_spec` names an unknown kind or is not a prefix of the output, if the
            policy does not divide the ensemble, or if a sharded or batched ensemble is not a
            `StackedEnsemble`.
    """
    if isinstance(ensemble, HostEnsemble):
        return _host_map_reduce(apply, ensemble, out_spec, args)
    return _resident_map_reduce(apply, ensemble, out_spec, args)


def lift(f: Callable[..., PyTree[Any]], *ensembles: Ensemble) -> Ensemble:
    """Member-wise `f` over ensembles of the same size: the algebra of ensembles.

    For list ensembles, `f` is applied to the member pytrees. For stacked ensembles, a
    **structural** `f` -- one that only rearranges its members' leaves, such as composing or
    transposing operators -- is detected by tracking leaf identity, and the output ensemble reuses
    the input arrays with their strides, at no cost. Otherwise `f` is vectorized by `jax.vmap`
    over the member axis, stride-s leaves expanded to one entry per member, and the output leaves
    come back with stride 1.

    Args:
        f: From one member per ensemble to a pytree: an operator constructor, a composition,
            a reduction rule.
        *ensembles: The members, all list or all stacked, of one size.

    Returns:
        The ensemble of `f`'s outputs, with `f`'s non-array leaves shared.
    """
    if not ensembles:
        raise ValueError('lift needs at least one ensemble')
    size = ensembles[0].size
    if any(e.size != size for e in ensembles):
        raise ValueError(f'lifted ensembles have different sizes {[e.size for e in ensembles]}')
    lists = [e for e in ensembles if isinstance(e, ListEnsemble)]
    if len(lists) == len(ensembles):
        members = [f(*(e.members[i] for e in lists)) for i in range(size)]
        return ListEnsemble(members, valid=lists[0].valid, policy=lists[0].policy)
    stacked = [e for e in ensembles if isinstance(e, StackedEnsemble)]
    if len(stacked) != len(ensembles):
        raise ValueError('lift takes ensembles that are all list or all stacked')
    return _lift_stacked(f, stacked)


def _lift_stacked(f: Callable[..., PyTree[Any]], ensembles: Sequence[StackedEnsemble]) -> Ensemble:
    structural = _lift_structural(f, ensembles)
    if structural is not None:
        return structural
    size = ensembles[0].size

    def on_members(flats: list[list[Any]]) -> list[Array]:
        members = [
            jax.tree.unflatten(jax.tree.structure(e.stacked), flat)
            for e, flat in zip(ensembles, flats)
        ]
        return [leaf for leaf in jax.tree.leaves(f(*members)) if _is_array(leaf)]

    flats, axes = [], []
    for e in ensembles:
        leaves, strides = e._leaves_and_strides()
        flats.append(
            [leaf if s is None else _expanded(leaf, s, size) for leaf, s in zip(leaves, strides)]
        )
        axes.append([None if s is None else 0 for s in strides])
    out_arrays = jax.vmap(on_members, in_axes=(axes,))(flats)

    member0 = f(*(e.fetch(0) for e in ensembles))
    out_leaves, out_treedef = jax.tree.flatten(member0)
    is_array = [_is_array(leaf) for leaf in out_leaves]
    stacked = _rebuild(out_treedef, out_leaves, is_array, list(out_arrays))
    strides = jax.tree.unflatten(out_treedef, [1 if flag else None for flag in is_array])
    return StackedEnsemble(
        size, stacked, strides, valid=ensembles[0].valid, policy=ensembles[0].policy
    )


def _lift_structural(
    f: Callable[..., PyTree[Any]], ensembles: Sequence[StackedEnsemble]
) -> StackedEnsemble | None:
    """`f` applied once to abstract members, the output leaves matched to the inputs by identity.

    A structural `f` passes its members' array leaves through untouched, so every array leaf of
    its output is one of the sentinel inputs: the output ensemble reuses the stacked arrays with
    their strides. Any computed array leaf, or any error on abstract leaves, falls back to None.
    """
    sources: dict[int, tuple[Array, Stride]] = {}
    members = []
    for e in ensembles:
        leaves, strides = e._leaves_and_strides()
        sentinels = []
        for leaf, s in zip(leaves, strides):
            if _is_array(leaf):
                shape = leaf.shape if s is None else leaf.shape[1:]
                sentinel = jax.ShapeDtypeStruct(shape, leaf.dtype)
                sources[id(sentinel)] = (leaf, s)
                sentinels.append(sentinel)
            else:
                sentinels.append(leaf)
        members.append(jax.tree.unflatten(jax.tree.structure(e.stacked), sentinels))
    try:
        out = f(*members)
    except (AttributeError, TypeError, ValueError):
        # f computes on its leaves, which the abstract sentinels cannot do: not structural.
        return None
    out_leaves, out_treedef = jax.tree.flatten(out)
    stacked, strides = [], []
    for leaf in out_leaves:
        if id(leaf) in sources:
            source, stride = sources[id(leaf)]
            stacked.append(source)
            strides.append(stride)
        elif isinstance(leaf, jax.ShapeDtypeStruct) or _is_array(leaf):
            return None  # a computed array leaf: f is not structural
        else:
            stacked.append(leaf)
            strides.append(None)
    return StackedEnsemble(
        ensembles[0].size,
        jax.tree.unflatten(out_treedef, stacked),
        jax.tree.unflatten(out_treedef, strides),
        valid=ensembles[0].valid,
        policy=ensembles[0].policy,
    )


def _resident_map_reduce(
    apply: Callable[..., PyTree[Array]],
    ensemble: Ensemble,
    out_spec: OutSpec,
    args: tuple[Any, ...],
) -> tuple[PyTree[Array], PyTree[Literal['sum'] | Stride]]:
    policy = ensemble.policy
    batch = policy.batch
    if not isinstance(ensemble, StackedEnsemble):
        if policy.axis is not None:
            raise ValueError('only a stacked ensemble is laid out over a mesh axis')
        if batch != 1:
            raise ValueError('only a stacked ensemble is evaluated in batched rounds')
    devices = 1 if policy.axis is None else _axis_size(policy.axis)
    per_device, remainder = divmod(ensemble.size, devices)
    if remainder:
        raise ValueError(f'{devices} devices do not divide {ensemble.size} members')
    rounds, remainder = divmod(per_device, batch)
    if remainder:
        raise ValueError(f'rounds of {batch} members do not divide {per_device} members')

    member_arrays, fixed_arrays = ensemble._arrays()
    args_leaves, args_def = jax.tree.flatten(args)
    body = ensemble._body(apply, args_def, batch)

    # The round body is traced once, on per-device shapes, with the member-carrying arrays as
    # inputs; the loop then evaluates the trace, and `'auto'` is resolved on it before any
    # collective is laid out.
    local = [
        jax.ShapeDtypeStruct((x.shape[0] // devices, *x.shape[1:]), x.dtype) for x in member_arrays
    ]
    closed, out_shape = jax.make_jaxpr(body, return_shape=True)(
        jnp.zeros((), jnp.int32), *local, *fixed_arrays, *args_leaves
    )
    kinds = _resolved_kinds(out_spec, out_shape, closed, 1 + len(member_arrays), per_device, batch)

    valid = ensemble.valid
    if valid is not None and not any(kind == 'sum' for kind in kinds):
        valid = None

    def run(
        member: list[Array], use: Array | None, fixed_and_args: list[Array]
    ) -> tuple[list[Array], ...]:
        local = closed
        if policy.axis is not None:
            # Avals are typed by the mesh context, so the trace made outside, under the explicit
            # mesh, cannot be evaluated inside the shard: trace the same body again in its context.
            local = jax.make_jaxpr(body)(jnp.zeros((), jnp.int32), *member, *fixed_and_args)
        return _loop(local, kinds, batch, rounds, per_device, [*member, *fixed_and_args], use)

    if policy.axis is None:
        sums, shareds, stacks = run(member_arrays, valid, [*fixed_arrays, *args_leaves])
    else:
        axis = policy.axis
        n_member = len(member_arrays)
        has_valid = valid is not None

        def block(*flat: Array) -> tuple[Array, ...]:
            member = list(flat[:n_member])
            use = flat[n_member] if has_valid else None
            rest = list(flat[n_member + has_valid :])
            sums, shareds, stacks = run(member, use, rest)
            sums = [jax.lax.psum(x, axis) for x in sums]
            return (*sums, *shareds, *stacks)

        n_sums = sum(kind == 'sum' for kind in kinds)
        n_stacks = sum(isinstance(kind, int) for kind in kinds)
        n_shareds = len(kinds) - n_sums - n_stacks
        in_specs = (
            (P(axis),) * n_member
            + ((P(axis),) if has_valid else ())
            + (P(),) * (len(fixed_arrays) + len(args_leaves))
        )
        out_specs = (P(),) * (n_sums + n_shareds) + (P(axis),) * n_stacks
        flat = jax.shard_map(block, in_specs=in_specs, out_specs=out_specs, check_vma=False)(
            *member_arrays, *([valid] if has_valid else []), *fixed_arrays, *args_leaves
        )
        sums = list(flat[:n_sums])
        shareds = list(flat[n_sums : n_sums + n_shareds])
        stacks = list(flat[n_sums + n_shareds :])

    return _unflattened(out_shape, kinds, sums, shareds, stacks)


def _host_map_reduce(
    apply: Callable[..., PyTree[Array]],
    ensemble: HostEnsemble,
    out_spec: OutSpec,
    args: tuple[Any, ...],
) -> tuple[PyTree[Array], PyTree[Literal['sum'] | Stride]]:
    policy = ensemble.policy
    batch = policy.batch
    axis = policy.axis
    devices = 1 if axis is None else _axis_size(axis)
    per_device, remainder = divmod(ensemble.size, devices)
    if remainder:
        raise ValueError(f'{devices} devices do not divide {ensemble.size} members')
    rounds, remainder = divmod(per_device, batch)
    if remainder:
        raise ValueError(f'rounds of {batch} members do not divide {per_device} members')

    leaves, is_member, _ = ensemble._parts()
    member_structs = [leaf for leaf, flag in zip(leaves, is_member) if flag]
    _, fixed_arrays = ensemble._arrays()
    args_leaves, args_def = jax.tree.flatten(args)
    body = ensemble._body(apply, args_def, batch)

    round_structs = [jax.ShapeDtypeStruct((batch, *x.shape), x.dtype) for x in member_structs]
    closed, out_shape = jax.make_jaxpr(body, return_shape=True)(
        *round_structs, *fixed_arrays, *args_leaves
    )
    kinds = _resolved_kinds(out_spec, out_shape, closed, len(member_structs), per_device, batch)
    traced = jaxpr_as_fun(closed)
    member_shapes = [
        (aval.shape[1:] if batch > 1 else aval.shape, aval.dtype) for aval in closed.out_avals
    ]

    valid = ensemble.valid
    if valid is not None and not any(kind == 'sum' for kind in kinds):
        valid = None

    def step(
        carry: tuple[list[Array], list[Array]],
        g: Int[Array, ''],
        round_arrays: list[Array],
        use: Array | None,
        fixed_and_args: list[Array],
    ) -> tuple[tuple[list[Array], list[Array]], list[Array]]:
        # The summed carries have a leading device axis, so each device accumulates its partial.
        # Inside a shard the mesh context differs from the trace's, so the body is traced again.
        run = traced
        if axis is not None:
            run = jaxpr_as_fun(jax.make_jaxpr(body)(*round_arrays, *fixed_and_args))
        sums, stacks = carry
        out = list(run(*round_arrays, *fixed_and_args))
        new_sums, new_stacks, shareds = _combined(
            out, kinds, [x[0] for x in sums], stacks, g, batch, use
        )
        return ([x[None] for x in new_sums], new_stacks), shareds

    launch: Callable[..., Any] = step
    if axis is not None:
        launch = jax.shard_map(
            step,
            in_specs=((P(axis), P(axis)), P(), P(axis), P(axis), P()),
            out_specs=((P(axis), P(axis)), P()),
            check_vma=False,
        )
    launch = jax.jit(launch, donate_argnums=0)

    init_sums = [
        jnp.zeros((devices, *shape), dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if kind == 'sum'
    ]
    init_stacks = [
        jnp.zeros((ensemble.size // kind, *shape), dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if isinstance(kind, int)
    ]
    if axis is not None:
        init_sums = [jax.device_put(x, P(axis)) for x in init_sums]
        init_stacks = [jax.device_put(x, P(axis)) for x in init_stacks]

    def read_round(g: int) -> tuple[list[np.ndarray], np.ndarray | None]:
        # device d evaluates positions d * per_device + g * batch + [0, batch)
        positions = [d * per_device + g * batch + i for d in range(devices) for i in range(batch)]
        with ThreadPoolExecutor(max(devices, 1)) as pool:
            members = list(pool.map(ensemble.read, positions))
        arrays = [np.stack([m[j] for m in members]) for j in range(len(member_structs))]
        use = None if valid is None else np.asarray([valid[p] for p in positions])
        return arrays, use

    def put(x: np.ndarray) -> Array:
        return jnp.asarray(x) if axis is None else jax.device_put(x, P(axis))

    def to_device(
        arrays: list[np.ndarray], use: np.ndarray | None
    ) -> tuple[list[Array], Array | None]:
        return [put(x) for x in arrays], None if use is None else put(use)

    carry = (init_sums, init_stacks)
    shareds: list[Array] = []
    pending = read_round(0)
    for g in range(rounds):
        round_arrays, use = to_device(*pending)
        carry, shareds = launch(
            carry, jnp.int32(g), round_arrays, use, [*fixed_arrays, *args_leaves]
        )
        if g + 1 < rounds:
            with ThreadPoolExecutor(1) as pool:  # read the next round while the devices compute
                future = pool.submit(read_round, g + 1)
                jax.block_until_ready(carry)
                pending = future.result()
    partials, stacks = carry
    sums = [x.sum(axis=0) for x in partials]  # one partial per device
    return _unflattened(out_shape, kinds, sums, shareds, stacks)


def _loop(
    closed: Any,
    kinds: list[Any],
    batch: int,
    rounds: int,
    per_device: int,
    inputs: list[Array],
    valid: Array | None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """The loop over the rounds of one device: evaluate the trace, combine by kind."""
    traced = jaxpr_as_fun(closed)
    member_shapes = [
        (aval.shape[1:] if batch > 1 else aval.shape, aval.dtype) for aval in closed.out_avals
    ]
    init_sums = [
        jnp.zeros(shape, dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if kind == 'sum'
    ]
    init_shareds = [
        jnp.zeros(shape, dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if kind is None
    ]
    init_stacks = [
        jnp.zeros((per_device // kind, *shape), dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if isinstance(kind, int)
    ]

    def step(
        g: Int[Array, ''], carry: tuple[list[Array], list[Array], list[Array]]
    ) -> tuple[list[Array], list[Array], list[Array]]:
        sums, _, stacks = carry  # the shared slots are replaced, the last round's survive
        out = list(traced(jnp.asarray(g, jnp.int32), *inputs))  # traced with an int32 index
        use: Array | None = None
        if valid is not None:
            if batch == 1:
                use = jax.lax.dynamic_index_in_dim(valid, g, keepdims=False)
            else:
                use = jax.lax.dynamic_slice_in_dim(valid, g * batch, batch)
        new_sums, new_stacks, new_shareds = _combined(out, kinds, sums, stacks, g, batch, use)
        return new_sums, new_shareds, new_stacks

    sums, shareds, stacks = jax.lax.fori_loop(
        0, rounds, step, (init_sums, init_shareds, init_stacks)
    )
    return sums, shareds, stacks


def _combined(
    out: list[Array],
    kinds: list[Any],
    sums: list[Array],
    stacks: list[Array],
    g: Int[Array, ''],
    batch: int,
    use: Array | None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """One round's outputs folded into the carries: sums masked and added, stacks written."""
    new_sums, new_stacks, shareds = [], [], []
    it_sums, it_stacks = iter(sums), iter(stacks)
    for leaf, kind in zip(out, kinds):
        if kind == 'sum':
            if use is not None:
                mask = use if batch == 1 else use.reshape((batch,) + (1,) * (leaf.ndim - 1))
                leaf = jnp.where(mask, leaf, 0)
            contribution = leaf if batch == 1 else leaf.sum(axis=0)
            new_sums.append(next(it_sums) + contribution)
        elif kind is None:
            shareds.append(leaf if batch == 1 else leaf[0])
        else:
            if batch == 1:
                chunk = leaf[None]
            elif kind >= batch:
                chunk = leaf[0][None]
            else:
                chunk = leaf[::kind]
            new_stacks.append(
                jax.lax.dynamic_update_slice_in_dim(next(it_stacks), chunk, g * batch // kind, 0)
            )
    return new_sums, new_stacks, shareds


def _resolved_kinds(
    out_spec: OutSpec,
    out_shape: PyTree[Any],
    closed: Any,
    n_seeds: int,
    per_device: int,
    batch: int,
) -> list[Any]:
    """The declared kinds broadcast over the output leaves, `'auto'` resolved from the trace."""
    declared = _spec_leaves(out_spec, out_shape)
    for kind in declared:
        if kind in ('sum', 'auto') or kind is None:
            continue
        if isinstance(kind, int) and not isinstance(kind, bool) and kind >= 1:
            if per_device % kind:
                raise ValueError(f'stride {kind} does not divide {per_device} members per device')
            if kind % batch and batch % kind:
                raise ValueError(f'stride {kind} is not aligned with rounds of {batch} members')
            continue
        raise ValueError(f'unknown output kind {kind!r}')
    if 'auto' not in declared:
        return list(declared)
    dependent = _dependent_outputs(closed.jaxpr, n_seeds)
    return [
        (1 if depends else None) if kind == 'auto' else kind
        for kind, depends in zip(declared, dependent, strict=True)
    ]


def _dependent_outputs(jaxpr: Any, n_seeds: int) -> list[bool]:
    """Which outputs depend on the first `n_seeds` inputs, rather than only on the others.

    One forward pass in program order: an operation with any dependent input makes all its
    outputs dependent. It is conservative, and sound: an output that does not depend on the
    member-carrying inputs cannot vary from one member to the next.
    """
    dependent = set(jaxpr.invars[:n_seeds])
    for eqn in jaxpr.eqns:
        if any(not isinstance(v, JaxprLiteral) and v in dependent for v in eqn.invars):
            dependent.update(eqn.outvars)
    return [not isinstance(v, JaxprLiteral) and v in dependent for v in jaxpr.outvars]


def _spec_leaves(out_spec: OutSpec, out_shape: PyTree[Any]) -> list[Any]:
    """The specification broadcast against the output tree, one kind per output leaf."""

    def is_kind(x: Any) -> bool:
        return x is None or isinstance(x, int | str)

    kinds, treedef = jax.tree.flatten(out_spec, is_leaf=is_kind)
    try:
        subtrees = treedef.flatten_up_to(out_shape)
    except ValueError as error:
        raise ValueError(f'out_spec is not a prefix of the output tree: {error}') from None
    return [kind for kind, sub in zip(kinds, subtrees) for _ in jax.tree.leaves(sub)]


def _unflattened(
    out_shape: PyTree[Any],
    kinds: list[Any],
    sums: list[Array],
    shareds: list[Array],
    stacks: list[Array],
) -> tuple[PyTree[Array], PyTree[Any]]:
    """The outputs and resolved kinds, as pytrees of the output structure."""
    treedef = jax.tree.structure(out_shape)
    it_sums, it_shareds, it_stacks = iter(sums), iter(shareds), iter(stacks)
    leaves = []
    for kind in kinds:
        if kind == 'sum':
            leaves.append(next(it_sums))
        elif kind is None:
            leaves.append(next(it_shareds))
        else:
            leaves.append(next(it_stacks))
    return jax.tree.unflatten(treedef, leaves), jax.tree.unflatten(treedef, kinds)


def _rebuild(
    treedef: Any, template: list[Any], flags: list[bool], selected: list[Any]
) -> PyTree[Any]:
    """The template leaves with the flagged positions replaced by `selected`, unflattened."""
    it = iter(selected)
    return jax.tree.unflatten(
        treedef, [next(it) if flag else leaf for leaf, flag in zip(template, flags)]
    )


def _expanded(leaf: Array, stride: int, size: int) -> Array:
    """A stride-s leaf with one entry per member, for leaves small enough to gather."""
    if stride == 1:
        return leaf
    return jnp.repeat(leaf, stride, axis=0, total_repeat_length=size)


def _axis_size(axis: str) -> int:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or axis not in mesh.shape:
        raise ValueError(f'no mesh axis {axis!r} is set: use `jax.set_mesh`')
    return mesh.shape[axis]


def _abstract(leaf: Any) -> Any:
    if _is_array(leaf):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
    return leaf


def _is_array(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array | np.ndarray)


# The resident ensembles are pytree nodes, so an operator can hold one where it holds a pytree of
# operators: its arrays are leaves of the operator, and the sizes, strides and policy are static.
# Unflattening bypasses validation, as tree transformations put placeholders in the leaves.


def _stacked_unflatten(aux: tuple[Any, ...], children: Iterable[Any]) -> StackedEnsemble:
    size, strides, policy = aux
    stacked, valid = children
    ensemble = object.__new__(StackedEnsemble)
    object.__setattr__(ensemble, 'size', size)
    object.__setattr__(ensemble, 'stacked', stacked)
    object.__setattr__(ensemble, 'strides', strides)
    object.__setattr__(ensemble, 'valid', valid)
    object.__setattr__(ensemble, 'policy', policy)
    return ensemble


def _list_unflatten(aux: tuple[Any, ...], children: Iterable[Any]) -> ListEnsemble:
    members, valid = children
    ensemble = object.__new__(ListEnsemble)
    object.__setattr__(ensemble, 'members', members)
    object.__setattr__(ensemble, 'valid', valid)
    object.__setattr__(ensemble, 'policy', aux[0])
    return ensemble


register_pytree_with_keys(
    StackedEnsemble,
    lambda e: (
        [(GetAttrKey('stacked'), e.stacked), (GetAttrKey('valid'), e.valid)],
        (e.size, tuple(e._leaves_and_strides()[1]), e.policy),
    ),
    _stacked_unflatten,
    lambda e: ([e.stacked, e.valid], (e.size, tuple(e._leaves_and_strides()[1]), e.policy)),
)

register_pytree_with_keys(
    ListEnsemble,
    lambda e: (
        [(GetAttrKey('members'), tuple(e.members)), (GetAttrKey('valid'), e.valid)],
        (e.policy,),
    ),
    _list_unflatten,
    lambda e: ([tuple(e.members), e.valid], (e.policy,)),
)
