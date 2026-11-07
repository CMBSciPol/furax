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

The members may be grouped into **segments** of consecutive members, such as the detector tiles
of one observation, through an `owner` index: `owner[i]` is the segment of member i. A stacked
leaf of stride `'segment'` has one entry per segment, read by every member of that segment, and
the outputs `'segment'` and `'segment_sum'` keep one entry per segment.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.extend.core import Literal as JaxprLiteral
from jax.extend.core import jaxpr_as_fun
from jax.sharding import AbstractMesh, Mesh
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

type Stride = int | None | Literal['segment']
"""Members per entry on a leaf's member axis: 1, s, None for a leaf shared by the ensemble, or
`'segment'` for one entry per segment, member i reading entry `owner[i]`."""

type OutSpec = PyTree[Literal['sum', 'segment_sum', 'auto'] | Stride]
"""How the members combine, per output leaf, as a prefix of the output tree: `'sum'` reduces the
leaf over the ensemble, `'segment_sum'` over the members of each segment; a stride keeps it, with
one entry per member (1), per run of s members (s), per segment (`'segment'`) or for the whole
ensemble (None); `'auto'` resolves to 1, `'segment'` or None from the traced computation."""


@dataclass(frozen=True)
class Policy:
    """How an ensemble is evaluated: in rounds of `batch` members, split over `axis`."""

    axis: str | None = None
    """The mesh axis the members are split over, each device evaluating its own, if any. The
    mesh is the one of the ensemble's laid-out leaves, or the current mesh when no leaf names
    it."""
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
        abstract_args = jax.tree.map(_abstract, args)
        compiled = jax.jit(apply).lower(ensemble.template, *abstract_args).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            raise ValueError('the backend reports no memory analysis: pass the policy explicitly')
        temporaries = analysis.temp_size_in_bytes
        devices = 1 if axis is None else _axis_size(axis, ensemble._mesh(axis))
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

    # Declared read-only, as the representations implement them (frozen dataclass fields or
    # properties). The declarations only exist for the type checker: runtime properties would
    # shadow the subclasses' instance attributes, as data descriptors win over instance dicts.
    if TYPE_CHECKING:

        @property
        def size(self) -> int:
            """The number of members."""
            ...

        @property
        def template(self) -> PyTree[Any]:
            """The member structure.

            `jax.ShapeDtypeStruct` on the per-member leaves, values on the shared ones.
            """
            ...

        @property
        def valid(self) -> Any:
            """One boolean per member, False on padded members; None when all members count."""
            ...

        @property
        def policy(self) -> Policy:
            """How [`map_reduce`][furax.core.map_reduce] walks this ensemble."""
            ...

        @property
        def owner(self) -> Any:
            """The segment of each member, None when the members are not grouped in segments."""
            ...

        @property
        def segments(self) -> int | None:
            """The number of segments, None without `owner`."""
            ...

    @abstractmethod
    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        """Member `i`."""

    def zip(self, x: PyTree[Any], axes: PyTree[int | None] = 0) -> 'Ensemble':
        """The members paired with their slices of a stacked pytree.

        Member i becomes the pair `(member, x[i])`: the array leaves of `x` carry a member axis
        of `size` entries, sliced per member. The member axis is leading by default; `axes`
        places it per leaf of `x`, a single value or an aligned pytree, so an input whose
        natural layout puts the member dimension elsewhere is sliced in place. The block
        composites use it to hand each block its own slice of the input.

        Only the resident ensembles support it: the members of a [`HostEnsemble`][] are read on
        the fly, they cannot be rebuilt into pairs.
        """
        raise NotImplementedError(f'the members of a {type(self).__name__} cannot be zipped')

    @abstractmethod
    def _arrays(self) -> tuple[list[Array], list[Array]]:
        """The ensemble's arrays: those that carry the members, and those every member holds."""

    def _member_axes(self) -> list[int]:
        """The member axis of each member-carrying array, aligned with `_arrays()[0]`."""
        return [0] * len(self._arrays()[0])

    @abstractmethod
    def _seeds(self) -> tuple[list[Any], list[Literal['member', 'segment'] | None], Any]:
        """The template's leaves, what each varies with, and the template's treedef.

        A leaf varies with the member, with the segment, or with neither (None). They seed the
        dataflow analysis that resolves `'auto'` outputs.
        """

    def _mesh(self, axis: str) -> Mesh | AbstractMesh | None:
        """The mesh carrying `axis`, read from the shardings of the ensemble's arrays.

        The mesh set by `jax.set_mesh` is thread-local: read back from the laid-out leaves, it
        lets the engine evaluate the ensemble from any thread -- such as the host-callback
        thread of `jax.pure_callback` -- however the policy was built. None when no array
        names the axis: the engine then falls back to the ambient mesh.
        """
        member, fixed = self._arrays()
        extra = [] if self.valid is None else [self.valid]
        return _leaves_mesh([*member, *extra, *fixed], axis)

    @abstractmethod
    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int, out_axes: Any = 0
    ) -> Callable[..., PyTree[Array]]:
        """The round body `(g, *member_arrays, *fixed_arrays, *args_leaves) -> outputs`.

        The outputs carry an axis of `batch` members when `batch > 1`, placed by `out_axes` (a
        prefix of the output tree, leading by default). The member-carrying arrays are arguments
        rather than closed-over values, so that an output depends on the member exactly where the
        traced computation does.
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

    @property
    def owner(self) -> None:
        return None

    @property
    def segments(self) -> None:
        return None

    @property
    def template(self) -> PyTree[Any]:
        leaves, _, is_array, treedef = self._parts()
        return jax.tree.unflatten(
            treedef,
            [
                jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) if flag else leaf
                for leaf, flag in zip(leaves, is_array)
            ],
        )

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, arrays, is_array, treedef = self._parts()
        if not arrays[0]:
            return self.members[0]
        selected = jax.lax.switch(i, [lambda a=a: list(a) for a in arrays])
        return _rebuild(treedef, leaves, is_array, selected)

    def zip(self, x: PyTree[Any], axes: PyTree[int | None] = 0) -> 'ListEnsemble':
        x_leaves, x_def = jax.tree.flatten(x)
        x_axes = _normalized_axes(x_leaves, _normalized_strides(x_leaves, 1), axes)
        members = [
            (
                member,
                jax.tree.unflatten(
                    x_def,
                    [
                        leaf if a is None else jax.lax.index_in_dim(leaf, i, axis=a, keepdims=False)
                        for leaf, a in zip(x_leaves, x_axes)
                    ],
                ),
            )
            for i, member in enumerate(self.members)
        ]
        return ListEnsemble(members, valid=self.valid, policy=self.policy)

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        _, arrays, _, _ = self._parts()
        return [leaf for member in arrays for leaf in member], []

    def _seeds(self) -> tuple[list[Any], list[Literal['member', 'segment'] | None], Any]:
        leaves, treedef = jax.tree.flatten(self.template)
        return leaves, ['member' if _is_abstract(leaf) else None for leaf in leaves], treedef

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int, out_axes: Any = 0
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
    """N members held as one pytree whose array leaves carry a member axis, leading by default.

    A leaf with stride s has a member axis of `size // s` entries, one per run of s consecutive
    members; a leaf with stride None, and any non-array leaf, is shared by the whole ensemble.
    `axes` places the member axis of each leaf: the engine slices, batches and shards along it
    in place, so a leaf whose natural layout puts the member dimension elsewhere needs no
    transposition. This is the only representation the engine can batch (`Policy.batch`) and
    shard (`Policy.axis`); the stacked outputs of `map_reduce` come back as one, with leading
    member axes.

    Members can also be grouped into segments of varying length through `owner`. A leaf with
    stride `'segment'` has `segments` entries on its member axis, member i reading entry
    `owner[i]`. A stride-s leaf shares an entry between s consecutive members; a segment leaf
    lets that number vary from one entry to the next.

    Examples:
        Five members in two segments, of three and two members: each reads its segment's offset.

        >>> ensemble = StackedEnsemble(
        ...     5,
        ...     {'x': jnp.arange(5.0), 'offset': jnp.array([10.0, 20.0])},
        ...     {'x': 1, 'offset': 'segment'},
        ...     owner=jnp.array([0, 0, 0, 1, 1]),
        ... )
        >>> out, _ = map_reduce(lambda m: m['x'] + m['offset'], ensemble, 1)
        >>> out.tolist()
        [10.0, 11.0, 12.0, 23.0, 24.0]
    """

    size: int
    """The number of members."""
    stacked: PyTree[Any]
    """The members' pytree; an array leaf of stride s has `size // s` entries on its member
    axis, one of stride `'segment'` has `segments`."""
    strides: PyTree[Stride] = 1
    """The stride of each leaf of `stacked`, or one stride for all; canonicalized at construction
    to one stride per leaf, non-array leaves getting None."""
    axes: PyTree[int | None] = 0
    """The member axis of each leaf of `stacked`, or one axis for all; canonicalized at
    construction to one non-negative axis per leaf, None on the shared leaves."""
    valid: Bool[Array, ' _n'] | None = None
    policy: Policy = Policy()
    owner: Int[Array, ' _n'] | None = None
    """The segment of each member, if the members are grouped in segments. Laid out over a mesh
    axis, the segments are split over the devices like the members: the members of device d own
    segments of the d-th block of `segments / devices`."""
    segments: int | None = None
    """The number of segments; by default, the entries of the segment leaves."""

    def __post_init__(self) -> None:
        # Canonicalize `strides` and `axes` to one value per leaf, so that flattening the ensemble
        # never has to look at the leaves again: tree transformations (eqx.partition,
        # jax.eval_shape) replace the arrays with placeholders, and an array test at flatten time
        # would flip the strides.
        leaves = jax.tree.leaves(self.stacked)
        object.__setattr__(self, 'strides', _normalized_strides(leaves, self.strides))
        object.__setattr__(self, 'axes', _normalized_axes(leaves, self.strides, self.axes))
        segments = self.segments
        for leaf, stride, axis in zip(leaves, self.strides, self.axes, strict=True):
            if stride is None:
                continue
            if stride == 'segment':
                if segments is None:
                    segments = leaf.shape[axis]
                if leaf.shape[axis] != segments:
                    raise ValueError(
                        f'a segment leaf of {segments} segments has shape {leaf.shape} '
                        f'on member axis {axis}'
                    )
                continue
            if self.size % stride:
                raise ValueError(f'stride {stride} does not divide {self.size} members')
            if leaf.shape[axis] != self.size // stride:
                raise ValueError(
                    f'a stride-{stride} leaf of {self.size} members has '
                    f'{self.size // stride} entries on axis {axis}, got shape {leaf.shape}'
                )
        object.__setattr__(self, 'segments', segments)
        _check_segments(self.owner, segments, self.size, 1)

    @property
    def size_per_device(self) -> int:
        """The members of one device."""
        axis = self.policy.axis
        return self.size // (1 if axis is None else _axis_size(axis, self._mesh(axis)))

    @property
    def template(self) -> PyTree[Any]:
        leaves, strides = self._leaves_and_strides()
        picked = [
            leaf
            if s is None
            else jax.ShapeDtypeStruct(leaf.shape[:a] + leaf.shape[a + 1 :], leaf.dtype)
            for leaf, s, a in zip(leaves, strides, self.axes)
        ]
        return jax.tree.unflatten(jax.tree.structure(self.stacked), picked)

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, strides = self._leaves_and_strides()
        picked = [
            leaf
            if s is None
            else jax.lax.dynamic_index_in_dim(
                leaf, _entry(i, s, self.owner), axis=a, keepdims=False
            )
            for leaf, s, a in zip(leaves, strides, self.axes)
        ]
        return jax.tree.unflatten(jax.tree.structure(self.stacked), picked)

    def zip(self, x: PyTree[Any], axes: PyTree[int | None] = 0) -> 'StackedEnsemble':
        x_leaves = jax.tree.leaves(x)
        x_strides = _normalized_strides(x_leaves, 1)
        x_axes = _normalized_axes(x_leaves, x_strides, axes)
        return StackedEnsemble(
            self.size,
            (self.stacked, x),
            self.strides + x_strides,
            axes=self.axes + x_axes,
            valid=self.valid,
            policy=self.policy,
            owner=self.owner,
            segments=self.segments,
        )

    def shard(self, axis: str) -> 'StackedEnsemble':
        """The ensemble laid out over a mesh axis: consecutive members on each device.

        The strided and segment leaves are split over their member axis, `valid` and `owner` over
        their leading axis, the shared leaves are replicated, and the returned ensemble's policy
        records the axis. The engine reads the mesh back from the laid-out leaves, so the
        ensemble evaluates from any thread, not only the one that set the mesh.

        Raises:
            ValueError: If the members or the segments do not split over the devices, or if a
                member owns a segment of another device.
        """
        devices = _axis_size(axis)
        leaves, strides = self._leaves_and_strides()
        for leaf, s, a in zip(leaves, strides, self.axes):
            if s is not None and leaf.shape[a] % devices:
                raise ValueError(f'{leaf.shape[a]} entries do not split over {devices} devices')
        _check_segments(self.owner, self.segments, self.size, devices)
        placed = [
            leaf if s is None else jax.device_put(leaf, P(*(None,) * a, axis))
            for leaf, s, a in zip(leaves, strides, self.axes)
        ]
        valid = None if self.valid is None else jax.device_put(self.valid, P(axis))
        owner = None if self.owner is None else jax.device_put(self.owner, P(axis))
        return dataclasses.replace(
            self,
            stacked=jax.tree.unflatten(jax.tree.structure(self.stacked), placed),
            valid=valid,
            owner=owner,
            policy=dataclasses.replace(self.policy, axis=axis),
        )

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        # the segment leaves and the owner index come last, see `_body`
        leaves, strides = self._leaves_and_strides()
        member = [leaf for leaf, s in zip(leaves, strides) if isinstance(s, int)]
        member += [leaf for leaf, s in zip(leaves, strides) if s == 'segment']
        if self.owner is not None and 'segment' in strides:
            member.append(self.owner)
        fixed = [leaf for leaf, s in zip(leaves, strides) if s is None and _is_array(leaf)]
        return member, fixed

    def _member_axes(self) -> list[int]:
        strided = [a for s, a in zip(self.strides, self.axes) if isinstance(s, int)]
        segment = [a for s, a in zip(self.strides, self.axes) if s == 'segment']
        return strided + segment + [0] * bool(segment and self.owner is not None)

    def _seeds(self) -> tuple[list[Any], list[Literal['member', 'segment'] | None], Any]:
        return _seeded(self.template, self.strides)

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int, out_axes: Any = 0
    ) -> Callable[..., PyTree[Array]]:
        leaves, strides = self._leaves_and_strides()
        axes = self.axes
        treedef = jax.tree.structure(self.stacked)
        member_at = [i for i, s in enumerate(strides) if isinstance(s, int)]
        member_at += [i for i, s in enumerate(strides) if s == 'segment']
        fixed_at = [i for i, s in enumerate(strides) if s is None and _is_array(leaves[i])]
        has_owner = 'segment' in strides
        n_member = len(member_at) + has_owner
        n_fixed = len(fixed_at)

        def merged(flat: tuple[Array, ...]) -> list[Any]:
            full = list(leaves)
            for i, leaf in zip(member_at, flat):
                full[i] = leaf
            for i, leaf in zip(fixed_at, flat[n_member : n_member + n_fixed]):
                full[i] = leaf
            return full

        def body(g: Int[Array, ''], *flat: Array) -> PyTree[Array]:
            args = jax.tree.unflatten(args_def, flat[n_member + n_fixed :])
            full = merged(flat)
            owner = None
            if has_owner:
                # the device's own block of segments: the remainder is the local entry
                first = member_at[-1]
                owner = flat[n_member - 1] % full[first].shape[axes[first]]
            start = g * batch
            if batch == 1:
                picked = [
                    leaf
                    if s is None
                    else jax.lax.dynamic_index_in_dim(
                        leaf, _entry(start, s, owner), axis=a, keepdims=False
                    )
                    for leaf, s, a in zip(full, strides, axes)
                ]
                return apply(jax.tree.unflatten(treedef, picked), *args)
            batched: list[Array | None] = []
            shared: list[Any] = []
            for leaf, s, a in zip(full, strides, axes):
                if s is None:
                    batched.append(None)
                    shared.append(leaf)
                elif s == 'segment':
                    assert owner is not None  # a segment leaf comes with an owner index
                    entries = jax.lax.dynamic_slice_in_dim(owner, start, batch)
                    batched.append(jnp.take(leaf, entries, axis=a))
                    shared.append(None)
                elif s >= batch:
                    batched.append(None)
                    shared.append(
                        jax.lax.dynamic_index_in_dim(leaf, start // s, axis=a, keepdims=False)
                    )
                else:
                    sliced = jax.lax.dynamic_slice_in_dim(leaf, start // s, batch // s, axis=a)
                    if s > 1:
                        sliced = jnp.repeat(sliced, s, axis=a, total_repeat_length=batch)
                    batched.append(sliced)
                    shared.append(None)
            picked = [(i, leaf) for i, leaf in enumerate(batched) if leaf is not None]

            def member(*entries: Array) -> PyTree[Array]:
                full = list(shared)
                for (i, _), leaf in zip(picked, entries):
                    full[i] = leaf
                return apply(jax.tree.unflatten(treedef, full), *args)

            return jax.vmap(member, in_axes=tuple(axes[i] for i, _ in picked), out_axes=out_axes)(
                *[leaf for _, leaf in picked]
            )

        return body

    def _leaves_and_strides(self) -> tuple[list[Any], tuple[Stride, ...]]:
        """The leaves of `stacked` and their strides, canonicalized at construction."""
        return jax.tree.leaves(self.stacked), self.strides


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

    With `owner`, the members are grouped in segments as in a `StackedEnsemble`, and the outputs
    `'segment'` and `'segment_sum'` land sharded over the mesh axis, device d holding the d-th
    block of `segments / devices` segments. `read` returns every per-member leaf, those of stride
    `'segment'` included: the stride only tells `'auto'` that they vary with the segment.
    """

    size: int
    """The number of members."""
    read: Callable[[int], tuple[np.ndarray, ...]]
    """The per-member leaves of member i, called from host threads. Under several processes, each
    process reads only the members of its own devices."""
    template: PyTree[Any]
    """The member structure: `jax.ShapeDtypeStruct` on the per-member leaves, values on the
    shared ones."""
    valid: np.ndarray | None = None
    policy: Policy = Policy()
    prefetch: bool = True
    """Whether to read the next round while the devices evaluate the current one; set it to False
    on devices that share the host's memory, where the next round's buffers would compete with the
    current round's working set."""
    strides: PyTree[Literal[1, 'segment'] | None] = 1
    """Per leaf of `template`, or one for all: 1 for a leaf that varies with the member,
    `'segment'` for one that varies with its segment only; canonicalized at construction, the
    shared leaves getting None."""
    owner: np.ndarray | None = None
    """The segment of each member, if the members are grouped in segments: the members of device
    d own segments of the d-th block of `segments / devices`."""
    segments: int | None = None
    """The number of segments, given with `owner`."""

    def __post_init__(self) -> None:
        leaves = jax.tree.leaves(self.template)
        strides = _normalized_strides(leaves, self.strides, is_member=_is_abstract)
        object.__setattr__(self, 'strides', strides)
        if any(s not in (1, 'segment', None) for s in strides):
            raise ValueError(f"the strides of a host ensemble are 1 or 'segment', got {strides}")
        if 'segment' in strides and self.owner is None:
            raise ValueError("a 'segment' leaf needs an owner index")
        _check_segments(self.owner, self.segments, self.size, 1)

    @property
    def size_per_device(self) -> int:
        """The members of one device."""
        axis = self.policy.axis
        return self.size // (1 if axis is None else _axis_size(axis, self._mesh(axis)))

    def fetch(self, i: Int[Array, ''] | int) -> PyTree[Any]:
        leaves, is_member, treedef = self._parts()
        arrays = [jnp.asarray(x) for x in self.read(int(i))]
        return _rebuild(treedef, leaves, is_member, arrays)

    def _arrays(self) -> tuple[list[Array], list[Array]]:
        leaves, is_member, _ = self._parts()
        return [], [leaf for leaf, flag in zip(leaves, is_member) if not flag and _is_array(leaf)]

    def _seeds(self) -> tuple[list[Any], list[Literal['member', 'segment'] | None], Any]:
        return _seeded(self.template, self.strides)

    def _body(
        self, apply: Callable[..., PyTree[Array]], args_def: Any, batch: int, out_axes: Any = 0
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
        is_member = [_is_abstract(leaf) for leaf in leaves]
        return leaves, is_member, treedef


def map_reduce(
    apply: Callable[..., PyTree[Array]],
    ensemble: Ensemble,
    out_spec: OutSpec,
    *args: Any,
    out_axes: PyTree[int] = 0,
) -> tuple[PyTree[Array], PyTree[Literal['sum', 'segment_sum'] | Stride]]:
    """`apply` on every member of an ensemble, its outputs combined over the members.

    The members are evaluated in rounds of `ensemble.policy.batch`, one round after the other by a
    single loop, so `apply` is traced once whatever the ensemble size. The members of a round are
    evaluated together, vectorized by `jax.vmap`, which holds the temporaries of all of them at
    once: rounds of one member suit heavy computations, a single round of every member the small
    ones, which a loop would serialize. A round of one member is `apply` itself, not vectorized.

    With `ensemble.policy.axis`, the members are first split over the devices of that axis -- the
    ensemble must then be a [`StackedEnsemble`][furax.core.StackedEnsemble] laid out over it (see
    [`StackedEnsemble.shard`][furax.core.StackedEnsemble.shard]) -- and each device loops over its
    own members. The mesh is read back from the shardings of the laid-out leaves, falling back to
    the current mesh, so the evaluation does not depend on the thread-local mesh of the calling
    thread. The summed outputs are summed across the devices and held
    by all of them; the stacked outputs stay with the devices that computed them, in position
    order, as do those kept per segment, device d holding its block of segments.

    Args:
        apply: The computation, from one member and `args` to a pytree of arrays.
        ensemble: The members.
        out_spec: How each output leaf combines over the members, as a prefix of the output tree:
            `'sum'` reduces it (padded members masked); stride 1 stacks one entry per member along
            a new leading axis; stride s keeps one entry per run of s members sharing it; stride
            None keeps one entry, the same for every member. With an owner index, `'segment_sum'`
            sums the members of each segment into one entry per segment (padded members masked),
            and `'segment'` keeps one entry per segment, the same for its members. A segment
            without valid members holds zeros. `'auto'` resolves to 1, `'segment'` or None by
            whether the leaf depends on the member, on its segment only, or on neither, in the
            traced computation. A declared stride is a promise the engine does not check.
        *args: The shared inputs, which every member receives whole.
        out_axes: Where the member axis of each stacked output leaf goes, leading by default, as
            a non-negative value or a prefix of the output tree: the engine writes each member's
            entry along that axis in place, without transposition, so an output can keep the
            layout of a [`StackedEnsemble`][furax.core.StackedEnsemble] input with the same
            `axes`. Ignored on the summed and shared leaves, and on those kept per segment, which
            are segment-leading. A host ensemble stacks its outputs member-leading only.

    Returns:
        The outputs of `apply` combined over the members, and the output specification with
        `'auto'` resolved.

    Raises:
        ValueError: If `out_spec` names an unknown kind or is not a prefix of the output, if the
            policy does not divide the ensemble, or if a sharded or batched ensemble is not a
            `StackedEnsemble`.
    """
    if isinstance(ensemble, HostEnsemble):
        if any(a != 0 for a in jax.tree.leaves(out_axes)):
            raise ValueError(
                'a host ensemble stacks its outputs member-leading: out_axes must be 0'
            )
        return _host_map_reduce(apply, ensemble, out_spec, args)
    return _resident_map_reduce(apply, ensemble, out_spec, args, out_axes)


def lift(f: Callable[..., PyTree[Any]], *ensembles: Ensemble) -> Ensemble:
    """Member-wise `f` over ensembles of the same size: the algebra of ensembles.

    For list ensembles, `f` is applied to the member pytrees. For stacked ensembles, a
    **structural** `f` -- one that only rearranges its members' leaves, such as composing or
    transposing operators -- is detected by tracking leaf identity, and the output ensemble reuses
    the input arrays with their strides, at no cost. Otherwise `f` is vectorized by `jax.vmap`
    over the member axis, stride-s and segment leaves expanded to one entry per member, and the
    output leaves come back with stride 1. Stacked ensembles grouped in segments must share their
    owner index, which the output keeps.

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

    flats, in_axes = [], []
    for e in ensembles:
        leaves, strides = e._leaves_and_strides()
        flats.append(
            [
                leaf if s is None else _expanded(leaf, s, size, e.owner, a)
                for leaf, s, a in zip(leaves, strides, e.axes)
            ]
        )
        in_axes.append(list(e.axes))  # the canonical axes are None exactly on the shared leaves
    out_arrays = jax.vmap(on_members, in_axes=(in_axes,))(flats)

    member0 = f(*(e.fetch(0) for e in ensembles))
    out_leaves, out_treedef = jax.tree.flatten(member0)
    is_array = [_is_array(leaf) for leaf in out_leaves]
    stacked = _rebuild(out_treedef, out_leaves, is_array, list(out_arrays))
    strides = jax.tree.unflatten(out_treedef, [1 if flag else None for flag in is_array])
    owner, segments = _common_owner(ensembles)
    return StackedEnsemble(
        size,
        stacked,
        strides,
        valid=ensembles[0].valid,
        policy=ensembles[0].policy,
        owner=owner,
        segments=segments,
    )


def _lift_structural(
    f: Callable[..., PyTree[Any]], ensembles: Sequence[StackedEnsemble]
) -> StackedEnsemble | None:
    """`f` applied once to abstract members, the output leaves matched to the inputs by identity.

    A structural `f` passes its members' array leaves through untouched, so every array leaf of
    its output is one of the sentinel inputs: the output ensemble reuses the stacked arrays with
    their strides and member axes. Any computed array leaf, or any error on abstract leaves,
    falls back to None.
    """
    sources: dict[int, tuple[Array, Stride, int | None]] = {}
    members = []
    for e in ensembles:
        leaves, strides = e._leaves_and_strides()
        sentinels = []
        for leaf, s, a in zip(leaves, strides, e.axes):
            if _is_array(leaf):
                shape = leaf.shape if s is None else leaf.shape[:a] + leaf.shape[a + 1 :]
                sentinel = jax.ShapeDtypeStruct(shape, leaf.dtype)
                sources[id(sentinel)] = (leaf, s, a)
                sentinels.append(sentinel)
            else:
                sentinels.append(leaf)
        members.append(jax.tree.unflatten(jax.tree.structure(e.stacked), sentinels))
    try:
        out = f(*members)
    except (AttributeError, IndexError, TypeError, ValueError):
        # f computes on its leaves, which the abstract sentinels cannot do: not structural.
        return None
    out_leaves, out_treedef = jax.tree.flatten(out)
    stacked, strides, axes = [], [], []
    for leaf in out_leaves:
        if id(leaf) in sources:
            source, stride, axis = sources[id(leaf)]
            stacked.append(source)
            strides.append(stride)
            axes.append(axis)
        elif isinstance(leaf, jax.ShapeDtypeStruct) or _is_array(leaf):
            return None  # a computed array leaf: f is not structural
        else:
            stacked.append(leaf)
            strides.append(None)
            axes.append(None)
    owner, segments = _common_owner(ensembles)
    return StackedEnsemble(
        ensembles[0].size,
        jax.tree.unflatten(out_treedef, stacked),
        jax.tree.unflatten(out_treedef, strides),
        axes=jax.tree.unflatten(out_treedef, axes),
        valid=ensembles[0].valid,
        policy=ensembles[0].policy,
        owner=owner,
        segments=segments,
    )


def _resident_map_reduce(
    apply: Callable[..., PyTree[Array]],
    ensemble: Ensemble,
    out_spec: OutSpec,
    args: tuple[Any, ...],
    out_axes: PyTree[int] = 0,
) -> tuple[PyTree[Array], PyTree[Literal['sum', 'segment_sum'] | Stride]]:
    if any(
        not isinstance(a, int) or isinstance(a, bool) or a < 0 for a in jax.tree.leaves(out_axes)
    ):
        raise ValueError('out_axes takes non-negative integers')
    policy = ensemble.policy
    batch = policy.batch
    if not isinstance(ensemble, StackedEnsemble):
        if policy.axis is not None:
            raise ValueError('only a stacked ensemble is laid out over a mesh axis')
        if batch != 1:
            raise ValueError('only a stacked ensemble is evaluated in batched rounds')
    mesh = None if policy.axis is None else ensemble._mesh(policy.axis)
    devices = 1 if policy.axis is None else _axis_size(policy.axis, mesh)
    per_device, remainder = divmod(ensemble.size, devices)
    if remainder:
        raise ValueError(f'{devices} devices do not divide {ensemble.size} members')
    rounds, remainder = divmod(per_device, batch)
    if remainder:
        raise ValueError(f'rounds of {batch} members do not divide {per_device} members')
    segments = _local_segments(ensemble, devices)

    member_arrays, fixed_arrays = ensemble._arrays()
    member_axes = ensemble._member_axes()
    args_leaves, args_def = jax.tree.flatten(args)
    body = ensemble._body(apply, args_def, batch)

    # The round body is traced once, on per-device shapes, with the member-carrying arrays as
    # inputs; the loop then evaluates the trace, and `'auto'` is resolved before any collective
    # is laid out.
    local = [
        jax.ShapeDtypeStruct((*x.shape[:a], x.shape[a] // devices, *x.shape[a + 1 :]), x.dtype)
        for x, a in zip(member_arrays, member_axes)
    ]
    closed, out_shape = jax.make_jaxpr(body, return_shape=True)(
        jnp.zeros((), jnp.int32), *local, *fixed_arrays, *args_leaves
    )
    seeds = closed.jaxpr.invars[: 1 + len(member_arrays)]  # the round index, the member arrays
    kinds = _resolved_kinds(
        out_spec,
        out_shape,
        lambda: _dependence(apply, ensemble, args, closed, seeds),
        per_device,
        batch,
        ensemble.owner is not None,
    )

    # The member axis of a stacked output leaf goes where `out_axes` says: the round outputs and
    # the accumulation buffers carry it there from the start, so nothing is transposed. The other
    # leaves, those kept per segment included, keep their round batch axis leading. A vectorized body places the batch axis itself,
    # which needs a second trace -- only when a non-leading axis is actually requested.
    axes_flat = [
        a if isinstance(kind, int) else 0
        for kind, a in zip(kinds, _spec_leaves(out_axes, out_shape))
    ]
    for (aval, kind), a in zip(zip(closed.out_avals, kinds), axes_flat):
        if isinstance(kind, int) and a > len(aval.shape) - (1 if batch > 1 else 0):
            raise ValueError(f'out axis {a} is out of bounds for a member output {aval.shape}')
    if batch > 1 and any(axes_flat):
        placement = jax.tree.unflatten(jax.tree.structure(out_shape), axes_flat)
        body = ensemble._body(apply, args_def, batch, out_axes=placement)
        closed, out_shape = jax.make_jaxpr(body, return_shape=True)(
            jnp.zeros((), jnp.int32), *local, *fixed_arrays, *args_leaves
        )

    valid = ensemble.valid
    if valid is not None and not any(kind in ('sum', 'segment_sum') for kind in kinds):
        valid = None
    owner = ensemble.owner if any(_is_segment(kind) for kind in kinds) else None

    def run(
        member: list[Array], use: Array | None, entries: Array | None, fixed_and_args: list[Array]
    ) -> tuple[list[Array], ...]:
        local = closed
        if policy.axis is not None:
            # Avals are typed by the mesh context, so the trace made outside, under the explicit
            # mesh, cannot be evaluated inside the shard: trace the same body again in its context.
            local = jax.make_jaxpr(body)(jnp.zeros((), jnp.int32), *member, *fixed_and_args)
        if entries is not None:
            assert segments is not None
            entries = entries % segments  # the device's own block of segments
        return _loop(
            local,
            kinds,
            axes_flat,
            batch,
            rounds,
            per_device,
            segments,
            [*member, *fixed_and_args],
            use,
            entries,
        )

    if policy.axis is None:
        sums, shareds, stacks = run(member_arrays, valid, owner, [*fixed_arrays, *args_leaves])
    else:
        axis = policy.axis
        n_member = len(member_arrays)
        has_valid = valid is not None
        has_owner = owner is not None

        def block(*flat: Array) -> tuple[Array, ...]:
            member = list(flat[:n_member])
            use = flat[n_member] if has_valid else None
            entries = flat[n_member + has_valid] if has_owner else None
            rest = list(flat[n_member + has_valid + has_owner :])
            sums, shareds, stacks = run(member, use, entries, rest)
            sums = [jax.lax.psum(x, axis) for x in sums]
            return (*sums, *shareds, *stacks)

        n_sums = sum(kind == 'sum' for kind in kinds)
        n_stacks = sum(_is_stacked(kind) for kind in kinds)
        n_shareds = len(kinds) - n_sums - n_stacks
        in_specs = (
            tuple(P(*(None,) * a, axis) for a in member_axes)
            + (P(axis),) * (has_valid + has_owner)
            + (P(),) * (len(fixed_arrays) + len(args_leaves))
        )
        out_specs = (P(),) * (n_sums + n_shareds) + tuple(
            P(*(None,) * a, axis) for kind, a in zip(kinds, axes_flat) if _is_stacked(kind)
        )
        flat = jax.shard_map(
            block, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        )(
            *member_arrays,
            *([valid] if has_valid else []),
            *([owner] if has_owner else []),
            *fixed_arrays,
            *args_leaves,
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
) -> tuple[PyTree[Array], PyTree[Literal['sum', 'segment_sum'] | Stride]]:
    policy = ensemble.policy
    batch = policy.batch
    axis = policy.axis
    mesh = None
    if axis is not None:
        mesh = ensemble._mesh(axis)
        if mesh is None:
            mesh = _leaves_mesh(jax.tree.leaves(args), axis)
    devices = 1 if axis is None else _axis_size(axis, mesh)
    per_device, remainder = divmod(ensemble.size, devices)
    if remainder:
        raise ValueError(f'{devices} devices do not divide {ensemble.size} members')
    rounds, remainder = divmod(per_device, batch)
    if remainder:
        raise ValueError(f'rounds of {batch} members do not divide {per_device} members')
    segments = _local_segments(ensemble, devices)
    _check_segments(ensemble.owner, ensemble.segments, ensemble.size, devices)

    leaves, is_member, _ = ensemble._parts()
    member_structs = [leaf for leaf, flag in zip(leaves, is_member) if flag]
    _, fixed_arrays = ensemble._arrays()
    args_leaves, args_def = jax.tree.flatten(args)
    body = ensemble._body(apply, args_def, batch)

    round_structs = [jax.ShapeDtypeStruct((batch, *x.shape), x.dtype) for x in member_structs]
    closed, out_shape = jax.make_jaxpr(body, return_shape=True)(
        *round_structs, *fixed_arrays, *args_leaves
    )
    seeds = closed.jaxpr.invars[: len(member_structs)]
    kinds = _resolved_kinds(
        out_spec,
        out_shape,
        lambda: _dependence(apply, ensemble, args, closed, seeds),
        per_device,
        batch,
        ensemble.owner is not None,
    )
    traced = jaxpr_as_fun(closed)
    member_shapes = [
        (aval.shape[1:] if batch > 1 else aval.shape, aval.dtype) for aval in closed.out_avals
    ]

    valid = ensemble.valid
    if valid is not None and not any(kind in ('sum', 'segment_sum') for kind in kinds):
        valid = None
    owner = ensemble.owner if any(_is_segment(kind) for kind in kinds) else None

    def step(
        carry: tuple[list[Array], list[Array]],
        g: Int[Array, ''],
        round_arrays: list[Array],
        use: Array | None,
        entries: Array | None,
        fixed_and_args: list[Array],
    ) -> tuple[tuple[list[Array], list[Array]], list[Array]]:
        # The summed carries have a leading device axis, so each device accumulates its partial.
        # Inside a shard the mesh context differs from the trace's, so the body is traced again.
        run = traced
        if axis is not None:
            run = jaxpr_as_fun(jax.make_jaxpr(body)(*round_arrays, *fixed_and_args))
        sums, stacks = carry
        out = list(run(*round_arrays, *fixed_and_args))
        if use is not None and batch == 1:
            use = use[0]
        if entries is not None and batch == 1:
            entries = entries[0]
        new_sums, new_stacks, shareds = _combined(
            out,
            kinds,
            [0] * len(kinds),
            [x[0] for x in sums],
            stacks,
            g,
            batch,
            segments,
            use,
            entries,
        )
        return ([x[None] for x in new_sums], new_stacks), shareds

    launch: Callable[..., Any] = step
    if axis is not None:
        launch = jax.shard_map(
            step,
            mesh=mesh,
            in_specs=((P(axis), P(axis)), P(), P(axis), P(axis), P(axis), P()),
            out_specs=((P(axis), P(axis)), P()),
            check_vma=False,
        )
    launch = jax.jit(launch, donate_argnums=0)

    # Each process reads and places the members of its own devices; the rows of the mesh axis
    # held by this process are contiguous, since a mesh orders its devices by process.
    if axis is None:
        sharding = None
        axis_devices: list[Any] = [None]
        local_rows = [0]
    else:
        if mesh is None:
            mesh = jax.sharding.get_mesh()
        if mesh.empty or mesh.axis_names != (axis,):
            raise ValueError(f'a host ensemble needs a mesh along the one axis {axis!r}')
        sharding = jax.sharding.NamedSharding(mesh, P(axis))
        axis_devices = list(mesh.devices.flat)
        process = jax.process_index()
        local_rows = [r for r, d in enumerate(axis_devices) if d.process_index == process]

    def zeros(shape: tuple[int, ...], dtype: Any) -> Array:
        if sharding is None:
            return jnp.zeros(shape, dtype)
        return jnp.zeros(shape, dtype, device=sharding)

    init_sums = [
        zeros((devices, *shape), dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if kind == 'sum'
    ]
    init_stacks = [
        zeros((devices * _stack_length(kind, per_device, segments), *shape), dtype)
        for (shape, dtype), kind in zip(member_shapes, kinds)
        if _is_stacked(kind)
    ]

    def assemble(stacks: list[list[np.ndarray]]) -> list[Array]:
        """Per-local-device round buffers, as arrays sharded over the mesh axis."""
        if sharding is None:
            return [jnp.asarray(by_row[0]) for by_row in stacks]
        placed = []
        for by_row in stacks:
            shards = [
                jax.device_put(block, axis_devices[row], may_alias=True)
                for row, block in zip(local_rows, by_row, strict=True)
            ]
            shape = (devices * by_row[0].shape[0], *by_row[0].shape[1:])
            placed.append(jax.make_array_from_single_device_arrays(shape, sharding, shards))
        return placed

    def per_row(values: np.ndarray, positions: list[int]) -> Array:
        """One value per member of the round, as an array sharded over the mesh axis."""
        rows = np.asarray([values[p] for p in positions]).reshape(len(local_rows), batch)
        return assemble([[rows[r] for r in range(len(local_rows))]])[0]

    def read_round(g: int) -> tuple[list[Array], Array | None, Array | None]:
        # the device of row r evaluates positions r * per_device + g * batch + [0, batch)
        positions = [r * per_device + g * batch + i for r in local_rows for i in range(batch)]
        with ThreadPoolExecutor(len(positions)) as pool:
            members = list(pool.map(ensemble.read, positions))
        by_member = [
            [
                np.stack([members[r * batch + i][j] for i in range(batch)])
                for r in range(len(local_rows))
            ]
            for j in range(len(member_structs))
        ]
        arrays = assemble(by_member)
        use = None if valid is None else per_row(np.asarray(valid), positions)
        entries = None
        if owner is not None:
            assert segments is not None
            entries = per_row(np.asarray(owner) % segments, positions)
        return arrays, use, entries

    carry = (init_sums, init_stacks)
    shareds: list[Array] = []
    members = read_round(0)
    for g in range(rounds):
        round_arrays, use, entries = members
        members = ([], None, None)  # a CPU read must not find the round's buffers still alive
        carry, shareds = launch(
            carry, jnp.int32(g), round_arrays, use, entries, [*fixed_arrays, *args_leaves]
        )
        del round_arrays, use, entries
        last = g + 1 == rounds
        if ensemble.prefetch and not last:
            members = read_round(g + 1)  # while the devices compute
        jax.block_until_ready(carry)
        if not ensemble.prefetch and not last:
            members = read_round(g + 1)
    partials, stacks = carry
    sums = [x.sum(axis=0) for x in partials]  # one partial per device
    return _unflattened(out_shape, kinds, sums, shareds, stacks)


def _loop(
    closed: Any,
    kinds: list[Any],
    axes: list[int],
    batch: int,
    rounds: int,
    per_device: int,
    segments: int | None,
    inputs: list[Array],
    valid: Array | None,
    entries: Array | None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """The loop over the rounds of one device: evaluate the trace, combine by kind.

    `axes` places the member axis of each stacked output, and locates the batch axis of each
    round output when the rounds are vectorized (0 on the summed, shared and per-segment
    leaves). `entries` holds the local segment of each member of the device, if any output is
    kept per segment.
    """
    traced = jaxpr_as_fun(closed)
    member_shapes = [
        (aval.shape[:a] + aval.shape[a + 1 :] if batch > 1 else aval.shape, aval.dtype)
        for aval, a in zip(closed.out_avals, axes)
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
        jnp.zeros((*shape[:a], _stack_length(kind, per_device, segments), *shape[a:]), dtype)
        for ((shape, dtype), kind), a in zip(zip(member_shapes, kinds), axes)
        if _is_stacked(kind)
    ]

    def per_round(values: Array | None, g: Int[Array, '']) -> Array | None:
        if values is None:
            return None
        if batch == 1:
            return jax.lax.dynamic_index_in_dim(values, g, keepdims=False)
        return jax.lax.dynamic_slice_in_dim(values, g * batch, batch)

    def step(
        g: Int[Array, ''], carry: tuple[list[Array], list[Array], list[Array]]
    ) -> tuple[list[Array], list[Array], list[Array]]:
        sums, _, stacks = carry  # the shared slots are replaced, the last round's survive
        out = list(traced(jnp.asarray(g, jnp.int32), *inputs))  # traced with an int32 index
        new_sums, new_stacks, new_shareds = _combined(
            out,
            kinds,
            axes,
            sums,
            stacks,
            g,
            batch,
            segments,
            per_round(valid, g),
            per_round(entries, g),
        )
        return new_sums, new_shareds, new_stacks

    sums, shareds, stacks = jax.lax.fori_loop(
        0, rounds, step, (init_sums, init_shareds, init_stacks)
    )
    return sums, shareds, stacks


def _combined(
    out: list[Array],
    kinds: list[Any],
    axes: list[int],
    sums: list[Array],
    stacks: list[Array],
    g: Int[Array, ''],
    batch: int,
    segments: int | None,
    use: Array | None,
    entries: Array | None,
) -> tuple[list[Array], list[Array], list[Array]]:
    """One round's outputs folded into the carries: sums masked and added, stacks written.

    `entries` holds the local segment of each member of the round, for the outputs kept per
    segment; a padded member's entry is out of bounds, so its write is dropped.
    """
    if entries is not None and use is not None:
        assert segments is not None  # entries come with the local segment count
        entries = jnp.where(use, entries, segments)
    new_sums, new_stacks, shareds = [], [], []
    it_sums, it_stacks = iter(sums), iter(stacks)
    for leaf, kind, a in zip(out, kinds, axes):
        if kind == 'sum':
            if use is not None:
                mask = use if batch == 1 else use.reshape((batch,) + (1,) * (leaf.ndim - 1))
                leaf = jnp.where(mask, leaf, 0)
            contribution = leaf if batch == 1 else leaf.sum(axis=0)
            new_sums.append(next(it_sums) + contribution)
        elif kind is None:
            shareds.append(leaf if batch == 1 else leaf[0])
        elif kind == 'segment_sum':
            assert entries is not None
            new_stacks.append(next(it_stacks).at[entries].add(leaf, mode='drop'))
        elif kind == 'segment':
            assert entries is not None
            new_stacks.append(next(it_stacks).at[entries].set(leaf, mode='drop'))
        else:
            if batch == 1:
                chunk = jnp.expand_dims(leaf, a)
            elif kind >= batch:
                chunk = jnp.expand_dims(jax.lax.index_in_dim(leaf, 0, axis=a, keepdims=False), a)
            else:
                chunk = leaf[(slice(None),) * a + (slice(None, None, kind),)]
            new_stacks.append(
                jax.lax.dynamic_update_slice_in_dim(next(it_stacks), chunk, g * batch // kind, a)
            )
    return new_sums, new_stacks, shareds


def _resolved_kinds(
    out_spec: OutSpec,
    out_shape: PyTree[Any],
    dependence: Callable[[], list[Literal['member', 'segment'] | None]],
    per_device: int,
    batch: int,
    has_owner: bool,
) -> list[Any]:
    """The declared kinds broadcast over the output leaves, `'auto'` resolved by `dependence`."""
    declared = _spec_leaves(out_spec, out_shape)
    for kind in declared:
        if kind in ('sum', 'auto') or kind is None:
            continue
        if _is_segment(kind):
            if not has_owner:
                raise ValueError(f'the output kind {kind!r} needs an ensemble with an owner index')
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
    resolved = {'member': 1, 'segment': 'segment', None: None}
    return [
        resolved[depends] if kind == 'auto' else kind
        for kind, depends in zip(declared, dependence(), strict=True)
    ]


def _dependence(
    apply: Callable[..., PyTree[Array]],
    ensemble: Ensemble,
    args: tuple[Any, ...],
    body: Any,
    body_seeds: Sequence[Any],
) -> list[Literal['member', 'segment'] | None]:
    """What each output of `apply` varies with: the member, its segment only, or neither.

    Without segment leaves, the traced round `body` answers, seeded by its member-carrying inputs
    `body_seeds`. Otherwise `apply` is traced again, on the template, whose per-member and
    per-segment leaves seed the analysis: in the body, a segment leaf is read at `owner[i]`,
    which depends on the member and would hide the difference.
    """
    leaves, classes, treedef = ensemble._seeds()
    if 'segment' not in classes:
        return [
            'member' if depends else None for depends in _dependent_outputs(body.jaxpr, body_seeds)
        ]
    seeded = [i for i, c in enumerate(classes) if c is not None]
    args_leaves, args_def = jax.tree.flatten(args)

    def member(*flat: Any) -> PyTree[Array]:
        full = list(leaves)
        for i, leaf in zip(seeded, flat[: len(seeded)]):
            full[i] = leaf
        return apply(
            jax.tree.unflatten(treedef, full), *jax.tree.unflatten(args_def, flat[len(seeded) :])
        )

    closed = jax.make_jaxpr(member)(*[leaves[i] for i in seeded], *args_leaves)
    invars = closed.jaxpr.invars
    by_member = _dependent_outputs(
        closed.jaxpr, [v for i, v in zip(seeded, invars) if classes[i] == 'member']
    )
    by_segment = _dependent_outputs(
        closed.jaxpr, [v for i, v in zip(seeded, invars) if classes[i] == 'segment']
    )
    return [
        'member' if m else 'segment' if s else None
        for m, s in zip(by_member, by_segment, strict=True)
    ]


def _dependent_outputs(jaxpr: Any, seeds: Iterable[Any]) -> list[bool]:
    """Which outputs depend on the input variables `seeds`, rather than only on the others.

    One forward pass in program order: an operation with any dependent input makes all its
    outputs dependent. It is conservative, and sound: an output that does not depend on the
    member-carrying inputs cannot vary from one member to the next.
    """
    dependent = set(seeds)
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


def _expanded(
    leaf: Array, stride: int | str, size: int, owner: Array | None, axis: int = 0
) -> Array:
    """A strided or segment leaf with one entry per member, for leaves small enough to gather."""
    if stride == 'segment':
        assert owner is not None  # a segment leaf comes with an owner index
        return jnp.take(leaf, owner, axis=axis)
    assert isinstance(stride, int)
    if stride == 1:
        return leaf
    return jnp.repeat(leaf, stride, axis=axis, total_repeat_length=size)


def _common_owner(ensembles: Sequence[StackedEnsemble]) -> tuple[Array | None, int | None]:
    """The owner index and segment count the ensembles share, if any of them has one.

    The owner indices are compared when their values are known; under a trace, sharing them is a
    promise the engine does not check.

    Raises:
        ValueError: If two ensembles group their members differently.
    """
    grouped = [e for e in ensembles if e.owner is not None]
    if not grouped:
        return None, None
    first = grouped[0]
    for e in grouped[1:]:
        if e.segments != first.segments:
            raise ValueError('the lifted ensembles group their members in different segments')
        if e.owner is first.owner or isinstance(e.owner, Tracer) or isinstance(first.owner, Tracer):
            continue
        if not np.array_equal(np.asarray(e.owner), np.asarray(first.owner)):
            raise ValueError('the lifted ensembles group their members in different segments')
    return first.owner, first.segments


def _is_segment(kind: Any) -> bool:
    return kind in ('segment', 'segment_sum')


def _is_stacked(kind: Any) -> bool:
    """Whether outputs of `kind` keep entries along a member axis, split over the devices."""
    return _is_segment(kind) or (isinstance(kind, int) and not isinstance(kind, bool))


def _stack_length(kind: Any, per_device: int, segments: int | None) -> int:
    """The entries of one device for a stacked output of `kind`."""
    if _is_segment(kind):
        assert segments is not None
        return segments
    return per_device // kind


def _local_segments(ensemble: Ensemble, devices: int) -> int | None:
    """The segments of one device, None without an owner index."""
    if ensemble.segments is None:
        return None
    if ensemble.segments % devices:
        raise ValueError(f'{ensemble.segments} segments do not split over {devices} devices')
    return ensemble.segments // devices


def _axis_size(axis: str, mesh: Mesh | AbstractMesh | None = None) -> int:
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty or axis not in mesh.shape:
        raise ValueError(f'no mesh axis {axis!r} is set: use `jax.set_mesh`')
    return mesh.shape[axis]


def _leaves_mesh(leaves: Iterable[Any], axis: str) -> Mesh | AbstractMesh | None:
    """The mesh carrying `axis`, read from the shardings of the leaves, or None.

    A traced leaf exposes no sharding (reading it raises, so `getattr` yields None): under a
    trace, every leaf is skipped and the engine falls back to the ambient mesh of the tracing
    thread.
    """
    found = None
    for leaf in leaves:
        mesh = getattr(getattr(leaf, 'sharding', None), 'mesh', None)
        if mesh is None or mesh.empty or axis not in mesh.shape:
            continue
        if found is None:
            found = mesh
        elif mesh != found:
            raise ValueError(f'the leaves are laid out over different meshes carrying {axis!r}')
    return found


def _abstract(leaf: Any) -> Any:
    if _is_array(leaf):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
    return leaf


def _normalized_strides(
    leaves: list[Any], strides: PyTree[Stride], is_member: Callable[[Any], bool] | None = None
) -> tuple[Stride, ...]:
    """One stride per leaf, the leaves that are not `is_member` forced to None (they are shared).

    By default, the member leaves are the arrays.
    """
    if is_member is None:
        is_member = _is_array
    if strides is None or isinstance(strides, int | str):
        flat: list[Stride] = [strides] * len(leaves)
    else:
        flat = jax.tree.flatten(strides, is_leaf=lambda x: x is None)[0]
        if len(flat) != len(leaves):
            raise ValueError(
                f'{len(flat)} strides for {len(leaves)} leaves: strides must be a single '
                f'value or a pytree aligned with the stacked leaves'
            )
    return tuple(s if is_member(leaf) else None for leaf, s in zip(leaves, flat))


def _normalized_axes(
    leaves: list[Any], strides: tuple[Stride, ...], axes: PyTree[int | None]
) -> tuple[int | None, ...]:
    """One non-negative member axis per leaf, None on the shared leaves."""
    if axes is None or isinstance(axes, int):
        flat: list[int | None] = [axes] * len(leaves)
    else:
        flat = jax.tree.flatten(axes, is_leaf=lambda x: x is None)[0]
        if len(flat) != len(leaves):
            raise ValueError(
                f'{len(flat)} axes for {len(leaves)} leaves: axes must be a single '
                f'value or a pytree aligned with the stacked leaves'
            )
    out: list[int | None] = []
    for leaf, s, a in zip(leaves, strides, flat):
        if s is None:
            out.append(None)
            continue
        if a is None:
            raise ValueError('a strided leaf needs a member axis')
        ndim = len(leaf.shape)
        if not -ndim <= a < ndim:
            raise ValueError(f'axis {a} is out of bounds for a leaf of shape {leaf.shape}')
        out.append(a % ndim)
    return tuple(out)


def _is_array(leaf: Any) -> bool:
    return isinstance(leaf, jax.Array | np.ndarray)


def _is_abstract(leaf: Any) -> bool:
    return isinstance(leaf, jax.ShapeDtypeStruct)


def _seeded(
    template: PyTree[Any], strides: tuple[Stride, ...]
) -> tuple[list[Any], list[Literal['member', 'segment'] | None], Any]:
    """The `_seeds` of an ensemble with one stride per leaf of its template."""
    leaves, treedef = jax.tree.flatten(template)
    classes: list[Literal['member', 'segment'] | None] = [
        None if s is None else 'segment' if s == 'segment' else 'member' for s in strides
    ]
    return leaves, classes, treedef


def _entry(i: Int[Array, ''] | int, stride: int | str, owner: Int[Array, ' _n'] | None) -> Any:
    """The entry of a leaf of `stride` that member `i` reads: `owner[i]` for a segment leaf."""
    if stride == 'segment':
        assert owner is not None  # a segment leaf comes with an owner index
        return owner[i]
    assert isinstance(stride, int)
    return i // stride


def _check_segments(owner: Any, segments: int | None, size: int, devices: int) -> None:
    """Check an owner index against its segment count, and its members' placement on devices.

    The members and the segments are split over `devices` in contiguous blocks; the members of
    block d must own segments of block d. Under a trace, or held in part by other processes,
    the values of `owner` are not known here and go unchecked.

    Raises:
        ValueError: If `owner` and `segments` do not come together, if `owner` is not one index
            per member, or if a member owns a segment out of range or of another device.
    """
    if (owner is None) != (segments is None):
        raise ValueError('segments and their owner index come together')
    if owner is None:
        return
    assert segments is not None
    if owner.shape != (size,):
        raise ValueError(f'owner has shape {owner.shape}, expected ({size},)')
    if size % devices or segments % devices:
        raise ValueError(
            f'{size} members in {segments} segments do not split over {devices} devices'
        )
    if isinstance(owner, Tracer) or owner.size == 0:
        return
    if isinstance(owner, jax.Array) and not owner.is_fully_addressable:
        return
    owner = np.asarray(owner)
    if owner.min() < 0 or owner.max() >= segments:
        raise ValueError(f'owner indices must lie in [0, {segments})')
    member_device = np.arange(owner.size) // (owner.size // devices)
    if np.any(owner // (segments // devices) != member_device):
        raise ValueError(
            f'over {devices} devices, the members of a device must own segments of that device'
        )


# The resident ensembles are pytree nodes, so an operator can hold one where it holds a pytree of
# operators: its arrays are leaves of the operator, and the sizes, strides and policy are static.
# Unflattening bypasses validation, as tree transformations put placeholders in the leaves.


def _stacked_unflatten(aux: tuple[Any, ...], children: Iterable[Any]) -> StackedEnsemble:
    size, strides, axes, policy, segments = aux
    stacked, valid, owner = children
    ensemble = object.__new__(StackedEnsemble)
    object.__setattr__(ensemble, 'size', size)
    object.__setattr__(ensemble, 'stacked', stacked)
    object.__setattr__(ensemble, 'strides', strides)
    object.__setattr__(ensemble, 'axes', axes)
    object.__setattr__(ensemble, 'valid', valid)
    object.__setattr__(ensemble, 'policy', policy)
    object.__setattr__(ensemble, 'owner', owner)
    object.__setattr__(ensemble, 'segments', segments)
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
        [
            (GetAttrKey('stacked'), e.stacked),
            (GetAttrKey('valid'), e.valid),
            (GetAttrKey('owner'), e.owner),
        ],
        (e.size, e.strides, e.axes, e.policy, e.segments),
    ),
    _stacked_unflatten,
    lambda e: ([e.stacked, e.valid, e.owner], (e.size, e.strides, e.axes, e.policy, e.segments)),
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
