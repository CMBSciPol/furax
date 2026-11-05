# Ensembles: one computation over many members

## Summary

An ensemble is N members sharing one structure: the same code, different array values. A member is
a pytree — an operator, the fields read for an observation, an operator paired with its input —
and two members only differ by the values of their array leaves. The engine,
[`map_reduce`][furax.core.map_reduce], applies a function to every member through a single traced
body, whatever N, and combines the outputs as declared per leaf: summed over the members, or kept
at the stride it varies with.

Three representations hold the members' data; a [`Policy`][furax.core.Policy] decides how many
members run together and over which devices; [`lift`][furax.core.lift] is the member-wise algebra;
and the block composites accept ensembles of operators, so a multi-observation system is an
ordinary operator expression.

## Motivation

The multi-observation mapmaker applies the same computation to every observation and to every
detector batch: pointing, weighting, Gram assembly. Written naively, this either

- compiles one body per member, so compilation time and program size grow with the dataset, or
- evaluates every member in one expression, which XLA is free to schedule together, every
  member's TOD-sized temporaries coexisting in memory.

The engine makes the parallelism explicit instead, with one traced body:

```
shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ
```

The members are split over the D devices of a mesh axis, each device runs G rounds one after the
other, and the K members of a round are evaluated together, vectorized. K bounds the memory peak
(one round's temporaries), the loop keeps the compiled program constant in N, and the mesh axis
sets the distribution. `Policy(axis, batch)` chooses D and K;
[`Policy.from_budget`][furax.core.Policy.from_budget] derives K from a device memory budget, using
XLA's memory analysis of one abstract member.

## Members, template, validity

Every [`Ensemble`][furax.core.Ensemble] exposes:

- `size`: the number of members;
- `template`: the member structure — `jax.ShapeDtypeStruct` on the per-member leaves, plain values
  on the leaves every member shares. Structure and tag queries go through the template, so they
  never materialize a member: no host read on a host ensemble, no gather on a sharded one;
- `valid`: one boolean per member, `False` on padded members, which are masked out of the summed
  outputs; `None` when every member counts;
- `policy`: how the engine walks the ensemble.

## The representations

### ListEnsemble

[`ListEnsemble`][furax.core.ListEnsemble] holds N member pytrees in a Python list. The engine
selects one member per iteration with a multiplexing `lax.switch`: each branch only routes the
arrays of one member to the single compute body, so the program holds one body plus a mux whose
size is linear in N, instead of N bodies. It runs on a single device and cannot be batched: it is
the representation of convenience for operators already at hand, such as the operands of an
[`AdditionOperator`][furax.core.AdditionOperator] sharing a structure.

### StackedEnsemble

[`StackedEnsemble`][furax.core.StackedEnsemble] holds one pytree whose array leaves carry a
member axis, with a stride per leaf ([`Stride`][furax.core.Stride]): a stride-s leaf has
one entry per run of s consecutive members, and a stride-`None` leaf — like any non-array leaf —
is shared by the whole ensemble. The member axis is leading by default; `axes` places it per
leaf, and the engine slices, batches and shards each leaf along its own axis in place, so a leaf
whose natural layout puts the member dimension elsewhere — a Stokes-first map stacked over its
frequency axis, a demodulated TOD over its detector axis — needs no transposition. Stacked engine
outputs always come back with a leading member axis. It is the only representation the engine can
batch (`Policy.batch`) and shard (`Policy.axis`).

Sharding is explicit: [`shard(axis)`][furax.core.StackedEnsemble.shard] lays the strided leaves
out over the mesh axis (`PartitionSpec` on each leaf's member axis) and records the axis in the
policy. At evaluation,
the engine reads the mesh back from the shardings of the laid-out leaves, falling back to the
ambient mesh — the mesh set by `jax.set_mesh` is thread-local, and reading it from the leaves lets
any thread evaluate the ensemble, such as the host-callback thread of `jax.pure_callback`. The
engine runs the members under `jax.shard_map`, which does not reshard: an ensemble whose leaves
are not laid out over the axis fails rather than silently gathering.

### HostEnsemble

[`HostEnsemble`][furax.core.HostEnsemble] reads its members from the host: `read(i)` returns the
per-member arrays of member i, and the engine runs a Python loop where each round is one compiled
launch on `Policy.batch` members per device while the host reads the next round in a thread
(`prefetch`). A pass over data too large for memory runs at the speed of the slower of the disk
and the devices, not their sum. The stacked outputs land sharded over the mesh axis in position
order, ready to be used as the leaves of a resident `StackedEnsemble` with the same layout: the
output of a pass over the data is the ensemble the solve then iterates on.

## The engine

```python
outputs, resolved = map_reduce(apply, ensemble, out_spec, *args)
```

evaluates `apply(member, *args)` on every member. `args` are the shared inputs every member
receives whole — arrays are best handed to the engine rather than closed over `apply`, so that a
sharded run treats them as explicit inputs instead of captured constants.

`out_spec` declares, per output leaf (as a prefix of the output tree, [`OutSpec`][furax.core.OutSpec]):

- `'sum'`: the leaf is reduced over the members (padded members masked), then across the devices,
  and held by all of them;
- stride 1: one entry per member, stacked along a new leading axis, staying with the device that
  computed it;
- stride s: one entry per run of s members that share it — a promise the engine does not check;
- stride `None`: the same for every member, returned once;
- `'auto'`: resolved to 1 or `None` by whether the leaf depends on the member in the traced
  computation.

The second return value is the specification with `'auto'` resolved: together with the stacked
outputs it is the template and strides of the next ensemble, so engine calls chain — the
mapmaker's pass over the data returns the ensemble its solver then runs on.

## Ensembles of operators

`lift(f, *ensembles)` applies `f` member-wise and returns the ensemble of its outputs. On stacked
ensembles, a *structural* `f` — one that only rearranges its members' leaves, such as composing,
transposing or reducing operators — is detected by tracking leaf identity, and the output reuses
the input arrays with their strides, at no cost; a computed `f` is vectorized by `jax.vmap` and
its outputs come back at stride 1.

The composites close the loop:

- `AdditionOperator(ensemble)` sums the members' outputs through the engine;
- [`BlockRowOperator`][furax.core.BlockRowOperator],
  [`BlockColumnOperator`][furax.core.BlockColumnOperator] and
  [`BlockDiagonalOperator`][furax.core.BlockDiagonalOperator] accept an ensemble of blocks;
- transposition, reduction rules and algebraic simplifications apply member-wise through `lift`,
  structurally where possible — `(H.T @ H).reduce()` on an ensemble of pointings never touches
  the data.

A sequential sum of plain operands uses the same machinery: operands sharing a structural
signature — same class, same static fields, same array shapes — form a group evaluated through a
`ListEnsemble`, and the groups are chained by `jax.lax.optimization_barrier`, so the temporaries
of two groups never coexist.

## Example

The toy multi-observation problem: each observation is a pure pointing `yᵢ = x[pᵢ]`, and solving
the normal equations `(Σᵢ Hᵢᵀ Hᵢ) x = Σᵢ Hᵢᵀ yᵢ` recovers the map. With the pointings stacked
over a mesh axis `'obs'`:

```python
hs = lift(
    lambda m: IndexOperator(m['p'], in_structure=structure),
    StackedEnsemble(n_obs, {'p': pointings}, policy=Policy(batch=2)),
).shard('obs')
H = BlockColumnOperator(hs)

y = H(x)  # the stacked TOD stays laid out over the devices
normal = (H.T @ H).reduce()  # structural: no data is moved
b = H.T(y)  # summed across the devices, held by all of them
x_hat, _ = jax.scipy.sparse.linalg.cg(normal, b)
```

This is the integration test `test_least_squares_solve_over_a_sharded_ensemble`.

## In the mapmaker

- The pass over the data is a `HostEnsemble` of each bucket's slots: every round launches one
  observation per device while its process reads the next, so one observation per device is in
  memory at a time. Its stacked outputs are the bucket's model, already laid out for the solve.
- The normal operator walks a `StackedEnsemble` of the bucket's slots, each slot running its
  detector batches one at a time, so the solve holds one batch of TOD-sized temporaries per
  device.
- Inside the RHS kernel, the detector batches of one observation form a `StackedEnsemble` of
  their starting detectors, each member slicing its batch out of the observation's fields.
