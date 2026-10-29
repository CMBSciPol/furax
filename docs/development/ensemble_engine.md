# Design note: ensembles of operators

*Status: proposal. This note is the first deliverable of the work on distributing per-observation
computations; the engine, the map-maker migration and the retirement of `StreamOperator` follow it
as separate pull requests.*

## Problem

The multi-observation map-maker solves A x = b with

    b   = Σᵢ Hᵢᵀ Nᵢ⁻¹ dᵢ         (right-hand side: one pass over the data)
    A x = Σᵢ Hᵢᵀ Nᵢ⁻¹ Hᵢ x       (left-hand side: applied at every CG iteration)

where i runs over observations. Four facts shape every design decision:

1. **The observations are heterogeneous.** Detector and sample counts differ from one observation
   to the next, and JAX needs static shapes: some padding is unavoidable, and the question is what
   to pad to and what to evaluate together.
2. **The time-ordered data does not fit.** The temporaries of one term, Hᵢ x and Nᵢ⁻¹ (Hᵢ x), are
   the size of an observation's TOD; the TOD volume of a run exceeds the memory of all the devices
   together. Terms must be evaluated a few at a time, reusing the same buffers, and the one pass
   that reads the data is bound by the disk, so reading the next items must overlap with computing
   the current ones.
3. **The map is small and replicated.** Every device holds the whole map, the accumulators are
   map-sized, and the only cross-device communication is a sum of map-sized objects. The map is
   never distributed.
4. **An observation is an integer.** The reader designates an item by its index
   (`AbstractReader.read_host(data_index)`), and that integer is all the distribution layer needs
   to know about it.

A fifth fact is easy to miss: the pass over the data is not only a sum. Besides b and the hit
map, it produces **per-observation outputs** -- noise fits per detector batch, the right-hand side
of the template amplitudes, and the compact leaves (quaternions, noise bands, masks) that the
left-hand side will need at every CG iteration. One pass must return both reduced and per-member
outputs.

Today these concerns are entangled in `StreamOperator` (`furax.mapmaking.streaming`), the buckets
of `furax.mapmaking.layout`, and the accumulation loop of `MultiObservationMapMaker`. The two sums
are evaluated by two different engines: a host-driven loop of `shard_map` launches for the pass
over the data, and resident stacked `StreamOperator`s summed by `AdditionOperator(sequential=True)`
for the left-hand side. The agreement between what the first engine stores and what the second one
reads -- which observation sits at which position of which device -- is maintained by three
separate pieces of index arithmetic that nothing checks.

### What `StreamOperator` got right, and what it conflated

Its *boundary* -- per component, whether the input and the output carry the ensemble index
(`in_stacked`/`out_stacked`) -- is the right description of how an ensemble closes into an operator;
below it is carried by the block composites and the engine's output specification instead of a
dedicated class. What it conflated are three things that this note keeps apart: the
**representation** of the ensemble's data (it only knows stacked, resident leaves, with `group_size`
as a layout trick for shared ones), the **policy** that walks the index (always shard over devices
then scan one slice at a time, on the device, never batched, never host-driven), and an
**implementation** built on equinox that the library is moving away from.

## The design in one paragraph

An **ensemble** is N members sharing one structure -- the same code, different array values -- held
as one template whose array leaves carry a member axis. Everything else follows from treating the
member axis uniformly, in three layers that know nothing about map-making. The **engine** is one
function, a memory-bounded, device-sharded map-reduce: apply a function to every member, sum some
output leaves over the ensemble, return the others stacked at the stride they vary with, declared
or inferred from the trace. The **algebra** is the existing operator algebra lifted member-wise:
the block composites close an ensemble into an operator, the composition rules combine ensembles
member by member, and `vmap` over the member axis is what lifts a leaf-computing operation. The
**policy** factorises the member axis onto the
machine, N = D × G × K: shard over D devices, loop over G rounds, batch K members per round, with
K chosen from the memory budget -- the members that fit together run in parallel, the rest are
sequenced. A non-summed output of the engine is itself an ensemble with the same layout, which is
the property that chains the two map-making sums without a copy.

## Concepts

### Ensemble

An **ensemble** is a set of N members sharing a static signature: the same code, different data. A
member is a pytree -- an operator, a parameter set, an operator paired with its input -- and two
members of an ensemble share their treedef, their static fields and their non-array leaves; they
differ only by the values of their array leaves. `AbstractLinearOperator._static_signature` is the
membership test for operators.

The ensemble's data can be held in three ways, and the way it is held decides what the engine may do
with it:

| representation | `fetch(i)` | batch of K | notes |
|---|---|---|---|
| **list**: N member pytrees | `lax.switch` over N branches returning the arrays of member i | no | HLO grows by ~2 kB per member: fine for tens of members, not thousands |
| **stacked**: one pytree whose array leaves carry a leading axis | `dynamic_index` on every leaf | `dynamic_slice` | the only representation `vmap` and `shard_map` can read |
| **host**: a callable returning host arrays | host read, then transfer | one transfer per round | lives on the host; every evaluation pays the transfer |

Converting a list into a stacked representation is the one operation that copies all the data;
it is never done implicitly. An ensemble whose members hold *compact* parameters and recompute
TOD-sized objects inside the applied function (on-the-fly pointing from quaternions) is just a
stacked ensemble with small leaves: nothing special is needed for it.

A leaf may be **shared** by a run of consecutive members: the detector batches of one
observation share its boresight quaternions. A per-leaf *stride* records this: stride 1 for a leaf
with one entry per member, stride s for one entry per s consecutive members, `None` for a leaf
shared by the whole ensemble. This replaces `StreamLayout.group_size`. It also expresses a two-level
index (observation × detector batch) without the engine knowing about levels: the ensemble flattens
the index and the strides keep the sharing. The same vocabulary describes the engine's outputs: a
leaf comes back summed, or at the stride it varies with (see the API sketch).

Padding makes an ensemble's size a multiple of what the policy requires; padded members are marked
in a boolean `valid` array, and the engine discards their contribution to the summed outputs.

The ensemble also carries its **placement**: a stacked ensemble laid out over a mesh axis holds
consecutive members per device, and that layout *is* the assignment of members to devices. The
policy that walks the ensemble belongs to the ensemble, not to the operators built on it: the same
ensemble data accepts any batch size K that divides its per-device count.

### Lifting: the algebra of ensembles

A function on members lifts to a function on ensembles. Two cases, by what the function does to
the leaves:

- **Structural** operations -- composing two operators with `@`, transposing, wrapping in a block
  composite -- only rearrange the treedef; the leaves pass through untouched, strides intact. An
  ensemble of Hᵢ and an ensemble of Wᵢ compose into an ensemble of Wᵢ ∘ Hᵢ at no cost.
- **Leaf-computing** operations -- a reduction rule that folds two homotheties into one, an
  operator constructor that precomputes from its arguments -- lift through `vmap` over the strided
  leaves. The control flow of `reduce` depends only on types and static fields, which all members
  share, so the output treedef is member-independent and the rule executes once, batched, for the
  whole ensemble: N homothety fusions are one elementwise multiply of length N.

Lifting is what makes "build Hᵢᵀ Wᵢ Hᵢ and reduce it, for all i" a single `vmap` of the
member-level expression the map-maker already writes.

### Closing an ensemble into an operator

At the scale of an ensemble there is no new operator class. The existing composites close an ensemble,
each fixing which side carries the member index:

| composite over an ensemble | input | output | meaning |
|---|---|---|---|
| `AdditionOperator` | shared | summed | Σᵢ Hᵢ x |
| `BlockRowOperator` | per member | summed | Σᵢ Hᵢ xᵢ |
| `BlockColumnOperator` | shared | stacked | (H₀ x, …, Hₙ₋₁ x) |
| `BlockDiagonalOperator` | per member | stacked | (H₀ x₀, …, Hₙ₋₁ xₙ₋₁) |

Their `blocks`/`operands` container accepts an ensemble where it accepts a list today (a list *is*
the degenerate ensemble), and their `mv` calls the engine with the corresponding input sharing and
output specification. The four composition rules of `furax.core._blocks` keep their statement --
`BlockRow(Aᵢ) @ BlockDiagonal(Bᵢ) = BlockRow(Aᵢ @ Bᵢ)` and so on -- with the member-wise
composition computed by lifting (structurally, so it is free); `BlockRow @ BlockColumn` produces
the `AdditionOperator` over the ensemble of compositions. The map-maker's algebra stays what it is
today, Hᵀ W H written with composites and reduced by rules; the ensemble enters as the container of
H's blocks and traverses the rules.

Two facts keep the closures this simple:

- **A stacked input or output at ensemble scale only exists when it fits.** The TOD-sized stacked
  objects never exist as vectors; a `BlockColumnOperator` over an ensemble is a symbolic intermediate
  that the rules eliminate inside Hᵀ W H before anything is applied. Applying one directly is
  legal exactly when its stacked side is compact, and that case is the pass over the data, which
  is not an operator application but an engine call.
- **Per-member unknowns are addressed inside the member.** The template amplitudes of the joint
  map–template system form one shared vector a, like the map: member i reads its slice through an
  index operator whose offset is a leaf of the ensemble, and the transpose scatter-adds the slice
  back, which is a summed output. The joint system is then a plain `AdditionOperator` over the
  ensemble of bordered members (Hᵢ, Tᵢ Sₚᵢ)ᵀ Nᵢ⁻¹ (Hᵢ, Tᵢ Sₚᵢ) -- shared input (map, a), summed
  output -- and needs no mixed-layout operator. Under a K-batched round the K scatter-adds cost K
  amplitude-sized temporaries; a is compact, and an indexed-add output specification is the
  escape hatch if that ever matters.

What survives of `StreamOperator`'s three reduction rules is two generic statements, both encoding
"one pass over the data": a plain composite (sum, block row) of same-ensemble closures is the
closure of the member-wise composite ensemble; and composing ensembles member-wise *before* closing
is always preferred, which the rules above already do. The third rule (scalar × ensemble) falls out
of lifting.

### Policy

Evaluating an ensemble means walking the member index. The JAX transforms are the ways of walking
it, and they compose by factorising the ensemble size, N = D × G × K:

    shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ      (devices ∘ sequential rounds ∘ batched members)

Summed outputs propagate level by level -- `sum` over the K members of a round, accumulation over
the G rounds, device sum over D -- and stacked outputs are concatenated along the same levels, so
that a device ends up holding, in position order, the outputs of the members it evaluated.

Each level is what it is for a reason:

- **shard_ᴅ** is the only level that brings more compute. Devices run independent streams; the one
  collective is the device sum of the summed outputs at the end.
- **vmap_ᴋ is the intra-device parallelism.** On a GPU, XLA executes one compute stream: the
  kernels of an unrolled Python loop, or of successive dispatches, run one after the other
  (horizontal fusion only merges small elementwise fusions, not FFTs, GEMMs or scatters). A Python
  loop pipelines -- dispatch and host reads overlap with compute -- but never overlaps two
  members' kernels. The only way to run K members in parallel on one device is to give them a
  batch axis and one kernel: `vmap`. (XLA:CPU's thunk executor *does* overlap independent
  kernels, which is why CPU timings of unrolled loops mislead about GPUs.)
- **loop_ɢ sequences what does not fit.** The batched level sits directly under the loop, so K
  members' temporaries, never G's, are live at once, reused from one round to the next.

K is therefore the memory ↔ parallelism knob, chosen from the budget: the members that fit
together in device memory run in parallel, the rest are sequenced by the loop. The loop level has
two implementations, chosen by the representation:

- a **device loop** (`lax.fori_loop`) for resident ensembles: one compiled body, temporaries
  allocated once for the whole loop;
- a **host loop** for host ensembles: one compiled body launched once per round, while the host
  reads and transfers the next round in a thread, the summed accumulators riding the launches as
  donated buffers. This is what the current accumulation does, and it is what makes the pass over
  the data run at the speed of the disk rather than the sum of disk and compute. An `io_callback`
  inside a device loop would serialise the two.

Rounds are **aligned on the strides**: a stride-s leaf is constant within a round (s a multiple
of K, passed to `vmap` unbatched) or covers whole entries (K a multiple of s). Aligning rounds on
observations keeps the per-observation leaves unbatched and lets the host loop prefetch one
observation at a time.

One invariant is easy to violate from outside: an ensemble laid out over a mesh axis must be
evaluated through the mesh (`shard_map`). Handing its sharded leaves to a plain `jit` makes GSPMD
insert all-gathers to honour the loop's dynamic slices, and the temporaries grow with N.

In this frame:

| today | in this frame |
|---|---|
| `StreamOperator` | a composite closing a stacked ensemble, shard_ᴅ ∘ loop_ɴ⁄ᴅ (K = 1), device loop |
| `_accumulate_bucket` and its rounds | engine call on a host ensemble, shard_ᴅ ∘ loop_ɢ (K = 1), host loop |
| `AdditionOperator(sequential=True)`, `BlockRowOperator` | closures of a list ensemble, loop_ɴ |
| `AdditionOperator(sequential=False)` | Python unroll |
| `StreamLayout.group_size` | a leaf stride |
| the constant segments of `StreamOperator` | leaves with stride `None` |
| `in_stacked` / `out_stacked` | which composite closes the ensemble; `out_spec` in the engine |
| buckets | shape classes: one ensemble per class, padded to D × G × K |
| `detector_batch_size` | the size of a member; K members per round |
| `_store_round` + `_merge_leading_axes` + slot arithmetic | stacked outputs of the engine, in position order |

## API sketch

Names are provisional; everything below lives in `furax.core`, knows nothing about observations,
and uses no equinox.

```python
type Stride = int | None   # entries per member on a leaf's leading axis: 1, s, or None (shared)


class Ensemble(Protocol):
    size: int                              # N, a multiple of D × K after padding
    template: PyTree                       # the structure of one member
    strides: PyTree[Stride]                # per leaf of the template
    valid: Bool[Array, ' N'] | None        # False on padded members
    policy: Policy                         # how the engine walks this ensemble

    def fetch(self, i: Int[Array, '']) -> PyTree: ...                 # member i, in a trace
    def round_slice(self, start, k) -> tuple[PyTree, PyTree]: ...     # (leaves with a leading k,
                                                                      #  leaves shared this round)


class ListEnsemble(Ensemble):     # members: list[PyTree]; fetch = lax.switch (the multiplexer of
                                  # AdditionOperator(sequential=True)); K = 1
class StackedEnsemble(Ensemble):  # one pytree, leaves with a leading axis; shard() lays it out
                                  # over the mesh axis, consecutive members per device
class HostEnsemble(Ensemble):     # read: Callable[[int], tuple[np.ndarray, ...]]; the host loop
                                  # reads it, one round ahead


@dataclass(frozen=True)
class Policy:
    axis: str | None = None      # mesh axis to shard over, None for a single device
    batch: int = 1               # K: members evaluated together in one round

    @classmethod
    def from_budget(cls, ensemble, apply, out_spec, budget: int, *, axis=None, margin=0.8):
        """K from a memory budget: compile `apply` on one member, read its temporaries."""


type OutSpec = PyTree[Literal['sum', 'auto'] | Stride]   # prefix of apply's output


def map_reduce(apply, ensemble, out_spec, *args) -> tuple[PyTree[Array], OutSpec]:
    """Apply `apply(member, *args)` to every member: `args` are the shared inputs, replicated.
    Per output leaf, `'sum'` reduces over the ensemble (padded members masked); a stride keeps it:
    1 stacked with a leading axis of size N in position order, sharded over the mesh axis like the
    ensemble; s one entry per run of s members sharing it; None one copy, member-invariant.
    `'auto'` resolves to 1 or None by whether the leaf depends on the member in the traced body.
    Returns the outputs and the specification with `'auto'` resolved.
    Evaluated as shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ with the ensemble's policy, a single traced body."""


def lift(f, *ensembles) -> StackedEnsemble:
    """Member-wise `f` over ensembles of the same size: `vmap` over the strided leaves.
    The algebra of ensembles: construction, composition, `reduce`, transpose."""
```

The engine checks that the ensemble size is a multiple of D × K, traces the round body once --
resolving the `'auto'` leaves on that abstract trace (see decisions) -- then emits `shard_map` over
the mesh axis when the ensemble is laid out (each device holding N / D consecutive members), then
the loop over the G rounds of the device -- `lax.fori_loop` for a resident ensemble, a Python loop
launching the same compiled round for a host ensemble -- and, inside a round, the round's slice of
the ensemble, `vmap(apply)` over the K per-member leaves with the round-shared leaves unbatched,
the `valid` mask zeroing the summed contributions of padded members, summed leaves reduced over K
and added to the carry, stride-1 leaves written at their positions, coarser leaves once per run --
computed once, since what does not depend on the member goes through the `vmap` unbatched. After
the loop, the device partials of the summed leaves are summed and replicated; the stacked leaves
stay where they were produced, sharded over the mesh axis in position order, and a stride-`None`
leaf is one copy, identical on every device. With K = 1 and a list ensemble the body is exactly
the multiplexer of `AdditionOperator(sequential=True)`.

A **non-summed output is an ensemble**: same size, same placement, and the strides `map_reduce`
returns are the strides of its leaves -- the output specification and the ensemble description are
one vocabulary, so an output whose strides were inferred feeds the next call unchanged. This is the
engine's central invariant -- what one call stacks at position p, another call with the same
ensemble layout reads at position p -- and it replaces the implicit agreement between today's
accumulation storage and the `StreamOperator`s with a property of the object.

The composites close ensembles through the same call: `AdditionOperator` over an ensemble evaluates
`map_reduce(lambda op, x: op(x), ensemble, 'sum', x)`, and the other three likewise with their input
sharing and output specification. `AdditionOperator(sequential=True)` and `BlockRowOperator` keep
their public behaviour and become closures of a `ListEnsemble` per signature group.

## Decisions

The brief asked for the following decisions to be explicit.

**No operator form of the engine.** An earlier draft closed ensembles with a dedicated
`EnsembleOperator` carrying per-component `in_stacked`/`out_stacked` flags. It is not needed: the
only closures the solver applies are sums (the LHS, the preconditioner's diagonal), the stacked
outputs are produced by the pass over the data, which is an engine call, not an operator, and the
one mixed case -- the joint map–template system -- is absorbed into the members by addressing the
amplitudes with per-member slices of one shared vector. The existing composites, with an ensemble as
their block container, say everything `EnsembleOperator` would have said, and the algebra (transpose,
rules, tags) comes with them instead of being reimplemented.

**Output strides, and `'auto'`.** The output specification reuses the stride vocabulary: `'sum'` is
the one reduction, and any other output leaf comes back at a stride -- 1 per member, s once per run
of s members, `None` once for the whole ensemble. The coarse strides exist because a pass also
produces leaves that vary less than the member index -- the bases an observation's detector batches
share, coarse amplitudes that follow from the bases alone -- and stacking those N times would waste
memory and lose the sharing the next ensemble needs. A declared stride is a promise the engine does
not check. `'auto'` infers 1 versus `None` for outputs whose variability the caller cannot spell
out -- an operator rebuilt per member mixes per-member leaves (its weight diagonal) with
ensemble-wide ones (shared bases) -- and the information is already in the program: the inference
is one dataflow pass over the round's trace, which the engine makes anyway. Variation between
members enters the trace only through its inputs (the member index, and the member-axis leaves of
the round's slice, passed as inputs for this purpose), so an output reachable from an input may
vary with the member -- stride 1 -- and an unreachable one is member-invariant by construction --
stride `None`. Inferred `None` is therefore sound exactly where declared `None` is trusted; the
other direction is conservative (x · 0 stacks: an extra axis, never a wrong value); `'auto'` never
resolves to `'sum'`, a reduction being an intent rather than a dataflow property; and a stride s is
never inferred, periodicity being invisible to dataflow. Resolution happens on the abstract trace
before the collective is laid out, so inference does not depend on the policy.

**Shape classes.** `partition_padded` stays: a dynamic programme over the observations sorted by
(detector count, sample count), minimising padded volume under `max_buckets`. Its cost moves from
"slots rounded to the device count" to the padding the policy actually imposes, D × G × K
members. Within a class every padded member has the same cost, so balance is a matter of counts
(see assignment); the only cross-class cost is one compiled body per class, which is what
`max_buckets` bounds. `max_buckets` stays a user knob with its current default. The sum over
classes is a plain Python-level addition of per-class closures: a handful of operands, each hiding
its own ensemble.

**K per class.** `Policy.from_budget` compiles `apply` on one member of the class (abstract
`ShapeDtypeStruct` leaves, no data), reads `memory_analysis().temp_size_in_bytes`, and sets
K = ⌊margin × (budget − resident) / temporaries⌋, clipped to [1, batches per observation].
`budget` defaults to the device's `memory_stats()['bytes_limit']` when the backend reports it and
must be given explicitly otherwise (host devices report none). `resident` is the byte size of the
leaves of the stacked ensembles already placed on the device, so classes are laid out one after the
other with a shrinking budget. The chosen K, the measured temporaries and the budget are logged.
If one member does not fit, `from_budget` raises and names the two knobs that shrink a member:
`detector_batch_size` and the shape classes. Past saturation of the device, growing K buys memory
and no time, so a future refinement may cap K from a measured throughput curve; the budget rule
stays the contract.

**Left-hand side that does not fit.** Version 1 refuses: if the resident ensembles of the
left-hand side exceed the budget, `make_maps` raises before the CG starts, stating the resident
size and the budget. Streaming the left-hand side from the host at every CG iteration is
implementable with the same engine (a host ensemble and a host loop) but pays a full transfer of the
parameters per iteration; it is a follow-up, enabled explicitly, never a silent fallback.

**Assignment.** The unit of assignment is the observation, not the detector batch: an
observation is read once, by the process owning the device that evaluates it. Within a class all
padded observations cost the same (same envelope, same batch count), so the balanced assignment
is a round-robin of the class's observations over the devices, the starting device rotating from
one class to the next so that the extra observations of successive classes land on different
devices. Observations are taken in index order, which makes the assignment a pure function of the
probed shapes and the device count: every process computes the same one. Padded observations are
the last of each device, so filler spreads across devices instead of piling up on the last ones.
The assignment is materialised once, as the order of the ensemble's positions; everything downstream
-- which process reads which observation, where a stacked output lands, what the CG fetches --
follows from the position, never recomputes the assignment. The order of floating-point reductions
-- K-sum, round carry, device sum in mesh order -- is fixed by construction and does not depend on
the process layout for a given device count; `tests/mapmaking/test_distributed.py` checks
agreement across process counts to its tolerance.

**Index levels.** The engine sees a flat index. The ensemble flattens observation × detector batch,
pads each observation to a multiple of K batches, and records the sharing with strides; the
map-making layer builds that ensemble. This keeps the engine small and leaves other second levels
(sample tiles, frequency bands) open without touching it.

**Inter-node.** Nothing in the engine is process-aware. Under `jax.distributed`, the mesh spans
the job, a device's positions belong to the process that holds the device, the host loop of each
process reads only its own devices' observations (as the accumulation does today), and the device
sum of the summed outputs is the one collective. The stacked outputs are global arrays whose
shards live where they were computed, which is exactly what the left-hand side consumes.

## The two map-making sums

**The pass over the data.** One `HostEnsemble` per shape class, whose `read` is the reader's
`read_host` on the observation assigned to a position (filler for padded positions), split into
detector batches; the applied function is today's observation kernel (hits, b, noise fit, template
right-hand side, model leaves); the output specification sums the hit map and b and keeps the rest
at its stride, `'auto'` sorting the leaves of the model structures; `Policy.from_budget` with the
mesh axis; the host loop with one observation prefetched. This replaces `_accumulate_bucket`, its
explicit rounds, the per-round storage and the slot arithmetic.

**The left-hand side.** The non-summed outputs of that pass, at the strides the engine returned --
1 for the compact per-batch leaves, s for the per-observation ones -- already form a
`StackedEnsemble` per class, on the device that will evaluate it, in the layout the engine
expects: the left-hand side's ensemble is an output of the right-hand side's pass, with no copy
and no host round trip. The member operator
Hᵢᵀ Wᵢ Hᵢ is built by lifting the map-maker's member-level expression over that ensemble, and
A = Σ_classes (`AdditionOperator` over the class ensemble), applied to the shared x with a summed
output; `A_diag` is the same closure with another member. The CG, the preconditioner and the pixel
selection are unchanged: they see one operator.

Both sums share the engine, the assignment and the budget; they differ only by the ensemble
representation and the loop that goes with it, which is the intended division of labour.

## Plan

Each step is one pull request with a single commit.

1. This note.
2. The engine in `furax.core`: `Ensemble` and its three representations, `Policy`, `map_reduce`,
   `lift`; `AdditionOperator(sequential=True)` and `BlockRowOperator` rewired as closures of a
   `ListEnsemble` per signature group (a strict refactor of the current multiplexer). Tests on the
   eight host devices of the suite: the compute instruction appears once in the compiled program
   whatever N (counted in the HLO text), temporaries do not grow with N and grow linearly with K
   (`memory_analysis()`), results match the unrolled evaluation for every representation × policy
   × stride layout, padded members contribute nothing, stacked outputs come back in position
   order with the ensemble's sharding, `'auto'` resolves each leaf as its dataflow dictates (and
   an inferred stride-`None` output equals every member's), and `shard_map` results match a
   single-device evaluation.
3. Ensembles as block containers: the composites accept an ensemble, the composition rules combine
   ensembles member-wise, `lift` carries construction and `reduce`.
4. The pass over the data on the engine, with the assignment; `test_distributed.py` unchanged
   and green.
5. The left-hand side on the ensemble produced by step 4, with `Policy.from_budget`; CG untouched.
6. `StreamOperator` reduced to a façade over the closures, then removed with its equinox
   dependency.

Benchmarks at steps 4 and 5, on the shapes of `test_distributed.py` scaled up: wall time of the
pass over the data and of one CG iteration against the device count (1, 2, 4, 8), maximum over
mean padded volume per device, HLO size and temporaries against N and K, compile time per class,
and the share of the pass spent waiting for reads.

## Follow-ups and non-goals

- Streaming the left-hand side when the resident parameters do not fit.
- An indexed-add output specification (scatter into a shared vector instead of K dense adds), if
  the joint template system ever makes it matter.
- Policies in another order than shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ, and a throughput-based cap on K.
- Distributing the map, and any change to the CG, are out of scope.
