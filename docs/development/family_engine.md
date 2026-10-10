# Design note: families of operators

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
left-hand side will need at every CG iteration. The engine must return both reduced and
per-member outputs from one pass.

Today these concerns are entangled in `StreamOperator` (`furax.mapmaking.streaming`), the buckets
of `furax.mapmaking.layout`, and the accumulation loop of `MultiObservationMapMaker`. The two sums
are evaluated by two different engines: a host-driven loop of `shard_map` launches for the pass
over the data, and resident stacked `StreamOperator`s summed by `AdditionOperator(sequential=True)`
for the left-hand side.

### What `StreamOperator` got right, and what it conflated

Its *boundary* -- per component, whether the input and the output carry the family index
(`in_stacked`/`out_stacked`) -- is the right description of how a family closes into an operator,
and it survives below as the output specification. What it conflated are three things that this
note keeps apart: the **representation** of the family's data (it only knows stacked, resident
leaves, with `group_size` as a layout trick for shared ones), the **policy** that walks the index
(always shard over devices then scan one slice at a time, on the device, never batched, never
host-driven), and an **implementation** built on equinox that the library is moving away from.

## Concepts

### Family

A **family** is a set of N members sharing a static signature: the same code, different data. A
member is a pytree -- an operator, or an operator paired with its input -- and two members of a
family share their treedef, their static fields and their non-array leaves; they differ only by
the values of their array leaves. `AbstractLinearOperator._static_signature` is the membership
test.

The family's data can be held in three ways, and the way it is held decides what the engine may do
with it:

| representation | `fetch(i)` | batch of K | notes |
|---|---|---|---|
| **list**: N member pytrees | `lax.switch` over N branches returning the arrays of member i | no | HLO grows by ~2 kB per member: fine for tens of members, not thousands |
| **stacked**: one pytree whose array leaves carry a leading axis | `dynamic_index` on every leaf | `dynamic_slice` | the only representation `vmap` and `shard_map` can read |
| **host**: a callable returning host arrays | host read, then transfer | one transfer per round | lives on the host; every evaluation pays the transfer |

Converting a list into a stacked representation is the one operation that copies all the data;
it is never done implicitly. A family whose members hold *compact* parameters and recompute
TOD-sized objects inside `apply` (on-the-fly pointing from quaternions) is just a stacked family
with small leaves: nothing special is needed for it.

A leaf may be **shared** by a run of consecutive members: the detector batches of one
observation share its boresight quaternions. A per-leaf *stride* records this: stride 1 for a leaf
with one entry per member, stride g for one entry per g consecutive members, `None` for a leaf
shared by the whole family. This replaces `StreamLayout.group_size`. It also expresses a two-level
index (observation × detector batch) without the engine knowing about levels: the family flattens
the index and the strides keep the sharing.

Padding makes a family's size a multiple of what the policy requires; padded members are marked
in a boolean `valid` array, and the engine discards their contribution.

### Combination

A family closes into one operator according to which side carries the index. Rather than four
operators, the engine takes one description: per member the input is either shared (the map x) or
per member (dᵢ, carried in the member through `fetch`), and per output leaf the result is either
**summed** over the family or **stacked**, one entry per member. The four block layouts are the
uniform cases -- sum (`AdditionOperator`: shared in, summed out), row (`BlockRowOperator`: per
member in, summed out), column (shared in, stacked out), diagonal (per member in, stacked out) --
and the pass over the data is a mixed one: summed hit map and b, stacked noise fits and model
leaves. Transposition swaps shared and per-member inputs with summed and stacked outputs.

### Policy

Evaluating the family means walking the index i. The JAX transforms are the ways of walking it,
and they compose by factorising the family size, N = D × G × K:

    shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ      (devices ∘ sequential rounds ∘ batched members)

Summed outputs propagate level by level -- `sum` over the K members of a round, accumulation over
the G rounds, `psum` over the D devices -- and stacked outputs are concatenated along the same
levels, so that a device ends up holding, in position order, the outputs of the members it
evaluated.

Two invariants follow from what each transform needs:

- `vmap` and `shard_map` need a leading axis in memory, so they require a stacked or host
  representation; a loop accepts any representation, including the list.
- The temporaries of a round are reused by the next one, so the batched level sits directly under
  the loop: K TODs of temporaries, never G, are live at once. The order shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ
  is the only one this note implements.

The loop level has two implementations, chosen by the representation:

- a **device loop** (`lax.fori_loop`) for resident families: one compiled body, temporaries
  allocated once for the whole loop;
- a **host loop** for host families: one compiled body launched once per round, while the host
  reads and transfers the next round in a thread. This is what the current accumulation does, and
  it is what makes the pass over the data run at the speed of the disk rather than the sum of disk
  and compute. An `io_callback` inside a device loop would serialise the two.

Rounds are **aligned on observations**: a round holds K detector batches of one observation, and
the family pads each observation's batch count to a multiple of K. A round therefore never
straddles two observations, the leaves shared by an observation are passed to `vmap` unbatched
(`in_axes=None`) instead of being gathered K times, and a device moves through its observations
one after the other, which is what the host loop needs to prefetch the next one.

In this frame:

| today | in this frame |
|---|---|
| `StreamOperator` | shard_ᴅ ∘ loop_ɴ⁄ᴅ (K = 1) on a stacked family, device loop |
| `_accumulate_bucket` and its rounds | shard_ᴅ ∘ loop_ɢ (K = 1) on a host family, host loop |
| `AdditionOperator(sequential=True)`, `BlockRowOperator` | loop_ɴ on a list family |
| `AdditionOperator(sequential=False)` | Python unroll |
| `StreamLayout.group_size` | a leaf stride |
| the constant segments of `StreamOperator` | leaves with stride `None` |
| `in_stacked` / `out_stacked` | shared or per-member inputs, summed or stacked outputs |
| buckets | shape classes: one family per class, padded to D × G × K |
| `detector_batch_size` | the size of a member; K members per round |

## API sketch

Names are provisional; everything below lives in `furax.core`, knows nothing about observations,
and uses no equinox.

```python
type Stride = int | None   # entries per member on a leaf's leading axis: 1, g, or None (shared)


class Family(Protocol):
    size: int                              # N, a multiple of D × K after padding
    template: PyTree                       # the structure of one member (ShapeDtypeStructs)
    strides: PyTree[Stride]                # per leaf of the template
    valid: Bool[Array, ' N'] | None        # False on padded members

    def fetch(self, i: Int[Array, '']) -> PyTree: ...                    # member i, in a trace
    def fetch_round(self, start: int, k: int) -> tuple[PyTree, PyTree]: ...  # (shared leaves,
                                                                            #  per-member leaves
                                                                            #  with a leading k)


class ListFamily(Family):      # members: list[PyTree]; fetch = lax.switch; no rounds
class StackedFamily(Family):   # stacked: PyTree; fetch = dynamic_index; fetch_round = dynamic_slice
class HostFamily(Family):      # read: Callable[[int], PyTree[np.ndarray]]; fetch_round = read,
                               # prefetched one round ahead, then transferred


@dataclass(frozen=True)
class Policy:
    devices: str | None = None   # mesh axis to shard over, None for a single device
    batch: int = 1               # K: members evaluated together in one round

    @classmethod
    def from_budget(cls, family, apply, out_spec, budget: int, *, devices=None, margin=0.8):
        """K from a memory budget: compile apply on one member, read its temporaries."""


type OutSpec = PyTree[Literal['sum', 'stack']]   # prefix of apply's output


def family_apply(family, apply, policy, out_spec) -> PyTree[Array]:
    """Apply `apply` to every member: summed leaves are reduced over the family, stacked leaves
    come back with a leading axis of size N, sharded over the devices when `policy.devices` is
    set. Evaluated as shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ."""


class FamilyOperator(AbstractLinearOperator):
    """A family of operators closed into one: shared or per-member input, summed or stacked
    output, per component."""
    family: Family
    policy: Policy = field(metadata={'static': True})
    in_stacked: PyTree[bool] = field(metadata={'static': True})
    out_stacked: PyTree[bool] = field(metadata={'static': True})
```

`family_apply` is the engine. It checks that the family size is a multiple of D × K, emits
`shard_map` over the mesh axis when asked (each device receives N / D consecutive members), then
the loop over the G rounds of the device -- a `lax.fori_loop` for a resident family, a Python
loop launching the same compiled round for a host family -- and, inside a round, `fetch_round`,
`vmap(apply)` over the K per-member leaves with the shared leaves closed over, the `valid` mask
applied, summed leaves reduced over K and added to the carry, stacked leaves written at their
positions. After the loop, `psum` over the mesh axis replicates the summed leaves; the stacked
leaves stay where they were produced, sharded over the mesh axis in position order. With K = 1
and a list family the round is the current `_sequential_sum`.

`FamilyOperator` is the operator form. Its transpose swaps `in_stacked` and `out_stacked` on the
family of transposes, its `reduce` reduces the template, and it reports the tags that all members
share. `AdditionOperator(sequential=True)` and `BlockRowOperator` keep their public behaviour
and become, internally, a `FamilyOperator` over a `ListFamily` per signature group.

## Decisions

The brief asked for the following decisions to be explicit.

**Shape classes.** `partition_padded` stays: a dynamic programme over the observations sorted by
(detector count, sample count), minimising padded volume under `max_buckets`. Its cost moves from
"slots rounded to the device count" to the padding the policy actually imposes, D × G × K
members. Within a class every padded member has the same cost, so balance is a matter of counts
(see assignment); the only cross-class cost is one compiled body per class, which is what
`max_buckets` bounds. `max_buckets` stays a user knob with its current default.

**K per class.** `Policy.from_budget` compiles `apply` on one member of the class (abstract
`ShapeDtypeStruct` leaves, no data), reads `memory_analysis().temp_size_in_bytes`, and sets
K = ⌊margin × (budget − resident) / temporaries⌋, clipped to [1, batches per observation].
`budget` defaults to the device's `memory_stats()['bytes_limit']` when the backend reports it and
must be given explicitly otherwise (host devices report none). `resident` is the byte size of the
leaves of the stacked families already placed on the device, so classes are laid out one after the
other with a shrinking budget. The chosen K, the measured temporaries and the budget are logged.
If one member does not fit, `from_budget` raises and names the two knobs that shrink a member:
`detector_batch_size` and the shape classes.

**Left-hand side that does not fit.** Version 1 refuses: if the resident families of the
left-hand side exceed the budget, `make_maps` raises before the CG starts, stating the resident
size and the budget. Streaming the left-hand side from the host at every CG iteration is
implementable with the same engine (a host family and a host loop) but pays a full transfer of the
parameters per iteration; it is a follow-up, enabled explicitly, never a silent fallback.

**Assignment.** The unit of assignment is the observation, not the detector batch: an
observation is read once, by the process owning the device that evaluates it. Within a class all
padded observations cost the same (same envelope, same batch count), so the balanced assignment
is a round-robin of the class's observations over the devices, the starting device rotating from
one class to the next so that the extra observations of successive classes land on different
devices. Observations are taken in index order, which makes the assignment a pure function of the
probed shapes and the device count: every process computes the same one. Padded observations are
the last of each device, so filler spreads across devices instead of piling up on the last ones.
The order of floating-point reductions -- K-sum, round carry, `psum` in mesh order -- is fixed by
construction and does not depend on the process layout for a given device count;
`tests/mapmaking/test_distributed.py` checks agreement across process counts to its tolerance.

**Index levels.** The engine sees a flat index. The family flattens observation × detector batch,
pads each observation to a multiple of K batches, and records the sharing with strides; the
map-making layer builds that family. This keeps the engine small and leaves other second levels
(sample tiles, frequency bands) open without touching it.

**Rules.** Three reduction rules of `StreamOperator` encode "one pass over the data" and survive
as rules on `FamilyOperator`: the sum of two families over the same family is one family of
member-wise sums; a block row of families over the same family is one family of block rows, which
is the joint map–template system of the templates path; a scalar times a family is a family. The
composition `H.T @ W @ H` is built member-wise inside the template before closing, as the
map-maker already does, so it needs no rule.

## The two map-making sums

**The pass over the data.** One `HostFamily` per shape class, whose `read` is the reader's
`read_host` on the observation assigned to a position (filler for padded positions), split into
detector batches; `apply` is today's observation kernel (hits, b, noise fit, template right-hand
side, model leaves); the output specification sums the hit map and b and stacks the rest;
`Policy.from_budget` with the mesh axis; a host loop with one observation prefetched. This
replaces `_accumulate_bucket`, its explicit rounds and the per-round storage.

**The left-hand side.** The stacked outputs of that pass -- the compact per-batch leaves and,
with their stride, the per-observation ones -- already form a `StackedFamily` per class, on the
device that will evaluate it, in the layout `shard_map` expects: the left-hand side's family is
an output of the right-hand side's pass, with no copy and no host round trip.
`A = Σ_classes FamilyOperator(family, policy)` with the member `H.T @ W @ H` applied to the shared
x and a summed output, and the same for `A_diag`. The CG, the preconditioner and the pixel
selection are unchanged: they see one operator.

Both sums share the engine, the assignment and the budget; they differ only by the family
representation and the loop that goes with it, which is the intended division of labour.

## Plan

Each step is one pull request with a single commit.

1. This note.
2. The engine in `furax.core`: `Family` and its three representations, `Policy`,
   `family_apply`, `FamilyOperator`; `AdditionOperator(sequential=True)` and
   `BlockRowOperator` rewired on it. Tests on the eight host devices of the suite: the compute
   instruction appears once in the compiled program whatever N (counted in the HLO text),
   temporaries do not grow with N (`memory_analysis()`), results match the unrolled evaluation
   for every representation × policy it accepts and every input/output layout, padded members
   contribute nothing, stacked outputs come back in position order, and `shard_map` results
   match a single-device evaluation.
3. The pass over the data on the engine, with the assignment; `test_distributed.py` unchanged
   and green.
4. The left-hand side on the family produced by step 3, with `Policy.from_budget`; CG untouched.
5. `StreamOperator` reduced to a façade over `FamilyOperator`, then removed with its equinox
   dependency.

Benchmarks at steps 3 and 4, on the shapes of `test_distributed.py` scaled up: wall time of the
pass over the data and of one CG iteration against the device count (1, 2, 4, 8), maximum over
mean padded volume per device, HLO size and temporaries against N, compile time per class, and
the share of the pass spent waiting for reads.

## Follow-ups and non-goals

- Streaming the left-hand side when the resident parameters do not fit.
- Policies in another order than shard_ᴅ ∘ loop_ɢ ∘ vmap_ᴋ.
- Distributing the map, and any change to the CG, are out of scope.
