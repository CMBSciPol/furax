# Templates and their Gram: one model for every template

## Summary

A template models a structured nuisance signal in the TOD as `s = T a`. The mapmaker either solves
for the amplitudes `a` together with the map (explicit templates), or marginalizes them out of the
weight, `W' = W − W T (Tᵀ W T)⁻¹ Tᵀ W` (implicit templates). Both need the Gram `G = Tᵀ W T`: the
preconditioner of the explicit system is its inverse, and the implicit weight is built from it.

This document describes every template with three ingredients:

- a **time basis** `B`: the functions of time, the existing `Basis` family;
- a **coupling** `C`: how each detector and each stream sees each mode of the template;
- the axes it has amplitudes **per**: stream, detector (or group of detectors, such as a wafer).
  Along the others, the amplitudes are shared.

The operator, the Gram and its factorization are built from that description alone. Fitting one
amplitude set per detector or per stream, sharing it across detectors or streams, coupling it
through a focal-plane pattern, and marginalizing a template all become values of these ingredients
rather than separate code paths, and Pomme a step of the Gram assembly.

## Motivation

Each template feature adds a branch to the operator, the Gram assembly and the mapmaker. The
table covers the template code on `main` and the work built on it for correlated weights, shared
amplitudes and the T2P field:

| Feature | Where it lives today |
|---|---|
| Amplitudes per detector and per stream | `StokesTemplateOperator` keys amplitudes by stream, groups bases by stream (`'qu'`); the Gram is split per stream (`_PerLegOperator`) |
| Amplitudes shared by the detectors | `shared` name tuples in the operator, a separate Gram summed over detectors, `_BatchSumOperator` in the mapmaker |
| A gain per detector on a shared signal | `CoupledBasis`, a basis flavour with its own `_gram` |
| T2P following a focal-plane field | couplings baked into the basis values, `(det, k, 2, samp)` |
| Pomme | interval sums and band corrections inside the Gram assembly (`_pomme_*`) |
| Implicit templates | `W' = W − W T G⁻¹ Tᵀ W` built in the mapmaker |
| Correlated (Toeplitz) weight | a second Gram assembly by probing (`_Probes`) |
| Several templates on one stream | the core/border layout heuristic (`_layout`) |

Most of these differ only in how amplitudes couple to (detector, stream), or in which amplitudes are
eliminated by a Schur complement. The next feature in line, a T2P field common to Q and U, would
need yet another branch: amplitudes shared across streams, with a different coupling on each stream.
The model below makes it a coupling.

## The model

### The TOD

A `TODFormat` (`furax.obs`) describes the data the templates add to:
`TODFormat(shape, dtype, stokes=None)`.

- `shape` is `(*det, samp)`: the detector timestreams, the samples last. There is usually one
  detector axis; a single timestream has none, and several are allowed.
- `stokes` is set only for demodulated data, which is then a `Stokes` of such arrays, one per
  stream.

A TOD sample is indexed by `(l, d, t)`: stream `l`, detector `d`, time `t`. Inside the template
code, the streams are an axis like the others, of size 1 unless the TOD is demodulated, and the
detector axes are flattened into one: the TOD is a `(stream, det, samp)` array, and a template
covers some indices along the stream axis. Only the edges (amplitudes, coupling, groups, basis
batch, TOD) carry the public shapes: there, only a demodulated TOD has a stream axis, and only then
do the amplitudes and couplings of its templates have one.

`TODFormat` is a frozen, hashable dataclass, so a template holds it as a static field. It also
gives `zeros`, `normal` and the other fills of the landscapes.

### A template

```
s[l, d, t] = Σₘ Σₖ C[l, d, m] · B_d[k, t] · a[o(l, d), k, m]
```

- `B_d[k, t]`: the time basis. It is one basis shared by every detector, or a batch with one basis
  per detector (`batch_shape=tod_format.det_shape`, as for T2P, whose basis is the detector's own
  temperature stream). `k` is the basis's amplitude index, including its time blocks for a
  time-local basis.
- `C[l, d, m]`: the coupling of detector `d` on stream `l` to mode `m`. Absent, it is 1 with a
  single mode, and the mode axis is dropped.
- `o(l, d)`: the set of amplitudes that sample `(l, d)` sees. `per` lists the axes that index the
  sets: `('stream', 'det')` gives every detector and stream its own amplitudes, `()` gives one set
  for the whole observation, `('det',)` with detector groups gives one set per group (wafer).

A template is a linear operator from its amplitudes to the whole TOD, zero on the streams it does
not cover:

```python
PerAxis = Literal['stream', 'det']


class Template(AbstractLinearOperator):
    basis: Basis
    tod_format: TODFormat = field(metadata={'static': True})
    # the streams covered, such as 'qu'; None unless demodulated
    streams: str | None = field(default=None, metadata={'static': True})
    per: tuple[PerAxis, ...] = field(default=('stream', 'det'), metadata={'static': True})
    coupling: Float[Array, '*stream *det mode'] | None = None  # each axis full size or 1
    det_groups: Bool[Array, '*det group'] | None = None  # one-hot, detectors sharing a set on 'det'
```

It checks itself when built: `per` and `streams`, the streams within `tod_format.stokes`, the
coupling and `det_groups` against `tod_format.det_shape`, and the basis against the detector
axes, sample count and dtype. `per` is normalized to a fixed order, without `'stream'` unless the
TOD is demodulated and without `'det'` on a single timestream.

The amplitudes of a template, its `in_structure`, have shape `(*sets, *basis.shape, *modes)`: the
sets along `per` (the streams, then the detector axes or the groups), then the basis's own shape,
then the modes when there is a coupling.

Internally, the amplitudes take a canonical layout with fixed axes, `(stream set, detector set,
*basis.shape, mode)`, an axis of size 1 where the sets are shared and a single mode without
coupling, so that every later step finds each axis at the same place for every template. The rows
`[l]` are the amplitudes stream `l` sees, and detector `d` sees row `o` through the weight
`W[d, o, m]`: the coupling of mode `m` where `d` belongs to `o`, zero elsewhere. Every choice of
`per` is then the same two contractions, one per direction, with a set per detector keeping the
weight diagonal.

A template has a single basis. Streams fitted with different bases, such as a polynomial of one
order on I and another on Q and U, are separate templates, one per group of streams fitted alike:
`polynomial` when every stream is fitted alike, `polynomial_<streams>` (`polynomial_i`,
`polynomial_qu`) otherwise. The same holds for the scan-synchronous template.

Every current template is an instance:

| Template | `basis` | `per` | `coupling` |
|---|---|---|---|
| Polynomial per interval | `SegmentedBasis` | `('stream', 'det')` | — |
| HWP-, scan-synchronous | `TensorBasis`, `KroneckerBasis` | `('stream', 'det')` | — |
| T2P per detector | detector's temperature, batched | `('stream', 'det')` | — |
| T2P field, common to Q and U (default) | detector's temperature, batched | `()` | Q: `Zₖ · (cos 2γ, −sin 2γ)`, U: `Zₖ · (sin 2γ, cos 2γ)` |
| T2P field, per stream (`joint_qu=False`) | detector's temperature, batched | `('stream',)` | `Zₖ(x_d) · (cos 2γ_d, sin 2γ_d)` |
| Signal common to every detector | any shared basis | `('stream',)` | — |
| Common mode with a gain per detector (`CoupledBasis`) | any shared basis | `('stream',)` | the gains |
| Common mode per wafer | `OffsetBasis` | `('det',)` or `('stream', 'det')`, grouped by wafer | the pattern across the wafer |
| Pomme offsets | `OffsetBasis(tau)` | `('det',)` | — (implicit) |

### The operator

```python
class TemplateOperator(BlockRowOperator):
    def __init__(self, templates: dict[str, Template]) -> None: ...  # all of the same TOD
```

The sum of the templates of one TOD, from amplitudes keyed by template name: their block row
`T = [T₁ | T₂ | …]`, whose transpose is the block column of the `Tᵢᵀ`. It holds nothing but the
templates, so stacking observations under a scan only adds a leading axis to their arrays. It
replaces the current `TemplateOperator`, `StokesTemplateOperator`, stream groups, `shared` and
`CoupledBasis`.

- **Synthesis**, for each stream: contract each detector's amplitude set with the coupling,
  `c[d, k] = Σₘ C[l, d, m] a[o(l, d), k, m]`, then expand with the basis. A template whose
  detectors share the basis but not one set each is expanded once per (set, mode), then weighted
  per detector, so the expansion does not scale with the detectors. A signal the same on every
  stream is expanded once.
- **Analysis** (the transpose): project with the basis, multiply by the coupling and scatter-add
  into the sets. Summing over the detectors of a shared template is that scatter-add.
- Only the synthesis departs from the block row's: each stream of the TOD sums the signals of the
  templates covering it, rather than adding up each template's whole TOD, and the per-detector
  templates on a dense shared basis are expanded in one matrix product, across templates. It
  stays only if it proves faster than the plain block row.

A basis is stored once, whatever its coupling. For the T2P field this replaces the baked
`(det, k, 2, samp)` values by the temperature `(det, samp)` and a `(stream, det, 2k)` coupling.

## The Gram

### Its sparsity

Write `L^{d}_{ll'} = B_{g,d}ᵀ W^{d}_{ll'} B_{h,d}` for the local Gram of templates `g` and `h` on
detector `d` and streams `l, l'`. The Gram is then

```
G_gh[(o, m), (o', m')] = Σ_{d, l, l'} [o_g(l, d) = o] [o_h(l', d) = o'] C_g[l, d, m] C_h[l', d, m'] L^{d}_{ll'}
```

Its sparsity follows from the template description and from what the weight couples. The weight may
couple samples up to `bandwidth` apart, and the streams of a detector if `streams_coupled`; it must
not couple detectors. Detector-correlated weights are not covered. The model admits them through
probing, at the cost of merging detectors into larger blocks.

The amplitudes split into two classes:

- **local**: the amplitudes of templates with a set per detector (`Template.per_detector`: `'det'`
  in `per`, without `det_groups`). Their Gram is block diagonal over **units**: one per detector, or
  one per (detector, stream) when every local template has `'stream'` in `per` and the weight does
  not couple streams.
- **global**: every other amplitude, in one block, or one per stream likewise. That includes
  templates grouped by wafer: a wafer's set is seen by all its detectors, which can span several
  detector batches.

Within a block, the time-local template with the most amplitudes keeps a banded Gram, its time
blocks first, bordered by the other templates of the block, flattened in turn.

### The linear algebra

The block structures are generic and live in `furax.linalg`:

- `BorderedBandedMatrix(bands, border, corner)`: a batch of symmetric matrices with a block-banded
  core and a dense border, each part optional, acting on `(core, border)` pairs. It factors by
  banded or bordered banded Cholesky, and `regularized` adds the ridge (see below).
- `probe_bordered_banded`: recovers such a matrix from the operator it represents.
- `ArrowheadCholeskyOperator`: solves independent blocks coupled to one dense block, by Schur
  complement.

```python
@register_dataclass_with_keys
@dataclass(frozen=True)
class BlockGram:
    """Tᵀ W T of one observation, or of one batch of its detectors, assembled but not factored."""

    local: _Part | None  # a BorderedBandedMatrix over the units, and its amplitude layout
    global_: _Part | None
    cross: Float[Array, 'global unit local'] | None  # kept only for an exact factorization
```

The layout maps the rows of each block to template amplitudes; it is static, and does not depend
on the number of detectors. The Grams of disjoint detector batches combine (`BlockGram.combine`):
the local blocks and the cross terms concatenate, the global blocks add up.

### Assembly

```python
def template_gram(
    operator,
    weight,
    *,
    bandwidth=0,
    streams_coupled=False,
    coupled=False,
    method=None,
    batch_size=1,
) -> BlockGram: ...
```

Two backends produce the same `BlockGram`, for the same layout. By default, a diagonal weight
(`bandwidth=0`, `streams_coupled=False`) is assembled from the structure, any other by probing;
`method` forces one.

- **Probing**, the reference. Apply `Tᵀ W T`, restricted to the templates of the
  block, to probe amplitudes and read the Gram off the responses. Units never interact, so every
  unit takes the same probe. The core is probed one color of blocks at a time, blocks far enough
  apart that their responses do not overlap: `2·half_width + 1` colors, the half-width estimated
  from the basis support and `bandwidth`. A core whose blocks are too short for that estimate is
  rejected (or, under a JAX transformation, gives a NaN Gram). The border is probed one amplitude
  at a time. With `coupled=True`, the global block is all border, and the responses to its probes
  on the local amplitudes are the cross block. It works for any weight whose reach is declared,
  and extends today's Toeplitz assembly to couplings, amplitude sets and streams. Cost: one
  application of `T`, `W` and `Tᵀ` per probe.
- **Structural**, for a weight diagonal in time and in streams, whose diagonal is read off by
  applying it to ones: `Basis.gram` (bands, for the core) and `_cross_gram` (dense, for the
  border and corner) with the weights of each (stream, detector), contracted with the weights
  through which each detector sees the amplitude sets. The local blocks are mapped over the
  detectors; the global ones are summed over them, in batches. For a global template with a
  per-detector basis, such as the T2P field, that is `Σ_d C_d C_dᵀ ⊗ (B_dᵀ w_d B_d)`. The streams
  are assembled one after the other, so that the transient memory is that of one stream, as in the
  current code.

Both backends are checked against the dense Gram in the tests. A new kind of weight works with
probing first, and gets a fast path only if it is worth it.

### Pomme

Pomme removes the offsets `Z = OffsetBasis(tau)`, one set per detector, by the projector `F`
applied to a TOD that is not demodulated. The Gram of every template, explicit or implicit, is
then taken under `W F`. Under the condition the Pomme path already enforces (a diagonal weight
constant on each interval), that is the Gram of the templates with the offsets marginalized, a
Schur complement:

```
G̃_TT = G_TT − G_TZ G_ZZ⁻¹ G_ZT
```

- **By probing**, nothing changes: `W F` is a weight like any other, coupling the samples of an
  interval, so its reach is `tau`.
- **From the structure**, the offsets are eliminated analytically, on each detector, from the
  weighted Grams of the bases, before their contraction with the couplings. There is one offset
  every `tau` samples, too many to hold even as a banded core. Their Gram `D` is diagonal, the
  weighted counts of the intervals, and each Gram of two bases loses `Cᵀ D⁻¹ C'`, with
  `C = Zᵀ W B` a basis summed over each interval. A template shared by the detectors takes the
  sum of these corrections over them.
- **A time-local core keeps its band**, one block wider: an interval straddling two blocks couples
  them. Its sums `C` are kept in the same band layout, so that no dense matrix of the core's size
  is formed. Its blocks must be longer than an interval, such as the knots of the spline
  HWP-synchronous template, which Pomme leaves since its harmonics are at high frequency.

A polynomial per interval does not combine with Pomme: the offsets absorb anything smooth over
`tau`, so the polynomial, smooth over much longer, would fit nothing they leave, and its Gram under
`W F` would be all but singular.

No other template is eliminated. The implicit templates are marginalized together, through one
Gram of all of them, so none needs eliminating from another's. The Gram of the explicit templates
is taken under `W`, or `W F`, and not under the marginalized weight `W'`: their preconditioner
leaves the implicit templates out, as today. On the sky side, the implicit weight stays the
operator `W − W Tᵢ G⁻¹ Tᵢᵀ W`, and Pomme stays the projector `F`.

### Factorization

```python
def factor(
    gram: BlockGram, *, regularization=0.0, batch_size=1
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]: ...
```

It returns the inverse on the amplitudes, and the ridge added before factoring.

- The units and the global block are factored by banded or bordered banded Cholesky.
- A Gram assembled with `coupled=True` is inverted exactly: the local blocks are eliminated into
  the Schur complement of the global one, `S = G_gg − Σ_u G_ugᵀ G_uu⁻¹ G_ug`
  (`ArrowheadCholeskyOperator`). That is cheap for a small global block, such as the 30
  amplitudes of a T2P field.
- Otherwise the coupling is dropped (block Jacobi), as the preconditioner does today. It is the
  only option when the global block is large and time-local, such as one common-mode amplitude per
  wafer every few seconds. Coupled to a local template spanning the observation (the Legendre
  polynomial), its Schur complement fills in to a dense matrix over every block.
- Implicit templates need an exact inverse. Being local, they always have one.

Amplitudes no weighted sample sees, whose rows are zero, take a unit diagonal and come back
unchanged. The ridge is `regularization` times the mean diagonal of each core block, and times
its own diagonal entry for each border amplitude, so that it is in the units of the amplitudes it
regularizes even when the border gathers several templates.

A **prior** per template would generalize the ridge: a Gaussian prior `a ~ N(0, Σₐ)` adds `Σₐ⁻¹`
to the diagonal of its block, and turns deprojection into Wiener filtering,
`W' = (N + T Σₐ Tᵀ)⁻¹`. It is not implemented yet.

## The mapmaker

- **The TOD.** `ObservationModel` exposes its TOD as a `TODFormat`, from which the templates are
  built, and which replaces the hand-written detector counts and stream selections.
- **Detector batches.** The amplitudes of templates with a set per detector live in their
  detector's batch. Every other amplitude, grouped ones included, is summed over the batches of
  its observation. The mapmaker reads this from `Template.per_detector`, not from a list of
  template names.
- **The system** keeps its form: `A = [H T]ᵀ W' [H T]` plus the priors, preconditioned by the
  block-Jacobi map preconditioner and `factor(gram)` on the amplitudes.
- **The amplitudes** are saved as one array per template, with the axes of `per` in front:
  `(obs, stream, det, ...)` for a per-detector template, `(obs, ...)` for one common to the
  observation. This replaces the `<template>/<stream>` entries of `amplitudes.npz` (see
  `MapMakingResults`).

## Adding a feature

- **A new kind of template**: a new `Basis`, if no existing flavour fits.
- **A new way of sharing amplitudes**: a coupling and a choice of `per`. A T2P field common to Q and
  U, or a common mode per wafer, is a few lines in the template builder.
- **A new weight structure**: its reach, `bandwidth` and `streams_coupled`. It works through
  probing, and may later get a structural fast path.
- **Marginalizing a template**: an implicit template, if it is local.

## Tests

One parametrized test covers the combinations: each basis flavour, each choice of `per`, with and
without coupling, on one stream or several, with one or several detector axes, under a diagonal,
banded and stream-coupled weight. On instances small enough for dense matrices, it checks:

1. synthesis against the definition, one detector at a time;
2. the transpose against the adjoint;
3. the probed Gram against the dense `Tᵀ W T`, alone and combined over detector batches;
4. the structural Gram against the dense one, including detectors summed in batches that do
   not divide their number;
5. `factor` against the dense Gram, exact and block by block, with and without the ridge;
6. the Gram under Pomme against the dense `Tᵀ W F T` (step 4).

The existing mapmaker tests (Pomme, implicit and explicit templates, detector batches) stay as
end-to-end checks.

## Migration

The work starts from `main`. Correlated (Toeplitz) weights under templates, amplitudes shared by
the detectors and the T2P field are built directly in the new model: the first through probing,
the others as couplings. Each step leaves the test suite green.

0. `TODFormat` in `furax.obs`, used by `ObservationReader` (#342).
1. Done: `Template` and the new `TemplateOperator`, with tests 1–2. The current code is untouched.
2. Done: the bordered banded matrices in `furax.linalg`, the probing assembly into `BlockGram`, and
   `factor`, exact or block by block, with the ridge, with tests 3 and 5. Probing is what
   correlated weights need, so they are supported from this step.
3. Done: the structural backend, with test 4. On CPU, for the per-detector templates of a
   demodulated SO SAT run (polynomial per interval and scan-synchronous on I, Q, U, T2P on Q, U;
   up to 512 detectors × 200 000 samples), it matches the current `gram_inverse` in run time and
   in scratch memory (within 3%). Probing takes 1.7 times as long and 16 times the memory. The
   same comparison on GPU, at the size of a full observation, remains to be run
   (`scripts/bench_template_gram.py`).
4. Pomme: the Gram under `W F`, by probing `W F`, and from the structure by eliminating the
   offsets from the weighted Grams of the bases, with test 6.
5. Done: `ObservationTemplates` and the mapmaker on the new model, `ObservationModel.tod_format`
   a `TODFormat`, one template per group of streams fitted alike (`polynomial`, or
   `polynomial_<streams>`), amplitudes saved as one array per template, `(obs, stream, det, ...)`.
   `StokesTemplateOperator` and the old Gram are removed, and the template modules gathered in the
   `furax.mapmaking.templates` subpackage (bases, builders, operator, Gram). Left: templates under a
   correlated weight, the Gram probed with the weight's correlation length as its bandwidth; the
   mapmaker still requires a diagonal weight with templates.
6. The first features on the new base: the T2P field (`T2PConfig.zernike_order`), common to Q
   and U unless `T2PConfig.joint_qu` is false, and common modes coupled to the detectors.

## Open questions

- **The explicit preconditioner under `W'`.** With implicit templates, the system operator uses the
  marginalized weight `W'`, but the preconditioner of the explicit amplitudes is the Gram under
  `W`. Eliminating the implicit templates from it would give the Gram under `W'`: worth building
  only if CG iteration counts, such as for T2P solved explicitly next to a deprojected polynomial,
  show the current preconditioner limits convergence.
- **Per-wafer blocks.** Templates grouped by wafer are global: the wafers end up in one block,
  coupled, although they never interact. One block per wafer needs the grouping to be static,
  such as detectors sorted by wafer with static boundaries, padded per wafer across observations.
- **A coupled global block is dense.** `coupled=True` puts the whole global block in the border,
  even a time-local one: one probe per amplitude, and a dense cross block, `n_global × units ×
  n_local`. That suits a T2P field but not a common mode per wafer, which stays block Jacobi. A
  time-sparse cross block would lift the limit.

## Risks

- **Performance.** The structural backend must match today's Gram assembly in time and memory;
  step 3 measures it before anything switches.
- **Probing cost.** One application of `T`, `W` and `Tᵀ` per probe. A global time-local template
  needs `colors × k × modes` probes: about 90 for one common-mode amplitude per wafer every few
  seconds.
- **Compile time.** Generic code over template combinations may compile more kernels than the
  specialized paths. The test suite's compile time is the first indicator.
