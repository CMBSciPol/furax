# Templates and their Gram: one model for every template

## Summary

A template models a structured nuisance signal in the TOD as `y = T a`. The mapmaker either solves
for the amplitudes `a` together with the map (explicit templates), or marginalizes them out of the
weight, `W' = W − W T (Tᵀ W T)⁻¹ Tᵀ W` (implicit templates). Both need the Gram `G = Tᵀ W T`: the
preconditioner of the explicit system is its inverse, and the implicit weight is built from it.

This document describes every template with three ingredients:

- a **time basis** `B`: the functions of time;
- a **coupling** `C`: how each detector and each stream sees each mode of the template;
- the axes it has amplitudes **per**: stream, detector (or group of detectors, such as a wafer).
  Along the others, the amplitudes are shared.

The operator, the Gram and its factorization follow from that description alone. Fitting one
amplitude set per detector or per stream, sharing it across detectors or streams, coupling it
through a focal-plane pattern, and marginalizing a template are values of these ingredients, not
separate cases, and Pomme is a step of the Gram assembly.

## Motivation

Templates differ in a few ways:

- amplitudes per detector and per stream, or shared by the detectors, by the streams, or by the
  detectors of a wafer;
- a gain per detector on a shared signal;
- a leakage that follows a field over the focal plane, the same for Q and U;
- the offsets of Pomme, eliminated from every other template;
- implicit templates, marginalized out of the weight;
- a weight that correlates samples, such as a Toeplitz noise model;
- several templates on one stream, of very different sizes.

Treated one by one, each would need its own operator, its own Gram assembly and its own path
through the mapmaker. Most differ only in how amplitudes couple to (detector, stream), or in which
amplitudes are eliminated by a Schur complement. The model makes each a choice of basis, coupling
and amplitude sets.

## The model

### The TOD

A TOD format describes the data the templates add to: its shape `(*det, samp)`, the detector
timestreams with the samples last, its dtype, and, for demodulated data only, its Stokes
components. There is usually one detector axis; a single timestream has none, and several are
allowed. Demodulated data has one such array per Stokes component, called a stream.

A TOD sample is indexed by `(s, d, t)`: stream `s`, detector `d`, time `t`. Inside the model, the
streams are an axis like the others, of size 1 unless the TOD is demodulated, and the detector
axes are flattened into one: the TOD is a `(stream, det, samp)` array, and a template covers some
indices along the stream axis. Only the edges (amplitudes, coupling, groups, basis batch, TOD)
carry the public shapes: there, only a demodulated TOD has a stream axis, and only then do the
amplitudes and couplings of its templates have one.

### A template

```
y[s, d, t] = Σₘ Σₖ C[s, d, m] · B_d[k, t] · a[o(s, d), k, m]
```

- `B_d[k, t]`: the time basis. It is one basis shared by every detector, or a batch with one basis
  per detector, as for the leakage of temperature into polarization (T2P), whose basis is the
  detector's own temperature stream. `k` is the basis's amplitude index, including its time
  blocks for a time-local basis.
- `C[s, d, m]`: the coupling of detector `d` on stream `s` to mode `m`. Absent, it is 1 with a
  single mode, and the mode axis is dropped.
- `o(s, d)`: the set of amplitudes that sample `(s, d)` sees. `per` lists the axes that index the
  sets: `('stream', 'det')` gives every detector and stream its own amplitudes, `()` gives one set
  for the whole observation, `('det',)` with detector groups gives one set per group (wafer).

A template is a linear operator from its amplitudes to the whole TOD, zero on the streams it does
not cover. In outline:

```python
class Template(AbstractLinearOperator):
    basis: Basis
    tod_format: TODFormat
    streams: str | None = None  # the streams covered, such as 'qu'; None unless demodulated
    per: tuple[Literal['stream', 'det'], ...] = ('stream', 'det')
    coupling: Float[Array, '*stream *det mode'] | None = None  # each axis full size or 1
    det_groups: Bool[Array, '*det group'] | None = None  # one-hot, detectors sharing a set on 'det'
```

The amplitudes of a template have shape `(*sets, *basis.shape, *modes)`: the sets along `per`
(the streams, then the detector axes or the groups), then the basis's own shape, then the modes
when there is a coupling.

Internally, the amplitudes take a canonical layout with fixed axes, `(stream set, detector set,
*basis.shape, mode)`, an axis of size 1 where the sets are shared and a single mode without
coupling, so that every later step finds each axis at the same place for every template. The rows
`[s]` are the amplitudes stream `s` sees, and detector `d` sees row `o` through the weight
`W[d, o, m]`: the coupling of mode `m` where `d` belongs to `o`, zero elsewhere. Every choice of
`per` is then the same two contractions, one per direction, with a set per detector keeping the
weight diagonal.

A template has a single basis. Streams fitted with different bases, such as a polynomial of one
order on I and another on Q and U, are separate templates, one per group of streams fitted alike.

Templates are instances of the model:

| Template | basis | `per` | coupling |
|---|---|---|---|
| Polynomial per scan interval | piecewise, one block per interval | `('stream', 'det')` | — |
| Drift over the observation | cubic B-spline of time, four knots read by each sample | `('stream', 'det')` | — |
| HWP- or scan-synchronous | dense, of the HWP angle or the azimuth | `('stream', 'det')` | — |
| T2P per detector | the detector's temperature, batched | `('stream', 'det')` | — |
| T2P field, common to Q and U | the detector's temperature, batched | `()` | Q: `Zₖ · (cos 2γ, −sin 2γ)`, U: `Zₖ · (sin 2γ, cos 2γ)` |
| T2P field, per stream | the detector's temperature, batched | `('stream',)` | `Zₖ(x_d) · (cos 2γ_d, sin 2γ_d)` |
| Signal common to every detector | any shared basis | `('stream',)` | — |
| Common mode with a gain per detector | any shared basis | `('stream',)` | the gains |
| Common mode per wafer | offsets | `('det',)` or `('stream', 'det')`, grouped by wafer | the pattern across the wafer |
| Pomme offsets | offsets, one per `tau` samples | `('det',)` | — (implicit) |

Here `Zₖ(x_d)` are the Zernike polynomials at the detector's focal-plane position, and `γ_d` its
polarization angle.

### The operator

```python
class TemplateOperator(BlockRowOperator):
    def __init__(self, templates: dict[str, Template]) -> None: ...  # all of the same TOD
```

The sum of the templates of one TOD, from amplitudes keyed by template name: their block row
`T = [T₁ | T₂ | …]`, whose transpose is the block column of the `Tᵢᵀ`. It holds nothing but the
templates, so stacking observations under a scan only adds a leading axis to their arrays.

- **Synthesis**, for each stream: contract each detector's amplitude set with the coupling,
  `c[d, k] = Σₘ C[s, d, m] a[o(s, d), k, m]`, then expand with the basis. A template whose
  detectors share the basis but not one set each is expanded once per (set, mode), then weighted
  per detector, so the expansion does not scale with the detectors. A signal the same on every
  stream is expanded once. Each stream sums the signals of the templates covering it, and the
  per-detector templates on a dense shared basis are expanded together, in one matrix product.
- **Analysis** (the transpose): project with the basis, multiply by the coupling and scatter-add
  into the sets. Summing over the detectors of a shared template is that scatter-add.

A basis is stored once, whatever its coupling. For the T2P field, that is the temperature
`(det, samp)` and a `(stream, det, 2k)` coupling, not a `(det, k, 2, samp)` array of values.

## The Gram

### Its sparsity

Write `L^{d}_{ss'} = B_{g,d}ᵀ W^{d}_{ss'} B_{h,d}` for the local Gram of templates `g` and `h` on
detector `d` and streams `s, s'`. The Gram is then

```
G_gh[(o, m), (o', m')] = Σ_{d, s, s'} [o_g(s, d) = o] [o_h(s', d) = o'] C_g[s, d, m] C_h[s', d, m'] L^{d}_{ss'}
```

Its sparsity follows from the template description and from what the weight couples. The weight may
couple samples up to `bandwidth` apart, and the streams of a detector if `streams_coupled`; it must
not couple detectors. Detector-correlated weights are not covered. The model admits them through
probing, at the cost of merging detectors into larger blocks.

The amplitudes split into two classes:

- **local**: the amplitudes of templates with a set per detector (`'det'` in `per`, without
  detector groups). Their Gram is block diagonal over **units**: one per detector, or one per
  (detector, stream) when every local template has `'stream'` in `per` and the weight does not
  couple streams.
- **global**: every other amplitude, in one block, or one per stream likewise. That includes
  templates grouped by wafer: a wafer's set is seen by all its detectors, which can span several
  detector batches.

Within a block, the time-local template with the most amplitudes keeps a banded Gram, its time
blocks first, bordered by the other templates of the block, flattened in turn.

### The linear algebra

The block structures are generic linear algebra:

- a **bordered banded matrix**: a batch of symmetric matrices with a block-banded core and a dense
  border, each part optional, acting on `(core, border)` pairs. It factors by banded or bordered
  banded Cholesky;
- its **recovery by probing**, from the operator it represents.

The Gram of one observation, or of one batch of its detectors, is a bordered banded matrix over
the units, and one over the global block:

```python
class BlockGram:
    local: BorderedBandedMatrix | None  # over the units
    global_: BorderedBandedMatrix | None
```

A static layout maps the rows of each block to template amplitudes; it does not depend on the
number of detectors. The Grams of disjoint detector batches combine: the local blocks
concatenate, the global blocks add up.

### Assembly

```python
def template_gram(
    operator, weight, *, bandwidth=0, streams_coupled=False, pomme_tau=None
) -> BlockGram: ...
```

Two backends produce the same Gram, for the same layout. A diagonal weight (`bandwidth=0`,
`streams_coupled=False`) is assembled from the structure, any other by probing. `pomme_tau` takes
the Gram under Pomme.

- **Probing**, the reference. Apply `Tᵀ W T`, restricted to the templates of the block, to probe
  amplitudes and read the Gram off the responses. Units never interact, so every unit takes the
  same probe. The core is probed one color of blocks at a time, blocks far enough apart that their
  responses do not overlap: `2·half_width + 1` colors, the half-width estimated from the basis
  support and `bandwidth`. A core whose blocks are too short for that estimate is rejected. The
  border is probed one amplitude at a time. Under a marginalized weight, the global block is all
  border, and the responses to its probes on the local amplitudes are their coupling. It works for any
  weight whose reach is declared, with any couplings, amplitude sets and streams. Cost: one
  application of `T`, `W` and `Tᵀ` per probe.
- **Structural**, for a weight diagonal in time and in streams, whose diagonal is read off by
  applying it to ones: the banded Grams of the bases, for the core, and their dense cross Grams,
  for the border and corner, with the weights of each (stream, detector), contracted with the
  weights through which each detector sees the amplitude sets. The local blocks are mapped over
  the detectors; the global ones are summed over them, in batches. For a global template with a
  per-detector basis, such as the T2P field, that is `Σ_d C_d C_dᵀ ⊗ (B_dᵀ w_d B_d)`. The streams
  are assembled one after the other, so that the transient memory is that of one stream.

A new kind of weight works with probing first, and gets a structural fast path only if it is
worth it. On CPU, for the per-detector templates of a demodulated SO SAT run (polynomial per
interval and scan-synchronous on I, Q, U, T2P on Q, U; up to 512 detectors × 200 000 samples),
probing takes 1.7 times as long as the structural assembly and 16 times its memory.

### Pomme

Pomme removes the offsets `Z`, one per `tau` samples and per detector, by the projector `F`
applied to a TOD that is not demodulated. The Gram of every template, explicit or implicit, is
then taken under `W F`. For a diagonal weight constant on each interval, as Pomme requires, that
is the Gram of the templates with the offsets marginalized, a Schur complement:

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
  is formed. Its blocks must be longer than an interval, such as the knots of a spline
  HWP-synchronous template, which Pomme leaves since its harmonics are at high frequency.

A polynomial per interval does not combine with Pomme: the offsets absorb anything smooth over
`tau`, so the polynomial, smooth over much longer, would fit nothing they leave, and its Gram under
`W F` would be all but singular.

No other template is eliminated. The implicit templates are marginalized together, through one
Gram of all of them, so none needs eliminating from another's. The Gram of the explicit templates
is taken under `W`, or `W F`, and not under the marginalized weight `W'`: their preconditioner
leaves the implicit templates out. On the sky side, the implicit weight stays the operator
`W − W Tᵢ G⁻¹ Tᵢᵀ W`, and Pomme stays the projector `F`.

### Factorization

```python
def factor(gram: BlockGram, *, regularization=0.0) -> tuple[inverse, ridge]: ...
```

It returns the inverse on the amplitudes, and the ridge added before factoring.

- The units and the global block are factored by banded or bordered banded Cholesky.
- The coupling of the local and global blocks is dropped (block Jacobi): the inverse
  preconditions their joint system.
- Implicit templates need an exact inverse. Having a set per detector, they always have one.
- The shared Gram under a marginalized weight `W'` is a preconditioner only, and inverted on two
  levels when both it and the implicit templates are time-local. The fine level is the band of
  the Gram under `W`, by banded Cholesky. The coarse level corrects the directions the implicit
  templates absorb, which only the ridge holds in the system: `Π`, the implicit functions of time
  projected onto the shared basis by least squares, mode by mode, each truncated to a band. The
  inverse is `B⁻¹ + Π E⁻¹ Πᵀ` with `E = Πᵀ (Tᵀ W' T) Π`, so that only `E`, of the implicit
  functions times the modes, is dense: a sixth of the shared amplitudes for the default common
  mode next to the default spline drift, and a thirty-sixth of the memory of the dense block.

Amplitudes no weighted sample sees, whose rows are zero, take a unit diagonal and come back
unchanged. The ridge is `regularization` times the mean diagonal of each core block, and times
its own diagonal entry for each border amplitude, so that it is in the units of the amplitudes it
regularizes even when the border gathers several templates.

A **prior** per template would generalize the ridge: a Gaussian prior `a ~ N(0, Σₐ)` adds `Σₐ⁻¹`
to the diagonal of its block, and turns deprojection into Wiener filtering,
`W' = (N + T Σₐ Tᵀ)⁻¹`.

## The mapmaker

- **The TOD.** Each observation's model exposes the format of its TOD, from which its templates
  are built.
- **Detector batches.** The amplitudes of templates with a set per detector live in their
  detector's batch. Every other amplitude, grouped ones included, is summed over the batches of
  its observation. Whether a template has a set per detector is read from its `per` and detector
  groups.
- **The weight.** A diagonal weight gives the Gram from the structure of the bases. A banded
  Toeplitz weight gives it by probing, with `bandwidth = correlation_length - 1`; the same
  bandwidth applies to the templates of each detector and to the shared ones, whose Gram is summed
  over the batches of an observation after being probed in each.
- **The system** is `A = [H T]ᵀ W' [H T]`, plus the ridge on the amplitudes, preconditioned by the
  block-Jacobi map preconditioner and `factor(gram)` on the amplitudes.
- **The amplitudes** are one array per template, with the axes of `per` in front:
  `(obs, stream, det, ...)` for a per-detector template, `(obs, ...)` for one common to the
  observation.

## Adding a feature

- **A new kind of template**: a new basis, if no existing one fits.
- **A new way of sharing amplitudes**: a coupling and a choice of `per`. The T2P field common to Q
  and U is a coupling with `per=()`; a common mode per wafer is a coupling with detector groups.
- **A new weight structure**: its reach, `bandwidth` and `streams_coupled`. It works through
  probing, and may later get a structural fast path.
- **Marginalizing a template**: an implicit template, if it is local.

## Verification

The model is checked over the combinations: each kind of basis, each choice of `per`, with and
without coupling, on one stream or several, with one or several detector axes, under a diagonal,
banded and stream-coupled weight. On instances small enough for dense matrices:

1. synthesis against the definition, one detector at a time;
2. the transpose against the adjoint;
3. the probed Gram against the dense `Tᵀ W T`;
4. the structural Gram against the dense one, including detectors summed in batches that do
   not divide their number;
5. `factor` against the dense Gram, exact and block by block, with and without the ridge;
6. the Gram under Pomme against the dense `Tᵀ W F T`.

Mapmaking runs check the whole: explicit and implicit templates give the same map, and detector
batches change neither the map nor the amplitudes.

## Open questions

- **The explicit preconditioner under `W'`.** With implicit templates, the system operator uses the
  marginalized weight `W'`, but the preconditioner of the explicit amplitudes is the Gram under
  `W`. Eliminating the implicit templates from it would give the Gram under `W'`: worth building
  only if CG iteration counts, such as for T2P solved explicitly next to a deprojected polynomial,
  show the preconditioner limits convergence.
- **Per-wafer blocks.** Templates grouped by wafer are global: the wafers end up in one block,
  coupled, although they never interact. One block per wafer needs the grouping to be static,
  such as detectors sorted by wafer with static boundaries, padded per wafer across observations.
- **A marginalized global block can be dense.** Under a marginalized weight, the coupling of a
  time-local global block to time-local local ones, such as a common-mode spline next to each
  detector's spline drift, is read off colored probes, as many as its band needs; a local template spanning the observation still takes one
  probe per global amplitude, and its marginalized Gram stays one dense block, `n_global²`,
  inverted dense, the directions the local templates absorb, singular without a ridge, at a
  cutoff. Time-local implicit templates avoid both: scan- and HWP-synchronous ones on a spline of
  time, as `spline_hwp_synchronous` is, would keep the two-level inverse.
- **Probing cost.** A global time-local template needs `colors × k × modes` probes: about 90 for
  one common-mode amplitude per wafer every few seconds, each an application of `T`, `W` and `Tᵀ`.
