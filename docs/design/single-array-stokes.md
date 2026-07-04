# Back Stokes with a single dense array (fastquat pattern)

**Status:** proposed — independent PR, to land before the demod TOD-as-array work on
`sb/templates` (which rebases onto it).

## Goal

Refactor `furax.obs.stokes` so each `Stokes` subclass is backed by **one** dense array of shape
`(..., n_stokes)`, with the Stokes components on the trailing axis — mirroring
[`fastquat.quaternion.Quaternion`](https://github.com/CMBSciPol/fastquat/blob/main/src/fastquat/quaternion.py)
(single `wxyz` array `(..., 4)`; `.w/.x/.y/.z` are trailing-axis slices; registered as a pytree
with the array as the sole child).

Today a `StokesIQU` is a dataclass with three separate array fields `i, q, u` → three pytree
leaves. Backing it with one array makes the Stokes axis a real array axis:

- Stokes-axis operations vectorise (QU rotation, HWP, component-separation mixing, per-leg noise)
  instead of `jax.tree.map` over 2–4 leaves.
- Conversion to/from a batched array is free (`.array` / `from_array`) — this is exactly the seam
  the demodulated-TOD-as-array work needs, so that work stops needing a bespoke stack operator.
- Fewer pytree leaves (n → 1) lightens `scan`/tree overhead.

It is also the smallest concrete instance of issue #18's `TreeArray`: a fixed irreducible treedef
(the Stokes labels) with one materialised axis. Good place to prove the pattern before generalising
`factorize` / `einsum` / `moveaxis`.

## Design

- **Backing field.** Each subclass holds a single field `array: (..., n)` with `n = len(stokes)`
  (`StokesI`→1, `StokesQU`→2, `StokesIQU`→3, `StokesIQUV`→4). The `stokes: ClassVar` label
  distinguishes I vs QU vs … so a 1-leaf pytree is still unambiguous by type/treedef.
- **Pytree.** Register with the backing array as the **only** child, aux `None`
  (`tree_flatten -> (array,), None`). `tree_unflatten` reconstructs without re-validation (avoid
  tracer issues under transforms), like fastquat.
- **Component access (source-compatible).** `.i/.q/.u/.v` are properties returning
  `self.array[..., self.stokes.index(<LETTER>)]`. Existing `.i/.q/.u` reads are unchanged.
- **Array view / conversion.** `.array` returns the backing array; `from_array(arr)` wraps it
  (assert trailing dim `== n`). These two are the entire Stokes↔array seam.
- **Shape / dtype.** `shape = array.shape[:-1]` (drop the Stokes axis); `dtype = array.dtype`.
- **Constructors — keep component-positional to bound churn.** `StokesIQU(i, q, u)` must keep
  working (~39 call sites). Stack the components into `array` (custom `__init__` on a non-init
  `array` field, or a `from_components` classmethod plus a mechanical call-site migration —
  implementer's choice, but positional-component construction stays available).
- **`structure_for(shape, dtype)`.** Returns a Stokes object whose backing is a single
  `ShapeDtypeStruct((*shape, n), dtype)` (duck-typed; it exposes `.shape/.dtype`). This turns a
  Stokes operator's `in/out_structure` from an n-leaf pytree into a **1-leaf** pytree — intended,
  but drives the audit below.
- **`__getitem__(index)`.** `type(self).from_array(self.array[index])` — `index` addresses the
  leading batch dims; the trailing Stokes axis is preserved.
- **Arithmetic / `*_like`.** Current ops route through `furax.tree` (`add`/`sub`/`mul`/…,
  `zeros_like`/`normal_like`); with one leaf they `tree.map` over the single array and keep
  working. Keep the `furax.tree` routing — no per-op rewrite.

## Call-site audit — the only semantic breakers (was n leaves, now 1)

- **Per-leg `tree.map` over a Stokes TOD** → an axis op. Chief case: `_model._noise_model`'s
  `jax.tree.map(_compute_Pxx_and_fit, SAMPLE_DATA)` (fit per leg) → `vmap`/axis over the trailing
  Stokes axis.
- **`np.stack([m.i, m.q, m.u], ...)`** in `results.py` / `mapmaker.py` (map save) → `m.array`.
- **Preconditioner** block-diagonal over Stokes leaves → over the trailing axis
  (`preconditioner.py`).
- **Operators constructing a Stokes from components** (pointing / polarizer / HWP / QURotation
  outputs, landscapes) → build via `from_array` / stacking. `.i/.q/.u` reads unaffected.
- **Any `jax.tree.leaves(stokes)` assuming n arrays** → now 1 array; grep and fix.
- Optional follow-up (not required for correctness): rewrite QU rotation / HWP / SED mixing as a
  single matmul over the trailing axis instead of per-component slicing.

## Files

- Core: `src/furax/obs/stokes.py`.
- Audited sites: `src/furax/obs/operators/_hwp.py`, `_qu_rotations.py`, `pointing.py`,
  `landscapes.py`, the SED / component-separation operators, and
  `src/furax/mapmaking/{_model,mapmaker,results,preconditioner}.py`, plus any `tree.leaves(stokes)`
  usage.
- Tests: `tests/core/test_pack.py`, `tests/core/test_trees.py`, `tests/mapmaking/test_*`,
  `tests/obs/operators/test_hwp.py`, `test_polarizers.py`, component-separation tests.

## Verification

- `.i/.q/.u` and `StokesIQU(i, q, u)` behave identically; round-trip `from_array(x.array) == x`.
- Full suite green — Stokes is cross-cutting, run all: `uv run pytest -q`.
- `uvx prek -a` (ruff, mypy — mind the jaxtyping annotations on the backing arrays).
