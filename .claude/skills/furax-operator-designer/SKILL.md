---
name: furax-operator-designer
description: >-
  Design and implement a furax linear operator (or, for a genuinely nonlinear map, a plain JAX
  function) from a mathematical description, and prove it matches the math. Use this whenever the
  user has math — a LaTeX file, a PDF, a paper, Markdown notes, or equations typed in chat —
  describing an operator, matrix, transform, projection, or linear map and wants working furax
  code for it. Triggers include: "implement this operator in furax", "turn these equations into a
  furax operator", "build an AbstractLinearOperator for ...", "I have a PDF/derivation of a
  matrix/operator, make it", "design a pointing / mixing / rotation / finite-difference /
  projection / noise-covariance operator", or any request to translate CMB / inverse-problem math
  into furax operator code. Produces: the operator module, a Markdown design doc that maps the math
  to the code with LaTeX matrix sketches and a rendered spy-plot of the operator's actual matrix,
  and an executed example notebook. Reach for it even when the user does not say "furax" but is
  clearly working in this repo and wants math turned into operator code.
---

# Furax Operator Designer

Turn a mathematical description of an operator into working, **verified** furax code, plus a design
doc and a runnable notebook that show — not just assert — that the code matches the math.

## What you produce

The default is a **researcher bundle**, a directory `./<operator_snake_name>/` containing:

1. **`<name>.py`** — the operator: a `furax.AbstractLinearOperator` subclass, or a composition of
   existing furax operators wrapped in a small factory function. (For a genuinely nonlinear map, a
   plain documented JAX function instead — see *Linear vs nonlinear*.)
2. **`design.md`** — a design doc mapping the math to the code, with LaTeX matrix sketches **and**
   `figs/structure.png`, a spy-plot of the operator's *actual* `as_matrix()`.
3. **`example.ipynb`** — an **executed** notebook: build the operator, apply it to example data,
   show the outputs and the structure plot.
4. **`verify.py`** — the math-anchored checks, so the bundle validates itself.

**Library mode** is for when the user asks to "add this to furax" / "open a PR": place the module
under `src/furax/...`, export it, and add a `tests/...` file in repo conventions. See *Library
mode* at the end. Default to the bundle unless they ask for integration.

## Step 1 — Extract the math and confirm it before writing anything

Math read from a PDF or dense LaTeX is easy to misread, and one misread equation poisons the code,
the doc, and the notebook simultaneously. So front-load a confirmation:

1. Read the source with the **Read** tool (it handles PDF, LaTeX, and Markdown directly).
2. Extract, then **echo back a short "here's what I understood — confirm or correct" block and
   wait** for the user before generating. Cover:
   - the operator's **action** — what it does to an input, in words and as the governing equation;
   - the **input and output spaces** — shapes, dtypes, and PyTree structure (a bare array? a
     `{'I','Q','U'}` dict? a list of detector arrays?);
   - the **parameters** — which become operator fields, and which are dynamic arrays vs static
     metadata (shapes, axes, flags);
   - declared **algebraic properties** (symmetric, orthogonal, diagonal, square?) and the
     **adjoint/transpose** if the source gives one;
   - whether the map is **linear** (next section).

This costs one short message and saves a whole wrong bundle.

## Step 2 — Linear vs nonlinear (center the linear case)

furax exists for **linear** operators: transpose, inverse, `as_matrix()`, tags, and `@`-composition
all assume linearity. Decide first:

- **Linear** → an `AbstractLinearOperator`. This is the main path; everything below targets it.
- **Affine** (`x ↦ A·x + b`) → implement the linear part `A` as the operator and surface the
  constant `b` separately; state this split explicitly.
- **Genuinely nonlinear** → a plain, documented JAX function. Do **not** fabricate a matrix sketch,
  a transpose, or an adjoint check — they are meaningless. Say clearly in `design.md` that this is a
  function, not a furax operator, and skip `as_matrix()` and the spy-plot. (`asoperator` does *not*
  rescue this: it still wraps a *linear* function and builds the transpose via `linear_transpose`.)

## Step 3 — Build the operator

**Ask first: can existing furax blocks express this by composition?** Composing tested operators is
more robust than a fresh subclass and gives you transpose/inverse for free. `references/operator-cookbook.md`
has the building-block catalog (Diagonal, Block{Diagonal,Row,Column}, Dense, Index, Mask, Sum,
Fourier, Toeplitz, Reshape/MoveAxis, `asoperator`, …) and worked **math → operator** patterns
(diagonal gain, pointing/sampling, finite difference, QU rotation, projection). Only write a fresh
subclass when no composition fits.

For a fresh subclass, **read `references/operator-cookbook.md` first** — it has the canonical
template and the rules that actually bite. The essentials:

- `mv(self, x)` is the single required method. It must return the correct **output shape**: furax
  derives `out_structure` from `mv` via `jax.eval_shape`, so a shape bug there is a structure bug.
- Fields are dynamic JAX arrays by default; mark shapes/axes/strings/flags static with
  `field(metadata={'static': True})`.
- Apply a property decorator (`@symmetric`, `@diagonal`, `@orthogonal`, `@square`, …) **only when it
  is provably true of the math.** Tags are static and unchecked — a wrong tag silently corrupts
  solvers. (Canonically: `DiagonalOperator` is tagged symmetric but deliberately *not* PSD.)
- Leave `transpose` to autodiff unless you can write a cheaper or clearer one. `@symmetric` already
  makes `T` return `self`.

Iterate with `uv run python …`. Format and lint generated code:
`uvx prek ruff-format --files <f>` then `uvx prek ruff-check --files <f>`.

## Step 4 — Verify against the math, not against itself

This is what makes the bundle trustworthy — do it deliberately. Full recipe and a `verify.py`
template are in `references/verification.md`. The principle: **build your reference independently
from the math, never from the operator.** The checks that actually catch a misread equation:

1. **Pointwise** — `op.mv(x)` on a few hand-chosen inputs equals values you computed by hand from
   the equation.
2. **Matrix** — on a small instance, `op.as_matrix()` equals a dense matrix you built directly from
   the math.
3. **Property / decorator checks** — validate *every* tag you applied: `@symmetric` ⇒ `M == M.T`;
   `@orthogonal` ⇒ `M @ M.T == I`; `@diagonal` ⇒ off-diagonals vanish.

Trap to avoid: `op.T.as_matrix().T == op.as_matrix()` is **near-vacuous when `transpose` is
autodiff-derived**, because both sides come from `mv` — it tests JAX against itself, not against
your math. Include the adjoint check **only when you hand-wrote `transpose`** (then it genuinely
tests your code against autodiff). `references/verification.md` explains why.

Run with `uv run python verify.py` and make every check pass before moving on.

## Step 5 — Write the design doc (`design.md`)

Use this structure:

1. **Summary** — one paragraph: what the operator does and its input/output spaces.
2. **Mathematical definition** — the source equation(s) in LaTeX (`$$ … $$`), every symbol defined.
3. **Matrix sketch** — a LaTeX block-matrix showing the *structure* (block-diagonal, banded, the
   pointing pattern, …) at a small illustrative size.
4. **Code mapping** — a table: each piece of the math ↦ the field / line / sub-operator in the code.
   This is the heart of the doc; it is the bridge the user asked for.
5. **Verified structure** — embed `figs/structure.png`, the spy-plot of the *actual* `as_matrix()`
   on a small instance, so the reader sees the code reproduce the sketch. (Generate it below.)
6. **Properties** — which tags hold and why, each backed by its verification result.

### Generating the spy-plot

`scripts/spy_plot.py` renders any dense matrix as a structure plot (signed colormap, zeros blank,
values annotated when small). In `verify.py`, after computing the small `as_matrix()`:

```python
import numpy as np
np.save("M.npy", np.asarray(op.as_matrix()))
```

then

```bash
uv run --with matplotlib python <SKILL_DIR>/scripts/spy_plot.py \
    M.npy <bundle>/figs/structure.png --title "A — short description"
```

(`<SKILL_DIR>` is this skill's directory.) For a side-by-side trust signal, also render the
difference `as_matrix() - reference`; if the code is right it is uniformly blank.

## Step 6 — Build and run the notebook (`example.ipynb`)

Author the notebook, then **execute it so outputs and plots are real** — an example that has not
run is not evidence. Cells, in order: imports → construct the operator → apply it to example data →
show the small `as_matrix()` and the inline spy-plot → a verification cell → a short narrative
tying each step back to the math.

Build it in Python with `nbformat` (less error-prone than hand-writing `.ipynb` JSON), then execute
with the **proven, kernel-robust** mechanism below. The subtlety it avoids: `nbconvert --execute`
defaults to the `python3` kernelspec, which can resolve to a system Python that lacks furax;
registering a kernel into the project env's own *sys-prefix* forces the right interpreter.

```bash
uv run --with nbconvert --with ipykernel bash -c '
  python -m ipykernel install --sys-prefix --name furaxk >/dev/null 2>&1
  python -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=furaxk \
    --output example.ipynb path/to/draft.ipynb'
```

Then confirm the executed notebook has **no error outputs** (read it back, or check
`nbconvert` exited 0). If a cell errored, fix the operator or the cell and re-run — do not ship a
notebook with tracebacks.

## Library mode (only when the user asks to add it to furax)

- Place the module in the right `src/furax/...` package and add the class to that package's
  `__init__.py` `__all__` (and the top-level `src/furax/__init__.py` if it is public).
- Give it a docstring with a runnable **doctest**, matching the style of `src/furax/core/_diagonal.py`.
- Add a test under `tests/...` mirroring the source path, in repo conventions: `@pytest.mark.parametrize`
  for similar cases, the float64 autouse fixture (already global via `tests/conftest.py`), and the
  math-anchored checks from Step 4 (`as_matrix` vs an independent reference, property checks, and the
  adjoint check only if `transpose` is overridden).
- Format / lint / type-check the touched files: `uvx prek ruff-format`, `uvx prek ruff-check`,
  `uvx prek mypy`. Then `uv run pytest <file> -v`.
- Still produce `design.md` and an executed notebook unless told otherwise.

## References

- `references/operator-cookbook.md` — `AbstractLinearOperator` anatomy, the canonical subclass
  template, decorator semantics, field rules, the building-block catalog, worked math→operator
  patterns, and the gotchas. **Read before writing operator code.**
- `references/verification.md` — the math-anchored verification recipe, the adjoint-check subtlety,
  and a `verify.py` template.
