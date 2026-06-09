# Furax operator cookbook

How to express a mathematical operator in furax. Read this before writing a fresh subclass. Source
of truth is `src/furax/core/` — every operator there has a docstring with doctests; when a
constructor signature is unclear, read the file.

## Contents
- [Anatomy of an operator](#anatomy)
- [The canonical subclass template](#template)
- [Fields: dynamic vs static](#fields)
- [Input/output structure](#structure)
- [Property decorators (and when to apply them)](#decorators)
- [Transpose, inverse, as_matrix](#transpose)
- [Gotchas](#gotchas)
- [Building-block catalog (compose first)](#catalog)
- [Worked math → operator patterns](#patterns)

<a id="anatomy"></a>
## Anatomy of an operator

Every operator subclasses `furax.AbstractLinearOperator`, a **frozen dataclass** that is also a JAX
PyTree. You get this for free: it is callable (`op(x) == op.mv(x)`), composes with `@`
(`CompositionOperator`), adds with `+`, scales with `*`/`/`, exposes `.T` (transpose), `.I`
(inverse), `.as_matrix()`, `in_structure`, `out_structure`, `in_size`, `out_size`, and `is_*`
property flags. You implement, at minimum, `mv`.

Because it is a frozen dataclass *and* a PyTree, an operator can be `jit`/`vmap`/`grad`-ed and its
array fields are traced like any other JAX leaf.

<a id="template"></a>
## The canonical subclass template

```python
from dataclasses import field

import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import Inexact, PyTree

from furax import AbstractLinearOperator


class MyOperator(AbstractLinearOperator):
    """One line: what it does. D(x) = ...

    Longer description, then a doctest that doubles as an example:

    Example:
        >>> import jax, jax.numpy as jnp
        >>> op = MyOperator(jnp.array([2.0, 3.0]),
        ...                 in_structure=jax.ShapeDtypeStruct((2,), jnp.float64))
        >>> op(jnp.array([1.0, 1.0]))
        Array([2., 3.], dtype=float64)
    """

    weights: Inexact[Array, ' n']                       # dynamic field (a JAX array, traced)
    axis: int = field(metadata={'static': True}, default=-1)   # static field (metadata, not traced)

    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return jax.tree.map(lambda leaf: self.weights * leaf, x)
```

Construct with the required keyword `in_structure`. There is no `__init__` to write: the dataclass
machinery and `AbstractLinearOperator.__post_init__` handle it. **If you do need custom construction
logic, prefer `__post_init__`** and set fields with `object.__setattr__` (the class is frozen). If
you write an `__init__`, you must call `super().__init__(in_structure=...)` yourself — see the long
comment in `_AbstractLazyDualOperator` in `src/furax/core/_base.py` for why.

<a id="fields"></a>
## Fields: dynamic vs static

- **Dynamic** (default): JAX arrays the operator multiplies/uses — diagonals, kernels, angles.
  These are PyTree leaves: traced under `jit`, differentiated under `grad`.
- **Static**: anything that defines *shape or behavior* but is not array data — an axis index, a
  mode string, a boolean flag, an output shape. Mark with `field(metadata={'static': True})`.
  Getting this wrong causes either retracing (static data left dynamic) or `TracerArrayConversion`
  errors (dynamic data marked static). When in doubt: "does autodiff/jit need to see through it?"
  yes → dynamic, no → static.

<a id="structure"></a>
## Input/output structure

- `in_structure` is a **required keyword-only field**: a PyTree of `jax.ShapeDtypeStruct` describing
  the input. For a bare array: `jax.ShapeDtypeStruct(shape, dtype)`. For structured input, mirror
  the PyTree: `{'Q': jax.ShapeDtypeStruct((npix,), jnp.float64), 'U': ...}`.
- Convenience: `furax.tree.as_structure(x)` turns an example array/PyTree into its structure.
  Several operators (e.g. `DiagonalOperator`) default `in_structure` from their data.
- `out_structure` is **derived** by `jax.eval_shape(self.mv, self.in_structure)`. You normally do
  not set it — but your `mv` must therefore produce the right shape. If input and output shapes are
  equal, the `@square` decorator short-circuits this (sets `out_structure = in_structure`).

<a id="decorators"></a>
## Property decorators (and when to apply them)

Apply from `furax` (`from furax import symmetric, diagonal, orthogonal, square, ...`). They set
*static* tags and sometimes override methods. Apply **only when the math guarantees the property** —
nothing checks them, and solvers trust them.

| Decorator | Implies | Also does | Apply when |
|---|---|---|---|
| `@square` | — | `out_structure = in_structure` | input space == output space |
| `@symmetric` | square | `transpose()` returns `self` | `A == Aᵀ` exactly |
| `@orthogonal` | square | `inverse() = transpose()` | `Aᵀ A = I` (rotations, permutations) |
| `@diagonal` | symmetric (→ square) | — | acts elementwise |
| `@tridiagonal` / `@lower_triangular` / `@upper_triangular` | square | — | that sparsity holds |
| `@positive_semidefinite` / `@negative_semidefinite` | square | — | quadratic form has that sign |

Note the deliberate gap: `DiagonalOperator` is `@symmetric` but **not** PSD, because positivity is
value-dependent and tags are static. Mirror that discipline — never tag PSD/orthogonality you cannot
guarantee for *all* values of the fields. Validate every tag you apply (see `verification.md`).

<a id="transpose"></a>
## Transpose, inverse, as_matrix

- **Transpose**: the default `transpose()` wraps `jax.linear_transpose(self.mv, ...)` — it just
  works for any linear `mv`, no code needed. Override only (a) for efficiency, or (b) to express a
  known closed form. `@symmetric` makes `T` return `self`.
- **Inverse**: default `.I` solves iteratively (lineax/CG) and needs a square operator. Override
  `inverse()` when there is a closed form (e.g. `DiagonalOperator` divides by its diagonal).
- **as_matrix**: the base builds the dense matrix by applying `mv` to basis vectors — free for any
  operator. Override only for a faster explicit construction. Use it for verification and the
  spy-plot; never inside hot paths.

<a id="gotchas"></a>
## Gotchas

- **jaxtyping 1-D shapes need a leading space**: write `Inexact[Array, ' n']` (note the space), not
  `Inexact[Array, 'n']`. Without it Ruff raises F821 (undefined name `n`); the space turns it into
  F722 (a forward-annotation string), which is ignored. Multi-dim like `'a b'` is fine as-is.
- **Frozen class**: assign fields only via `object.__setattr__` inside `__post_init__`/`__init__`.
- **mv decides the output shape** — there is no separate place to declare it. A wrong shape in `mv`
  is a wrong `out_structure`.
- **Tags are not validated** — a wrong decorator is a silent correctness bug. Always verify.
- **`asoperator(func, in_structure=...)`** wraps a *linear* function into an operator (transpose via
  autodiff). It is for linear maps only; it does not make a nonlinear function an operator.

<a id="catalog"></a>
## Building-block catalog (compose first)

Prefer composing these over a new subclass. Compose with `@` (matmul) and `+`. All take
`in_structure=` unless they infer it.

| Operator | Action | Notes |
|---|---|---|
| `IdentityOperator(in_structure=…)` | `x` | orthogonal, diagonal, PSD |
| `HomothetyOperator(k, in_structure=…)` | `k·x` | scalar multiple; diagonal |
| `DiagonalOperator(d, axis_destination=-1)` | `d ⊙ x` elementwise | symmetric, square; infers `in_structure` from `d` |
| `BroadcastDiagonalOperator(d, axis_destination=…, in_structure=…)` | diagonal **with broadcasting** | can change shape (non-square) |
| `DenseBlockDiagonalOperator(blocks, in_structure=…, subscripts='ij...,j...->i...')` | per-block dense matmul via einsum | small dense blocks (e.g. per-pixel 2×2/3×3) |
| `BlockDiagonalOperator(pytree_of_ops)` | block-diagonal `diag(A,B,…)` | operands in any PyTree container |
| `BlockRowOperator([A,B,…])` | `[A B …]`, horizontal | input is a list/tuple of blocks; outputs their sum |
| `BlockColumnOperator({…})` | vertical stack | one input → a PyTree of outputs |
| `IndexOperator(indices, in_structure=…)` | gather `x[indices]` | **pointing / sampling**; transpose scatters (co-adds) |
| `MaskOperator(...)` | select/zero by boolean mask | see `_mask.py` |
| `SumOperator(...)` | sum over an axis | see `_sum.py`; transpose broadcasts |
| `FourierOperator(...)` | FFT-based transform | see `_fourier.py` |
| `SymmetricBandToeplitzOperator(bands, in_structure=…)` | symmetric banded Toeplitz matvec | **stationary noise covariance**; PSD-ish, solver-friendly |
| `TreeOperator(pytree_of_ops)` | apply a per-leaf operator across a PyTree | component-wise processing |
| `ReshapeOperator(shape, in_structure=…)` / `MoveAxisOperator(source, destination, …)` / `RavelOperator(…)` | shape plumbing | orthogonal reshapes |
| `asoperator(func, in_structure=…)` | wrap a linear `func` | transpose via autodiff |

<a id="patterns"></a>
## Worked math → operator patterns

**(a) Diagonal gain / weighting** — `y_i = g_i x_i`. Compose, no subclass:
```python
from furax import DiagonalOperator
G = DiagonalOperator(gains)              # gains: (n,) array; in_structure inferred
```

**(b) Pointing / sampling** `P` — sample a sky map onto a timestream, `(P x)_t = x_{p(t)}`:
```python
import jax
from furax import IndexOperator
P = IndexOperator(pixel_index, in_structure=jax.ShapeDtypeStruct((npix,), jnp.float64))
# P.T is the classic "co-add into pixels" scatter — for free.
```

**(c) Finite difference** `D` — `(D x)_i = x_{i+1} − x_i`, non-square `(n → n−1)`. Fresh subclass;
autodiff transpose is correct, so no `transpose` override needed:
```python
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import Inexact, PyTree
from furax import AbstractLinearOperator

class ForwardDifferenceOperator(AbstractLinearOperator):
    """(D x)_i = x_{i+1} - x_i, mapping length n to length n-1."""
    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        return x[1:] - x[:-1]

# D = ForwardDifferenceOperator(in_structure=jax.ShapeDtypeStruct((n,), jnp.float64))
```
If you instead hand-write `transpose` here (Dᵀ is the backward difference with boundary terms), the
adjoint check in `verification.md` becomes a *real* test of it.

**(d) QU polarization rotation** by angle `ψ` per pixel — orthogonal, validate `M Mᵀ = I`:
```python
import jax.numpy as jnp
from jax import Array
from jaxtyping import Inexact, PyTree
from furax import AbstractLinearOperator, orthogonal

@orthogonal
class QURotationOperator(AbstractLinearOperator):
    """Rotate (Q, U) by 2ψ per pixel: Q' = cos2ψ·Q − sin2ψ·U, U' = sin2ψ·Q + cos2ψ·U."""
    angle: Inexact[Array, ' p']                      # per-pixel ψ, dynamic
    def mv(self, x: PyTree[Inexact[Array, ' _a']]) -> PyTree[Inexact[Array, ' _b']]:
        c, s = jnp.cos(2 * self.angle), jnp.sin(2 * self.angle)
        return {'Q': c * x['Q'] - s * x['U'], 'U': s * x['Q'] + c * x['U']}

# in_structure = {'Q': ShapeDtypeStruct((p,), f64), 'U': ShapeDtypeStruct((p,), f64)}
```
`@orthogonal` sets `.I = .T`; verify `M @ M.T == I` to confirm the tag is earned.

**(e) Projection / mixing matrix** (e.g. component separation `d = A s`, sky components → frequency
channels) — small dense per-pixel block: use `DenseBlockDiagonalOperator` with `blocks` of shape
`(npix, n_freq, n_comp)` and matching `subscripts`, or a custom `mv` using `jnp.einsum`. Prefer the
block operator when the same dense matrix structure repeats over an axis.

When a pattern is *almost* a catalog operator, reach for composition (`A @ B`, `BlockDiagonal`,
`A + B`) before subclassing — you inherit tested transpose/inverse behavior.
