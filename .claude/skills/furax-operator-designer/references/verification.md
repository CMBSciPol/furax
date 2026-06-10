# Verifying an operator against its math

The point of verification here is **trust**: the user handed you an equation and wants confidence
the code computes *that* equation. So every check must compare the operator to a reference derived
**independently from the math** — never from the operator itself. A check that derives both sides
from `mv` proves only that JAX is self-consistent.

## The one trap to understand

`op.T.as_matrix().T == op.as_matrix()` looks like an adjoint test, but it is **near-vacuous when
`transpose` is autodiff-derived** (the default). Why: `as_matrix()` is built by applying `mv` to
basis vectors, and the default `op.T` is `jax.linear_transpose(op.mv)`. Both sides come from `mv`,
so the identity holds by construction regardless of whether `mv` matches the user's equation.

It becomes a **real** test only when you **hand-wrote `transpose`** — then it checks your explicit
transpose against autodiff's, which is worth doing. So:

- **Default (autodiff transpose)** → skip the adjoint check; it is theater.
- **Hand-written transpose** → include it; it earns its place.

## The checks that actually catch a misread equation

Use a small instance (sizes ~3–6) so you can build references by hand. Run under float64 to match
furax test precision: `jax.config.update("jax_enable_x64", True)` at the top.

### 1. Pointwise: `mv` vs hand-computed values
Pick a couple of simple inputs (a basis vector, a vector of ones) and write down the expected output
*from the equation*, then assert. This is the most direct test of intent.

### 2. Matrix: `as_matrix()` vs an independent dense matrix
Construct the dense matrix straight from the math — `np.diag(...)`, an explicit gather/scatter
pattern, a hand-typed small array — and compare to `op.as_matrix()`. This catches off-by-one
indexing, wrong axes, and transposed conventions.

### 3. Property checks that validate the decorators you applied
A wrong tag is a silent solver bug, so prove each tag from the matrix:
- `@symmetric` ⇒ `assert_allclose(M, M.T)`
- `@orthogonal` ⇒ `assert_allclose(M @ M.T, I)` (and `M.T @ M == I`)
- `@diagonal` ⇒ off-diagonal entries are zero
- `@square` ⇒ `M.shape[0] == M.shape[1]`
- `@positive_semidefinite` ⇒ `eigvalsh(M).min() >= -tol` **on your test instance** (a sanity check,
  not a proof for all values — only tag PSD when the math guarantees it)

### 4. (If applicable) inverse / round-trip
If you overrode `inverse()` or the operator is meant to be invertible: `assert_allclose(op.I(op(x)), x)`
on a well-conditioned instance, or `op.I.as_matrix() == inv(reference)`.

## `verify.py` template

Drop this in the bundle, fill the `# TODO` spots, and run `uv run python verify.py`. It both checks
the operator and saves `M.npy` for the spy-plot.

```python
"""Verify <Operator> against its mathematical definition.

Run:  uv run python verify.py
The reference matrix/values are built directly from the math, independently of the operator.
"""

import jax

jax.config.update("jax_enable_x64", True)  # match furax test precision

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from my_operator import MyOperator  # <- the generated module / factory

ATOL = 1e-12


def reference_matrix() -> np.ndarray:
    """Dense matrix built DIRECTLY from the equation — must not call MyOperator."""
    # TODO: e.g. return np.diag([2.0, 3.0, 4.0])
    raise NotImplementedError


def main() -> None:
    # --- small, hand-checkable instance ---
    op = MyOperator(  # TODO: construct with small fields + in_structure
        # ...,
        # in_structure=jax.ShapeDtypeStruct((3,), jnp.float64),
    )
    M = np.asarray(op.as_matrix())

    # 1) pointwise vs hand values
    x = jnp.ones(op.in_size)              # TODO: shape/structure to match in_structure
    expected = None                        # TODO: value(s) computed by hand from the equation
    if expected is not None:
        flat_out = np.concatenate([np.asarray(v).ravel() for v in jax.tree.leaves(op(x))])
        assert_allclose(flat_out, expected, atol=ATOL)

    # 2) matrix vs independent reference
    assert_allclose(M, reference_matrix(), atol=ATOL)

    # 3) property checks for the tags you applied
    if op.is_symmetric:
        assert_allclose(M, M.T, atol=ATOL)
    if op.is_orthogonal:
        assert_allclose(M @ M.T, np.eye(M.shape[0]), atol=ATOL)
    if op.is_diagonal:
        assert_allclose(M - np.diag(np.diag(M)), 0.0, atol=ATOL)

    # 4) adjoint check — ONLY if you hand-wrote transpose (else it is vacuous)
    HAND_WRITTEN_TRANSPOSE = False  # TODO: set True if you overrode transpose()
    if HAND_WRITTEN_TRANSPOSE:
        assert_allclose(np.asarray(op.T.as_matrix()).T, M, atol=ATOL)

    np.save("M.npy", M)  # for scripts/spy_plot.py
    print("all checks passed; wrote M.npy")


if __name__ == "__main__":
    main()
```

## Tolerances

With `jax_enable_x64=True`, exact-arithmetic operators (gather, diagonal, permutation) match to
`atol=1e-12` or tighter. Operators with transcendental entries (rotations, FFTs) may need `1e-10`.
If a check needs a loose tolerance to pass, suspect a real discrepancy before relaxing it.
