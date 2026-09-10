# `furax.linalg`

## Enlarged conjugate gradients

[`ecg`][furax.linalg.ecg] solves a single real symmetric positive-definite system
`A(x) = b` using several search directions per iteration. Its inputs and solution
can be arbitrary PyTrees of real floating-point arrays, including leaves with
different shapes. No preconditioner is applied. With `enlargement=1`, it delegates
to [`cg`][furax.linalg.cg].

```python
import jax.numpy as jnp

from furax import HomothetyOperator, tree
from furax.linalg import ecg

b = {'map': jnp.ones((2, 3)), 'offset': jnp.array(1.0)}
A = HomothetyOperator(2.0, in_structure=tree.as_structure(b))
result = ecg(A, b, enlargement=3)
# result.solution has the same structure as b; every coordinate equals 0.5.
```

### Splitting the residual

The enlargement factor $t$ determines how many components are used to split the
initial residual $r_0=b-Ax_0$. Disjoint coordinate masks define a block $S(r_0)$
whose columns sum to $r_0$. The default partition assigns balanced contiguous
groups across all coordinates, using JAX's leaf order and row-major order within
each leaf. A group can cross a leaf boundary; physical leaves are never concatenated.

For a problem-specific partition, pass integer arrays matching the structure and
shapes of `b`, with labels from zero through `enlargement - 1`:

```python
partition = {
    'map': jnp.array([[0, 0, 0], [1, 1, 1]]),
    'offset': jnp.array(2),
}
result = ecg(A, b, enlargement=3, partition=partition)
```

Empty groups and groups with zero initial residual are allowed. A larger block
does not guarantee better performance: the partition and the spectrum of `A`
both affect convergence.

### Recurrence and numerical stability

The solver uses the Orthodir ECG recurrence described by
[Grigori and Tissot (2019)](https://doi.org/10.1137/18M1196285).
Between restarts, it searches

$$
\mathcal K_{k,t}(A,S(r_0)) =
\operatorname{span}\{S(r_0), AS(r_0), \ldots, A^{k-1}S(r_0)\}.
$$

For an active block $P_k$ with $P_k^\mathsf{T}AP_k=I$, the update is

$$
\alpha_k=P_k^\mathsf{T}r_k,\qquad
x_{k+1}=x_k+P_k\alpha_k,\qquad
r_{k+1}=r_k-AP_k\alpha_k.
$$

The next block is formed from $AP_k$ by subtracting its $A$-projections onto
$P_k$ and $P_{k-1}$, then $A$-orthonormalizing the remaining directions. The two
previous blocks are projected out together, with two passes to limit loss of
orthogonality.

Each new block is normalized with two-pass Gram-Schmidt in the $A$ inner product.
No additional application of `A` is needed for normalization. Numerically dependent
directions are replaced by zeros, keeping the block shapes fixed for JIT compilation.
This deflation prevents division by vanishing direction norms; it does not reduce the
allocated block width or guarantee a reduction in operator work.

The rank threshold is 100 times the largest machine epsilon of the input leaves,
relative to each candidate's norm before projection. Significant negative curvature,
or zero curvature along an independent direction, raises an error. If every direction
is lost, the solver tries a restart from the true residual and raises an error if no
direction remains and the solve has not finished.

By default, `stabilise_every=10` replaces the residual with `b - A(x)` and restarts
the recurrence every ten iterations. Set it to zero to disable periodic restarts.
Restarts limit residual drift but discard the accumulated Krylov space and can
increase the iteration count. For enlargement greater than one, a candidate
convergence is always checked with the true residual using
`norm(b - A(x)) <= atol + rtol * norm(b)`.

With both tolerances zero, the solver counts exactly `max_steps` iterations.
Updates become inert once the true residual falls below the numerical floor,
100 times machine epsilon times the initial residual norm. Otherwise reaching
`max_steps` returns the current approximation, which need not have converged.
`ECGResult.residuals` follows CG's convention: it stores the initial norm followed
by post-step norms in an array of length `max_steps`. The norm after the final
step is therefore omitted when the budget is exhausted; compute `tree.norm(tree.sub(b,
A(result.solution)))` to check the returned solution directly.

### Cost and differentiation

Block vectors are PyTrees with a leading axis of length $t$ on each leaf. `A` is
applied through `jax.vmap`. Block inner products and basis transformations use
matrix contractions within each leaf; the small Gram matrices are summed across
leaves and replicated across the device mesh. The block axis is independent of any
device sharding of the physical coordinates.

Persistent vector storage is $O(nt)$, where $n$ is the number of scalar
coordinates. Each ordinary block recurrence advances using one application of
`A` to a block of $t$ vectors and $O(nt^2)$ orthogonalization work. Initialization,
restarts, and true-residual checks incur additional operator applications. Compare
operator work and warmed-up runtime as well as `num_steps`: a block iteration
processes up to $t$ times as many vectors as a CG iteration.

The default `loop_kind='lax'` supports forward differentiation. Use `'bounded'`
for both forward and reverse differentiation, or `'checkpointed'` for reverse
differentiation. Derivatives follow the executed iterations and are valid locally
where the rank decisions remain fixed. Rank-selection boundaries are nonsmooth.

::: furax.linalg
    options:
      show_submodules: true
      show_root_heading: false
      show_root_members_full_path: true
