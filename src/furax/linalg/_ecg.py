"""Enlarged conjugate gradients for PyTree-valued linear systems."""

from collections.abc import Callable
from typing import Literal, NamedTuple

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.tree_util import PyTreeDef
from jaxtyping import Array, Float, Int, PyTree

from furax import AbstractLinearOperator, tree

from ._cg import cg


class ECGResult(NamedTuple):
    """Result of the Enlarged Conjugate Gradient solver.

    Attributes:
        solution: Approximate solution of `A(x) = b`, with the same PyTree structure as `b`.
        residuals: Residual norms, shape `(max_steps,)`. Entry zero is the initial norm;
            entry `i` is the norm after `i` steps. Unused entries are zero. As in
            [`CGResult`][], the last norm is omitted when all `max_steps` steps are taken.
        num_steps: Number of block iterations, each processing up to `enlargement` vectors.
    """

    solution: PyTree[Float[Array, '...']]
    residuals: Float[Array, ' max_steps']
    num_steps: Array


class _ECGCarry(NamedTuple):
    x: PyTree
    r: PyTree
    p: PyTree
    ap: PyTree
    previous: PyTree
    a_previous: PyTree
    step: Array
    converged: Array
    inert: Array
    residuals: Array


def _combine(block: PyTree, coefficients: Array) -> PyTree:
    """Combine block vectors along their leading axis, preserving each leaf's dtype."""
    return jax.tree.map(
        lambda v: jnp.tensordot(
            coefficients.astype(v.dtype), v, axes=1, precision=jax.lax.Precision.HIGHEST
        ),
        block,
    )


def _gram(left: PyTree, right: PyTree) -> Array:
    """Return pairwise inner products, reducing over physical coordinates and leaves."""
    # Explicitly replicate the result when contracting over explicitly sharded coordinates.
    out_sharding = P() if not jax.sharding.get_abstract_mesh().empty else None
    products = jax.tree.map(
        lambda a, b: jnp.tensordot(
            a,
            b,
            axes=(tuple(range(1, a.ndim)), tuple(range(1, b.ndim))),
            precision=jax.lax.Precision.HIGHEST,
            out_sharding=out_sharding,
        ),
        left,
        right,
    )
    return sum(jax.tree.leaves(products), start=jnp.array(0))


def _mix(block: PyTree, coefficients: Array) -> PyTree:
    return jax.tree.map(
        lambda v: jnp.tensordot(
            coefficients.T.astype(v.dtype), v, axes=1, precision=jax.lax.Precision.HIGHEST
        ),
        block,
    )


def _scale(vector: PyTree, coefficient: Array) -> PyTree:
    return jax.tree.map(lambda v: v * coefficient.astype(v.dtype), vector)


def _same_structure(left: PyTree, right: PyTree) -> bool:
    """Compare array spaces independently of JAX's scalar weak-type promotion flag."""
    left_def: PyTreeDef
    right_def: PyTreeDef
    left_leaves, left_def = jax.tree.flatten(left)
    right_leaves, right_def = jax.tree.flatten(right)
    return left_def == right_def and all(
        a.shape == b.shape and a.dtype == b.dtype for a, b in zip(left_leaves, right_leaves)
    )


def _orthonormalize(
    vectors: PyTree,
    images: PyTree,
    reference_norms: Array,
    rank_rtol: float,
) -> tuple[PyTree, PyTree, Array]:
    """A-orthonormalize a block with two-pass Gram-Schmidt and fixed-shape rank deflation.

    `images` contains A applied to `vectors`. Reference norms are measured before
    projection against older blocks, so roundoff left by an exhausted Krylov space
    is not amplified into a new direction. Inactive columns of both outputs are zero.
    """
    width = reference_norms.size
    basis = tree.zeros_like(vectors)
    a_basis = tree.zeros_like(images)
    active = jnp.zeros(width, dtype=bool)

    def column(i: int | Array, state: tuple[PyTree, PyTree, Array]) -> tuple[PyTree, PyTree, Array]:
        q, aq, active = state
        v = jax.tree.map(lambda a: a[i], vectors)
        av = jax.tree.map(lambda a: a[i], images)
        for _ in range(2):
            coefficients = jax.vmap(tree.dot, in_axes=(0, None))(aq, v)
            v = tree.sub(v, _combine(q, coefficients))
            av = tree.sub(av, _combine(aq, coefficients))
        norm = tree.norm(v)
        energy = tree.dot(v, av)
        independent = norm > rank_rtol * reference_norms[i]
        energy = eqx.error_if(
            energy,
            independent & (energy < -rank_rtol * norm * tree.norm(av)),
            'ecg: negative curvature; A must be symmetric positive definite',
        )
        energy = eqx.error_if(
            energy,
            independent & (energy == 0),
            'ecg: zero curvature; A must be positive definite',
        )
        keep = independent & (energy > 0)
        # Mask before sqrt/division as well as afterwards: inactive columns must have
        # finite derivatives in bounded and checkpointed loops.
        inverse_norm = jnp.where(keep, jax.lax.rsqrt(jnp.where(keep, energy, 1)), 0)
        v, av = _scale(v, inverse_norm), _scale(av, inverse_norm)
        q = jax.tree.map(lambda a, b: a.at[i].set(b), q, v)
        aq = jax.tree.map(lambda a, b: a.at[i].set(b), aq, av)
        return q, aq, active.at[i].set(keep)

    result: tuple[PyTree, PyTree, Array] = jax.lax.fori_loop(
        0, width, column, (basis, a_basis, active)
    )
    return result


def _partition_labels(b: PyTree, partition: PyTree | None, width: int) -> PyTree:
    treedef: PyTreeDef
    leaves, treedef = jax.tree.flatten(b)
    if partition is None:
        size = sum(v.size for v in leaves)
        group_size, extra = divmod(size, width)
        long_group_end = extra * (group_size + 1)
        offset = 0
        labels = []
        for v in leaves:
            index = jnp.arange(v.size) + offset
            # The first `extra` groups get one additional coordinate. Division avoids
            # an index * width intermediate that can overflow int32 on large maps.
            label = jnp.where(
                index < long_group_end,
                index // (group_size + 1),
                extra + (index - long_group_end) // max(group_size, 1),
            )
            labels.append(label.reshape(v.shape))
            offset += v.size
        return treedef.unflatten(labels)
    partition_def: PyTreeDef
    labels, partition_def = jax.tree.flatten(partition)
    if partition_def != treedef:
        raise ValueError('partition must have the same PyTree structure as b')
    checked = []
    for label, v in zip(labels, leaves):
        label = jnp.asarray(label)
        if label.shape != v.shape or not jnp.issubdtype(label.dtype, jnp.integer):
            raise ValueError('partition leaves must be integer arrays with the same shapes as b')
        checked.append(
            eqx.error_if(
                label,
                jnp.any((label < 0) | (label >= width)),
                'partition labels must be in [0, enlargement)',
            )
        )
    return treedef.unflatten(checked)


def ecg(
    A: AbstractLinearOperator,
    b: PyTree[Float[Array, '...']],
    x0: PyTree[Float[Array, '...']] | None = None,
    *,
    enlargement: int = 4,
    partition: PyTree[Int[Array, '...']] | None = None,
    max_steps: int = 500,
    atol: float = 0.0,
    rtol: float = 1e-5,
    stabilise_every: int = 10,
    loop_kind: Literal['lax', 'checkpointed', 'bounded'] = 'lax',
    iteration_callback: Callable[[Array, Array], None] | None = None,
) -> ECGResult:
    r"""Solve a real symmetric positive-definite system using enlarged conjugate gradients.

    The Orthodir recurrence searches the enlarged Krylov space
    $\mathcal K_{k,t}(A,S(r_0))$, where the columns of $S(r_0)$ sum to the initial
    residual. Each iteration processes up to `enlargement` directions. Operators
    receive ordinary vectors through `jax.vmap`; no dense matrix or flattened
    system vector is constructed. Vector storage is $O(nt)$, with $O(nt^2)$ work
    for block orthogonalization, where $n$ is the total number of scalar coordinates.

    Convergence uses the true residual: $\|b-Ax\| \leq \mathrm{atol} +
    \mathrm{rtol}\|b\|$. With both tolerances zero, exactly `max_steps` iterations
    are counted, and updates become inert at the floating-point residual floor.
    Dependent directions are masked without changing array shapes. The numerical
    rank threshold is 100 times the largest machine epsilon of the input leaves.
    The solver raises an Equinox runtime error on detected negative or zero curvature,
    or when the search space is exhausted with a significant true residual.

    Forward differentiation is supported by `lax` and `bounded` loops; reverse
    differentiation by `bounded` and `checkpointed`. Derivatives describe the
    executed iterations with locally fixed rank decisions; rank changes are
    nonsmooth. No preconditioner is applied.

    Args:
        A: A real symmetric positive-definite Furax operator on the structure of `b`.
        b: Nonempty PyTree of real floating-point arrays.
        x0: Initial guess matching `b`, or `None` for zeros.
        enlargement: Positive static block width. With one direction, delegates to
            [`cg`][], including its residual replacement and stopping behavior.
        partition: Integer labels with the same PyTree structure and leaf shapes as
            `b`, in `[0, enlargement)`. Each label selects one residual component.
            Empty groups are allowed. Defaults to balanced contiguous groups in
            JAX leaf order and row-major coordinate order across all leaves.
        max_steps: Nonnegative static ceiling on block iterations.
        atol: Nonnegative absolute residual tolerance.
        rtol: Nonnegative relative residual tolerance.
        stabilise_every: Recompute the true residual and restart the recurrence every
            this many steps. Zero disables periodic restarts. Convergence candidates
            are always checked using the true residual.
        loop_kind: Equinox loop lowering: `lax`, `bounded`, or `checkpointed`.
        iteration_callback: Optional JIT-compatible host callback receiving the zero-based
            step index and post-step residual norm via `jax.debug.callback`.

    Returns:
        Solution, residual history, and block iteration count in an [`ECGResult`][].

    References:
        Grigori and Tissot, *Scalable Linear Solvers Based on Enlarged Krylov Subspaces
        with Dynamic Reduction of Search Directions*, SIAM J. Sci. Comput. 41(5),
        C522-C547 (2019). https://doi.org/10.1137/18M1196285.

    Examples:
        >>> import jax.numpy as jnp
        >>> from furax import DiagonalOperator, tree
        >>> from furax.linalg import ecg
        >>> d = jnp.array([1., 2., 3., 4.])
        >>> b = jnp.ones(4)
        >>> A = DiagonalOperator(d, in_structure=tree.as_structure(b))
        >>> result = ecg(A, b, enlargement=2)
        >>> bool(jnp.allclose(result.solution, b / d))
        True
    """
    if isinstance(enlargement, bool) or not isinstance(enlargement, int) or enlargement < 1:
        raise ValueError('enlargement must be a positive integer')
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError('max_steps must be a nonnegative integer')
    if not isinstance(stabilise_every, int) or stabilise_every < 0:
        raise ValueError('stabilise_every must be a nonnegative integer')
    if atol < 0 or rtol < 0:
        raise ValueError('atol and rtol must be nonnegative')
    if loop_kind not in ('lax', 'bounded', 'checkpointed'):
        raise ValueError('loop_kind must be lax, bounded, or checkpointed')
    leaves = jax.tree.leaves(b)
    if not leaves or not sum(v.size for v in leaves):
        raise ValueError('b must contain at least one scalar coordinate')
    if any(not jnp.issubdtype(v.dtype, jnp.floating) for v in leaves):
        raise ValueError('b must contain real floating-point arrays')
    if not _same_structure(b, A.in_structure) or not _same_structure(b, A.out_structure):
        raise ValueError('A must map the structure of b to itself')
    if x0 is None:
        x0 = tree.zeros_like(b)
    elif not _same_structure(x0, b):
        raise ValueError('x0 must match the structure, shapes, and dtypes of b')
    labels = _partition_labels(b, partition, enlargement)
    if max_steps == 0:
        return ECGResult(x0, jnp.zeros(0, dtype=tree.norm(b).dtype), jnp.int32(0))
    if enlargement == 1:
        result = cg(
            A,
            b,
            x0,
            max_steps=max_steps,
            atol=atol,
            rtol=rtol,
            stabilise_every=stabilise_every,
            negative_curvature='error',
            loop_kind=loop_kind,
            iteration_callback=iteration_callback,
        )
        return ECGResult(*result)

    rank_rtol = 100 * max(float(jnp.finfo(v.dtype).eps) for v in leaves)
    tolerance = atol + rtol * tree.norm(b)
    has_tolerance = atol > 0 or rtol > 0
    r = tree.sub(b, A(x0))
    initial_norm = tree.norm(r)
    floor = rank_rtol * initial_norm

    def converged(norm: Array) -> Array:
        return has_tolerance & (norm <= tolerance)

    def split(r: PyTree) -> PyTree:
        return jax.vmap(
            lambda i: jax.tree.map(lambda v, label: jnp.where(label == i, v, 0), r, labels)
        )(jnp.arange(enlargement))

    zero_block = tree.zeros_like(split(r))

    def restart(r: PyTree) -> tuple[PyTree, PyTree, Array]:
        vectors = split(r)
        return _orthonormalize(
            vectors, jax.vmap(A)(vectors), jax.vmap(tree.norm)(vectors), rank_rtol
        )

    def advance(c: _ECGCarry) -> tuple[PyTree, PyTree, Array]:
        vectors, images = c.ap, jax.vmap(A)(c.ap)
        reference_norms = jax.vmap(tree.norm)(vectors)
        p = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), c.previous, c.p)
        ap = jax.tree.map(lambda a, b: jnp.concatenate((a, b)), c.a_previous, c.ap)
        for _ in range(2):
            coefficients = _gram(ap, vectors)
            vectors = tree.sub(vectors, _mix(p, coefficients))
            images = tree.sub(images, _mix(ap, coefficients))
        return _orthonormalize(vectors, images, reference_norms, rank_rtol)

    p, ap, active = restart(r)
    p = eqx.error_if(p, ~jnp.any(active) & (initial_norm > 0), 'ecg: no positive search direction')
    history = jnp.zeros(max_steps, dtype=initial_norm.dtype).at[0].set(initial_norm)

    def body(c: _ECGCarry) -> _ECGCarry:
        alpha = jax.vmap(tree.dot, in_axes=(0, None))(c.p, c.r)
        alpha = jnp.where(c.inert, 0, alpha)
        x = tree.add(c.x, _combine(c.p, alpha))
        r = tree.sub(c.r, _combine(c.ap, alpha))
        norm = tree.norm(r)
        periodic = (c.step + 1) % stabilise_every == 0 if stabilise_every else jnp.array(False)
        verify = periodic | converged(norm) | (norm <= floor)
        r = jax.lax.cond(verify, lambda: tree.sub(b, A(x)), lambda: r)
        norm = tree.norm(r)
        done = converged(norm)
        inert = c.inert | ((not has_tolerance) & (norm <= floor))

        def next_directions() -> tuple[PyTree, PyTree, PyTree, PyTree, PyTree]:
            p, ap, active = jax.lax.cond(verify, lambda: restart(r), lambda: advance(c))
            exhausted = ~jnp.any(active)
            # Finite precision can exhaust the block recurrence before the true
            # residual vanishes. Rebuild from that residual before declaring failure.
            true_r = jax.lax.cond(exhausted, lambda: tree.sub(b, A(x)), lambda: r)
            p, ap, active = jax.lax.cond(
                exhausted, lambda: restart(true_r), lambda: (p, ap, active)
            )
            true_norm = tree.norm(true_r)
            finished = converged(true_norm) | ((not has_tolerance) & (true_norm <= floor))
            p = eqx.error_if(
                p, ~jnp.any(active) & ~finished, 'ecg: search space exhausted before convergence'
            )
            previous = jax.tree.map(lambda v: jnp.where(verify | exhausted, 0, v), c.p)
            a_previous = jax.tree.map(lambda v: jnp.where(verify | exhausted, 0, v), c.ap)
            return p, ap, previous, a_previous, true_r

        p, ap, previous, a_previous, r = jax.lax.cond(
            ~done & ~inert & (c.step + 1 < max_steps),
            next_directions,
            lambda: (zero_block, zero_block, zero_block, zero_block, r),
        )
        norm = tree.norm(r)
        done = converged(norm)
        inert = inert | ((not has_tolerance) & (norm <= floor))
        if iteration_callback is not None:
            jax.debug.callback(iteration_callback, c.step, norm)
        return _ECGCarry(
            x,
            r,
            p,
            ap,
            previous,
            a_previous,
            c.step + 1,
            done,
            inert,
            c.residuals.at[c.step + 1].set(norm),
        )

    out = eqxi.while_loop(
        lambda c: ~c.converged & (c.step < max_steps),
        body,
        _ECGCarry(
            x0,
            r,
            p,
            ap,
            zero_block,
            zero_block,
            jnp.int32(0),
            converged(initial_norm),
            initial_norm == 0,
            history,
        ),
        max_steps=max_steps,
        buffers=lambda c: c.residuals,
        kind=loop_kind,
    )
    return ECGResult(out.x, out.residuals, out.step)
