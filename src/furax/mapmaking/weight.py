import math
from dataclasses import field, replace
from typing import Any, Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, HomothetyOperator, MaskOperator, symmetric, tree
from furax.core import DiagonalOperator, IndexOperator
from furax.linalg import cg
from furax.math.bspline import cardinal_bspline

from .config import NestedConfig


@symmetric
class WeightOperator(AbstractLinearOperator):
    """Masked noise weight `M W M` as a single operator."""

    weight: AbstractLinearOperator  # symmetric
    mask: MaskOperator

    @classmethod
    def create(cls, weight: AbstractLinearOperator, mask: MaskOperator) -> Self:
        return cls(weight, mask, in_structure=mask.in_structure)

    def with_mask(self, mask: MaskOperator) -> Self:
        """Rebuild the weight around a new mask."""
        return type(self).create(self.weight, mask)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        W, M = self.weight, self.mask
        if W.is_diagonal:
            return W(M(x))
        return M(W(M(x)))


@symmetric
class MarginalizedWeightOperator(AbstractLinearOperator):
    r"""A weight with the amplitudes of some templates marginalized.

    Marginalizing the amplitudes of templates $T$ turns the weight $W$ into

    $$ W' = W - W T G^{-1} T^\top W, $$

    with $G^{-1}$ the inverse of their Gram $T^\top W T$, possibly regularized. When $G^{-1}$ is
    exact, $W'$ is blind to the templates: $W' T = 0$.
    """

    weight: AbstractLinearOperator
    """The weight $W$."""
    templates: AbstractLinearOperator
    """The templates $T$ whose amplitudes are marginalized."""
    gram_inverse: AbstractLinearOperator
    """The inverse $G^{-1}$ of their Gram under $W$."""

    @classmethod
    def create(
        cls,
        weight: AbstractLinearOperator,
        templates: AbstractLinearOperator,
        gram_inverse: AbstractLinearOperator,
    ) -> Self:
        return cls(weight, templates, gram_inverse, in_structure=weight.in_structure)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        W, T = self.weight, self.templates
        wx = W(x)
        return tree.sub(wx, W(T(self.gram_inverse(T.T(wx)))))


@symmetric
class NestedWeightOperator(AbstractLinearOperator):
    r"""Inverse-noise weighting for gappy data that models the flagged samples.

    The minimum-variance weight for gappy data is the inverse of the *good-good block* of the noise
    covariance, `W_exact = Pᵀ N_gg⁻¹ P` (with `P` selecting the good samples). Forming `N_gg⁻¹`
    directly would need the covariance `N`; instead, block-inverse algebra moves the whole correction
    onto the *flagged* samples, expressing the weight through `N⁻¹` alone:

        W = M [N⁻¹ − N⁻¹ C (Cᵀ N⁻¹ C)⁻¹ Cᵀ N⁻¹] M,

    where the columns of `C` span the values the flagged samples may take. With one column per
    flagged sample (`resolution = 1`), `W` is `W_exact`. With a coarser `resolution`, each gap is
    instead modelled by a B-spline across it, whose amplitudes are marginalized: the flagged samples
    carry no data, so `W` stays unbiased, with a much smaller inner system. Only `N⁻¹` appears, so
    this is valid for any symmetric PSD inverse-noise -- including det-det correlations, where
    forming `N = (N⁻¹)⁻¹` is intractable.

    **Inner solve.** With the per-channel zero-lag weight `white`, `(Cᵀ N⁻¹ C)⁻¹` is applied by a
    fixed polynomial in it, preconditioned by the inverse of `Cᵀ D C` under that diagonal weight
    `D`: `inner_steps` steps of Richardson iteration from `y = P b`. Its coefficients do not depend
    on the vector it acts on, so `W` is linear and symmetric. Without `white`, a CG solve of
    `inner_steps` steps is used instead, which is linear only once converged.

    **Fixed budgets.** Under jit the flagged subspace must have a static size, but the flag count is
    a runtime value: the flagged samples, gaps and amplitudes are padded to fixed budgets, and the
    padding acts as the identity in the inner system. When an observation exceeds a budget the
    weight falls back (per ``jax.lax.cond``) to the inner-mask weight `M N⁻¹ M` -- still unbiased,
    just suboptimal. A fully-masked input falls back to `M N⁻¹ M = 0`. Assumes a single-leaf
    time-ordered structure (the SO TOD layout).

    **Optional covariance preconditioner.** Passing the covariance `N` (``cov``) preconditions the
    CG solve with the flagged block `Cᵀ N C`, which helps only for a few gaps much wider than the
    correlation length.
    """

    ninv: AbstractLinearOperator  # N⁻¹, symmetric PSD
    mask: MaskOperator  # M (defines the flagged set and the .mask contract)
    gaps: AbstractLinearOperator  # Cᵀ, from the TOD to the gap amplitudes
    n_active: Array  # number of gap amplitudes in use
    within_budget: Array
    preconditioner: AbstractLinearOperator | None  # P ≈ (Cᵀ N⁻¹ C)⁻¹
    max_flag_fraction: float = field(metadata={'static': True})
    inner_steps: int = field(metadata={'static': True})
    rtol: float = field(metadata={'static': True})
    atol: float = field(metadata={'static': True})
    resolution: int = field(metadata={'static': True})
    max_gaps_per_channel: int = field(metadata={'static': True})
    cov: AbstractLinearOperator | None = None
    white: Array | None = None  # zero-lag of N⁻¹ per channel

    @classmethod
    def create(
        cls,
        ninv: AbstractLinearOperator,
        mask: MaskOperator,
        config: NestedConfig,
        cov: AbstractLinearOperator | None = None,
        white: Array | None = None,
    ) -> Self:
        """Build the weight.

        Args:
            ninv: The inverse noise covariance $N^{-1}$.
            mask: The mask of the valid samples.
            config: The inner-solve options.
            cov: The noise covariance $N$, to precondition a CG inner solve.
            white: The zero-lag of $N^{-1}$ of each channel, flattened: the inner solve is then a
                linear polynomial preconditioned under this diagonal weight.
        """
        gaps, n_active, within, preconditioner = _gap_model(
            mask, config.resolution, config.max_flag_fraction, config.max_gaps_per_channel, white
        )
        return cls(
            ninv,
            mask,
            gaps,
            n_active,
            within,
            preconditioner,
            max_flag_fraction=config.max_flag_fraction,
            inner_steps=config.inner_steps,
            rtol=config.rtol,
            atol=config.atol,
            resolution=config.resolution,
            max_gaps_per_channel=config.max_gaps_per_channel,
            cov=cov,
            white=white,
            in_structure=mask.in_structure,
        )._damped()

    def with_mask(self, mask: MaskOperator) -> Self:
        """Rebuild the weight around a new mask."""
        gaps, n_active, within, preconditioner = _gap_model(
            mask, self.resolution, self.max_flag_fraction, self.max_gaps_per_channel, self.white
        )
        return replace(
            self,
            mask=mask,
            gaps=gaps,
            n_active=n_active,
            within_budget=within,
            preconditioner=preconditioner,
            in_structure=mask.in_structure,
        )._damped()

    @property
    def n_flag_max(self) -> int:
        """The budget of flagged samples."""
        return _resolve_n_flag_max(self.mask, self.max_flag_fraction)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        # Over-budget inputs (including fully-masked ones) fall back to the inner-mask
        # weight; both branches are traced, one is selected at runtime.
        return jax.lax.cond(self.within_budget, self._woodbury, self._inner_mask, x)

    def _inner_mask(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        """Fallback weight `M N⁻¹ M` for an over-budget (or fully-masked) input."""
        return (self.mask @ self.ninv @ self.mask).reduce()(x)

    def _damped(self) -> Self:
        """The weight with its inner preconditioner scaled to bring the spectrum of `P A` in (0, 1].

        Richardson iteration then converges, and its polynomial approximates `A⁻¹` from below,
        which keeps `W` positive semidefinite for any number of steps. The largest eigenvalue of
        `P A` is estimated by power iteration, with a margin for its underestimate.
        """
        if self.preconditioner is None or self.cov is not None:
            return self
        inner, _, _ = self._inner()
        P = self.preconditioner
        z = jax.random.normal(jax.random.key(0), P.in_structure.shape, P.in_structure.dtype)
        for _ in range(_POWER_STEPS):
            z = P(inner(z))
            z = z / jnp.linalg.norm(z)
        largest = jnp.linalg.norm(P(inner(z)))
        scale = 1 / (_POWER_MARGIN * jnp.maximum(largest, 1e-30))
        damped = (HomothetyOperator(scale, in_structure=P.in_structure) @ P).reduce()
        return replace(self, preconditioner=damped)

    def _inner(self) -> tuple[AbstractLinearOperator, DiagonalOperator, DiagonalOperator]:
        """The inner operator `V (Cᵀ N⁻¹ C) V + (I − V)`, and `V`, `I − V`.

        `V` marks the gap amplitudes in use; the identity on padding solves to 0.
        """
        q = self.gaps
        kstruct = q.out_structure
        v = (jnp.arange(kstruct.shape[0]) < self.n_active).astype(kstruct.dtype)
        vop = DiagonalOperator(v, in_structure=kstruct)
        vcomp = DiagonalOperator(1 - v, in_structure=kstruct)  # I − V
        return vop @ q @ self.ninv @ q.T @ vop + vcomp, vop, vcomp

    def _woodbury(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        ninv, q = self.ninv, self.gaps
        inner, vop, vcomp = self._inner()

        x = self.mask(x)  # the flagged samples carry no data
        nx = ninv(x)
        rhs = vop(q(nx))
        if self.preconditioner is not None and self.cov is None:
            y = _richardson(inner, self.preconditioner, rhs, self.inner_steps)
        else:
            preconditioner = None
            if self.cov is not None:
                preconditioner = (vop @ q @ self.cov @ q.T @ vop + vcomp).reduce()
            y = cg(
                inner,
                rhs,
                preconditioner=preconditioner,
                max_steps=self.inner_steps,
                rtol=self.rtol,
                atol=self.atol,
            ).solution
        correction = ninv(q.T(vop(y)))
        return self.mask(tree.sub(nx, correction))


_POWER_STEPS = 10  # power iterations estimating the largest eigenvalue of P A
_POWER_MARGIN = 1.2  # margin over that estimate, which power iteration only approaches from below


def _richardson(
    operator: AbstractLinearOperator,
    preconditioner: AbstractLinearOperator,
    b: Array,
    steps: int,
) -> Array:
    """`steps` preconditioned Richardson steps for `A y = b`, starting from `P b`.

    The result is a fixed polynomial in `P A` applied to `P b`: linear in `b`, and symmetric when
    `A` and `P` are.
    """
    y = preconditioner(b)
    for _ in range(steps):
        y = y + preconditioner(b - operator(y))
    return y


def _gap_model(
    mask: MaskOperator,
    resolution: int,
    max_flag_fraction: float,
    max_gaps_per_channel: int,
    white: Array | None,
) -> tuple[AbstractLinearOperator, Array, Array, AbstractLinearOperator | None]:
    """The gap amplitudes `Cᵀ`, their count, whether they fit the budgets, and the preconditioner."""
    structure = mask.in_structure
    leaf = jax.tree.leaves(structure)[0]
    bad = jax.tree.leaves(mask.to_boolean_mask())[0] == 0
    n_flag = bad.sum()
    n_flag_max = _resolve_n_flag_max(mask, max_flag_fraction)
    if resolution == 1:
        # one amplitude per flagged sample; padding slots point at index 0
        indices = jnp.nonzero(bad, size=n_flag_max, fill_value=0)
        gaps: AbstractLinearOperator = IndexOperator(indices, in_structure=structure)
        preconditioner = None
        if white is not None:
            channel = jnp.ravel_multi_index(indices[:-1], leaf.shape[:-1], mode='clip')
            kstruct = jax.ShapeDtypeStruct((n_flag_max,), leaf.dtype)
            preconditioner = DiagonalOperator(1 / white[channel], in_structure=kstruct)
        return gaps, n_flag, n_flag <= n_flag_max, preconditioner
    basis = _GapSplineBasis.create(bad, resolution, n_flag_max, max_gaps_per_channel, structure)
    preconditioner = None if white is None else _GapBlockInverse.create(basis, white)
    return basis.T, basis.n_coefficients, basis.within_budget(n_flag), preconditioner


class _GapSplineBasis(AbstractLinearOperator):
    r"""A B-spline across each gap of each channel, from its amplitudes to the TOD.

    A gap of $L$ flagged samples is cut into $n = \lceil L / r \rceil$ intervals of about
    `resolution` $= r$ samples, and gets the $n + p$ uniform B-splines of degree
    $p = \min(3, n - 1)$ over them: a gap no longer than $r$ is a constant, a long one a cubic
    spline. A gap with no more samples than that keeps one amplitude per sample instead. The
    amplitudes of all the gaps are packed in a fixed-size vector; those past `n_coefficients` are
    padding.
    """

    flagged: Array  # (n_flag_max,) flat indices of the flagged samples, padded with 0
    first: Array  # (n_flag_max,) amplitude index of the first B-spline under each sample
    weights: Array  # (n_flag_max, 4) the B-spline values under each sample, 0 on padding
    gap: Array  # (n_flag_max,) gap of each sample
    local: Array  # (n_flag_max,) index, within its gap, of the first B-spline under each sample
    base: Array  # (n_gap_max,) index of the first amplitude of each gap
    functions: Array  # (n_gap_max,) number of amplitudes of each gap
    n_coefficients: Array
    n_gaps: Array
    n_gap_max: int = field(metadata={'static': True})
    tod_leaf: jax.ShapeDtypeStruct = field(metadata={'static': True})
    tod_treedef: Any = field(metadata={'static': True})

    @classmethod
    def create(
        cls,
        bad: Array,
        resolution: int,
        n_flag_max: int,
        max_gaps_per_channel: int,
        tod_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> Self:
        n_samples = bad.shape[-1]
        n_gap_max = bad.size // n_samples * max_gaps_per_channel
        n_coefficient_max = -(-n_flag_max // resolution) + 4 * n_gap_max

        flat = bad.ravel()
        flagged = jnp.nonzero(flat, size=n_flag_max, fill_value=0)[0]
        valid = jnp.arange(n_flag_max) < flat.sum()
        # a gap starts where the previous flagged sample is not the neighbour in the same channel
        previous = jnp.concatenate([jnp.array([-2]), flagged[:-1]])
        starts = valid & ((flagged != previous + 1) | (flagged % n_samples == 0))
        gap = jnp.where(valid, jnp.cumsum(starts) - 1, n_gap_max)  # padding past the last gap
        length = jax.ops.segment_sum(valid.astype(jnp.int32), gap, num_segments=n_gap_max)
        start = jax.ops.segment_min(
            jnp.where(valid, flagged, flat.size), gap, num_segments=n_gap_max
        )
        intervals = jnp.maximum(1, -(-length // resolution))
        degree = jnp.minimum(3, intervals - 1)
        per_sample = length <= intervals + degree
        functions = jnp.where(per_sample, length, intervals + degree)
        base = jnp.cumsum(functions) - functions

        g = jnp.clip(gap, 0, n_gap_max - 1)
        position = flagged - start[g]
        # position on the knot grid of the gap, from 0 at its first sample to `intervals` at its last
        u = position / jnp.maximum(length[g] - 1, 1) * intervals[g]
        offset = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, intervals[g] - 1)
        o = jnp.arange(4)
        # B-spline j of degree p is cardinal_bspline(u - j + p), nonzero for j = offset, ..., + p
        p = degree[g][:, None]
        values = cardinal_bspline(u[:, None] - (offset[:, None] + o) + p, p)
        # beyond its degree a window slot is not a B-spline of the gap; a constant takes every sample
        values = jnp.where(o <= p, values, 0)
        values = jnp.where(p == 0, (o == 0).astype(values.dtype), values)
        sample = per_sample[g]
        values = jnp.where(sample[:, None], (o == 0).astype(values.dtype), values)
        local = jnp.where(sample, position, offset)
        (leaf,), treedef = jax.tree.flatten(tod_structure)
        return cls(
            flagged,
            jnp.where(valid, base[g] + local, 0),
            jnp.where(valid[:, None], values, 0).astype(leaf.dtype),
            g,
            local,
            base,
            functions,
            functions.sum(),
            starts.sum(),
            n_gap_max=n_gap_max,
            tod_leaf=leaf,
            tod_treedef=treedef,
            in_structure=jax.ShapeDtypeStruct((n_coefficient_max,), leaf.dtype),
        )

    def within_budget(self, n_flag: Array) -> Array:
        """Whether the flagged samples, their gaps and their amplitudes fit the budgets."""
        return (
            (n_flag <= self.flagged.size)
            & (self.n_gaps <= self.n_gap_max)
            & (self.n_coefficients <= self.in_structure.shape[0])
        )

    def mv(self, x: Array) -> PyTree[Array]:
        knots = jnp.clip(self.first[:, None] + jnp.arange(4), 0, x.size - 1)
        values = jnp.sum(self.weights * x[knots], axis=1)
        tod = jnp.zeros(math.prod(self.tod_leaf.shape), x.dtype).at[self.flagged].add(values)
        return jax.tree.unflatten(self.tod_treedef, [tod.reshape(self.tod_leaf.shape)])


_CHUNK = 4  # the preconditioner inverts the Gram of each run of 4 amplitudes of a gap


@symmetric
class _GapBlockInverse(AbstractLinearOperator):
    """Block-Jacobi inverse of a gap basis's Gram `Bᵀ D B` under a per-channel diagonal weight.

    The amplitudes of each gap are split in consecutive chunks of 4, and the Gram restricted to
    each chunk is inverted: exactly the Gram of a gap with at most 4 amplitudes.
    """

    block_inverse: Array  # (n_chunk_max, 4, 4)
    chunk: Array  # (n_coefficient_max,) chunk of each amplitude
    slot: Array  # (n_coefficient_max,) position of each amplitude in its chunk
    active: Array  # (n_coefficient_max,)

    @classmethod
    def create(cls, basis: _GapSplineBasis, white: Array) -> Self:
        chunks = -(-basis.functions // _CHUNK)
        chunk_base = jnp.cumsum(chunks) - chunks
        n_coefficient_max = basis.in_structure.shape[0]
        n_chunk_max = basis.n_gap_max + n_coefficient_max // _CHUNK

        weight = white[basis.flagged // basis.tod_leaf.shape[-1]]
        local = basis.local[:, None] + jnp.arange(4)  # (flag, 4) local index of each B-spline
        chunk = chunk_base[basis.gap][:, None] + local // _CHUNK
        slot = local % _CHUNK
        same = chunk[:, :, None] == chunk[:, None, :]
        products = weight[:, None, None] * basis.weights[:, :, None] * basis.weights[:, None, :]
        index = (chunk[:, :, None] * _CHUNK + slot[:, :, None]) * _CHUNK + slot[:, None, :]
        blocks = jnp.zeros(n_chunk_max * _CHUNK * _CHUNK, products.dtype)
        blocks = blocks.at[index.ravel()].add(jnp.where(same, products, 0).ravel(), mode='drop')
        blocks = blocks.reshape(n_chunk_max, _CHUNK, _CHUNK)
        # unused slots, of padding chunks or of a gap's last partial chunk, get the identity
        diagonal = jnp.diagonal(blocks, axis1=1, axis2=2)
        blocks = blocks + jax.vmap(jnp.diag)((diagonal == 0).astype(blocks.dtype))

        c = jnp.arange(n_coefficient_max)
        gap = jnp.clip(jnp.searchsorted(basis.base, c, side='right') - 1, 0, basis.n_gap_max - 1)
        within = c - basis.base[gap]
        return cls(
            jnp.linalg.inv(blocks),
            jnp.clip(chunk_base[gap] + within // _CHUNK, 0, n_chunk_max - 1),
            within % _CHUNK,
            c < basis.n_coefficients,
            in_structure=basis.in_structure,
        )

    def mv(self, x: Array) -> Array:
        blocks = jnp.zeros(self.block_inverse.shape[:2], x.dtype)
        blocks = blocks.at[self.chunk, self.slot].add(jnp.where(self.active, x, 0))
        y = jnp.einsum('gij,gj->gi', self.block_inverse, blocks)
        return jnp.where(self.active, y[self.chunk, self.slot], x)


def _resolve_n_flag_max(mask: MaskOperator, max_flag_fraction: float) -> int:
    """Resolve the static flagged-subspace size from a mask."""
    n_total: int = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(mask.in_structure))
    return math.ceil(max_flag_fraction * n_total)
