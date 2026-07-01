import math
from dataclasses import field, replace
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, PyTree

from furax import AbstractLinearOperator, MaskOperator, symmetric, tree
from furax.core import DiagonalOperator, IndexOperator
from furax.linalg import cg

from .config import NestedConfig


@symmetric
class WeightOperator(AbstractLinearOperator):
    """Masked noise weight `M W M` as a single operator."""

    weight: AbstractLinearOperator  # symmetric
    mask: MaskOperator

    @classmethod
    def create(cls, weight: AbstractLinearOperator, mask: MaskOperator) -> Self:
        return cls(weight, mask, in_structure=mask.in_structure)

    def with_mask(self, mask: MaskOperator) -> 'WeightOperator':
        """Rebuild the weight around a new mask."""
        return WeightOperator.create(self.weight, mask)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        W, M = self.weight, self.mask
        if W.is_diagonal:
            return W(M(x))
        return M(W(M(x)))


@symmetric
class NestedWeightOperator(AbstractLinearOperator):
    r"""Minimum-variance inverse-noise weighting for gappy data, using an iterative algorithm.

    The minimum-variance weight for gappy data is the inverse of the *good-good block* of the noise
    covariance, `W_exact = Pᵀ N_gg⁻¹ P` (with `P` selecting the good samples). Forming `N_gg⁻¹`
    directly would need the covariance `N`; instead, block-inverse algebra moves the whole correction
    onto the *flagged* samples, expressing the weight through `N⁻¹` alone:

        W_exact = N⁻¹ − N⁻¹ Qᵀ (Q N⁻¹ Qᵀ)⁻¹ Q N⁻¹,

    where `Q` packs the flagged samples. Sketch: order the samples good/bad and split `N⁻¹` into
    `2×2` blocks; the Schur complement of its good-good block is exactly `N_gg⁻¹`, and rearranging
    that relation gives the line above. The correction lives entirely in the flagged subspace, whose
    block `Q N⁻¹ Qᵀ` is small and well-conditioned; the flagged rows of the output are zeroed
    automatically (no outer mask needed). Only `N⁻¹` appears, so this is valid for any symmetric PSD
    inverse-noise -- including det-det correlations, where forming `N = (N⁻¹)⁻¹` is intractable. The
    inner factor `(Q N⁻¹ Qᵀ)⁻¹` is applied by a CG solve, never materialised.

    **Fixed flagged-subspace budget.** Under jit the flagged subspace must have a static size, but
    the flag count is a runtime value. We therefore pad the flagged-index set to a fixed
    `n_flag_max` (``jnp.nonzero(..., size=n_flag_max)``); a validity vector `v` marks the real
    flagged slots, and the inner system is `V (Q N⁻¹ Qᵀ) V + (I − V)` so padding slots act as the
    identity and contribute nothing. When the flag count exceeds the budget the weight falls back
    (per ``jax.lax.cond``) to the cheap inner-mask weight `M N⁻¹ M` -- still unbiased, just
    suboptimal -- so correctness never depends on the budget. A fully-masked input exceeds the
    budget and falls back to `M N⁻¹ M = 0`, contributing nothing.

    The inner solve runs a fixed number of iterations, so `W` is a constant linear operator --
    identical on the RHS and on every system-operator apply. A tolerance-based inner solve would
    make `W` depend on its input, i.e. no longer linear. Assumes a single-leaf time-ordered
    structure (the SO TOD layout).

    **Optional preconditioner.** The inner block `Q N⁻¹ Qᵀ` is ill-conditioned for correlated noise,
    so passing the covariance `N` (``cov``, the banded/Fourier approximation from the noise model)
    builds the flagged-block preconditioner `Q N Qᵀ`. Its product with the inner operator differs
    from the identity only by a boundary term of rank ≈ 2×(correlation bandwidth)×(number of gap
    edges) -- set by how many separate gaps there are, not their width -- and inner CG converges in
    roughly that many steps: a few for a handful of gaps, more when the flags are fragmented, but
    always far fewer than unpreconditioned. Without ``cov`` the inner solve is unpreconditioned.
    """

    ninv: AbstractLinearOperator  # N⁻¹, symmetric PSD
    mask: MaskOperator  # M (defines the flagged set and the .mask contract)
    max_flag_fraction: float = field(metadata={'static': True})
    n_flag_max: int = field(metadata={'static': True})
    inner_steps: int = field(metadata={'static': True})
    rtol: float = field(metadata={'static': True})
    atol: float = field(metadata={'static': True})
    cov: AbstractLinearOperator | None = None

    @classmethod
    def create(
        cls,
        ninv: AbstractLinearOperator,
        mask: MaskOperator,
        config: NestedConfig,
        cov: AbstractLinearOperator | None = None,
    ) -> Self:
        return cls(
            ninv,
            mask,
            max_flag_fraction=config.max_flag_fraction,
            n_flag_max=_resolve_n_flag_max(mask, config.max_flag_fraction),
            inner_steps=config.inner_steps,
            rtol=config.rtol,
            atol=config.atol,
            cov=cov,
            in_structure=mask.in_structure,
        )

    def with_mask(self, mask: MaskOperator) -> 'NestedWeightOperator':
        """Rebuild the weight around a new mask, re-resolving the budget."""
        new_flag_max = _resolve_n_flag_max(mask, self.max_flag_fraction)
        return replace(self, mask=mask, n_flag_max=new_flag_max)

    def _inner_mask(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        """Fallback weight `M N⁻¹ M` for an over-budget (or fully-masked) input."""
        return (self.mask @ self.ninv @ self.mask).reduce()(x)

    def _woodbury(
        self,
        x: PyTree[Inexact[Array, '...']],
        bad: Array,
        n_flag: Array,
    ) -> PyTree[Inexact[Array, '...']]:
        ninv, k = self.ninv, self.n_flag_max
        dtype = jax.tree.leaves(self.in_structure)[0].dtype

        # Pack the flagged samples into a fixed-size block; padding slots point at index 0 and are
        # deactivated by the validity vector v.
        idx = jnp.nonzero(bad, size=k, fill_value=0)
        q = IndexOperator(idx, in_structure=self.in_structure)
        kstruct = jax.ShapeDtypeStruct((k,), dtype)
        v = (jnp.arange(k) < n_flag).astype(dtype)
        vop = DiagonalOperator(v, in_structure=kstruct)
        vcomp = DiagonalOperator(1 - v, in_structure=kstruct)  # I − V

        # Inner system A_in = V (Q N⁻¹ Qᵀ) V + (I − V): the real flagged block on valid slots,
        # the identity on padding (so padded slots solve to 0 and drop out of the correction).
        a_in = vop @ q @ ninv @ q.T @ vop + vcomp
        # Preconditioner V (Q N Qᵀ) V + (I − V): the flagged block of the covariance N.
        preconditioner = None
        if self.cov is not None:
            preconditioner = (vop @ q @ self.cov @ q.T @ vop + vcomp).reduce()
        nx = ninv(x)
        rhs = vop(q(nx))
        y = cg(
            a_in,
            rhs,
            preconditioner=preconditioner,
            max_steps=self.inner_steps,
            rtol=self.rtol,
            atol=self.atol,
        ).solution
        correction = ninv(q.T(vop(y)))
        return tree.sub(nx, correction)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        bad = jax.tree.leaves(self.mask.to_boolean_mask())[0] == 0
        n_flag = bad.sum()
        # Over-budget inputs (including fully-masked ones) fall back to the inner-mask
        # weight; both branches are traced, one is selected at runtime.
        return jax.lax.cond(
            n_flag <= self.n_flag_max,
            lambda operand: self._woodbury(operand, bad, n_flag),
            self._inner_mask,
            x,
        )


def _resolve_n_flag_max(mask: MaskOperator, max_flag_fraction: float) -> int:
    """Resolve the static flagged-subspace size from a mask."""
    n_total = sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(mask.in_structure))
    return int(math.ceil(max_flag_fraction * n_total))
