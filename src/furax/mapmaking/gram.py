"""Per-detector Gram inversion for template deprojection.

Deprojection: implicit vs explicit families
-------------------------------------------
A template family ``T`` (amplitudes → TOD) can be removed from the GLS problem two ways.
The *explicit* path keeps its amplitudes as unknowns and returns them. The *implicit* path
marginalises them away by replacing the noise weight ``W`` with the template-marginalised
weight

    W' = W − W T (Tᵀ W T)⁻¹ Tᵀ W ,

the ``W``-metric projector off ``range(T)`` (so ``W' T = 0``). This is *exactly*
equivalent to jointly solving for the amplitudes and discarding them — the map and any
retained (explicit) amplitudes are identical — but the implicit degrees of freedom never
enter the joint conjugate-gradient solve. Both paths can be active at once.

The per-detector inverse Gram ``(Tᵀ W T)⁻¹`` is assembled structurally from each basis
(``Basis.weighted_gram``), not by probing every amplitude column.

Restrictions (kept deliberately simple)
---------------------------------------
- ``W`` must be *diagonal* (white / ATOP weighting). The Gram ``G = Tᵀ W T`` is then
  block-diagonal over detectors — each detector fits its own amplitudes. Correlated
  (Toeplitz) ``W`` is not handled here.
- Tuned for shared-basis families ([`TemplateFamily.shared`][]). Several
  families are combined into one ``T`` (e.g. a ``TemplateOperator``); cross-family coupling
  within a detector is captured jointly, so the families need not be mutually orthogonal.

Design note: Gaussian amplitude priors (not implemented)
--------------------------------------------------------
A Gaussian prior ``a ~ N(0, Σ_a)`` on the amplitudes would generalise the weight to

    W' = W − W T (Tᵀ W T + Σ_a⁻¹)⁻¹ Tᵀ W = (N + T Σ_a Tᵀ)⁻¹   (Woodbury, W = N⁻¹),

i.e. Wiener-filter the amplitudes instead of fitting them exactly; the current
deprojection is the improper (flat-prior) limit. Since ``W' T = W T M⁻¹ Σ_a⁻¹`` with
``M = Tᵀ W T + Σ_a⁻¹``, modes in ``ker(Σ_a⁻¹)`` stay *exactly* annihilated even when
coupled to proper ones. Any prior that is block-diagonal per detector (diagonal or dense
``K×K``) slots straight into the block inversion here. Cross-detector priors (e.g. HWPSS
is strongly correlated across detectors) break the per-detector blocks; tractable routes:

- low-rank detector correlation (common mode): Woodbury correction on top of the
  per-detector inverses — one small ``rK×rK`` dense solve;
- reparametrise the correlated part as a *shared* template family (one amplitude set
  expanded to all detectors) plus per-detector residuals: Gram becomes arrow-shaped,
  solved by Schur complement on the small shared block;
- otherwise keep the family explicit and add ``Σ_a⁻¹`` to the amplitude block of the
  joint CG system (one prior apply per iteration, no dense inverse).

The ``regularization`` ridge is itself the MAP limit of an isotropic prior with
data-adaptive precision ``λ·mean(diag G)``; it is kept as a numerical safeguard, not a
statistical statement.
"""

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, PyTree

import furax.tree
from furax import AbstractLinearOperator, IdentityOperator
from furax.core import BlockDiagonalOperator
from furax.linalg import BandedCholeskyOperator

from .templates import Basis, TemplateOperator

__all__ = [
    'gram_inverse',
    'pairwise_gram',
]


class _NoStructuredGram(Exception):
    """Internal control-flow signal: no structured Gram assembly for this operator/basis."""


def pairwise_gram(
    a: Basis, b: Basis, weights: Float[Array, ' samp']
) -> Float[Array, 'a_size b_size']:
    """Cross weighted Gram ``B_aᵀ diag(weights) B_b`` as a dense ``(n_a·k_a, n_b·k_b)`` block.

    Built structurally from each basis's column supports ([`Basis.support`][]), not by probing.
    ``a is b`` recovers the self-Gram; ``a is not b`` is the family-coupling block. Dense; for a
    single family the structured [`Basis.weighted_gram`][] is cheaper.

    Raises:
        _NoStructuredGram: If either basis has no [`Basis.support`][] view.
    """
    try:
        ca, cb = a.support(), b.support()
    except NotImplementedError:
        raise _NoStructuredGram from None
    ka, kb = ca.values.shape[0], cb.values.shape[0]
    if ka <= kb:  # fold the weight into whichever side is smaller, cheaper elementwise multiply
        vwa, vwb = ca.values * weights[None, :], cb.values
    else:
        vwa, vwb = ca.values, cb.values * weights[None, :]
    gram = jnp.zeros((ca.n_blocks, ka, cb.n_blocks, kb), a.dtype)
    for wa in range(ca.blocks.shape[1]):  # window slots (single slot for non-overlapping bases)
        lhs = ca.taps[:, wa][None, :] * vwa  # (k_a, samp)
        for wb in range(cb.blocks.shape[1]):
            rhs = cb.taps[:, wb][None, :] * vwb  # (k_b, samp)
            contrib = jnp.einsum('at,bt->tab', lhs, rhs)  # (samp, k_a, k_b)
            gram = gram.at[ca.blocks[:, wa], :, cb.blocks[:, wb], :].add(contrib)
    return gram.reshape(ca.n_blocks * ka, cb.n_blocks * kb)


def _coupled_gram_inverse(
    template: TemplateOperator, diag: PyTree[Array], regularization: float
) -> AbstractLinearOperator:
    """Dense per-detector inverse Gram for several coupled shared-basis families.

    Built structurally from [`pairwise_gram`][], not the ``O(K)`` dense-probe fallback. No known
    band structure across families to exploit, so the joint ``K×K`` block (spanning every
    family/leg at once) is inverted dense per detector.

    Raises `_NoStructuredGram` if any family is not [`TemplateFamily.shared`][], or, propagated
    from [`pairwise_gram`][], if a basis has no [`Basis.support`][] view.
    """
    if any(not f.shared for f in template.families):
        raise _NoStructuredGram

    if template.stokes is None:
        bases_tree: PyTree[Any] = {f.name: f.items()[0][1] for f in template.families}
        legs_tree: PyTree[Any] = {f.name: '' for f in template.families}
        diag_tree: PyTree[Array] = {f.name: diag for f in template.families}
    else:
        legs_lower = template.stokes.lower()
        bases_tree = {f.name: dict(f.items()) for f in template.families}
        legs_tree = {f.name: {leg: leg for leg, _ in f.items()} for f in template.families}
        diag_tree = {
            f.name: {
                leg: diag.data[legs_lower.index(leg)]
                for leg, _ in f.items()
                if leg is not None  # demod families never carry a bare Basis
            }
            for f in template.families
        }

    is_basis = lambda x: isinstance(x, Basis)  # noqa: E731
    bases: list[Basis] = jax.tree.leaves(bases_tree, is_leaf=is_basis)
    legs: list[str] = jax.tree.leaves(legs_tree)
    diags: list[Array] = jax.tree.leaves(diag_tree)

    sizes = [basis.size for basis in bases]
    offsets = np.cumsum([0, *sizes])
    n = int(offsets[-1])
    dtype = bases[0].dtype

    def build(*per_det_diags: Array) -> Array:
        block = jnp.zeros((n, n), dtype)
        for i, a in enumerate(bases):
            for j, b in enumerate(bases):
                if legs[i] != legs[j]:  # different Stokes legs never share a weighted sample
                    continue
                blk = pairwise_gram(a, b, per_det_diags[i])
                block = block.at[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]].set(blk)
        return block

    block = jax.vmap(build)(*diags)  # (n_dets, n, n)
    return BandedCholeskyOperator.from_dense(block, template.in_structure, regularization)


def _structured_gram_inverse(
    template: TemplateOperator, diag: PyTree[Array], regularization: float
) -> AbstractLinearOperator:
    """Structured inverse Gram for a shared-basis ``TemplateOperator``.

    ``diag`` is the per-sample weight diagonal matching ``template.out_structure`` (the TOD). A
    single family delegates to [`Basis.weighted_gram`][], recursing leg-by-leg for a demodulated
    (Stokes-valued) template; several coupled families go through [`_coupled_gram_inverse`][]
    instead.

    Raises `_NoStructuredGram` for anything unsupported (per-detector basis, non-structured
    basis).
    """
    if len(template.families) > 1:
        return _coupled_gram_inverse(template, diag, regularization)

    family = template.families[0]
    if not family.shared:
        raise _NoStructuredGram
    amp_structure = template.in_structure[family.name]

    def _leg_gram_inverse(basis: Basis, leg_diag: Array, amp: PyTree[Any]) -> Any:
        try:
            bands = jax.vmap(basis.weighted_gram)(leg_diag)
        except NotImplementedError:
            raise _NoStructuredGram from None
        return BandedCholeskyOperator.from_bands(bands, amp, regularization)

    if template.stokes is None:
        leg_diags: PyTree[Array] = diag
    else:
        leg_diags = {leg: getattr(diag, leg) for leg, _ in family.items() if leg is not None}

    sub = jax.tree.map(
        _leg_gram_inverse,
        family.bases,
        leg_diags,
        amp_structure,
        is_leaf=lambda x: isinstance(x, Basis),
    )
    return BlockDiagonalOperator({family.name: sub})


def _probed_gram_inverse(
    operator: AbstractLinearOperator,
    weight: AbstractLinearOperator,
    regularization: float,
    n_diagonal_axes: int = 1,
) -> AbstractLinearOperator:
    """Probe ``G = Aᵀ W A`` to assemble and factor its block-diagonal Gram.

    Assumes A is diagonal in its leaves' first ``n_diagonal_axes`` dimensions (e.g. detectors),
    so G is block-diagonal along them.
    """
    gram = (operator.T @ weight @ operator).reduce()
    in_structure = gram.in_structure
    leaves, treedef = jax.tree.flatten(in_structure)
    leading_shape = leaves[0].shape[:n_diagonal_axes]
    dtype = leaves[0].dtype
    sizes = [prod(s.shape[n_diagonal_axes:]) for s in leaves]
    split_points = np.cumsum(sizes)[:-1]  # interior cut points between leaves
    n_basis = sum(sizes)

    def probe(col: Array) -> Array:
        # one-hot on the concatenated coupled index, split at the (static) leaf sizes and
        # broadcast over the leading (diagonal) axes.
        flat = jnp.zeros((n_basis,), dtype).at[col].set(1.0)
        parts = [
            jnp.broadcast_to(part.reshape(s.shape[n_diagonal_axes:]), s.shape)
            for part, s in zip(jnp.split(flat, split_points), leaves, strict=True)
        ]
        response = gram(treedef.unflatten(parts))  # type: ignore[attr-defined]
        flats = [leaf.reshape(*leading_shape, -1) for leaf in jax.tree.leaves(response)]
        return jnp.concatenate(flats, axis=-1)  # (*n_leading, n_basis) = column `col`

    cols = jax.lax.map(probe, jnp.arange(n_basis))  # (col, *n_leading, row)
    block = jnp.moveaxis(cols, 0, -1)  # (*n_leading, row, col)
    return BandedCholeskyOperator.from_dense(block, in_structure, regularization)


def gram_inverse(
    operator: AbstractLinearOperator,
    weight: AbstractLinearOperator,
    regularization: float = 0.0,
    *,
    allow_dense_probe: bool = False,
) -> AbstractLinearOperator:
    """Per-detector inverse Gram matrix ``(Aᵀ W A)⁻¹`` for operator ``A`` and weight ``W``.

    ``A`` (``operator``) maps amplitudes to TOD, per detector; ``weight`` is assumed diagonal, so
    ``G = Aᵀ W A`` is block-diagonal over detectors. A structural assembly (never the ``O(K)``
    column-probe) is always tried first, but only applies when ``operator`` is a
    [`TemplateOperator`][] over shared-basis families. Otherwise —
    or if that structural path turns out to be unavailable (a per-detector, ``shared=False``
    family, or a basis without ``weighted_gram``/``support``) — raises ``NotImplementedError``
    unless ``allow_dense_probe=True``, in which case falls back to the dense probe: correct for
    any ``operator`` but ``O(K)`` in the amplitude size ``K``, so only ever safe for small-``K``
    explicit families (e.g. T2P, which config already forces ``explicit=True``, i.e. this
    fallback is never reached on the implicit deprojection path).

    Args:
        operator: The linear operator ``A``, e.g., a TemplateOperator instance.
        weight: The (assumed diagonal) noise weight ``W``.
        regularization: Relative ridge added to each detector's Gram block before factoring.
        allow_dense_probe: Allow the ``O(K)`` dense-probe fallback when the structured path is
            unavailable. Defaults to ``False`` (raise instead) — flip only for small-``K``
            families known to have no structural Gram view.

    Returns:
        The per-detector inverse Gram operator.
    """
    if operator.in_size == 0:
        return IdentityOperator(in_structure=operator.in_structure)

    if isinstance(operator, TemplateOperator):
        try:
            ones = furax.tree.ones_like(weight.in_structure)
            diag = weight(ones)
            return _structured_gram_inverse(operator, diag, regularization)
        except _NoStructuredGram:
            pass  # fall back to dense probe, or raise below
    if allow_dense_probe:
        return _probed_gram_inverse(operator, weight, regularization)
    msg = f'structured Gram construction not possible for {operator}, pass `allow_dense_probe=True`'
    raise NotImplementedError(msg)
