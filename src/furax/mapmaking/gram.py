r"""Gram matrix construction and inversion for template deprojection.

Implicit deprojection replaces the weight matrix $W$ with the template-marginalised weight

$$ W' = W - W T (T^\top W T)^{-1} T^\top W, $$

the $W$-metric projector off $\mathrm{range}(T)$.

This module assembles the Gram matrix $G \equiv T^\top W T$ from the bases of an
[`AbstractTemplateOperator`][]. Inversion is performed via [`furax.linalg.cholesky`][].

Limitations:

- $W$ must be *diagonal*. Correlated (Toeplitz) weights are not supported. Interaction with ATOP
  deprojection is not handled (also results in a non-diagonal effective weight).
- When assembling the Gram, basis structure (column support) is only exploited if all bases of the
  template operator are shared over detectors.
- With several shared bases, the per-detector Gram blocks are stored as dense objects, even where
  the cross blocks are sparse (e.g. from the product of two time-local bases).

A Gaussian prior $a \sim \mathcal{N}(0, \Sigma_a)$ on the amplitudes would generalise implicit
deprojection to Wiener filtering:

$$ W' = W - W T (T^\top W T + \Sigma_a^{-1})^{-1} T^\top W = (N + T \Sigma_a T^\top)^{-1}, $$

with $W = N^{-1}$ (Woodbury). The deprojection above is its improper, flat-prior limit.

The only prior this module currently offers is the isotropic case, via `regularization`: a ridge
$\lambda \cdot \mathrm{mean}(\mathrm{diag}\, G)$ added to each block before factoring, meant as a
numerical safeguard rather than a statistical choice.
"""

from math import prod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, PyTree

import furax.tree
from furax import AbstractLinearOperator
from furax.core import BlockDiagonalOperator
from furax.linalg import BandedCholeskyOperator

from .templates import AbstractTemplateOperator, Basis, NoStructuredView, is_basis

__all__ = [
    'cross_gram',
    'gram_inverse',
]


def gram_inverse(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    regularization: float = 0.0,
    *,
    allow_probe: bool = False,
    batch_size: int = 32,
) -> AbstractLinearOperator:
    """Inverse Gram matrix `(Tᵀ W T)⁻¹` for template operator `T` and weight `W`.

    `T` maps amplitudes to TOD, per detector. `W` is *assumed* diagonal, in both detector and
    sample space, and its diagonal is read off by applying it to a vector of ones: a weight that
    does not respect that contract silently changes the result. `G = Tᵀ W T` thus does not couple
    different detectors: each gets its own block, making `G` block-diagonal.

    When every basis is shared across detectors and exposes a structured (column-support) view,
    that structure is used to assemble the Gram matrix efficiently.

    When any basis is a per-detector stack or has no structured form, the fallback is a column
    probe: correct for any `T`, but `O(K)` in the amplitude count `K`. This requires `allow_probe`
    to be `True`.

    Args:
        operator: The template operator `T`.
        weight: The diagonal weights `W`.
        regularization: Relative ridge added to each detector's Gram block before factoring.
        allow_probe: Allow the `O(K)` dense-probe fallback.
        batch_size: Detector batch size to bound transient memory usage in the structured
            per-detector Gram assembly path.

    Returns:
        The per-detector inverse Gram operator.

    Raises:
        NotImplementedError: If the structured path does not apply and `allow_probe` is `False`.
    """
    try:
        ones = furax.tree.ones_like(weight.in_structure)
        diag = weight(ones)
        # if we wanted to guard against a non-diagonal W, one extra application on a random
        # vector `x` and a comparison against `diag * x` would catch it with very high probability
        return _structured_gram_inverse(operator, diag, regularization, batch_size)
    except NoStructuredView:
        pass  # fall back to dense probe, or raise below
    if allow_probe:
        return _probed_gram_inverse(operator, weight, regularization)
    msg = f'structured Gram construction not possible for {operator}, pass `allow_probe=True`'
    raise NotImplementedError(msg)


def cross_gram(a: Basis, b: Basis, weights: Float[Array, ' samp']) -> Float[Array, 'a_size b_size']:
    """The weighted cross Gram `B_aᵀ diag(weights) B_b`, as one `(n_a·k_a, n_b·k_b)` block.

    `a is b` recovers the self-Gram, though [`Basis.gram`][] is cheaper for a single template,
    returning bands instead.

    Raises:
        NoStructuredView: If either basis has no [`Basis.support`][] view.
    """
    ca, cb = a.support(), b.support()
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


def _zero_sub_identity(diagonal_blocks: Float[Array, '*batch k k']) -> Float[Array, '*batch k k']:
    """Substitute the identity for zero Gram blocks, which are singular and factor to NaN."""
    k = diagonal_blocks.shape[-1]
    unconstrained = jnp.all(diagonal_blocks == 0, axis=(-2, -1), keepdims=True)
    return jnp.where(unconstrained, jnp.eye(k, dtype=diagonal_blocks.dtype), diagonal_blocks)


def _structured_gram_inverse(
    template: AbstractTemplateOperator,
    diag: PyTree[Array],
    regularization: float,
    batch_size: int,
) -> AbstractLinearOperator:
    if len(template.bases) > 1:
        return _coupled_gram_inverse(template, diag, regularization, batch_size)

    def leg_inverse(path: Any, basis: Basis, amp: PyTree[Any]) -> Any:
        leg = path[-1].key if len(path) > 1 else None
        leg_diag = diag if leg is None else getattr(diag, leg)
        bands = jax.lax.map(basis.gram, leg_diag, batch_size=batch_size)
        bands = bands.at[..., 0, :, :].set(_zero_sub_identity(bands[..., 0, :, :]))
        return BandedCholeskyOperator.from_bands(bands, amp, regularization)

    # one factored block per template and Stokes leg, keyed as the amplitudes are: legs are
    # independent, and detectors are already the leading axis inside each block
    return BlockDiagonalOperator(
        jax.tree.map_with_path(
            leg_inverse,
            template.bases,
            template.in_structure,
            is_leaf=is_basis,
        )
    )


def _coupled_gram_inverse(
    template: AbstractTemplateOperator,
    diag: PyTree[Array],
    regularization: float,
    batch_size: int,
) -> AbstractLinearOperator:
    # Flattening keeps the keys: `('poly', 'q')` for a Stokes-valued template, `('poly',)` without
    # a Stokes axis. The leg says which stream a basis is weighted by, and two bases on different
    # legs never share a weighted sample.
    entries, _ = jax.tree.flatten_with_path(template.bases, is_leaf=is_basis)
    bases: list[Basis] = [basis for _, basis in entries]
    legs = [path[-1].key if len(path) > 1 else None for path, _ in entries]
    diags = tuple(diag if leg is None else getattr(diag, leg) for leg in legs)

    # each basis owns a contiguous slice of the joint block, laid out in that same order
    sizes = [basis.size for basis in bases]
    offsets = np.cumsum([0, *sizes])
    n_amps = int(offsets[-1])
    dtype = bases[0].dtype

    def build(per_basis_diags: tuple[Array, ...]) -> Array:
        """One detector's joint block, filled in one basis pair at a time."""
        block = jnp.zeros((n_amps, n_amps), dtype)
        for i, a in enumerate(bases):
            rows = slice(offsets[i], offsets[i + 1])
            for j, b in enumerate(bases):
                if legs[i] != legs[j]:  # different Stokes legs never share a weighted sample
                    continue
                cols = slice(offsets[j], offsets[j + 1])
                block = block.at[rows, cols].set(cross_gram(a, b, per_basis_diags[i]))
        return block

    blocks = jax.lax.map(build, diags, batch_size=batch_size)  # (n_dets, n_amps, n_amps)
    blocks = _zero_sub_identity(blocks)
    return BandedCholeskyOperator.from_dense(blocks, template.in_structure, regularization)


def _probed_gram_inverse(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    regularization: float,
) -> AbstractLinearOperator:
    """The fallback: recover `G = Tᵀ W T` by applying it to one amplitude at a time.

    Costs `O(K)` applications for `K` amplitudes, but needs nothing of the bases beyond `T` itself.
    Amplitudes carry detectors on their leading axis and `T` couples none of them, so `G` is
    block-diagonal there and each detector's block is factored on its own.
    """
    gram_op = (operator.T @ weight @ operator).reduce()
    in_structure = gram_op.in_structure
    leaves, treedef = jax.tree.flatten(in_structure)
    n_dets = leaves[0].shape[0]
    dtype = leaves[0].dtype
    # amplitudes of every template, concatenated into one index; each leaf owns a slice of it,
    # the same slice its column of `G` occupies
    sizes = [prod(s.shape[1:]) for s in leaves]
    split_points = np.cumsum(sizes)[:-1]  # interior cut points between leaves
    n_amps = sum(sizes)

    def probe(col: Array) -> Array:
        """Column `col` of `G`, for every detector at once."""
        # one amplitude set to 1, the rest 0, split back into leaves and shared by all detectors:
        # `G` couples none of them, so one application gives every detector's column at once
        flat = jnp.zeros((n_amps,), dtype).at[col].set(1.0)
        parts = [
            jnp.broadcast_to(part.reshape(s.shape[1:]), s.shape)
            for part, s in zip(jnp.split(flat, split_points), leaves, strict=True)
        ]
        response = gram_op(treedef.unflatten(parts))  # type: ignore[attr-defined]
        per_leaf = [leaf.reshape(n_dets, -1) for leaf in jax.tree.leaves(response)]
        return jnp.concatenate(per_leaf, axis=-1)  # (n_dets, n_amps)

    columns = jax.lax.map(probe, jnp.arange(n_amps))  # (col, n_dets, row)
    blocks = jnp.moveaxis(columns, 0, -1)  # (n_dets, row, col)
    blocks = _zero_sub_identity(blocks)
    return BandedCholeskyOperator.from_dense(blocks, in_structure, regularization)
