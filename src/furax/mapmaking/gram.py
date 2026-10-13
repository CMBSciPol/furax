r"""Gram matrix construction and inversion for template deprojection.

Implicit deprojection replaces the weight matrix $W$ with the template-marginalised weight

$$ W' = W - W T (T^\top W T)^{-1} T^\top W, $$

the $W$-metric projector off $\mathrm{range}(T)$.

This module assembles the Gram matrix $G \equiv T^\top W T$ from the bases of an
[`AbstractTemplateOperator`][]. Inversion is performed via [`furax.linalg.cholesky`][].

Combined with Pomme (`pomme_tau`), the effective weight is $W F$ with $F$ the
[`PommeProjectionOperator`][], and the Gram becomes $\tilde G = T^\top W F T = (FT)^\top W (FT)$:
the Gram of the Pomme-filtered templates. It is assembled without forming $FT$, by
Schur-eliminating the Pomme template $Z$ (the interval indicators) from the joint Gram of $[Z\ T]$:

$$ \tilde G = T^\top W T - C^\top D^{-1} C, \qquad C = Z^\top W T, \quad D = Z^\top W Z, $$

with $D$ diagonal (weighted interval sample counts) and $C$ the weighted per-interval sums of the
basis columns. This equals $T^\top W D_Z T$ for any diagonal $W$, and $T^\top W F T$ when $W$
and $F$ commute, which the whole-interval masking of the Pomme path guarantees. An interval
straddling two blocks of a time-local basis couples them, so the Gram band widens by one block.
$F$ annihilates constant columns, so a template containing one (polynomial order 0, a DC harmonic,
binned azimuth) makes $\tilde G$ singular: a `regularization` ridge is then required.

Limitations:

- $W$ must be *diagonal*. Correlated (Toeplitz) weights are not supported.
- When assembling the Gram, basis structure (column support) is only exploited if all bases of the
  template operator are shared over detectors.
- Several templates on one stream (the TOD, or one Stokes leg) are coupled through the weight.
  The time-local template with the most amplitudes keeps its band structure, and the others
  border it, eliminated by a Schur complement. A second time-local template in the border is
  stored dense, even though its coupling to the banded one is sparse.

A Gaussian prior $a \sim \mathcal{N}(0, \Sigma_a)$ on the amplitudes would generalise implicit
deprojection to Wiener filtering:

$$ W' = W - W T (T^\top W T + \Sigma_a^{-1})^{-1} T^\top W = (N + T \Sigma_a T^\top)^{-1}, $$

with $W = N^{-1}$ (Woodbury). The deprojection above is its improper, flat-prior limit.

The only prior this module currently offers is the isotropic case, via `regularization`: a ridge
$\lambda \cdot \mathrm{mean}(\mathrm{diag}\, G)$ added to each block before factoring, meant as a
numerical safeguard rather than a statistical choice.
"""

from dataclasses import field
from math import prod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PyTree

import furax.tree
from furax import AbstractLinearOperator, DiagonalOperator, IdentityOperator, symmetric
from furax.core import BlockDiagonalOperator
from furax.linalg import BandedCholeskyOperator, BorderedBandedCholeskyOperator

from .pomme import PommeProjectionOperator
from .templates import (
    AbstractTemplateOperator,
    Basis,
    NoStructuredView,
    OffsetBasis,
    StokesTemplateOperator,
    TemplateOperator,
    _BasisColumns,
    is_basis,
)

__all__ = [
    'cross_gram',
    'gram_inverse',
]


def gram_inverse(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    regularization: float = 0.0,
    *,
    pomme_tau: int | None = None,
    allow_probe: bool = False,
    batch_size: int = 8,
) -> AbstractLinearOperator:
    """Inverse Gram matrix `(Tᵀ W T)⁻¹` for template operator `T` and weight `W`.

    `T` maps amplitudes to TOD, per detector. `W` is *assumed* diagonal, in both detector and
    sample space, and its diagonal is read off by applying it to a vector of ones: a weight that
    does not respect that contract silently changes the result. `G = Tᵀ W T` thus does not couple
    different detectors: each gets its own block, making `G` block-diagonal.

    Each stream (the TOD, or each Stokes leg of a demodulated one) is inverted on its own. When
    every basis on a stream is shared across detectors and exposes a structured (column-support)
    view, that structure is used to assemble its Gram efficiently. Otherwise, the fallback is a
    column probe: correct for any `T`, but `O(K)` in the amplitude count `K`. This requires
    `allow_probe` to be `True`.

    With `pomme_tau`, the effective weight is `W F` with `F` the [`PommeProjectionOperator`][] of
    that interval length, and the inverse of `Tᵀ W F T` is returned instead (see the module
    docstring). `W` itself stays the diagonal weight.

    Args:
        operator: The template operator `T`.
        weight: The diagonal weights `W`.
        regularization: Relative ridge added to each detector's Gram block before factoring.
        pomme_tau: Pomme interval length, when the templates are combined with Pomme.
        allow_probe: Allow the `O(K)` dense-probe fallback.
        batch_size: Detector batch size to bound transient memory usage in the structured
            per-detector Gram assembly path.

    Returns:
        The per-detector inverse Gram operator.

    Raises:
        NotImplementedError: If the structured path does not apply and `allow_probe` is `False`.
        ValueError: If `pomme_tau` is given for a Stokes TOD rather than a single stream.
    """
    inverse, _ = _gram_inverse_and_ridge(
        operator,
        weight,
        regularization,
        pomme_tau=pomme_tau,
        allow_probe=allow_probe,
        batch_size=batch_size,
    )
    return inverse


def _gram_inverse_and_ridge(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    regularization: float = 0.0,
    *,
    pomme_tau: int | None = None,
    allow_probe: bool = False,
    batch_size: int = 8,
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """[`gram_inverse`][], paired with the ridge added to the Gram before factoring it.

    The explicit normal system is built from the Gram itself, not from its factor, so it must add
    the same ridge to stay consistent with the preconditioner. Where the Gram is singular, as under
    Pomme with a constant template column, that ridge is also what makes the system well posed.
    `None` when `regularization` is 0.
    """
    is_stokes = isinstance(operator, StokesTemplateOperator)
    # Pomme acts on the modulated TOD, which demodulation splits into one stream per leg.
    if pomme_tau is not None and is_stokes:
        raise ValueError('Pomme filtering applies to a single stream, not a Stokes TOD')

    ones = furax.tree.ones_like(weight.in_structure)
    diag = weight(ones)
    # if we wanted to guard against a non-diagonal W, one extra application on a random
    # vector `x` and a comparison against `diag * x` would catch it with very high probability

    ridges: PyTree[Array]
    if not is_stokes:  # a single stream
        pomme = None if pomme_tau is None else OffsetBasis(pomme_tau, diag.shape[-1], diag.dtype)
        stream = _Stream(operator.bases, diag, operator.in_structure, pomme)
        inverse, ridges = _stream_gram_inverse(stream, regularization, batch_size, allow_probe)
    else:
        # Two bases on different legs never share a weighted sample, so each leg is a stream of
        # its own, with one block over the templates it carries rather than one over every leg.
        blocks = {}
        ridges = {name: {} for name in operator.bases_by_leg}
        for leg in operator.legs:
            bases = {name: on[leg] for name, on in operator.bases_by_leg.items() if leg in on}
            if bases:
                structure = {name: operator.in_structure[name][leg] for name in bases}
                stream = _Stream(bases, getattr(diag, leg), structure)
                blocks[leg], leg_ridges = _stream_gram_inverse(
                    stream, regularization, batch_size, allow_probe
                )
                for name, ridge in leg_ridges.items():
                    ridges[name][leg] = ridge
        inverse = _PerLegOperator(blocks, in_structure=operator.in_structure)

    if not regularization:
        return inverse, None
    return inverse, _ridge_operator(ridges, operator.in_structure)


def cross_gram(a: Basis, b: Basis, weights: Float[Array, ' samp']) -> Float[Array, 'a_size b_size']:
    """The weighted cross Gram `B_aᵀ diag(weights) B_b`, as one `(n_a·k_a, n_b·k_b)` block.

    `a is b` recovers the self-Gram, though [`Basis.gram`][] is cheaper for a single template,
    returning bands instead.

    Raises:
        NoStructuredView: If either basis has no [`Basis.support`][] view.
    """
    ca, cb = a.support(), b.support()
    ka, kb = ca.values.shape[0], cb.values.shape[0]
    if ka < kb:  # step through the smaller side's sub-basis functions (below), keep the larger
        return cross_gram(b, a, weights).T
    vwa = (ca.values * weights[None, :]).T  # (samp, k_a)
    if ca.blocks.shape[1] == cb.blocks.shape[1] == 1 and ca.n_blocks == cb.n_blocks == 1:
        # both global: a plain matrix product, with no per-sample products at all
        taps = ca.taps[:, 0] * cb.taps[:, 0]
        return jnp.einsum(
            'ta,bt->ab', taps[:, None] * vwa, cb.values, precision=jax.lax.Precision.HIGHEST
        )

    def column(values_b: Array) -> Array:
        """The block for one sub-basis function of `b`, `(n_a, k_a, n_b)`."""
        gram = jnp.zeros((ca.n_blocks, ka, cb.n_blocks), a.dtype)
        for wa in range(ca.blocks.shape[1]):  # window slots (single slot for non-overlapping bases)
            for wb in range(cb.blocks.shape[1]):
                taps = ca.taps[:, wa] * cb.taps[:, wb] * values_b  # (samp,)
                gram = gram.at[ca.blocks[:, wa], :, cb.blocks[:, wb]].add(taps[:, None] * vwa)
        return gram

    columns = jax.lax.map(column, cb.values)  # (k_b, n_a, k_a, n_b)
    return jnp.moveaxis(columns, 0, -1).reshape(ca.n_blocks * ka, cb.n_blocks * kb)


def _unit_on_zero_rows(matrix: Float[Array, '*batch k k']) -> Float[Array, '*batch k k']:
    """Put a unit diagonal on the zero rows of a Gram: amplitudes no weighted sample sees.

    Such a row (and its column, by symmetry) makes the Gram singular and its factor NaN. The unit
    diagonal leaves the other amplitudes' solution unchanged and returns the unseen ones as they
    came in. An all-zero block, e.g. an interval no sample sees, becomes the identity.
    """
    unseen = jnp.all(matrix == 0, axis=-1)  # (*batch, k)
    return matrix + unseen[..., None] * jnp.eye(matrix.shape[-1], dtype=matrix.dtype)


class _Stream(NamedTuple):
    """The templates on one stream (the TOD, or one Stokes leg of it) and its diagonal weights."""

    bases: dict[str, Basis]
    weights: Float[Array, 'det samp']
    in_structure: PyTree[jax.ShapeDtypeStruct]
    """The amplitudes of `bases`."""
    pomme: OffsetBasis | None = None
    """The Pomme template to eliminate from the Gram, when Pomme is enabled."""


def _ridge_operator(
    ridges: PyTree[Array], structure: PyTree[jax.ShapeDtypeStruct]
) -> AbstractLinearOperator:
    """The ridge added to the Gram before factoring, as a diagonal operator on the amplitudes.

    `ridges` matches `structure`, each leaf holding one value per amplitude, in any shape.
    """
    diagonals = jax.tree.map(
        lambda ridge, s: DiagonalOperator(ridge.reshape(s.shape), in_structure=s),
        ridges,
        structure,
    )
    return BlockDiagonalOperator(diagonals)


def _per_detector(
    ridge: Float[Array, ' det'], structure: PyTree[jax.ShapeDtypeStruct]
) -> PyTree[Array]:
    """A ridge of one value per detector, spread over every amplitude of that detector."""
    return jax.tree.map(
        lambda s: jnp.broadcast_to(ridge.reshape(-1, *(1,) * (len(s.shape) - 1)), s.shape),
        structure,
    )


def _stream_gram_inverse(
    stream: _Stream, regularization: float, batch_size: int, allow_probe: bool
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """Inverse of the Gram of every template on one stream, one block per detector.

    The bases' structure is used when every one of them exposes it; otherwise the Gram is probed,
    if allowed. Paired with the ridge added before factoring, on each amplitude of
    `stream.in_structure` (zero without regularization).
    """
    try:
        return _structured_gram_inverse(stream, regularization, batch_size)
    except NoStructuredView:
        if allow_probe:
            return _probed_gram_inverse(stream, regularization)
    msg = f'structured Gram construction not possible for {list(stream.bases)}, pass `allow_probe=True`'
    raise NotImplementedError(msg)


def _structured_gram_inverse(
    stream: _Stream, regularization: float, batch_size: int
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """The Gram inverse built from the bases' structure.

    A single template keeps the band structure of its own Gram. Several are coupled through the
    shared weight: the time-local one with the most amplitudes keeps its band structure, the
    others bordering it. Without any time-local template, the few amplitudes share a dense block.

    Raises:
        NoStructuredView: If a basis has no structured view.
    """
    bases, weights, pomme = stream.bases, stream.weights, stream.pomme
    if len(bases) == 1:
        ((name, basis),) = bases.items()
        bands = jax.lax.map(lambda w: _banded_gram(basis, pomme, w), weights, batch_size=batch_size)
        bands, ridge = _regularize_bands(bands, regularization)
        return BandedCholeskyOperator.from_bands(bands, stream.in_structure), {name: ridge}

    # a basis split into several blocks of time is time-local; one block sees every sample
    local = [name for name, basis in bases.items() if basis._n_blocks > 1]
    if local:
        core = max(local, key=lambda name: bases[name].size)
        return _bordered_gram_inverse(stream, core, regularization, batch_size)

    ordered: list[Basis] = jax.tree.leaves(bases, is_leaf=is_basis)
    blocks = jax.lax.map(
        lambda w: _dense_gram(ordered, w, _pomme_sums(ordered, w, pomme)),
        weights,
        batch_size=batch_size,
    )
    blocks, ridge = _regularize_dense(blocks, regularization)
    inverse = BandedCholeskyOperator.from_dense(blocks, stream.in_structure)
    return inverse, _per_detector(ridge, stream.in_structure)


def _regularize_bands(
    bands: Float[Array, '*batch n w1 k k'], regularization: float
) -> tuple[Float[Array, '*batch n w1 k k'], Float[Array, '*batch n k']]:
    """Make a banded Gram factorable: a unit diagonal on unseen amplitudes, then the ridge.

    Each diagonal block takes `regularization` times its mean diagonal, so that the ridge is in the
    units of the Gram. It is scaled from the Gram as assembled, before the unit placeholders go in,
    so that it depends on the data alone. The ridge is returned for the explicit normal system to
    add the same one.
    """
    diagonal = bands[..., 0, :, :]
    scale = jnp.mean(jnp.diagonal(diagonal, axis1=-2, axis2=-1), axis=-1, keepdims=True)
    ridge = jnp.broadcast_to(regularization * scale, diagonal.shape[:-1])
    eye = jnp.eye(diagonal.shape[-1], dtype=bands.dtype)
    diagonal = _unit_on_zero_rows(diagonal) + ridge[..., None] * eye
    return bands.at[..., 0, :, :].set(diagonal), ridge


def _regularize_dense(
    blocks: Float[Array, '*batch k k'], regularization: float
) -> tuple[Float[Array, '*batch k k'], Float[Array, ' *batch']]:
    """`_regularize_bands` for dense blocks, whose ridge is one value per block."""
    bands, ridge = _regularize_bands(blocks[..., None, None, :, :], regularization)
    return bands[..., 0, 0, :, :], ridge[..., 0, 0]


def _banded_gram(
    basis: Basis, pomme: OffsetBasis | None, weights: Float[Array, ' samp']
) -> Float[Array, 'n w1 k k']:
    """One detector's `basis.gram`, Pomme-corrected when Pomme is enabled."""
    if pomme is None:
        return basis.gram(weights)
    if basis._n_blocks > 1:
        bands, _ = _pomme_local_gram(basis, pomme, weights)
        return bands
    # a global basis: one dense block, with no band for an interval to widen
    sums = _interval_sums(pomme, basis, weights)
    correction = sums.T @ (_inverse_counts(pomme, weights)[:, None] * sums)
    return basis.gram(weights) - correction[None, None]


class _PommeCoreSums(NamedTuple):
    """One detector's factors of the Pomme Schur correction for a time-local basis.

    `C = Zᵀ W B` is kept in the band layout: an interval reads at most `w1` consecutive blocks, so
    the dense `(n_int, n_blocks·k)` matrix is never formed.
    """

    first: Int[Array, ' n_int']
    """The first block each interval couples."""
    sums: Float[Array, 'n_int w1 k']
    """`C` on each interval's `w1` consecutive blocks, from `first`."""
    inverse_counts: Float[Array, ' n_int']
    """`D⁻¹ = (Zᵀ W Z)⁻¹`."""


def _pomme_local_gram(
    basis: Basis, pomme: OffsetBasis, weights: Float[Array, ' samp']
) -> tuple[Float[Array, 'n w1 k k'], _PommeCoreSums]:
    """One detector's Pomme-corrected Gram of a time-local basis, and the factors it used.

    An interval straddles at most two of the basis's blocks, so the corrected Gram needs one more
    band than the plain one. The factors are returned for the bordered path to reuse.
    """
    view = basis.support()
    bands = basis.gram(weights)
    w1 = min(bands.shape[1] + 1, view.n_blocks)
    bands = jnp.pad(bands, [(0, 0), (0, w1 - bands.shape[1]), (0, 0), (0, 0)])
    core = _pomme_core_sums(pomme, view, weights, w1)
    return bands - _pomme_band_correction(core, view.n_blocks), core


def _pomme_core_sums(
    pomme: OffsetBasis, view: _BasisColumns, weights: Float[Array, ' samp'], w1: int
) -> _PommeCoreSums:
    """`C = Zᵀ W B` in the band layout, and `D⁻¹`.

    Interval `i` couples the consecutive blocks its weighted samples read, `first[i] + s` for
    `s < w1`. A wider span means a block shorter than an interval, which the band cannot hold; it
    is reported as a runtime error. Masked samples couple nothing, so a short block reached only
    by them passes. No `(samp, k, k)` intermediate is formed.
    """
    intervals = pomme.support()
    n_int, n_blocks, k = intervals.n_blocks, view.n_blocks, view.values.shape[0]
    ids = intervals.blocks[:, 0]
    zw = weights * intervals.values[0]  # each sample's entry of Zᵀ W, zero past the last interval

    # the blocks each interval couples; the accumulators take the block index dtype, which varies
    # with the basis flavour
    dtype = view.blocks.dtype
    first, last = jnp.full(n_int, n_blocks, dtype), jnp.full(n_int, -1, dtype)
    active = (zw != 0) & jnp.any(view.values != 0, axis=0)
    for slot in range(view.blocks.shape[1]):  # window width is static
        reads = active & (view.taps[:, slot] != 0)
        block = view.blocks[:, slot]
        first = first.at[ids].min(jnp.where(reads, block, n_blocks).astype(dtype))
        last = last.at[ids].max(jnp.where(reads, block, -1).astype(dtype))
    first = eqx.error_if(
        first,
        jnp.any(last - first >= w1),
        'a Pomme interval spans more template blocks than the Gram band can hold: '
        'template blocks (scan intervals, spline knots) must be longer than pomme_tau',
    )

    wv = (view.values * zw[None, :]).T  # (samp, k)
    sums = jnp.zeros((n_int, w1, k), view.values.dtype)
    for slot in range(view.blocks.shape[1]):
        s = jnp.clip(view.blocks[:, slot] - first[ids], 0, w1 - 1)
        sums = sums.at[ids, s].add(view.taps[:, slot][:, None] * wv)
    return _PommeCoreSums(first, sums, _inverse_counts(pomme, weights))


def _pomme_band_correction(core: _PommeCoreSums, n_blocks: int) -> Float[Array, 'n_blocks w1 k k']:
    """The Pomme Schur correction `Cᵀ D⁻¹ C` of a banded Gram, in its band layout.

    Interval `i`'s outer product lands in the band at `(first[i] + s, s' − s)`.
    """
    _, w1, k = core.sums.shape
    bands = jnp.zeros((n_blocks, w1, k, k), core.sums.dtype)
    for s in range(w1):
        # an interval coupling fewer than w1 blocks has zero sums past its last: clip the index
        rows = jnp.clip(core.first + s, 0, n_blocks - 1)
        for s2 in range(s, w1):
            contrib = jnp.einsum(
                'ia,ib,i->iab', core.sums[:, s], core.sums[:, s2], core.inverse_counts
            )
            bands = bands.at[rows, s2 - s].add(contrib)
    return bands


def _dense_gram(
    bases: list[Basis],
    weights: Float[Array, ' samp'],
    sums: tuple[Float[Array, ' n_int'], list[Float[Array, 'n_int k']]] | None = None,
) -> Float[Array, 'k k']:
    """One detector's joint Gram of `bases`, each owning a contiguous slice in the given order.

    The lower blocks are the transposes of the upper ones, so each pair is computed once. With the
    Pomme factors `sums` (`D⁻¹` and each basis's `C`, see `_pomme_sums`), every block carries the
    Schur correction that eliminates the Pomme template.
    """
    offsets = np.cumsum([0, *(basis.size for basis in bases)])
    n_amps = int(offsets[-1])
    block = jnp.zeros((n_amps, n_amps), bases[0].dtype)
    for i, a in enumerate(bases):
        rows = slice(offsets[i], offsets[i + 1])
        for j in range(i, len(bases)):
            cols = slice(offsets[j], offsets[j + 1])
            cross = cross_gram(a, bases[j], weights)
            if sums is not None:
                inverse_counts, per_basis = sums
                cross = cross - per_basis[i].T @ (inverse_counts[:, None] * per_basis[j])
            block = block.at[rows, cols].set(cross)
            if j > i:
                block = block.at[cols, rows].set(cross.T)
    return block


def _pomme_sums(
    bases: list[Basis],
    weights: Float[Array, ' samp'],
    pomme: OffsetBasis | None,
) -> tuple[Float[Array, ' n_int'], list[Float[Array, 'n_int k']]] | None:
    """`D⁻¹` and each basis's `C = Zᵀ W B`, the factors of the Pomme Schur correction."""
    if pomme is None:
        return None
    sums = [_interval_sums(pomme, basis, weights) for basis in bases]
    return _inverse_counts(pomme, weights), sums


def _interval_sums(
    pomme: OffsetBasis, basis: Basis, weights: Float[Array, ' samp']
) -> Float[Array, 'n_int k']:
    """`C = Zᵀ W B` for one basis: its weighted columns summed over each interval."""
    if basis._n_blocks > 1:
        return cross_gram(pomme, basis, weights)
    # a global basis: its few columns span every sample, so weighting them and projecting each
    # onto the regular intervals avoids scattering a `(samp, k)` copy by interval
    view = basis.support()
    columns = view.values * (view.taps.sum(axis=1) * weights)[None, :]
    return pomme._project_detectors(columns).T


def _inverse_counts(pomme: OffsetBasis, weights: Float[Array, ' samp']) -> Float[Array, ' n_int']:
    """`D⁻¹ = (Zᵀ W Z)⁻¹`, zero on intervals of zero weight, whose rows of `C` vanish too."""
    counts = pomme.gram(weights)[:, 0, 0, 0]  # Z is block diagonal: one weighted count each
    return 1.0 / jnp.where(counts > 0, counts, jnp.inf)


def _bordered_gram_inverse(
    stream: _Stream, core_name: str, regularization: float, batch_size: int
) -> tuple['_BorderedGramInverse', PyTree[Array]]:
    """Inverse Gram of one time-local template bordered by the others, one per detector.

    The core template's own Gram is block-banded; the others couple to it through a dense border
    and to each other through a dense corner.
    """
    bases, pomme = stream.bases, stream.pomme
    core = bases[core_name]
    border_names = tuple(name for name in bases if name != core_name)
    border_bases = [bases[name] for name in border_names]

    def build(weights: Array) -> tuple[Array, Array, Array]:
        border = [cross_gram(core, basis, weights) for basis in border_bases]
        if pomme is None:
            corner = _dense_gram(border_bases, weights)
            return core.gram(weights), jnp.concatenate(border, axis=1), corner

        bands, core_sums = _pomme_local_gram(core, pomme, weights)
        border_sums = [_interval_sums(pomme, basis, weights) for basis in border_bases]
        border = [
            block - _pomme_border_correction(core_sums, sums, core._n_blocks)
            for block, sums in zip(border, border_sums, strict=True)
        ]
        corner = _dense_gram(border_bases, weights, (core_sums.inverse_counts, border_sums))
        return bands, jnp.concatenate(border, axis=1), corner

    bands, border, corner = jax.lax.map(build, stream.weights, batch_size=batch_size)
    # the core's diagonal blocks and the corner are ridged separately, each scaled by its own
    # diagonal
    bands, core_ridge = _regularize_bands(bands, regularization)
    corner, corner_ridge = _regularize_dense(corner, regularization)

    structure = stream.in_structure
    ridges = {core_name: core_ridge}
    ridges |= {name: _per_detector(corner_ridge, structure[name]) for name in border_names}
    factor = BorderedBandedCholeskyOperator.from_blocks(bands, border, corner)
    inverse = _BorderedGramInverse(factor, core_name, border_names, in_structure=structure)
    return inverse, ridges


def _pomme_border_correction(
    core: _PommeCoreSums, border_sums: Float[Array, 'n_int m'], n_blocks: int
) -> Float[Array, 'n_blocks_k m']:
    """The Pomme Schur correction `C_coreᵀ D⁻¹ C_border` of a border block, dense.

    Interval `i` adds the outer product of its core sums and its border sums to the rows of the
    `w1` blocks from `first[i]`, in the core's `(block, k)` amplitude order.
    """
    _, w1, k = core.sums.shape
    scaled = core.inverse_counts[:, None] * border_sums  # (n_int, m)
    out = jnp.zeros((n_blocks, k, border_sums.shape[1]), core.sums.dtype)
    for s in range(w1):
        rows = jnp.clip(core.first + s, 0, n_blocks - 1)
        out = out.at[rows].add(jnp.einsum('ia,ib->iab', core.sums[:, s], scaled))
    return out.reshape(n_blocks * k, -1)


@symmetric
class _BorderedGramInverse(AbstractLinearOperator):
    """A bordered banded Cholesky inverse, on template amplitudes.

    One time-local template is the banded core of
    [`BorderedBandedCholeskyOperator`][furax.linalg.BorderedBandedCholeskyOperator], the other
    templates its border.
    """

    factor: BorderedBandedCholeskyOperator
    core_name: str = field(metadata={'static': True})
    border_names: tuple[str, ...] = field(metadata={'static': True})

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        r_a = x[self.core_name]
        n_dets = r_a.shape[0]
        r_c = jnp.concatenate([x[name].reshape(n_dets, -1) for name in self.border_names], axis=1)
        x_a, x_c = self.factor((r_a, r_c))

        out = {self.core_name: x_a}
        cuts = np.cumsum([prod(x[name].shape[1:]) for name in self.border_names])[:-1]
        for name, part in zip(self.border_names, jnp.split(x_c, cuts, axis=1), strict=True):
            out[name] = part.reshape(x[name].shape)
        return {name: out[name] for name in x}


@symmetric
class _PerLegOperator(AbstractLinearOperator):
    """Block diagonal over Stokes legs, each block acting on every template the leg carries.

    The amplitudes are keyed by template, then leg, so a block's input is gathered across
    templates and its output scattered back.
    """

    blocks: dict[str, AbstractLinearOperator]

    def mv(self, x: PyTree[Array]) -> PyTree[Array]:
        out: dict[str, dict[str, Array]] = {name: {} for name in x}
        for leg, block in self.blocks.items():
            y = block({name: amps[leg] for name, amps in x.items() if leg in amps})
            for name, amp in y.items():
                out[name][leg] = amp
        return out


def _probed_gram_inverse(
    stream: _Stream, regularization: float
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """The Gram inverse from `G = Tᵀ W T` applied to one amplitude at a time.

    Costs `O(K)` applications for `K` amplitudes, but needs nothing of the bases beyond `T` itself.
    Amplitudes carry detectors on their leading axis and `T` couples none of them, so `G` is
    block-diagonal there and each detector's block is factored on its own. Each application
    expands to a single `(det, samp)` stream, not to every Stokes leg of the TOD. With Pomme the
    probed operator is `Tᵀ W F T`, exact for any `T`.
    """
    diag, in_structure = stream.weights, stream.in_structure
    n_dets = diag.shape[0]
    operator = TemplateOperator(stream.bases, n_dets)
    filter_: AbstractLinearOperator = IdentityOperator(in_structure=operator.out_structure)
    if stream.pomme is not None:
        filter_ = PommeProjectionOperator(stream.pomme.tau, in_structure=operator.out_structure)
    leaves, treedef = jax.tree.flatten(in_structure)
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
        response = operator.T(diag * filter_(operator(treedef.unflatten(parts))))
        per_leaf = [leaf.reshape(n_dets, -1) for leaf in jax.tree.leaves(response)]
        return jnp.concatenate(per_leaf, axis=-1)  # (n_dets, n_amps)

    columns = jax.lax.map(probe, jnp.arange(n_amps))  # (col, n_dets, row)
    blocks = jnp.moveaxis(columns, 0, -1)  # (n_dets, row, col)
    blocks, ridge = _regularize_dense(blocks, regularization)
    inverse = BandedCholeskyOperator.from_dense(blocks, in_structure)
    return inverse, _per_detector(ridge, in_structure)
