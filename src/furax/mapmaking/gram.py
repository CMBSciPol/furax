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

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, PyTree

import furax.tree
from furax import AbstractLinearOperator, symmetric
from furax.linalg import BandedCholeskyOperator, BorderedBandedCholeskyOperator

from .pomme import PommeIntervals, PommeProjectionOperator
from .templates import (
    AbstractTemplateOperator,
    Basis,
    NoStructuredView,
    StokesTemplateOperator,
    TemplateOperator,
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
    ones = furax.tree.ones_like(weight.in_structure)
    diag = weight(ones)
    # if we wanted to guard against a non-diagonal W, one extra application on a random
    # vector `x` and a comparison against `diag * x` would catch it with very high probability

    if not isinstance(operator, StokesTemplateOperator):  # a single stream
        intervals = _intervals(operator.bases, pomme_tau)
        stream = _Stream(operator.bases, diag, operator.in_structure, intervals)
        return _stream_gram_inverse(stream, regularization, batch_size, allow_probe)

    # Pomme acts on the modulated TOD, which demodulation splits into one stream per leg.
    if pomme_tau is not None:
        raise ValueError('Pomme filtering applies to a single stream, not a Stokes TOD')

    # Two bases on different legs never share a weighted sample, so each leg is a stream of its
    # own, with one block over the templates it carries rather than one block over every leg.
    blocks = {}
    for leg in operator.legs:
        bases = {name: on[leg] for name, on in operator.bases_by_leg.items() if leg in on}
        if bases:
            structure = {name: operator.in_structure[name][leg] for name in bases}
            stream = _Stream(bases, getattr(diag, leg), structure, None)
            blocks[leg] = _stream_gram_inverse(stream, regularization, batch_size, allow_probe)
    return _PerLegOperator(blocks, in_structure=operator.in_structure)


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


def _unit_on_zero_bands(bands: Float[Array, '*batch n w1 k k']) -> Float[Array, '*batch n w1 k k']:
    """`_unit_on_zero_rows` on the diagonal blocks of a banded Gram (`d = 0`)."""
    return bands.at[..., 0, :, :].set(_unit_on_zero_rows(bands[..., 0, :, :]))


class _Stream(NamedTuple):
    """The templates on one stream (the TOD, or one Stokes leg of it) and its diagonal weights."""

    bases: dict[str, Basis]
    weights: Float[Array, 'det samp']
    in_structure: PyTree[jax.ShapeDtypeStruct]
    """The amplitudes of `bases`."""
    intervals: PommeIntervals | None
    """The Pomme intervals to eliminate from the Gram, when Pomme is enabled."""


def _intervals(bases: dict[str, Basis], pomme_tau: int | None) -> PommeIntervals | None:
    """The Pomme intervals over the stream the `bases` are evaluated on."""
    if pomme_tau is None:
        return None
    basis: Basis = jax.tree.leaves(bases, is_leaf=is_basis)[0]  # all agree on the sample count
    return PommeIntervals(basis.n_points, pomme_tau)


def _stream_gram_inverse(
    stream: _Stream, regularization: float, batch_size: int, allow_probe: bool
) -> AbstractLinearOperator:
    """Inverse of the Gram of every template on one stream, one block per detector.

    The bases' structure is used when every one of them exposes it; otherwise the Gram is probed,
    if allowed.
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
) -> AbstractLinearOperator:
    """The Gram inverse built from the bases' structure.

    A single template keeps the band structure of its own Gram. Several are coupled through the
    shared weight: the time-local one with the most amplitudes keeps its band structure, the
    others bordering it. Without any time-local template, the few amplitudes share a dense block.

    Raises:
        NoStructuredView: If a basis has no structured view.
    """
    bases, weights = stream.bases, stream.weights
    if len(bases) == 1:
        (basis,) = bases.values()
        gram = _pomme_banded_gram(basis, stream)
        bands = _unit_on_zero_bands(jax.lax.map(gram, weights, batch_size=batch_size))
        return BandedCholeskyOperator.from_bands(bands, stream.in_structure, regularization)

    # a basis split into several blocks of time is time-local; one block sees every sample
    local = [name for name, basis in bases.items() if basis._n_blocks > 1]
    if local:
        core = max(local, key=lambda name: bases[name].size)
        return _bordered_gram_inverse(stream, core, regularization, batch_size)

    ordered: list[Basis] = jax.tree.leaves(bases, is_leaf=is_basis)
    blocks = jax.lax.map(
        lambda w: _dense_gram(ordered, w, stream.intervals), weights, batch_size=batch_size
    )
    blocks = _unit_on_zero_rows(blocks)
    return BandedCholeskyOperator.from_dense(blocks, stream.in_structure, regularization)


def _pomme_banded_gram(basis: Basis, stream: _Stream):
    """`basis.gram`, Pomme-corrected when the stream carries intervals.

    An interval straddles at most two of the basis's blocks, so the corrected Gram needs one more
    band than the plain one, and a basis of a single block (a dense one) keeps its single band.
    """
    intervals = stream.intervals
    if intervals is None:
        return basis.gram

    view = basis.support()
    weights = stream.weights
    w1 = min(jax.eval_shape(basis.gram, weights[0]).shape[1] + 1, view.n_blocks)
    first = intervals.first_blocks(view, w1)

    def gram(w: Float[Array, ' samp']) -> Float[Array, 'n w1 k k']:
        bands = basis.gram(w)
        bands = jnp.pad(bands, [(0, 0), (0, w1 - bands.shape[1]), (0, 0), (0, 0)])
        return bands - intervals.band_correction(view, first, w, w1)

    return gram


def _dense_gram(
    bases: list[Basis],
    weights: Float[Array, ' samp'],
    intervals: PommeIntervals | None = None,
) -> Float[Array, 'k k']:
    """One detector's joint Gram of `bases`, each owning a contiguous slice in the given order.

    The lower blocks are the transposes of the upper ones, so each pair is computed once. With
    Pomme intervals, every block carries the Schur correction that eliminates them.
    """
    offsets = np.cumsum([0, *(basis.size for basis in bases)])
    n_amps = int(offsets[-1])
    block = jnp.zeros((n_amps, n_amps), bases[0].dtype)
    sums = _pomme_sums(bases, weights, intervals)
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
    intervals: PommeIntervals | None,
) -> tuple[Float[Array, ' n_int'], list[Float[Array, 'n_int k']]] | None:
    """`D⁻¹` and each basis's `C = Zᵀ W B`, the factors of the Pomme Schur correction."""
    if intervals is None:
        return None
    return intervals.inverse_counts(weights), [
        intervals.sums(basis.support(), weights) for basis in bases
    ]


def _bordered_gram_inverse(
    stream: _Stream, core_name: str, regularization: float, batch_size: int
) -> '_BorderedGramInverse':
    """Inverse Gram of one time-local template bordered by the others, one per detector.

    The core template's own Gram is block-banded; the others couple to it through a dense border
    and to each other through a dense corner.
    """
    bases = stream.bases
    core = bases[core_name]
    border_names = tuple(name for name in bases if name != core_name)
    border_bases = [bases[name] for name in border_names]

    intervals = stream.intervals
    core_gram = _pomme_banded_gram(core, stream)

    def build(weights: Array) -> tuple[Array, Array, Array]:
        border = [cross_gram(core, basis, weights) for basis in border_bases]
        sums = _pomme_sums([core, *border_bases], weights, intervals)
        if sums is not None:
            inverse_counts, (core_sums, *border_sums) = sums
            border = [
                block - core_sums.T @ (inverse_counts[:, None] * basis_sums)
                for block, basis_sums in zip(border, border_sums, strict=True)
            ]
        corner = _dense_gram(border_bases, weights, intervals)
        return core_gram(weights), jnp.concatenate(border, axis=1), corner

    bands, border, corner = jax.lax.map(build, stream.weights, batch_size=batch_size)
    bands, corner = _unit_on_zero_bands(bands), _unit_on_zero_rows(corner)
    factor = BorderedBandedCholeskyOperator.from_blocks(bands, border, corner, regularization)
    return _BorderedGramInverse(factor, core_name, border_names, in_structure=stream.in_structure)


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


def _pomme_filter(stream: _Stream, structure: jax.ShapeDtypeStruct):
    """The Pomme projector over one stream, or the identity when Pomme is disabled."""
    if stream.intervals is None:
        return lambda x: x
    return PommeProjectionOperator(stream.intervals.tau, in_structure=structure)


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


def _probed_gram_inverse(stream: _Stream, regularization: float) -> AbstractLinearOperator:
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
    filter_ = _pomme_filter(stream, operator.out_structure)
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
    blocks = _unit_on_zero_rows(blocks)
    return BandedCholeskyOperator.from_dense(blocks, in_structure, regularization)
