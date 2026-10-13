r"""Gram matrix construction and inversion for template deprojection.

Implicit deprojection replaces the weight matrix $W$ with the template-marginalised weight

$$ W' = W - W T (T^\top W T)^{-1} T^\top W, $$

the $W$-metric projector off $\mathrm{range}(T)$.

This module assembles the Gram matrix $G \equiv T^\top W T$ from the bases of an
[`AbstractTemplateOperator`][]. Inversion is performed via [`furax.linalg.cholesky`][].

Combined with Pomme (`pomme_tau`), the effective weight is $W F$ with $F$ the
[`PommeProjectionOperator`][furax.mapmaking.pomme.PommeProjectionOperator], and the Gram becomes
$\tilde G = T^\top W F T = (FT)^\top W (FT)$: the Gram of the Pomme-filtered templates. It is
assembled without forming $FT$, by Schur-eliminating the Pomme template $Z$ (the interval
indicators) from the joint Gram of $[Z\ T]$:

$$ \tilde G = T^\top W T - C^\top D^{-1} C, \qquad C = Z^\top W T, \quad D = Z^\top W Z, $$

with $D$ diagonal (weighted interval sample counts) and $C$ the weighted per-interval sums of the
basis columns. This equals $T^\top W D_Z T$ for any diagonal $W$, and $T^\top W F T$ when $W$
and $F$ commute, which the whole-interval masking of the Pomme path guarantees. An interval
straddling two blocks of a time-local basis couples them, so the Gram band widens by one block.
$F$ annihilates constant columns, so a template containing one (polynomial order 0, a DC harmonic,
binned azimuth) makes $\tilde G$ singular: a `regularization` ridge is then required.

A correlated weight, such as the masked Toeplitz $M N^{-1} M$ of a $1/f$ noise model, still
couples only samples a bounded distance apart, so a time-local template keeps a banded Gram, only
wider. Its entries are read off the responses $T^\top W T x$ to a few probe amplitudes $x$, each
probe setting blocks spaced far enough apart that their responses do not overlap.

Limitations:

- $W$ must not couple detectors, nor the Stokes legs of a demodulated TOD.
- Pomme requires a diagonal $W$.
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

from collections.abc import Callable
from dataclasses import field
from math import prod
from typing import Any, NamedTuple, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, Int, PyTree

import furax.tree
from furax import AbstractLinearOperator, DiagonalOperator, symmetric
from furax.core import BlockDiagonalOperator
from furax.linalg import BandedCholeskyOperator, BorderedBandedCholeskyOperator

from .bases import Basis, OffsetBasis, WindowedBasis
from .templates import AbstractTemplateOperator, StokesTemplateOperator

__all__ = [
    'gram_inverse',
]


def gram_inverse(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    regularization: float = 0.0,
    *,
    bandwidth: int = 0,
    pomme_tau: int | None = None,
    batch_size: int = 8,
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """Inverse Gram matrix `(Tᵀ W T)⁻¹` for template operator `T` and weight `W`, and its ridge.

    `T` maps amplitudes to TOD, per detector. `W` is *assumed* not to couple detectors, nor
    samples more than `bandwidth` apart: a weight that does not respect that contract silently
    changes the result. `G = Tᵀ W T` thus does not couple different detectors: each gets its own
    block, making `G` block-diagonal. Templates whose amplitudes are shared by the detectors
    (`operator.shared`) have a single block instead, summed over the detectors; their Gram couples
    them to every per-detector block, so the two kinds cannot be mixed in one operator.

    Each stream (the TOD, or each Stokes leg of a demodulated one) is inverted on its own. For a
    diagonal `W` (`bandwidth=0`), read off by applying it to a vector of ones, the Gram is
    assembled from the structure ([`Basis.gram`][], [`Basis.support`][]) of the bases it carries.
    Otherwise, it is read off the responses `Tᵀ W T x` to a few probe amplitudes `x`, as many as a
    time-local template has blocks within reach of one another (`bandwidth` samples, plus its
    window), times its sub-basis functions, plus one per amplitude of the other templates.

    With `pomme_tau`, the effective weight is `W F` with `F` the
    [`PommeProjectionOperator`][furax.mapmaking.pomme.PommeProjectionOperator] of that interval
    length, and the inverse of `Tᵀ W F T` is returned instead (see the module
    docstring). `W` itself stays the diagonal weight.

    The ridge is returned alongside the inverse because a system built from the Gram itself, not
    from its factor, must add the same ridge to stay consistent with it. Where the Gram is
    singular, as under Pomme with a constant template column, that ridge is also what makes the
    system well posed.

    Args:
        operator: The template operator `T`.
        weight: The weights `W`.
        regularization: Relative ridge added to each detector's Gram block before factoring.
        bandwidth: The largest distance, in samples, between two samples that `W` couples.
        pomme_tau: Pomme interval length, when the templates are combined with Pomme.
        batch_size: Detector batch size, bounding the transient memory of the Gram assembly of a
            diagonal `W`.

    Returns:
        The per-detector inverse Gram operator, and the ridge added to the Gram before factoring
        it, as a diagonal operator on the amplitudes; `None` when `regularization` is 0.

    Raises:
        ValueError: If `pomme_tau` is given for a Stokes TOD rather than a single stream, or
            together with a nonzero `bandwidth`, or if `operator` mixes templates with shared and
            per-detector amplitudes.
    """
    is_stokes = isinstance(operator, StokesTemplateOperator)
    # Pomme acts on the modulated TOD, which demodulation splits into one stream per leg.
    if pomme_tau is not None and is_stokes:
        raise ValueError('Pomme filtering applies to a single stream, not a Stokes TOD')
    if pomme_tau is not None and bandwidth:
        raise ValueError('Pomme filtering requires a diagonal weight')
    shared = bool(operator.shared)
    if shared and set(operator.shared) != set(operator.bases):
        raise ValueError('templates with shared and per-detector amplitudes are coupled')

    # Two bases on different legs never share a weighted sample, so each leg is a stream of its
    # own, with one block over the templates it carries rather than one over every leg.
    streams: dict[str | None, dict[str, Basis]]
    if is_stokes:
        streams = {
            leg: bases
            for leg in operator.legs
            if (bases := {name: on[leg] for name, on in operator.bases_by_leg.items() if leg in on})
        }
    else:
        streams = {None: operator.bases}

    if bandwidth:
        results = _banded_weight_gram_inverses(operator, weight, streams, regularization, bandwidth)
    else:
        diag = weight(furax.tree.ones_like(weight.in_structure))
        # if we wanted to guard against a non-diagonal W, one extra application on a random
        # vector `x` and a comparison against `diag * x` would catch it with very high probability
        results = {}
        for leg, bases in streams.items():
            if leg is None:
                n_samps = diag.shape[-1]
                pomme = None if pomme_tau is None else OffsetBasis(pomme_tau, n_samps, diag.dtype)
                stream = _Stream(bases, diag, operator.in_structure, pomme, shared)
            else:
                structure = _on_leg(operator.in_structure, bases, leg)
                stream = _Stream(bases, getattr(diag, leg), structure, shared=shared)
            results[leg] = _stream_gram_inverse(stream, regularization, batch_size)

    ridges: PyTree[Array]
    if not is_stokes:  # a single stream
        inverse, ridges = results[None]
    else:
        ridges = {name: {} for name in operator.bases_by_leg}
        for leg, (_, leg_ridges) in results.items():
            for name, ridge in leg_ridges.items():
                ridges[name][leg] = ridge
        blocks = {leg: block for leg, (block, _) in results.items() if leg is not None}
        inverse = _PerLegOperator(blocks, in_structure=operator.in_structure)

    if not regularization:
        return inverse, None
    return inverse, _ridge_operator(ridges, operator.in_structure)


def _cross_gram(
    a: Basis, b: Basis, weights: Float[Array, ' samp']
) -> Float[Array, 'a_size b_size']:
    """The weighted cross Gram `B_aᵀ diag(weights) B_b`, as one `(n_a·k_a, n_b·k_b)` block.

    `a is b` recovers the self-Gram, though [`Basis.gram`][] is cheaper for a single template,
    returning bands instead.
    """
    ca, cb = a.support(), b.support()
    ka, kb = ca.values.shape[0], cb.values.shape[0]
    if ka < kb:  # step through the smaller side's sub-basis functions (below), keep the larger
        return _cross_gram(b, a, weights).T
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
    shared: bool = False
    """Whether the amplitudes are shared by the detectors, their Gram summed over them."""


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
    stream: _Stream, regularization: float, batch_size: int
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """Inverse of the Gram of every template on one stream, one block per detector.

    A single template keeps the band structure of its own Gram. Several are coupled through the
    shared weight: the time-local one with the most amplitudes keeps its band structure, the
    others bordering it. Without any time-local template, the few amplitudes share a dense block.
    Paired with the ridge added before factoring, on each amplitude of `stream.in_structure` (zero
    without regularization).
    """
    bases, pomme = stream.bases, stream.pomme
    core, border = _layout(bases)
    if not border:
        assert core is not None
        bands = _map_detectors(
            lambda b, w: _banded_gram(b[core], pomme, w), stream, batch_size=batch_size
        )
        return _banded_inverse(bands, core, stream.in_structure, regularization)
    if core is not None:
        return _bordered_gram_inverse(stream, core, border, regularization, batch_size)

    def dense(bases: dict[str, Basis], weights: Array) -> Array:
        ordered = [bases[name] for name in border]
        return _dense_gram(ordered, weights, _pomme_sums(ordered, weights, pomme))

    blocks = _map_detectors(dense, stream, batch_size=batch_size)
    return _dense_inverse(blocks, stream.in_structure, regularization)


def _layout(bases: dict[str, Basis]) -> tuple[str | None, tuple[str, ...]]:
    """How the Gram of the templates on one stream is laid out: a banded core, and a border.

    A single template is its own core. Among several, the time-local one with the most amplitudes
    is the core, bordered by the others. Without any time-local template, every template is in the
    border, in the order of the amplitude tree, sharing one dense block.
    """
    if len(bases) == 1:
        (name,) = bases
        return name, ()
    # a basis split into several blocks of time is time-local; one block sees every sample
    local = [name for name, basis in bases.items() if basis._n_blocks > 1]
    if not local:
        return None, tuple(sorted(bases))  # the key order of the amplitude tree
    core = max(local, key=lambda name: bases[name].size)
    return core, tuple(name for name in bases if name != core)


def _banded_inverse(
    bands: Float[Array, 'det n w1 k k'],
    name: str,
    structure: PyTree[jax.ShapeDtypeStruct],
    regularization: float,
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """Factor the banded Gram of a single template, and its ridge."""
    bands, ridge = _regularize_bands(bands, regularization)
    return BandedCholeskyOperator.from_bands(bands, structure), {name: ridge}


def _dense_inverse(
    blocks: Float[Array, 'det k k'],
    structure: PyTree[jax.ShapeDtypeStruct],
    regularization: float,
) -> tuple[AbstractLinearOperator, PyTree[Array]]:
    """Factor the dense Gram of templates sharing one block, and its ridge."""
    blocks, ridge = _regularize_dense(blocks, regularization)
    inverse = BandedCholeskyOperator.from_dense(blocks, structure)
    return inverse, _per_detector(ridge, structure)


def _on_leg(
    structure: PyTree[jax.ShapeDtypeStruct], bases: dict[str, Basis], leg: str
) -> PyTree[jax.ShapeDtypeStruct]:
    """The part of a Stokes template operator's amplitude tree on one leg, for `bases`."""
    return {name: structure[name][leg] for name in bases}


def _map_detectors[T](
    f: Callable[[dict[str, Basis], Float[Array, ' samp']], T], stream: _Stream, batch_size: int
) -> T:
    """`f(bases, weights)` for each detector, results stacked on a leading detector axis.

    A basis batched over detectors is passed as that detector's own; a shared one as is. With
    amplitudes shared by the detectors, the results are summed instead, keeping an axis of size 1.
    """
    batched = {name: basis for name, basis in stream.bases.items() if basis.batch_shape}
    parts = {name: basis._partition_batch() for name, basis in batched.items()}

    def one(args: tuple[Array, dict[str, Basis]]) -> T:
        weights, own = args
        own = {
            name: eqx.combine(part, parts[name][1])._with_batch_shape(batched[name].batch_shape[1:])
            for name, part in own.items()
        }
        return f(stream.bases | own, weights)

    weights = stream.weights
    if stream.shared and not batched:
        # every detector sees the same functions, and a Gram is linear in the weights
        weights = weights.sum(axis=0, keepdims=True)
    own = {name: part for name, (part, _) in parts.items()}
    results = jax.lax.map(one, (weights, own), batch_size=batch_size)
    if stream.shared:
        return jax.tree.map(lambda a: a.sum(axis=0, keepdims=True), results)
    return results


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
    pomme: OffsetBasis, view: WindowedBasis, weights: Float[Array, ' samp'], w1: int
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
            cross = _cross_gram(a, bases[j], weights)
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
        return _cross_gram(pomme, basis, weights)
    # a global basis: its few columns span every sample, so weighting them and projecting each
    # onto the regular intervals avoids scattering a `(samp, k)` copy by interval
    view = basis.support()
    columns = view.values * (view.taps.sum(axis=1) * weights)[None, :]
    return pomme.T(columns).T  # (n_int, k)


def _inverse_counts(pomme: OffsetBasis, weights: Float[Array, ' samp']) -> Float[Array, ' n_int']:
    """`D⁻¹ = (Zᵀ W Z)⁻¹`, zero on intervals of zero weight, whose rows of `C` vanish too."""
    counts = pomme.gram(weights)[:, 0, 0, 0]  # Z is block diagonal: one weighted count each
    return 1.0 / jnp.where(counts > 0, counts, jnp.inf)


def _bordered_gram_inverse(
    stream: _Stream,
    core_name: str,
    border_names: tuple[str, ...],
    regularization: float,
    batch_size: int,
) -> tuple['_BorderedGramInverse', PyTree[Array]]:
    """Inverse Gram of one time-local template bordered by the others, one per detector.

    The core template's own Gram is block-banded; the others couple to it through a dense border
    and to each other through a dense corner.
    """
    pomme = stream.pomme

    def build(bases: dict[str, Basis], weights: Array) -> tuple[Array, Array, Array]:
        core = bases[core_name]
        border_bases = [bases[name] for name in border_names]
        border = [_cross_gram(core, basis, weights) for basis in border_bases]
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

    bands, border, corner = _map_detectors(build, stream, batch_size=batch_size)
    return _bordered_inverse(
        bands, border, corner, core_name, border_names, stream.in_structure, regularization
    )


def _bordered_inverse(
    bands: Float[Array, 'det n w1 k k'],
    border: Float[Array, 'det nk m'],
    corner: Float[Array, 'det m m'],
    core_name: str,
    border_names: tuple[str, ...],
    structure: PyTree[jax.ShapeDtypeStruct],
    regularization: float,
) -> tuple['_BorderedGramInverse', PyTree[Array]]:
    """Factor the bordered banded Gram of one stream, and its ridge."""
    # the core's diagonal blocks and the corner are ridged separately, each scaled by its own
    # diagonal
    bands, core_ridge = _regularize_bands(bands, regularization)
    corner, corner_ridge = _regularize_dense(corner, regularization)

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


_BAND_MARGIN = 1
"""Blocks added to the band half-width estimated from the mean block length.

Blocks shorter than the mean, such as a partial scan interval, let a sample couple one more block
than the estimate; the margin absorbs one such block.
"""


def _banded_weight_gram_inverses(
    operator: AbstractTemplateOperator,
    weight: AbstractLinearOperator,
    streams: dict[str | None, dict[str, Basis]],
    regularization: float,
    bandwidth: int,
) -> dict[str | None, tuple[AbstractLinearOperator, PyTree[Array]]]:
    """The inverse Gram of every stream, and its ridge, under a weight of the given bandwidth.

    Each probe sets amplitudes on every stream at once, and its response `Tᵀ W T x` costs one
    application of `W` to the whole TOD, every detector included.
    """

    def part(tree: PyTree[Any], leg: str | None) -> PyTree[Any]:
        return tree if leg is None else _on_leg(tree, streams[leg], leg)

    probes = {
        leg: _Probes.create(bases, part(operator.in_structure, leg), bandwidth)
        for leg, bases in streams.items()
    }

    def respond(p: Array) -> PyTree[Array]:
        x = furax.tree.zeros_like(operator.in_structure)
        for leg, plan in probes.items():
            for name, amps in plan.probe(p).items():
                if leg is None:
                    x[name] = amps
                else:
                    x[name][leg] = amps
        return operator.T(weight(operator(x)))

    n_probes = max(plan.n_probes for plan in probes.values())
    responses = jax.lax.map(respond, jnp.arange(n_probes))
    return {
        leg: plan.gram_inverse(part(responses, leg), regularization) for leg, plan in probes.items()
    }


class _Probes(NamedTuple):
    """Probe amplitudes for the templates on one stream, and how to read their Gram off.

    The response `Tᵀ W T x` to a probe `x` is a sum of Gram columns. A border template is probed
    one amplitude at a time, its response the Gram column itself. A core template is probed one
    sub-basis function at a time, but on every block of one color: the blocks `n_colors` apart.
    A block couples to those at most `half_width` away, so with `n_colors = 2·half_width + 1` the
    response on a block within reach of a colored one holds that block's coupling alone.
    """

    structure: dict[str, jax.ShapeDtypeStruct]
    """The amplitudes of every template on the stream, with a leading detector axis."""
    core: str | None
    border: tuple[str, ...]
    n_blocks: int
    """Blocks of the core template, 0 without one."""
    k: int
    """Sub-basis functions of the core template on each of its blocks."""
    half_width: int
    """Blocks the core couples a block to, on either side."""
    reach: Array
    """The half-width the core actually needs, checked against `half_width`."""

    @classmethod
    def create(
        cls, bases: dict[str, Basis], structure: dict[str, jax.ShapeDtypeStruct], bandwidth: int
    ) -> Self:
        core, border = _layout(bases)
        if core is None:
            return cls(structure, None, border, 0, 0, 0, jnp.array(0))
        basis = bases[core]
        n_blocks = basis._n_blocks
        if n_blocks == 1:
            return cls(structure, core, border, 1, basis.size, 0, jnp.array(0))
        # Static, the half-width is estimated from the mean block length: a sample reaches the
        # blocks of its window, and those started within `bandwidth` samples of it.
        crossed = -(-bandwidth * n_blocks // basis.n_points)
        half_width = min(n_blocks - 1, _support_window(basis) - 1 + crossed + _BAND_MARGIN)
        reach = _support_reach(basis, bandwidth)
        return cls(structure, core, border, n_blocks, basis.size // n_blocks, half_width, reach)

    @property
    def n_colors(self) -> int:
        return min(self.n_blocks, 2 * self.half_width + 1)

    @property
    def n_core(self) -> int:
        """Probes of the core template."""
        return self.n_colors * self.k

    @property
    def n_border(self) -> int:
        """Probes of the border templates: their amplitudes per detector."""
        return sum(prod(self.structure[name].shape[1:]) for name in self.border)

    @property
    def n_probes(self) -> int:
        return self.n_core + self.n_border

    def probe(self, p: Array) -> dict[str, Array]:
        """Probe `p`, the same on every detector; zero past `n_probes`."""
        amps = {}
        if self.core is not None:
            s = self.structure[self.core]
            color, a = p // self.k, p % self.k  # no block has a color past `n_colors`
            blocks = jnp.arange(self.n_blocks) % self.n_colors == color
            pattern = blocks[:, None] & (jnp.arange(self.k) == a)
            amps[self.core] = jnp.broadcast_to(pattern.reshape(s.shape[1:]), s.shape).astype(
                s.dtype
            )
        if self.border:
            unit = jax.nn.one_hot(p - self.n_core, self.n_border)  # zero outside the border probes
            cuts = np.cumsum([prod(self.structure[name].shape[1:]) for name in self.border])[:-1]
            for name, part in zip(self.border, jnp.split(unit, cuts), strict=True):
                s = self.structure[name]
                amps[name] = jnp.broadcast_to(part.reshape(s.shape[1:]), s.shape).astype(s.dtype)
        return amps

    def gram_inverse(
        self, responses: dict[str, Array], regularization: float
    ) -> tuple[AbstractLinearOperator, PyTree[Array]]:
        """The inverse Gram, and its ridge, from the responses to every probe of every stream."""
        # stacked over the probes of every stream: those past this one's are zero
        responses = {name: r[: self.n_probes] for name, r in responses.items()}
        if self.core is None:
            corner = self._border_columns(responses)
            return _dense_inverse(corner, self.structure, regularization)

        bands = self._core_bands(responses[self.core][: self.n_core])
        if not self.border:
            return _banded_inverse(bands, self.core, self.structure, regularization)
        n_dets = bands.shape[0]
        core_columns = responses[self.core][self.n_core :].reshape(self.n_border, n_dets, -1)
        border = jnp.moveaxis(core_columns, 0, -1)  # (det, n·k, m)
        corner = self._border_columns(responses)
        return _bordered_inverse(
            bands, border, corner, self.core, self.border, self.structure, regularization
        )

    def _core_bands(self, responses: Float[Array, 'probe det ...']) -> Float[Array, 'det n w1 k k']:
        """The core's Gram bands: `bands[i, d, a, b]` couples `(i, a)` to `(i + d, b)`."""
        n, k, w1 = self.n_blocks, self.k, self.half_width + 1
        y = responses.reshape(self.n_colors, k, -1, n, k)  # (color, b, det, i, a)
        y = y.transpose(2, 3, 0, 4, 1)  # (det, i, color, a, b)
        i, d = jnp.arange(n)[:, None], jnp.arange(w1)[None, :]
        # block `i + d` has color `(i + d) % n_colors`, the only one of that color within reach
        bands = jnp.where((i + d < n)[..., None, None], y[:, i, (i + d) % self.n_colors], 0)
        # the diagonal blocks are symmetric, up to the rounding of the responses
        diagonal = bands[:, :, 0]
        bands = bands.at[:, :, 0].set((diagonal + jnp.swapaxes(diagonal, -1, -2)) / 2)
        return eqx.error_if(
            bands,
            self.reach > self.half_width,
            'template blocks (scan intervals, spline knots) are too short for the weight '
            'correlation length: a sample couples more blocks than the Gram band can hold',
        )

    def _border_columns(self, responses: dict[str, Array]) -> Float[Array, 'det m m']:
        """The Gram block of the border templates, from their probes."""
        probed = [responses[name][self.n_core :] for name in self.border]
        columns = jnp.concatenate(
            [r.reshape(self.n_border, r.shape[1], -1) for r in probed], axis=-1
        )  # (column, det, row)
        corner = jnp.moveaxis(columns, 0, -1)
        return (corner + jnp.swapaxes(corner, -1, -2)) / 2


def _support_window(basis: Basis) -> int:
    """How many consecutive blocks each sample of `basis` reads."""

    def taps(b: Basis) -> Array:
        return b._map_batch(lambda single, _: single.support().taps, jnp.zeros(b.batch_shape), 0)

    return jax.eval_shape(taps, basis).shape[-1]


def _support_reach(basis: Basis, bandwidth: int) -> Int[Array, '']:
    """How many blocks apart two blocks are that samples at most `bandwidth` apart read.

    Only samples where some function is nonzero count. The largest over a batch of bases. Assumes
    blocks are non-decreasing in time.
    """

    def single(b: Basis, _: Array) -> Array:
        support = b.support()
        n = support.n_points
        active = jnp.any(support.values != 0, axis=0)
        t = jnp.arange(n)
        last = jax.lax.cummax(jnp.where(active, t, -1))  # the last active sample up to each
        ahead = last[jnp.minimum(t + bandwidth, n - 1)]
        span = support.offset[ahead] - support.offset + support.taps.shape[-1] - 1
        return jnp.max(jnp.where(active, span, 0))

    return jnp.max(basis._map_batch(single, jnp.zeros(basis.batch_shape), 0))
