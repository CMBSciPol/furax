r"""The Gram $T^\top W T$ of a template operator, its assembly and its inverse.

Under a weight $W$ diagonal in time and in streams, the Gram is assembled from the structure of the
bases, their weighted Grams ([`Basis.gram`][furax.mapmaking.templates.bases.Basis.gram]).
Otherwise it is assembled by probing: applying $T^\top W T$ to a few probe amplitudes and reading
its entries off the responses. That needs no structure from the
weight beyond its reach: it must not couple detectors, nor samples more than `bandwidth` apart, nor
streams unless declared. Both lay out the Gram alike, the probed bands possibly wider.

The amplitudes split in two. The **local** ones belong to templates with a set per detector
([`Template.per_detector`][furax.mapmaking.templates.Template.per_detector]): a detector's
amplitudes interact with no other detector's, so their Gram is block diagonal, one block per
detector, or per detector and stream when every local template has a set per stream and the weight
does not couple streams. Every other amplitude is **global**, in one block, or one per stream
likewise. Within a block, the time-local template with the most amplitudes keeps a banded Gram,
bordered by the others.

[`factor`][] inverts the Gram block by block, leaving out the coupling between the local and the
global amplitudes.

Templates whose amplitudes are marginalized, rather than solved for, change the weight under which
the others are seen: the Gram of global templates is also assembled under a
[`MarginalizedWeightOperator`][furax.mapmaking.weight.MarginalizedWeightOperator] of local ones.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, PyTree

from furax import AbstractLinearOperator, DiagonalOperator, symmetric, tree
from furax.core import BlockDiagonalOperator
from furax.core.utils import register_dataclass_with_keys
from furax.linalg import (
    BandedCholeskyOperator,
    BorderedBandedMatrix,
    DensePseudoInverseOperator,
    probe_bordered_banded,
)

from ..weight import MarginalizedWeightOperator
from . import _structure
from ._operator import TemplateOperator
from .bases import Basis, OffsetBasis, _einsum
from .pomme import PommeProjectionOperator

__all__ = [
    'BlockGram',
    'factor',
    'template_gram',
]


@register_dataclass_with_keys
@dataclass(frozen=True)
class BlockGram:
    r"""The Gram $T^\top W T$ of a template operator, assembled but not factored.

    `local` holds the blocks of the detectors' own amplitudes, and `global_` those of the others.
    `marginal` is what the global block loses under a marginalized weight, when assembled under one.
    """

    local: '_Part | None'
    global_: '_Part | None'
    marginal: Float[Array, '1 global global'] | None = None
    """Subtracted from the global block, then all border, when factored; with `coarse`, its
    projection onto the coarse amplitudes."""
    coarse: '_Coarse | None' = None
    """The coarse amplitudes of a time-local global block assembled under a marginalized weight,
    which then keeps its band (see [`template_gram`][])."""


def template_gram(
    operator: TemplateOperator,
    weight: AbstractLinearOperator,
    *,
    bandwidth: int = 0,
    streams_coupled: bool = False,
    pomme_tau: int | None = None,
    method: Literal['structure', 'probe'] | None = None,
    batch_size: int = 1,
) -> BlockGram:
    r"""Assemble the Gram $T^\top W T$, or $T^\top W F T$ with the Pomme projector $F$.

    A weight diagonal in time and in streams (`bandwidth=0`, `streams_coupled=False`) is read off by
    applying it to ones, and the Gram assembled from the structure of the bases. Any other weight
    is probed.

    With `pomme_tau`, the Gram is that under $W F$, with $F$ the
    [`PommeProjectionOperator`][furax.mapmaking.templates.pomme.PommeProjectionOperator] removing
    the mean of each interval of `pomme_tau` samples. $W$ must then be constant on each interval, as
    the Pomme path makes it by masking intervals whole: $W F$ is the weight with the interval offsets
    marginalized. Probed, $W F$ is a weight reaching `pomme_tau` samples; from the structure, the
    offsets are eliminated from the Gram. A time-local template next to Pomme must have blocks
    longer than an interval.

    Under a [`MarginalizedWeightOperator`][furax.mapmaking.weight.MarginalizedWeightOperator]
    $W' = W - W T_i G_i^{-1} T_i^\top W$ of templates $T_i$ whose amplitudes are each
    detector's own, the Gram of global templates $T_g$ is

    $$ T_g^\top W' T_g = T_g^\top W T_g - C^\top G_i^{-1} C, $$

    with $C = T_i^\top W T_g$ their coupling. `bandwidth` and `streams_coupled` then describe
    $W$, and `pomme_tau` applies $F$ to it. As the local templates couple all the global amplitudes
    they see, the Gram is one dense block, the subtracted term held apart (`marginal`) so that
    [`factor`][] takes the ridge from $T_g^\top W T_g$. When both the global block and the local
    templates are time-local, the global block keeps its band instead, $T_g^\top W T_g$, and the
    subtracted term is taken on coarse amplitudes only (`coarse`): those whose signal, mode by
    mode, the local templates' functions of time reproduce, which they absorb.

    Args:
        operator: The template operator $T$.
        weight: The weight $W$. It must not couple detectors, nor samples more than `bandwidth`
            apart, nor the streams of a detector unless `streams_coupled`. It may also be such a
            weight with local templates marginalized, when `operator` has global templates only.
        bandwidth: The largest distance, in samples, between two samples that $W$ couples.
        streams_coupled: Whether $W$ couples the streams of a detector, such as Q and U.
        pomme_tau: The length of the Pomme intervals, when the templates are combined with Pomme.
        method: `'structure'` or `'probe'` to choose the assembly; by default, from the
            structure when the weight is diagonal, by probing otherwise.
        batch_size: The number of probes applied at once, or of detectors assembled at once
            from the structure, bounding the memory of the assembly.

    Raises:
        TypeError: If the templates of a marginalized weight are not a [`TemplateOperator`][].
        ValueError: If the weight is marginalized and `operator` has local templates. If the assembly from the structure is asked for a weight that is not
            diagonal, or Pomme for a demodulated TOD, or, from the structure, Pomme next to a
            time-local template whose `block_lengths` do not state blocks at least an interval
            long.
    """
    marginalized = None
    if isinstance(weight, MarginalizedWeightOperator):
        marginalized, weight = weight, weight.weight
        if not isinstance(marginalized.templates, TemplateOperator):
            raise TypeError('a marginalized weight takes its templates as a TemplateOperator')
        if any(t.per_detector for t in operator.templates.values()):
            raise ValueError('under a marginalized weight, the Gram is that of global templates')
        # the marginalized templates join the operator, so that their coupling is assembled
        operator = TemplateOperator(marginalized.templates.templates | operator.templates)
    tod_format = operator.tod_format
    pomme = None
    if pomme_tau is not None:
        if tod_format.stokes is not None:
            raise ValueError('Pomme filtering applies to a TOD that is not demodulated')
        pomme = OffsetBasis(pomme_tau, tod_format.n_samples, tod_format.dtype)
    diagonal = bandwidth == 0 and not streams_coupled
    if method is None:
        method = 'structure' if diagonal else 'probe'
    if method == 'structure' and not diagonal:
        raise ValueError('the Gram is assembled from the structure for a diagonal weight only')
    local = tuple(name for name, t in operator.templates.items() if t.per_detector)
    rest = tuple(name for name in operator.templates if name not in local)
    local_layout = global_layout = None
    if local:
        local_layout = _Layout.of(
            operator, local, local=True, dense=False, streams_coupled=streams_coupled
        )
    if rest:
        global_layout = _Layout.of(
            operator,
            rest,
            local=False,
            dense=marginalized is not None,
            streams_coupled=streams_coupled,
        )

    if method == 'structure' and pomme is not None:
        # an interval straddles two blocks of a time-local core at most, its band one wider
        for layout in (local_layout, global_layout):
            if layout is not None and layout.core is not None:
                lengths = operator.templates[layout.core.name].basis.block_lengths
                if lengths is None or lengths[0] < pomme.tau - 1:
                    msg = (
                        f'template {layout.core.name!r} must have blocks of at least '
                        f'{pomme.tau - 1} samples, as stated by its block_lengths, to be combined '
                        f'with Pomme intervals of {pomme.tau}'
                    )
                    raise ValueError(msg)

    diagonal_weights = None
    if method == 'structure':
        weights = tod_format.as_array(weight(tod_format.ones()))
        diagonal_weights = tuple(
            weights.reshape(tod_format.n_streams, tod_format.n_dets, tod_format.n_samples)
        )
    elif pomme is not None:  # W F couples the samples of an interval
        weight = weight @ PommeProjectionOperator(pomme.tau, in_structure=tod_format.structure)
        bandwidth = max(bandwidth, pomme.tau - 1)

    def assemble(layout, reading):
        if diagonal_weights is None:
            matrix, cross = _probe(operator, weight, layout, reading, bandwidth, batch_size)
        else:
            matrix, cross = _structure.assemble(
                operator, diagonal_weights, layout, reading, batch_size, pomme
            )
        return _Part(matrix, layout), cross

    if marginalized is not None:  # the local block is that of the marginalized templates
        assert global_layout is not None and local_layout is not None
        # a single unit, as the dense layout, but with the time-local core of the global block
        banded = _Layout.of(operator, rest, local=False, dense=False, streams_coupled=True)
        time_local = _time_local(operator, banded, local_layout)
        # The coarse amplitudes need the block probed under `weight` itself, which carries the
        # Pomme projector wherever Pomme applies, except on the structure path.
        coarse = None
        if time_local and (diagonal_weights is None or pomme is None):
            coarse = _coarse_projection(operator, banded, local_layout, batch_size)
        if coarse is not None or (time_local and diagonal_weights is None):
            dense = None if coarse is not None else global_layout
            matrix, cross = _probe_time_local(
                operator, weight, banded, local_layout, dense, bandwidth, batch_size
            )
            global_part = _Part(matrix, global_layout if dense else banded)
        else:
            global_part, rows = assemble(global_layout, local_layout)
            assert rows is not None
            cross = _DenseCoupling(rows)
        marginal = _marginal_term(
            cross, local_layout, marginalized.gram_inverse, batch_size, coarse
        )
        return BlockGram(None, global_part, marginal, coarse)

    local_part = global_part = None
    if local_layout is not None:
        local_part, _ = assemble(local_layout, None)
    if global_layout is not None:
        global_part, _ = assemble(global_layout, None)
    return BlockGram(local_part, global_part)


def factor(
    gram: BlockGram, *, regularization: float = 0.0, batch_size: int = 1
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    r"""The inverse of the Gram, and the ridge added to it before factoring.

    The Gram is inverted block by block, the coupling of the local and global amplitudes left out:
    the inverse preconditions their joint system. A global block assembled under a marginalized weight is inverted as one dense block,
    its marginal term subtracted: the directions the marginalized templates absorb, held only by
    the ridge, are inverted at the ridge, or at the rounding error of the subtraction if larger,
    so that the inverse still preconditions a system that holds them. With coarse amplitudes, the
    block keeps its band, factored by banded Cholesky, and only the coarse amplitudes $\Pi$ are
    inverted dense: the inverse is that of the band plus $\Pi E^{-1} \Pi^\top$, with
    $E = \Pi^\top S \Pi$ the marginalized Gram $S$ on them, a two-level preconditioner.
    Amplitudes that no weighted sample sees, whose rows are zero, take a unit diagonal and come
    back unchanged. The ridge is `regularization` times the mean diagonal of each core block, and
    times its own diagonal entry for each border amplitude, so that it is in the units of the
    amplitudes it regularizes: those of a core block are of one basis, while the border may gather
    several templates. It is taken before the marginal term is subtracted.

    Args:
        gram: The assembled Gram.
        regularization: The relative ridge of each block.
        batch_size: The number of coarse amplitudes whose Gram columns are assembled at once.

    Returns:
        The inverse, on the amplitudes, and the ridge as a diagonal operator on them; `None`
        without regularization.
    """
    parts: list[_FactorPart] = []
    local = None
    ridges: dict[str, Array] = {}
    if gram.local is not None:
        matrix, ridge = _regularized(gram.local.matrix, regularization)
        local = _FactorPart(matrix.cholesky(), gram.local.layout)
        ridges |= gram.local.layout.scatter(*ridge)
    if gram.global_ is not None:
        matrix, ridge = _regularized(gram.global_.matrix, regularization)
        ridges |= gram.global_.layout.scatter(*ridge)
        if gram.coarse is not None:
            assert gram.marginal is not None
            inverse = _two_level(matrix, gram.marginal, gram.coarse, regularization, batch_size)
            parts.append(_FactorPart(inverse, gram.global_.layout))
        elif gram.marginal is not None:
            assert matrix.corner is not None  # a marginalized block is all border
            structure = BorderedBandedMatrix(corner=matrix.corner).in_structure
            inverse = _absorbed_inverse(matrix.corner, gram.marginal, regularization, structure)
            parts.append(_FactorPart(inverse, gram.global_.layout))
        else:
            parts.append(_FactorPart(matrix.cholesky(), gram.global_.layout))
    if local is not None:
        parts.append(local)

    # shapes and dtypes only: the ridges may be laid out over a mesh the inverse is not used on
    structure = {name: jax.ShapeDtypeStruct(r.shape, r.dtype) for name, r in ridges.items()}
    inverse = _GramInverse(tuple(parts), in_structure=structure)
    if not regularization:
        return inverse, None
    return inverse, _ridge_operator(ridges, structure)


def _marginal_term(
    coupling: '_DenseCoupling | _BandedCoupling',
    layout: '_Layout',
    gram_inverse: AbstractLinearOperator,
    batch_size: int,
    coarse: '_Coarse | None' = None,
) -> Float[Array, '1 global global']:
    r"""$C^\top G_i^{-1} C$, from the coupling $C$ to the amplitudes laid out by `layout`.

    It is built a column at a time, `batch_size` at once: $C^\top G_i^{-1} C_q$ for the coupling
    $C_q$ of each global amplitude, which only the columns of a batch hold whole. With `coarse`,
    it is $\Pi^\top C^\top G_i^{-1} C \Pi$ on the coarse amplitudes $\Pi$ instead, a column
    per coarse amplitude, the coupling then in the order of the banded layout.
    """

    def solve(local):
        solved = gram_inverse(layout.scatter(*layout.unflat(local)))
        return coupling.contract(layout.flat(*layout.gather(solved)))

    if coarse is None:
        rows = jnp.arange(coupling.n_rows)
        return tree.stacked_map(lambda q: solve(coupling.row(q)), rows, batch_size=batch_size)[None]

    assert isinstance(coupling, _BandedCoupling)  # coarse amplitudes come with the colored probes
    n_core = coarse.n_blocks * coarse.k

    def apply(core):
        border = jnp.zeros(coupling.n_rows - n_core, core.dtype)
        out = solve(coupling.apply(jnp.concatenate([core.ravel(), border])))
        return out[:n_core].reshape(coarse.n_blocks, coarse.k)

    return coarse.restrict(apply, batch_size)[None]


class _DenseCoupling(NamedTuple):
    """The coupling $C$ of global amplitudes to local ones, held whole."""

    rows: Float[Array, 'global unit local']

    @property
    def n_rows(self) -> int:
        return self.rows.shape[0]

    def row(self, p: Int[Array, '']) -> Float[Array, 'unit local']:
        """The coupling of the global amplitude `p`."""
        return self.rows[p]

    def contract(self, v: Float[Array, 'unit local']) -> Float[Array, ' global']:
        r"""$C^\top v$, from local amplitudes to global ones."""
        return _einsum('pui,ui->p', self.rows, v)


class _BandedCoupling(NamedTuple):
    """The coupling $C$ of a time-local global block to time-local amplitudes, held by color.

    The responses to the colored probes of the global core (see `_probe_time_local`): a local
    block's response to a color is its coupling to the one global block of that color it reaches,
    if any. The global amplitudes are those of the core, block by block, then of the border, in
    the order `order` gives.
    """

    core: Float[Array, 'color k unit local_block local_k']
    reached: Int[Array, 'color local_block']
    """The global block of each color a local block reaches, `n_blocks` for none."""
    border: Float[Array, 'border unit local']
    order: Int[Array, ' global']
    """Where each global amplitude is among the core ones, then the border ones."""

    @property
    def n_rows(self) -> int:
        return self.order.shape[0]

    def row(self, p: Int[Array, '']) -> Float[Array, 'unit local']:
        """The coupling of the global amplitude `p`."""
        n_colors, k, n_units = self.core.shape[:3]
        i = self.order[p]
        block = i // k
        color = block % n_colors
        reaches = (self.reached[color] == block)[None, :, None]
        row = jnp.where(reaches, self.core[color, i % k], 0).reshape(n_units, -1)
        if n_border := self.border.shape[0]:
            n_core = self.n_rows - n_border
            border = self.border[jnp.clip(i - n_core, 0, n_border - 1)]
            row = jnp.where(i < n_core, row, border)
        return row

    def apply(self, v: Float[Array, ' global']) -> Float[Array, 'unit local']:
        """$C v$, from global amplitudes to local ones."""
        _, k, n_units, _, _ = self.core.shape
        n_border = self.border.shape[0]
        n_blocks = (self.n_rows - n_border) // k
        banded = jnp.zeros_like(v).at[self.order].set(v)
        core = jnp.concatenate(
            [banded[: n_blocks * k].reshape(n_blocks, k), jnp.zeros((1, k), v.dtype)]
        )
        reached = core[self.reached]  # (color, local block, k), zero where none is reached
        out = _einsum('caujl,cja->ujl', self.core, reached).reshape(n_units, -1)
        return out + _einsum('rui,r->ui', self.border, banded[n_blocks * k :])

    def contract(self, v: Float[Array, 'unit local']) -> Float[Array, ' global']:
        r"""$C^\top v$, from local amplitudes to global ones."""
        _, k, n_units, n_local, local_k = self.core.shape
        n_blocks = (self.n_rows - self.border.shape[0]) // k
        by_color = _einsum('caujl,ujl->caj', self.core, v.reshape(n_units, n_local, local_k))
        core = jnp.zeros((n_blocks, k), v.dtype)
        core = core.at[self.reached[:, None, :], jnp.arange(k)[:, None]].add(by_color, mode='drop')
        border = _einsum('rui,ui->r', self.border, v)
        return jnp.concatenate([core.ravel(), border])[self.order]


@register_dataclass_with_keys
@dataclass(frozen=True)
class _Coarse:
    r"""Coarse amplitudes $\Pi$ of the time-local core of a global block.

    Coarse function `j` weighs the core blocks from `start[j]` on by `band[j]`, and each of the
    `k` amplitudes of a block alike: a coarse amplitude is a pair (function, `k` index). The border
    has none.
    """

    start: Int[Array, ' coarse']
    band: Float[Array, 'coarse width']
    n_blocks: int = field(metadata={'static': True})
    k: int = field(metadata={'static': True})

    @property
    def size(self) -> int:
        """The number of coarse amplitudes."""
        return self.start.shape[0] * self.k

    def lift(self, y: Float[Array, 'coarse k']) -> Float[Array, 'block k']:
        r"""$\Pi y$, on the core."""
        core = jnp.zeros((self.n_blocks, self.k), y.dtype)
        return core.at[self._blocks].add(self.band[..., None] * y[:, None, :])

    def project(self, x: Float[Array, 'block k']) -> Float[Array, 'coarse k']:
        r"""$\Pi^\top x$, from the core."""
        return _einsum('jw,jwa->ja', self.band, x[self._blocks])

    def column(self, c: Int[Array, '']) -> Float[Array, 'block k']:
        r"""$\Pi$ of the coarse amplitude `c`, on the core: its function's band, on its index."""
        j, a = c // self.k, c % self.k
        band = jnp.zeros((self.band.shape[1], self.k), self.band.dtype).at[:, a].set(self.band[j])
        core = jnp.zeros((self.n_blocks, self.k), self.band.dtype)
        return jax.lax.dynamic_update_slice_in_dim(core, band, self.start[j], axis=0)

    def restrict(
        self, apply: Callable[[Array], Array], batch_size: int
    ) -> Float[Array, 'coarse coarse']:
        r"""$\Pi^\top A \Pi$, for `apply` the action of $A$ on the core, a batch of columns at once."""

        def column(c):
            return self.project(apply(self.column(c))).ravel()

        return tree.stacked_map(column, jnp.arange(self.size), batch_size=batch_size)

    @property
    def _blocks(self) -> Int[Array, 'coarse width']:
        """The core blocks of each coarse function's band."""
        return self.start[:, None] + jnp.arange(self.band.shape[1])


def _coarse_projection(
    operator: TemplateOperator,
    layout: '_Layout',
    reading: '_Layout',
    batch_size: int,
) -> _Coarse | None:
    """The coarse amplitudes of the time-local core of `layout` that the core of `reading` absorbs.

    The functions of time of the local core, projected by least squares onto the basis of the
    global core: a coarse function each, which the global core synthesizes, mode by mode, as close
    to that local function as it can, so that the local templates of every detector absorb it.
    Each is truncated to the global blocks its local function overlaps, two more on either side,
    a band of a static width bounded by the `block_lengths` of both bases. `None` unless the
    global core has a single function per block and both bases one function for every detector
    and stated block lengths.
    """
    assert layout.core is not None and reading.core is not None
    global_ = operator.templates[layout.core.name].basis
    local = operator.templates[reading.core.name].basis
    if global_.batch_shape or local.batch_shape or global_.shape != (global_._n_blocks,):
        return None
    if global_.block_lengths is None or local.block_lengths is None:
        return None
    margin = 2  # global blocks either side of those a local function overlaps
    n_blocks = global_._n_blocks
    width = min(n_blocks, _reach_bound(local, global_, 0) + 2 * margin)

    ones = jnp.ones(global_.n_points, global_.dtype)
    cross = _structure._cross_gram(global_, local, ones)  # (block, function)
    # a unit diagonal on the blocks no sample reads
    gram, _ = _regularized(BorderedBandedMatrix(bands=global_.gram(ones)), 0.0)
    assert gram.bands is not None
    solve = BandedCholeskyOperator.from_bands(gram.bands)

    def function(reads):
        solved = solve(reads[:, None])[:, 0]
        start = jnp.clip(jnp.argmax(reads != 0) - margin, 0, n_blocks - width)
        return start, jax.lax.dynamic_slice_in_dim(solved, start, width)

    start, band = tree.stacked_map(function, cross.T, batch_size=batch_size)
    return _Coarse(start.astype(jnp.int32), band, n_blocks, layout.k)


def _two_level(
    matrix: BorderedBandedMatrix,
    marginal: Float[Array, '1 coarse coarse'],
    coarse: _Coarse,
    regularization: float,
    batch_size: int,
) -> '_TwoLevelInverse':
    """The two-level inverse of a banded block whose marginalized Gram is projected on `coarse`.

    `matrix` is the regularized block under the plain weight, `marginal` what the marginalized
    weight takes from it on the coarse amplitudes.
    """
    border = None
    if matrix.border is not None:
        border = jnp.zeros((1, matrix.border.shape[-1]), matrix.border.dtype)

    def apply(core):
        return matrix((core[None], border))[0][0]

    plain = coarse.restrict(apply, batch_size)
    correction = _absorbed_inverse(plain[None], marginal, regularization)
    return _TwoLevelInverse(matrix.cholesky(), coarse, correction, in_structure=matrix.in_structure)


def _absorbed_inverse(
    plain: Float[Array, '1 n n'],
    marginal: Float[Array, '1 n n'],
    regularization: float,
    in_structure: PyTree[jax.ShapeDtypeStruct] | None = None,
) -> DensePseudoInverseOperator:
    """The inverse of `plain - marginal`, a regularized Gram less what a marginalized weight takes.

    The marginal term cancels the Gram on the directions it absorbs, up to a rounding error
    amplified by the conditioning of the marginalized Gram: they are told apart by the weight left
    of each amplitude. Normalized by the diagonal of `plain`, the ridge is `r / (1 + r)` on every
    amplitude, a lower bound of the eigenvalues that holds wherever rounding resolves it: those
    directions are inverted there, or at the rounding error if larger.
    """
    dense = plain - marginal
    dense = (dense + jnp.swapaxes(dense, -1, -2)) / 2  # symmetric up to rounding
    scale = jnp.diagonal(plain, axis1=-2, axis2=-1)
    rounding = float(jnp.finfo(dense.dtype).eps) ** 0.5
    return DensePseudoInverseOperator.from_dense(
        dense,
        in_structure,
        scale=jnp.where(scale > 0, scale, 1),  # an amplitude no sample sees
        rtol=max(rounding, regularization / (1 + regularization)),
        clamp=True,
    )


def _regularized(
    matrix: BorderedBandedMatrix, regularization: float
) -> tuple[BorderedBandedMatrix, tuple[Array | None, Array | None]]:
    """A block of the Gram made factorable, and the ridge added to it (see [`factor`][]).

    Returns:
        The regularized block, and the ridge on its core and border, shaped like them.
    """
    # A positive semi-definite matrix has a zero row exactly where its diagonal is zero.
    bands, corner = matrix.bands, matrix.corner
    ridge_a = ridge_c = None
    if bands is not None:
        entries = jnp.diagonal(bands[..., 0, :, :], axis1=-2, axis2=-1)  # (*units, n, k)
        scale = jnp.mean(entries, axis=-1, keepdims=True)
        ridge_a = jnp.where(entries == 0, 0, regularization * scale)
        bands = bands.at[..., 0, :, :].add(_diag((entries == 0) + ridge_a))
    if corner is not None:
        entries = jnp.diagonal(corner, axis1=-2, axis2=-1)
        ridge_c = regularization * entries
        corner = corner + _diag((entries == 0) + ridge_c)
    return BorderedBandedMatrix(bands, matrix.border, corner), (ridge_a, ridge_c)


def _diag(values: Float[Array, '*batch k']) -> Float[Array, '*batch k k']:
    return values[..., None] * jnp.eye(values.shape[-1], dtype=values.dtype)


@register_dataclass_with_keys
@dataclass(frozen=True)
class _Part:
    """A block of the Gram, and the layout of its amplitudes."""

    matrix: BorderedBandedMatrix
    layout: '_Layout' = field(metadata={'static': True})


class _Layout(NamedTuple):
    """How the amplitudes of some templates are laid out in one block of the Gram.

    The block is split into units that no two templates of it, nor the weight, couple: one per
    detector or a single one, times one per stream of the TOD if `per_stream`. A template that
    covers only some streams has no amplitudes on the units of the others: they are padded with
    zeros there. Per unit, the amplitudes form a core `(n_blocks, k)`, those of the time-local
    template with the most amplitudes, its time blocks first, then a border `(n_border,)`, those of
    the other templates, flattened in turn.
    """

    local: bool
    """Whether the block has one unit per detector, rather than a single one."""
    per_stream: bool
    """Whether the block has one unit per stream."""
    n_streams: int
    """The number of streams of the TOD."""
    core: '_Entry | None'
    border: tuple['_Entry', ...]

    @classmethod
    def of(
        cls,
        operator: TemplateOperator,
        names: tuple[str, ...],
        *,
        local: bool,
        dense: bool,
        streams_coupled: bool,
    ) -> Self:
        """The layout of the templates `names` in one block; all in the border if `dense`.

        The streams are units of their own if every template has its own amplitudes on each stream,
        the weight does not couple them, and the block is not laid out dense.
        """
        templates = {name: operator.templates[name] for name in names}
        per_stream = (
            not streams_coupled
            and not dense
            and all('stream' in template.per for template in templates.values())
        )
        entries = {}
        for name, template in templates.items():
            public = operator.in_structure[name].shape
            canonical = template._canonical_shape
            if local:  # the detectors, along the first axis, vary from one batch to another
                axis = template.per.index('det')
                public = public[:axis] + (-1,) + public[axis + 1 :]
                canonical = canonical[:1] + (-1,) + canonical[2:]
            entries[name] = _Entry(name, public, canonical, template._stream_indices)
        time_local = [n for n, t in templates.items() if not dense and t.basis._n_blocks > 1]
        core = None
        if time_local:
            name = max(time_local, key=lambda name: abs(prod(entries[name].canonical)))
            basis = templates[name].basis
            # the time blocks lead the amplitudes of a time-local basis
            assert basis.shape[0] == basis._n_blocks, f'{name!r}: no leading axis of time blocks'
            core = entries.pop(name)
        return cls(local, per_stream, operator.tod_format.n_streams, core, tuple(entries.values()))

    @property
    def entries(self) -> tuple['_Entry', ...]:
        """The templates of the block, in the order of its amplitudes: the core, then the border."""
        return (() if self.core is None else (self.core,)) + self.border

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def n_blocks(self) -> int:
        """The number of time blocks of the core; 0 without one."""
        return 0 if self.core is None else self.core.canonical[2]

    @property
    def k(self) -> int:
        return 0 if self.core is None else prod(self.unit(self.core)) // self.n_blocks

    @property
    def n_border(self) -> int:
        return sum(prod(self.unit(entry)) for entry in self.border)

    @property
    def block_axis(self) -> int:
        """The axis of the time blocks in a template's amplitudes per unit, the units first."""
        return 3 - self.local - self.per_stream  # (unit, [stream,] [det,] block, ...)

    def unit(self, entry: '_Entry') -> tuple[int, ...]:
        """The shape of one unit's amplitudes of a template: its canonical ones, per unit."""
        shape = entry.canonical
        if self.local:
            shape = shape[:1] + shape[2:]
        return shape[1:] if self.per_stream else shape

    def n_units(self, n_dets: int) -> int:
        return (n_dets if self.local else 1) * (self.n_streams if self.per_stream else 1)

    def gather(self, amplitudes: dict[str, Array]) -> tuple[Array | None, Array | None]:
        """The block's amplitudes, per unit: the core `(unit, n, k)` and the border `(unit, m)`."""
        core = border = None
        if self.core is not None:
            units = self.to_units(self.core, amplitudes[self.core.name])
            units = jnp.moveaxis(units, self.block_axis, 1)
            core = units.reshape(units.shape[0], self.n_blocks, self.k)
        if self.border:
            border = tree.ravel([self.to_units(e, amplitudes[e.name]) for e in self.border], 1)
        return core, border

    def scatter(self, core: Array | None, border: Array | None) -> dict[str, Array]:
        """The transpose, and inverse, of `gather`: the amplitudes of the block's templates."""
        out = {}
        if core is not None and self.core is not None:
            unit = self.unit(self.core)
            axis = self.block_axis - 1  # in the unit's shape
            moved = (core.shape[0], unit[axis], *unit[:axis], *unit[axis + 1 :])
            units = jnp.moveaxis(core.reshape(moved), 1, axis + 1)
            out[self.core.name] = self.from_units(self.core, units)
        if border is not None:
            units = [jax.ShapeDtypeStruct(self.unit(e), border.dtype) for e in self.border]
            for entry, unit in zip(self.border, tree.unravel(units, border), strict=True):
                out[entry.name] = self.from_units(entry, unit)
        return out

    @staticmethod
    def flat(core: Array | None, border: Array | None) -> Array:
        """Each unit's amplitudes as one vector, the core then the border."""
        if core is None:
            assert border is not None
            return border
        return tree.ravel((core, border), core.ndim - 2)

    def unflat(self, flat: Array) -> tuple[Array | None, Array | None]:
        """The inverse of `flat`."""
        core = border = None
        if self.core is not None:
            core = jax.ShapeDtypeStruct((self.n_blocks, self.k), flat.dtype)
        if self.n_border:
            border = jax.ShapeDtypeStruct((self.n_border,), flat.dtype)
        return tree.unravel((core, border), flat)

    def to_units(self, entry: '_Entry', a: Array) -> Array:
        """A template's amplitudes, canonical, the units first: `(unit, *self.unit(entry))`."""
        canonical = a.reshape(entry.canonical)
        if self.per_stream:  # on every stream of the TOD, zero on those the template does not cover
            padded = jnp.zeros((self.n_streams, *canonical.shape[1:]), canonical.dtype)
            canonical = padded.at[jnp.array(entry.streams)].set(canonical)
        if not self.local:
            return canonical if self.per_stream else canonical[None]
        units = jnp.moveaxis(canonical, 1, 0)  # (det, stream, ...)
        return units.reshape(-1, *units.shape[2:]) if self.per_stream else units

    def from_units(self, entry: '_Entry', units: Array) -> Array:
        """The inverse of `to_units`, and its transpose but for the padding."""
        if self.local:
            if self.per_stream:
                units = units.reshape(-1, self.n_streams, *units.shape[1:])
            canonical = jnp.moveaxis(units, 0, 1)
        else:
            canonical = units if self.per_stream else units[0]
        if self.per_stream:
            canonical = canonical[jnp.array(entry.streams)]
        return canonical.reshape(entry.public)


class _Entry(NamedTuple):
    """The amplitude shapes of one template of a block; -1 for the detectors of a local block."""

    name: str
    public: tuple[int, ...]
    canonical: tuple[int, ...]
    """`(stream set, detector set, *basis.shape, mode)`: the public amplitudes, reshaped."""
    streams: tuple[int, ...]
    """The indices along the TOD's stream axis of the streams the template covers."""


def _probe(
    operator: TemplateOperator,
    weight: AbstractLinearOperator,
    layout: _Layout,
    reading: _Layout | None,
    bandwidth: int,
    batch_size: int,
) -> tuple[BorderedBandedMatrix, Array | None]:
    """One block of the Gram, read off the responses to its probes.

    Every unit takes the same probe, as no two units interact (see
    [`probe_bordered_banded`][furax.linalg.probe_bordered_banded]). With `reading`, the responses
    on the local amplitudes of that layout are read too: the coupling of each unit's amplitudes to
    the block's, the block being all border, so probed one amplitude at a time.
    """
    apply = _probe_apply(operator, weight, layout, reading)
    half_width = _band_half_width(operator, layout, bandwidth)
    matrix, cross = probe_bordered_banded(
        apply,
        batch_shape=(layout.n_units(operator.tod_format.n_dets),),
        n_blocks=layout.n_blocks,
        k=layout.k,
        n_border=layout.n_border,
        half_width=half_width,
        dtype=operator.tod_format.dtype,
        batch_size=batch_size,
    )
    return matrix, cross


def _probe_apply(
    operator: TemplateOperator,
    weight: AbstractLinearOperator,
    layout: _Layout,
    reading: _Layout | None,
) -> Callable[
    [tuple[Array | None, Array | None]], tuple[tuple[Array | None, Array | None], Array | None]
]:
    """The Gram's response to probes of the amplitudes of `layout`, on those and on `reading`."""
    # the probes need only the templates whose responses are read
    names = layout.names + (() if reading is None else reading.names)
    probed = TemplateOperator({name: operator.templates[name] for name in names})
    gram = probed.T @ weight @ probed
    zeros = tree.zeros_like(probed.in_structure)

    def apply(x):
        response = gram(zeros | layout.scatter(*x))
        local = None if reading is None else reading.flat(*reading.gather(response))
        return layout.gather(response), local

    return apply


def _band_half_width(operator: TemplateOperator, layout: _Layout, bandwidth: int) -> int:
    """How many blocks of the core of `layout` samples up to `bandwidth` apart couple, either side.

    Samples that far apart start their windows at most `⌈bandwidth / fewest⌉` blocks apart, with
    `fewest` the shortest block between the first and the last (see `Basis.block_lengths`).
    Without block lengths, any two blocks may couple.
    """
    if layout.core is None:
        return 0
    basis = operator.templates[layout.core.name].basis
    if basis.block_lengths is None:
        return layout.n_blocks - 1
    fewest, _ = basis.block_lengths
    return min(layout.n_blocks - 1, basis._support_window - 1 + -(-bandwidth // fewest))


def _time_local(operator: TemplateOperator, layout: _Layout, reading: _Layout) -> bool:
    """Whether the coupling of the global block of `layout` to `reading` is probed by color.

    It is when both are time-local, the local block all core, and the global core one basis.
    """
    if layout.core is None or reading.core is None or reading.border:
        return False
    return not operator.templates[layout.core.name].basis.batch_shape


def _probe_time_local(
    operator: TemplateOperator,
    weight: AbstractLinearOperator,
    layout: _Layout,
    reading: _Layout,
    dense: _Layout | None,
    bandwidth: int,
    batch_size: int,
) -> tuple[BorderedBandedMatrix, '_BandedCoupling']:
    """A time-local global block of the Gram, and its coupling to time-local amplitudes.

    The block is dense in the layout `dense`, or banded in `layout` without one, its coupling
    that to the local amplitudes of `reading`, both read off colored probes of the global core.
    A local time block couples to the global blocks read by samples within `bandwidth` of those
    reading it: a few consecutive ones. With the probes of the global core spread so that no two
    blocks of a color lie within the reach of one local block, the response of a local block to a
    probe is its coupling to the one block of that color it reaches. The probes are those of the
    band of the global block, with that many colors, rather than one per global amplitude.
    """
    apply = _probe_apply(operator, weight, layout, reading)
    assert layout.core is not None and reading.core is not None
    half_width = _band_half_width(operator, layout, bandwidth)
    global_basis = operator.templates[layout.core.name].basis
    local_basis = operator.templates[reading.core.name].basis
    first, last = _reach(local_basis, global_basis, bandwidth)
    n_colors = min(
        layout.n_blocks, max(2 * half_width + 1, _reach_bound(local_basis, global_basis, bandwidth))
    )

    n_units = reading.n_units(operator.tod_format.n_dets)
    dtype = operator.tod_format.dtype
    matrix, responses = probe_bordered_banded(
        apply,
        batch_shape=(1,),
        n_blocks=layout.n_blocks,
        k=layout.k,
        n_border=layout.n_border,
        half_width=half_width,
        dtype=dtype,
        batch_size=batch_size,
        n_colors=n_colors,
        all_extras=True,
    )

    # A local block's response to the probe of a color is its coupling to the block of that color
    # it reaches, if any.
    n_core = n_colors * layout.k
    core = responses[:n_core].reshape(n_colors, layout.k, n_units, reading.n_blocks, reading.k)
    reached = first + (jnp.arange(n_colors)[:, None] - first) % n_colors  # (color, local block)
    reached = jnp.where(reached <= last, reached, layout.n_blocks)  # none
    border = responses[n_core:]  # probed one at a time

    if dense is None:
        order = jnp.arange(layout.n_blocks * layout.k + layout.n_border)
        return matrix, _BandedCoupling(core, reached, border, order)
    # in the order of the dense layout, its global block all border
    order = _order(layout, dense)
    corner = matrix.as_matrix()[order][:, order]
    coupling = _BandedCoupling(core, reached, border, order)
    return BorderedBandedMatrix(corner=corner[None]), coupling


def _reach_bound(local: Basis, global_: Basis, bandwidth: int) -> int:
    """At most how many blocks of `global_` one block of `local` reaches (see [`_reach`][]).

    The samples reading a block of `local` are those of the `window` runs whose windows contain
    it, at most `window · most` samples; those up to `bandwidth` from them span `bandwidth` more on
    either side, over which the windows of `global_` start at most `⌈(span − 1) / fewest⌉` blocks
    apart (see `Basis.block_lengths`). Without block lengths, all of them.
    """
    if local.block_lengths is None or global_.block_lengths is None:
        return global_._n_blocks
    _, most = local.block_lengths
    fewest, _ = global_.block_lengths
    span = local._support_window * most + 2 * bandwidth
    return -(-(span - 1) // fewest) + global_._support_window


def _reach(
    local: Basis, global_: Basis, bandwidth: int
) -> tuple[Int[Array, ' block'], Int[Array, ' block']]:
    """The first and the last block of `global_` each block of `local` reaches.

    A block reaches those read by samples up to `bandwidth` apart from those reading it, over the
    batch of `local` if it has one. Blocks are non-decreasing in time.
    """
    support = global_.support()
    n = support.n_points
    t = jnp.arange(n)
    lowest = support.offset[jnp.maximum(t - bandwidth, 0)]
    highest = support.offset[jnp.minimum(t + bandwidth, n - 1)] + support.taps.shape[-1] - 1

    def single(basis: Basis, _: Array) -> Array:
        own = basis.support()
        blocks = own.offset[:, None] + jnp.arange(own.taps.shape[-1])  # (samp, window)
        blocks = jnp.where(own.taps != 0, blocks, own.n_blocks).ravel()  # a block of no sample
        n_segments = own.n_blocks + 1
        low = jnp.broadcast_to(lowest[:, None], own.taps.shape).ravel()
        high = jnp.broadcast_to(highest[:, None], own.taps.shape).ravel()
        first = jax.ops.segment_min(low, blocks, num_segments=n_segments)[:-1]
        last = jax.ops.segment_max(high, blocks, num_segments=n_segments)[:-1]
        return jnp.stack([first, last])

    ends = local._map_batch(single, jnp.zeros(local.batch_shape), 0)  # (*batch, 2, block)
    ends = ends.reshape(-1, *ends.shape[-2:])
    return ends[:, 0].min(axis=0), ends[:, 1].max(axis=0)


def _order(layout: _Layout, dense: _Layout) -> Int[Array, ' m']:
    """Where each amplitude of `dense`, all border in a single unit, is in `layout.flat`."""
    n = layout.n_blocks * layout.k + layout.n_border
    amplitudes = layout.scatter(*layout.unflat(jnp.arange(n, dtype=jnp.float64)[None]))
    _, border = dense.gather(amplitudes)
    assert border is not None
    return border[0].astype(jnp.int32)


@register_dataclass_with_keys
@dataclass(frozen=True)
class _FactorPart:
    """The inverse of one block, on its core and border per unit, and their layout."""

    inverse: AbstractLinearOperator
    layout: _Layout = field(metadata={'static': True})

    def solve(self, x: dict[str, Array]) -> dict[str, Array]:
        return self.layout.scatter(*self.inverse(self.layout.gather(x)))


@symmetric
class _GramInverse(AbstractLinearOperator):
    """The inverse of a block Gram, on the amplitudes of its templates."""

    parts: tuple[_FactorPart, ...]
    """The inverted blocks."""

    def mv(self, x: dict[str, Array]) -> dict[str, Array]:
        out: dict[str, Array] = {}
        for part in self.parts:
            out |= part.solve(x)
        return {name: out[name] for name in x}


@symmetric
class _TwoLevelInverse(AbstractLinearOperator):
    r"""$B^{-1} + \Pi E^{-1} \Pi^\top$, on the core and border of a single unit.

    $B^{-1}$ inverts the band of a block, and $E^{-1}$ the marginalized Gram on its coarse
    amplitudes $\Pi$, which the band holds as well determined although the marginalized templates
    absorb them.
    """

    band: AbstractLinearOperator
    coarse: _Coarse
    correction: DensePseudoInverseOperator
    """$E^{-1}$, on the coarse amplitudes, flat."""

    def mv(self, x: tuple[Array, Array | None]) -> tuple[Array, Array | None]:
        core, border = self.band(x)
        y = self.correction(self.coarse.project(x[0][0]).reshape(1, -1))[0]
        return core + self.coarse.lift(y.reshape(-1, self.coarse.k))[None], border


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
