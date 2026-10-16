r"""The Gram $T^\top W T$ of a template operator, its assembly and its inverse.

Under a weight $W$ diagonal in time and in streams, the Gram is assembled from the structure of the
bases (see `_structure`). Otherwise it is assembled by probing: applying $T^\top W T$ to a
few probe amplitudes and reading its entries off the responses. That needs no structure from the
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
global amplitudes, unless the Gram was assembled with that coupling: it is then inverted exactly,
eliminating the local amplitudes into the Schur complement of the global block. The coupling is
one dense block per detector, which suits a small global block, such as the few amplitudes of a
leakage field.

The Grams of disjoint sets of detectors, such as the detector batches of an observation, combine
into the Gram of their union ([`BlockGram.combine`][]).
"""

from dataclasses import dataclass, field
from functools import reduce
from math import prod
from typing import Literal, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PyTree

from furax import AbstractLinearOperator, DiagonalOperator, symmetric, tree
from furax.core import BlockDiagonalOperator
from furax.core.utils import register_dataclass_with_keys
from furax.linalg import ArrowheadCholeskyOperator, BorderedBandedMatrix, probe_bordered_banded

from ..pomme import PommeProjectionOperator
from . import _structure
from ._operator import TemplateOperator
from .bases import OffsetBasis

__all__ = [
    'BlockGram',
    'factor',
    'template_gram',
]

_BAND_MARGIN = 1
"""Blocks added to the band half-width estimated from the mean block length.

A block shorter than the mean, such as a partial scan interval, lets a sample couple one more
block than the estimate; the margin absorbs one such block.
"""


@register_dataclass_with_keys
@dataclass(frozen=True)
class BlockGram:
    r"""The Gram $T^\top W T$ of a template operator, assembled but not factored.

    `local` holds the blocks of the detectors' own amplitudes, `global_` those of the others, and
    `cross` their coupling, `(n_global, n_local_units, n_local)`, when assembled with it.
    """

    local: '_Part | None'
    global_: '_Part | None'
    cross: Float[Array, 'global unit local'] | None

    @classmethod
    def combine(cls, grams: list[Self]) -> Self:
        """The Gram over the union of the disjoint sets of detectors of `grams`, in order.

        The detectors' own blocks and couplings join; the global blocks, each summed over its
        detectors, add up. The Grams are assembled alike, by the same method.
        """
        first = grams[0]
        local = global_ = cross = None
        if first.local is not None:
            matrices = [gram.local.matrix for gram in grams if gram.local is not None]
            local = _Part(_concatenate(matrices), first.local.layout)
        if first.global_ is not None:
            matrices = [gram.global_.matrix for gram in grams if gram.global_ is not None]
            global_ = _Part(reduce(tree.add, matrices), first.global_.layout)
        if first.cross is not None:
            cross = jnp.concatenate([gram.cross for gram in grams if gram.cross is not None], 1)
        return cls(local, global_, cross)


def template_gram(
    operator: TemplateOperator,
    weight: AbstractLinearOperator,
    *,
    bandwidth: int = 0,
    streams_coupled: bool = False,
    coupled: bool = False,
    pomme_tau: int | None = None,
    method: Literal['structure', 'probe'] | None = None,
    batch_size: int = 1,
) -> BlockGram:
    r"""Assemble the Gram $T^\top W T$, or $T^\top W F T$ with the Pomme projector $F$.

    A weight diagonal in time and in streams (`bandwidth=0`, `streams_coupled=False`) is read off by
    applying it to ones, and the Gram assembled from the structure of the bases. Any other weight
    is probed.

    With `pomme_tau`, the Gram is that under $W F$, with $F$ the
    [`PommeProjectionOperator`][furax.mapmaking.pomme.PommeProjectionOperator] removing the mean
    of each interval of `pomme_tau` samples. $W$ must then be constant on each interval, as the
    Pomme path makes it by masking intervals whole: $W F$ is the weight with the interval offsets
    marginalized. Probed, $W F$ is a weight reaching `pomme_tau` samples; from the structure, the
    offsets are eliminated from the Gram. A time-local template next to Pomme must have blocks
    longer than an interval.

    Args:
        operator: The template operator $T$.
        weight: The weight $W$. It must not couple detectors, nor samples more than `bandwidth`
            apart, nor the streams of a detector unless `streams_coupled`.
        bandwidth: The largest distance, in samples, between two samples that $W$ couples.
        streams_coupled: Whether $W$ couples the streams of a detector, such as Q and U.
        coupled: Whether to also assemble the coupling of the local and the global amplitudes,
            for an exact inverse.
        pomme_tau: The length of the Pomme intervals, when the templates are combined with Pomme.
        method: `'structure'` or `'probe'` to choose the assembly; by default, from the
            structure when the weight is diagonal, by probing otherwise.
        batch_size: The number of probes applied at once, or of detectors assembled at once
            from the structure, bounding the memory of the assembly.

    Raises:
        ValueError: If the assembly from the structure is asked for a weight that is not
            diagonal, or Pomme for a demodulated TOD. If, when probing, the time blocks of a
            template are too short for the probes to resolve its banded Gram; under a
            transformation such as `jax.jit`, where this cannot be checked, the Gram is NaN
            instead, as it is from the structure when a time-local template has blocks shorter
            than a Pomme interval.
    """
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
            operator, rest, local=False, dense=coupled, streams_coupled=streams_coupled
        )

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

    local_part = global_part = cross = None
    if local_layout is not None:
        local_part, _ = assemble(local_layout, None)
    if global_layout is not None:
        global_part, cross = assemble(global_layout, local_layout if coupled else None)
    return BlockGram(local_part, global_part, cross)


def factor(
    gram: BlockGram, *, regularization: float = 0.0, batch_size: int = 1
) -> tuple[AbstractLinearOperator, AbstractLinearOperator | None]:
    """The inverse of the Gram, and the ridge added to it before factoring.

    The Gram is inverted exactly if it was assembled with its local-global coupling, block by block
    otherwise (see
    [`BorderedBandedMatrix.regularized`][furax.linalg.BorderedBandedMatrix.regularized] for the
    ridge).

    Args:
        gram: The assembled Gram.
        regularization: The relative ridge of each block.
        batch_size: The number of global amplitudes whose coupling is solved at once, for an
            exact inverse.

    Returns:
        The inverse, on the amplitudes, and the ridge as a diagonal operator on them; `None`
        without regularization.
    """
    parts: list[_FactorPart] = []
    exact = local = None
    ridges: dict[str, Array] = {}
    if gram.local is not None:
        matrix, ridge = gram.local.matrix.regularized(regularization)
        local = _FactorPart(matrix.cholesky(), gram.local.layout)
        ridges |= gram.local.layout.scatter(*ridge)
    if gram.global_ is not None:
        matrix, ridge = gram.global_.matrix.regularized(regularization)
        ridges |= gram.global_.layout.scatter(*ridge)
        if gram.cross is None:
            parts.append(_FactorPart(matrix.cholesky(), gram.global_.layout))
        else:  # the local blocks eliminated into the Schur complement of the global one
            assert local is not None and matrix.corner is not None  # a coupled block is all border
            cross = local.layout.unflat(gram.cross)
            arrow = ArrowheadCholeskyOperator.from_blocks(
                local.inverse, cross, matrix.corner[0], batch_size
            )
            exact = _Exact(arrow, local.layout, gram.global_.layout)
            local = None
    if local is not None:
        parts.append(local)

    # shapes and dtypes only: the ridges may be laid out over a mesh the inverse is not used on
    structure = {name: jax.ShapeDtypeStruct(r.shape, r.dtype) for name, r in ridges.items()}
    inverse = _GramInverse(tuple(parts), exact, in_structure=structure)
    if not regularization:
        return inverse, None
    return inverse, _ridge_operator(ridges, structure)


@register_dataclass_with_keys
@dataclass(frozen=True)
class _Part:
    """A block of the Gram, and the layout of its amplitudes."""

    matrix: BorderedBandedMatrix
    layout: '_Layout' = field(metadata={'static': True})


def _concatenate(matrices: list[BorderedBandedMatrix]) -> BorderedBandedMatrix:
    """The matrix whose batch concatenates those of `matrices`."""

    def joined(arrays):
        return None if arrays[0] is None else jnp.concatenate(arrays)

    return BorderedBandedMatrix(
        joined([m.bands for m in matrices]),
        joined([m.border for m in matrices]),
        joined([m.corner for m in matrices]),
    )


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
        the weight does not couple them, and the block is not to be coupled to the local amplitudes.
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
            parts = [self.to_units(e, amplitudes[e.name]) for e in self.border]
            border = jnp.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1)
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
            start = 0
            for entry in self.border:
                unit = self.unit(entry)
                units = border[:, start : start + prod(unit)].reshape(border.shape[0], *unit)
                out[entry.name] = self.from_units(entry, units)
                start += prod(unit)
        return out

    @staticmethod
    def flat(core: Array | None, border: Array | None) -> Array:
        """Each unit's amplitudes as one vector, the core then the border."""
        parts = [] if core is None else [core.reshape(*core.shape[:-2], -1)]
        return jnp.concatenate([*parts, *([] if border is None else [border])], axis=-1)

    def unflat(self, flat: Array) -> tuple[Array | None, Array | None]:
        """The inverse of `flat`."""
        n_core = self.n_blocks * self.k
        core = None
        if self.core is not None:
            core = flat[..., :n_core].reshape(*flat.shape[:-1], self.n_blocks, self.k)
        return core, flat[..., n_core:] if self.n_border else None

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
    # the probes need only the templates whose responses are read
    names = layout.names + (() if reading is None else reading.names)
    probed = TemplateOperator({name: operator.templates[name] for name in names})
    gram = probed.T @ weight @ probed
    zeros = tree.zeros_like(probed.in_structure)

    def apply(x):
        response = gram(zeros | layout.scatter(*x))
        local = None if reading is None else reading.flat(*reading.gather(response))
        return layout.gather(response), local

    half_width, too_narrow = 0, jnp.array(False)
    if layout.core is not None:
        basis = operator.templates[layout.core.name].basis
        crossed = -(-bandwidth * layout.n_blocks // basis.n_points)
        half_width = min(layout.n_blocks - 1, basis._support_window - 1 + crossed + _BAND_MARGIN)
        too_narrow = basis._support_reach(bandwidth) > half_width
        try:
            if too_narrow:
                msg = (
                    f'template {layout.core.name!r} has time blocks short enough that samples '
                    f'{bandwidth} apart couple blocks beyond the {half_width} probed on either side'
                )
                raise ValueError(msg)
        except jax.errors.ConcretizationTypeError:
            pass  # traced: the bands are made NaN below instead
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
    if matrix.bands is not None:  # so that a band too narrow cannot pass unnoticed
        bands = jnp.where(too_narrow, jnp.nan, matrix.bands)
        matrix = BorderedBandedMatrix(bands, matrix.border, matrix.corner)
    return matrix, cross


@register_dataclass_with_keys
@dataclass(frozen=True)
class _FactorPart:
    """The inverse of one block, on its core and border per unit, and their layout."""

    inverse: AbstractLinearOperator
    layout: _Layout = field(metadata={'static': True})

    def solve(self, x: dict[str, Array]) -> dict[str, Array]:
        return self.layout.scatter(*self.inverse(self.layout.gather(x)))


@register_dataclass_with_keys
@dataclass(frozen=True)
class _Exact:
    """The exact inverse: the local and global blocks solved together."""

    arrow: ArrowheadCholeskyOperator
    local: _Layout = field(metadata={'static': True})
    global_: _Layout = field(metadata={'static': True})

    def solve(self, x: dict[str, Array]) -> dict[str, Array]:
        r_global = self.global_.flat(*self.global_.gather(x))[0]
        x_local, x_global = self.arrow((self.local.gather(x), r_global))
        out = self.local.scatter(*x_local)
        return out | self.global_.scatter(*self.global_.unflat(x_global[None]))


@symmetric
class _GramInverse(AbstractLinearOperator):
    """The inverse of a block Gram, on the amplitudes of its templates."""

    parts: tuple[_FactorPart, ...]
    """The blocks inverted on their own."""
    exact: _Exact | None
    """The blocks inverted together, for an exact inverse."""

    def mv(self, x: dict[str, Array]) -> dict[str, Array]:
        out = {} if self.exact is None else self.exact.solve(x)
        for part in self.parts:
            out |= part.solve(x)
        return {name: out[name] for name in x}


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
