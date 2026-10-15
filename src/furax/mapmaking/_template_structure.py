r"""The Gram $T^\top W T$ of templates under a diagonal weight, assembled from their bases.

Under a weight diagonal in time and in streams, $W = \mathrm{diag}(w_{ld})$, a template $g$ couples
to a template $h$ only through the weighted Grams of their bases on each detector and stream,

$$ L^{ld}_{gh} = B_{g,d}^\top\, \mathrm{diag}(w_{ld})\, B_{h,d}, $$

which [`Basis.gram`][furax.mapmaking.bases.Basis.gram] gives in bands and `_cross_gram` as a
dense block. The Gram is then their contraction with the weights through which each detector sees
the amplitude sets of each template (see [`Template`][furax.mapmaking._template_model.Template]):

$$ G_{gh}[(o, m), (o', m')] = \sum_{l, d} W_{g, ld}[o, m]\, W_{h, ld}[o', m']\, L^{ld}_{gh}. $$

The streams are assembled one after the other, so that the transient memory is that of one stream.
Each block is then laid out as the probed one, for the same layout.
"""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from furax import tree
from furax.linalg import BorderedBandedMatrix

from ._template_model import Template, TemplateOperator
from .bases import Basis, _einsum
from .gram import _cross_gram

if TYPE_CHECKING:  # the Gram module calls this one
    from ._template_gram import _Entry, _Layout


def assemble(
    operator: TemplateOperator,
    weights: tuple[Float[Array, 'det samp'], ...],
    layout: _Layout,
    reading: _Layout | None,
    batch_size: int,
) -> tuple[BorderedBandedMatrix, Array | None]:
    """One block of the Gram, and its coupling to the local amplitudes of `reading` if given.

    `weights` is the diagonal of the weight, one `(n_dets, n_samples)` stream per stream of the TOD.
    The coupling has the shape of the probed one: `(n_global, n_units, n_local)`, the block being
    all border.
    """
    names = layout.names + (() if reading is None else reading.names)
    inputs = {name: _GramInputs.of(operator.templates[name]) for name in names}
    border = layout.border
    # The Gram is symmetric: only the blocks on and above the diagonal are assembled.
    pairs = [_Pair(layout, e, layout, f) for i, e in enumerate(border) for f in border[i:]]
    if layout.core is not None:
        pairs += [_Pair(layout, layout.core, layout, f) for f in border]
    local_rows = () if reading is None else reading.entries
    crosses = []
    if reading is not None:
        crosses = [_Pair(reading, e, layout, f) for e in local_rows for f in border]

    parts: dict[Any, list[tuple[int, Array]]] = {}
    for stream, w in enumerate(weights):
        core = layout.core if layout.core is not None and stream in layout.core.streams else None
        on_stream = [p for p in pairs if p.covers(stream)]
        crossing = [p for p in crosses if p.covers(stream)]
        done = {}
        if on_stream or core is not None:
            done |= _on_stream(
                inputs, on_stream, core, stream, w, batch_size, total=not layout.local
            )
        if crossing:
            done |= _on_stream(inputs, crossing, None, stream, w, batch_size, total=False)
        for key, block in done.items():
            parts.setdefault(key, []).append((stream, block))

    bands = None if layout.core is None else _place_bands(operator, layout, parts[layout.core])
    placed = {(p.g.name, p.h.name): p.place(operator, inputs, parts) for p in pairs + crosses}
    corner = core_border = cross = None
    if border:
        corner = jnp.block(
            [
                [
                    placed[e.name, f.name]
                    if i <= j
                    else jnp.swapaxes(placed[f.name, e.name], -1, -2)
                    for j, f in enumerate(border)
                ]
                for i, e in enumerate(border)
            ]
        )
        if layout.core is not None:
            core_border = jnp.block([[placed[layout.core.name, f.name] for f in border]])
        if reading is not None:
            cross = jnp.block([[placed[e.name, f.name] for f in border] for e in local_rows])
            cross = jnp.moveaxis(cross, -1, 0)  # (global, unit, local)
    return BorderedBandedMatrix(bands, core_border, corner), cross


class _GramInputs(NamedTuple):
    """What the assembly reads of a template: its basis, and the detectors' weights on its sets."""

    basis: Basis
    """The basis, batched over the flattened detectors if it has one per detector."""
    support: Basis
    """The basis written as its support, computed once if shared by the detectors."""
    weights: Float[Array, 'det j o m']
    """On each stream `j` of the template, each detector's weight on each set `o` and mode `m`."""

    @classmethod
    def of(cls, template: Template) -> Self:
        tod = template.tod
        on_streams = []
        for j in range(len(template._stream_indices)):
            w = template._weights(j)
            on_streams.append(jnp.broadcast_to(w, (tod.n_dets, *w.shape[1:])))
        weights = jnp.stack(on_streams, axis=1)
        basis = template.basis
        if not basis.batch_shape:
            return cls(basis, basis.support(), weights)
        # one per detector, over the flattened detectors
        n_axes = len(basis.batch_shape)
        basis = jax.tree.map(lambda a: a.reshape(tod.n_dets, *a.shape[n_axes:]), basis)
        basis = basis._with_batch_shape((tod.n_dets,))
        return cls(basis, basis, weights)

    @property
    def set_shape(self) -> tuple[int, int, int]:
        """The shape of the template's amplitudes on one stream: `(set, basis size, mode)`."""
        n_sets, n_modes = self.weights.shape[-2:]
        return n_sets, self.basis.size, n_modes


class _Pair(NamedTuple):
    """The block coupling template `g`, in the block `rows`, to template `h`, in `columns`."""

    rows: _Layout
    g: _Entry
    columns: _Layout
    h: _Entry

    def covers(self, stream: int) -> bool:
        """Whether both templates cover the TOD's stream number `stream`."""
        return stream in self.g.streams and stream in self.h.streams

    def place(
        self,
        operator: TemplateOperator,
        inputs: dict[str, _GramInputs],
        parts: dict[Any, list[tuple[int, Array]]],
    ) -> Float[Array, 'unit rows columns']:
        """The block, from its parts on each stream, laid out as the units of `rows`.

        The rows are those of a unit, the core's time blocks first; the units are the detectors of a
        local block, times its streams if it has one per stream. The columns are those of a unit of
        `columns`, which shares the streams of the rows as units if it has one per stream too.
        """
        rows, g, columns, h = self
        tg, th = operator.templates[g.name], operator.templates[h.name]
        shape = (*inputs[g.name].set_shape, *inputs[h.name].set_shape)
        lead = (tg.tod.n_dets,) if rows.local else ()
        n_slots = rows.n_streams if rows.per_stream else tg._canonical_shape[0]
        n_column_slots = 1 if columns.per_stream else th._canonical_shape[0]
        out = jnp.zeros((*lead, n_slots, n_column_slots, *shape), tg.tod.dtype)
        for stream, block in parts.get(self, []):
            slot = stream if rows.per_stream else tg._stream_set(g.streams.index(stream))
            column_slot = 0 if columns.per_stream else th._stream_set(h.streams.index(stream))
            out = out.at[..., slot, column_slot, *(slice(None),) * len(shape)].add(
                block.reshape(*lead, *shape)
            )
        # the stream sets lead the amplitudes of a unit, then (set, *basis.shape, mode)
        n_lead = len(lead)
        out = jnp.moveaxis(out, n_lead + 1, n_lead + 4)
        if not rows.per_stream:
            out = jnp.expand_dims(out, n_lead)
        n_units = prod(out.shape[: n_lead + 1])
        out = out.reshape(n_units, *rows.unit(g), -1)
        if g == rows.core:  # the time blocks first
            out = jnp.moveaxis(out, rows.block_axis, 1)
        return out.reshape(n_units, prod(rows.unit(g)), -1)


def _on_stream(
    inputs: dict[str, _GramInputs],
    pairs: list[_Pair],
    core: _Entry | None,
    stream: int,
    w: Float[Array, 'det samp'],
    batch_size: int,
    *,
    total: bool,
) -> dict[Any, Array]:
    """The blocks of `pairs`, and the bands of `core`, on stream `stream` of the TOD.

    A block is `(o, b, m, o', c, n)`: the sets, the flattened basis index and the modes of each
    template, and the bands `(i, w, o, r, m, o', s, n)`, the time block `i` coupled to `i + w`,
    `r`, `s` within them. They are stacked over the detectors, or summed over them if `total`, in
    batches of `batch_size` detectors.
    """
    names = {name for p in pairs for name in (p.g.name, p.h.name)}
    names |= set() if core is None else {core.name}
    # each template's weights on the sets, on its own stream that is the TOD's stream `stream`
    streams = {e.name: e.streams.index(stream) for p in pairs for e in (p.g, p.h)}
    if core is not None:
        streams[core.name] = core.streams.index(stream)
    weights = {name: inputs[name].weights[:, streams[name]] for name in names}
    batched = {name: inputs[name].basis for name in names if inputs[name].basis.batch_shape}

    def grams(w_d, own):
        """One detector's Grams of the bases of each pair, and the bands of the core's."""
        bases = {name: basis._with_batch_shape(()) for name, basis in own.items()}
        blocks = []
        for p in pairs:
            g = bases.get(p.g.name, inputs[p.g.name].support)
            h = bases.get(p.h.name, inputs[p.h.name].support)
            blocks.append(_cross_gram(g, h, w_d))
        bands = None
        if core is not None:
            bands = bases.get(core.name, inputs[core.name].basis).gram(w_d)
        return blocks, bands

    def by_key(blocks, bands):
        out: dict[Any, Array] = dict(zip(pairs, blocks, strict=True))
        if core is not None:
            out[core] = bands
        return out

    if not total:
        blocks, bands = tree.stacked_map(grams, w, batched, batch_size=batch_size)
        return by_key(*_contract(pairs, core, blocks, bands, weights, None))

    n_dets = w.shape[0]
    size = min(batch_size, n_dets)

    def batch(i):
        # The last batch ends on the last detector, overlapping the one before: its detectors
        # already counted are scaled to zero, rather than the inputs copied to pad them.
        start = jnp.minimum(i * size, n_dets - size)
        w_b, own, weights_b = jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, start, size), (w, batched, weights)
        )
        fresh = (start + jnp.arange(size) >= i * size).astype(w.dtype)
        blocks, bands = jax.vmap(grams)(w_b, own)
        return _contract(pairs, core, blocks, bands, weights_b, fresh)

    def add(acc, i):
        return tree.add(acc, batch(i)), None

    n_batches = -(-n_dets // size)
    return by_key(*jax.lax.scan(add, batch(0), jnp.arange(1, n_batches))[0])


def _contract(
    pairs: list[_Pair],
    core: _Entry | None,
    blocks: list[Array],
    bands: Array | None,
    weights: dict[str, Array],
    scale: Array | None,
) -> tuple[list[Array], Array | None]:
    """Each detector's Grams contracted with its weights on the sets, `(det, o, m)`.

    Kept per detector, or summed over them, each scaled by `scale`, if given.
    """

    def apply(spec, wg, wh, gram):
        if scale is None:
            return _einsum(spec.format('d'), wg, wh, gram)
        return _einsum('d,' + spec.format(''), scale, wg, wh, gram)

    blocks = [
        apply('dom,dpn,dbc->{}obmpcn', weights[p.g.name], weights[p.h.name], gram)
        for p, gram in zip(pairs, blocks, strict=True)
    ]
    if core is not None:
        w = weights[core.name]
        bands = apply('dom,dpn,diwrs->{}iwormpsn', w, w, bands)
    return blocks, bands


def _place_bands(
    operator: TemplateOperator, layout: _Layout, parts: list[tuple[int, Array]]
) -> Float[Array, 'unit n w1 k k']:
    """The bands of the core's own Gram, from their parts on each stream, per unit of `layout`.

    `bands[u, i, w]` couples the time blocks `i` and `i + w` of unit `u`.
    """
    assert layout.core is not None
    template = operator.templates[layout.core.name]
    lead = (template.tod.n_dets,) if layout.local else ()
    block_shape = parts[0][1].shape[len(lead) :]  # (i, w, o, r, m, o', s, n)
    n_blocks, n_w1 = block_shape[:2]
    k = prod(block_shape[2:5])
    if layout.per_stream:  # one unit per stream: the streams on the diagonal
        out = jnp.zeros((*lead, layout.n_streams, *block_shape), template.tod.dtype)
        for stream, block in parts:
            out = out.at[..., stream, *(slice(None),) * len(block_shape)].add(block)
        return out.reshape(-1, n_blocks, n_w1, k, k)
    n_sets = template._canonical_shape[0]
    out = jnp.zeros((*lead, n_sets, n_sets, *block_shape), template.tod.dtype)
    for stream, block in parts:
        s = template._stream_set(layout.core.streams.index(stream))
        out = out.at[..., s, s, *(slice(None),) * len(block_shape)].add(block)
    # the stream sets z, z' next to their sets: (z, z', i, w, o, r, m, o', s, n) to
    # (i, w, z, o, r, m, z', o', s, n)
    n_lead = len(lead)
    out = jnp.moveaxis(out, (n_lead, n_lead + 1), (n_lead + 2, n_lead + 6))
    return out.reshape(-1, n_blocks, n_w1, n_sets * k, n_sets * k)
