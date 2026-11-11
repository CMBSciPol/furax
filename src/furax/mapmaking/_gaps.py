r"""The gaps of a TOD as splines whose amplitudes are solved for with the map.

Under `GapTreatment.EXPLICIT`, the flagged samples of each channel are modelled by a spline per
gap: a gap of $L$ samples is cut into $n = \lceil L / r \rceil$ intervals of about $r$ samples,
and gets the $n + p$ uniform B-splines of degree $p = \min(3, n - 1)$ over them, or one amplitude
per sample when that is fewer. A short gap is then a constant, a long one a cubic spline.

The amplitudes of a channel are packed one gap after the other in a static `budget` of slots:
under jit their count, set by the flags, is a runtime value. A channel needing more amplitudes
than the budget is dropped (see [`GapSplines.counts`][]).

The samples padding a TOD to the length of its bucket are not data: they are neither good nor
flagged, and take no amplitudes.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Int, PyTree

from furax import AbstractLinearOperator, MaskOperator
from furax.math.bspline import cardinal_bspline


@dataclass(frozen=True)
class GapSplines:
    """The spline model of the gaps of a tile."""

    resolution: int
    """The knot spacing $r$, in samples; 1 for one amplitude per flagged sample."""
    budget: int
    """The number of amplitude slots of each channel for its gaps."""
    bridge: int = 0
    """The number of amplitudes of the cubic spline over the padding after the data, in the slots
    after the budget; 0 for none. Under a circulant weight, the padding joins the end of the data
    to its start: the spline bridges them, so that the weight sees no step there."""

    def counts(
        self, good: Bool[Array, '... samp'], n_samples: Int[Array, ''] | int
    ) -> Int[Array, '...']:
        """The number of amplitudes each channel needs, from its mask of good samples.

        Only the first `n_samples` samples are data, those after them padding. A channel without
        good samples has no gaps: it carries no data, so its flagged samples need no model.
        """
        layout = _GapLayout.create(_flagged(good, n_samples), self.resolution)
        return jnp.sum(jnp.where(layout.first, layout.functions, 0), axis=-1)

    def basis(self, mask: MaskOperator, n_samples: Int[Array, ''] | int) -> 'GapSplineOperator':
        """The basis $C$ of the gaps of the mask $M$ of the good samples of a TOD.

        Only the first `n_samples` samples are data, those after them padding, which the bridge
        spans, if any. Each channel of the TOD has amplitudes of its own: the streams of a
        demodulated TOD, which share the mask of their detector, too.
        """
        leaf = jax.tree.leaves(mask.in_structure)[0]
        good = jnp.broadcast_to(jax.tree.leaves(mask.to_boolean_mask())[0] != 0, leaf.shape)
        basis = GapSplineOperator.create(
            _flagged(good, n_samples), self.resolution, self.budget, mask, self.bridge
        )
        if not self.bridge:
            return basis
        # only the channels holding data have something to bridge
        return basis.with_bridge(n_samples, jnp.any(good, axis=-1), self.budget)


def knot_spacing(resolution: float, sample_rate: float) -> int:
    """The knot spacing in samples of the gap splines, from `resolution` in seconds; at least 1.

    Examples:
        >>> knot_spacing(1.25, 8.0)
        10
        >>> knot_spacing(0.0, 8.0)
        1
    """
    return max(1, round(resolution * sample_rate))


def gap_budget(n_samples: int, max_flagged_fraction: float) -> int:
    """A budget of amplitudes per channel that fits any mask of an observation.

    The observation has at most `n_samples` samples, and at most `max_flagged_fraction` of them
    flagged. A gap has no more amplitudes than samples.
    """
    return math.ceil(max_flagged_fraction * n_samples)


def max_gap_amplitudes(
    good: Bool[np.ndarray, 'det samp'],
    resolution: int,
    max_flagged_fraction: float,
) -> int:
    """The most amplitudes the explicit gaps of a detector kept need, from the masks of all.

    This is the `gap_amplitudes` of an [`ObservationProbe`][furax.mapmaking.ObservationProbe],
    for an interface that knows the masks, at a knot spacing of `resolution` samples. The
    detectors with more than `max_flagged_fraction` of their samples flagged are left out.
    """
    kept = good[np.mean(~good, axis=-1) <= max_flagged_fraction]
    flagged = ~kept & kept.any(axis=-1, keepdims=True)
    edges = np.diff(np.pad(flagged, [(0, 0), (1, 1)]).astype(np.int8), axis=-1)
    rows, starts = np.nonzero(edges == 1)
    _, ends = np.nonzero(edges == -1)  # row-major: paired with the starts
    length = ends - starts
    intervals = -(-length // resolution)
    amplitudes = np.minimum(length, intervals + np.minimum(3, intervals - 1))
    counts = np.bincount(rows, weights=amplitudes, minlength=len(kept))
    return int(counts.max(initial=0))


class GapSplineOperator(AbstractLinearOperator):
    """The gap basis $C$: from the amplitudes of each channel, `(..., budget)`, to the TOD."""

    index: Int[Array, '... samp 4']
    """The slot of each of the four B-splines under each sample."""
    values: Array
    """The values of those B-splines, 0 on the good samples and past the degree, `(..., samp, 4)`."""
    tod_treedef: Any = field(metadata={'static': True})
    """The pytree structure of the TOD, whose single leaf the basis fills."""

    @classmethod
    def create(
        cls,
        flagged: Bool[Array, '... samp'],
        resolution: int,
        budget: int,
        mask: MaskOperator,
        extra: int = 0,
    ) -> Self:
        """The basis of the gaps of `flagged`, on the TOD of `mask`, in `budget` slots.

        `extra` more slots follow them, left unused.
        """
        (leaf,), treedef = jax.tree.flatten(mask.in_structure)
        layout = _GapLayout.create(flagged, resolution)
        dtype = leaf.dtype
        # position on the knot grid of the gap, from 0 at its first sample to `intervals` at its last
        u = layout.position.astype(dtype) / jnp.maximum(layout.length - 1, 1) * layout.intervals
        offset = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, layout.intervals - 1)
        o = jnp.arange(4, dtype=jnp.int32)
        # B-spline j of degree p is cardinal_bspline(u - j + p), nonzero for j = offset, ..., + p
        p = layout.degree[..., None]
        values = cardinal_bspline(u[..., None] - (offset[..., None] + o) + p, p)
        # a slot past the degree reads B_p at or below 0, where it vanishes; a constant takes every
        # sample of its gap, the closed right end included
        constant = (layout.per_sample | (layout.degree == 0))[..., None]
        values = jnp.where(constant, (o == 0).astype(values.dtype), values)
        values = jnp.where(flagged[..., None], values, 0).astype(dtype)
        local = jnp.where(layout.per_sample, layout.position, offset)
        # the amplitudes of a gap follow those of the earlier gaps of its channel
        first_slot = jnp.cumsum(jnp.where(layout.first, layout.functions, 0), axis=-1)
        first_slot = first_slot - layout.functions
        index = jnp.clip(first_slot[..., None] + local[..., None] + o, 0, budget - 1)
        structure = jax.ShapeDtypeStruct((*leaf.shape[:-1], budget + extra), dtype)
        return cls(index, values, treedef, in_structure=structure)

    def with_bridge(
        self, n_samples: Int[Array, ''] | int, channels: Bool[Array, '...'], first: int
    ) -> Self:
        """The basis with a cubic spline over the samples from `n_samples` on.

        The spline takes the slots from `first` to the last one, on the `channels` given.
        """
        bridge = self.in_structure.shape[-1] - first
        n = self.index.shape[-2]
        dtype = self.values.dtype
        t = jnp.arange(n)
        span = (t >= n_samples) & channels[..., None]
        # uniform cubic B-splines on `bridge - 3` intervals over the span
        u = (t - n_samples).astype(dtype) / jnp.maximum(n - n_samples - 1, 1) * (bridge - 3)
        offset = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, bridge - 4)[:, None]
        o = jnp.arange(4, dtype=jnp.int32)
        values = cardinal_bspline(u[:, None] - (offset + o) + 3, 3).astype(dtype)
        return replace(
            self,
            index=jnp.where(span[..., None], first + offset + o, self.index),
            values=jnp.where(span[..., None], values, self.values),
        )

    def squared(self) -> Self:
        """The basis with its values squared."""
        return replace(self, values=self.values**2)

    def mv(self, x: Array) -> PyTree[Array]:
        shape = self.index.shape
        gathered = jnp.take_along_axis(x, self.index.reshape(*shape[:-2], -1), axis=-1)
        tod = jnp.sum(self.values * gathered.reshape(shape), axis=-1)
        return jax.tree.unflatten(self.tod_treedef, [tod])


def _flagged(
    good: Bool[Array, '... samp'], n_samples: Int[Array, ''] | int
) -> Bool[Array, '... samp']:
    """The flagged samples among the first `n_samples`, of the channels holding good ones."""
    data = jnp.arange(good.shape[-1]) < n_samples
    return ~good & data & jnp.any(good, axis=-1, keepdims=True)


class _GapLayout(NamedTuple):
    """Per sample, the gap it belongs to and the splines of that gap."""

    first: Bool[Array, '... samp']
    """Whether the sample is the first of its gap."""
    length: Int[Array, '... samp']
    """The number of samples of its gap."""
    position: Int[Array, '... samp']
    """Its position in its gap."""
    intervals: Int[Array, '... samp']
    """The number of knot intervals of its gap."""
    degree: Int[Array, '... samp']
    """The degree of the splines of its gap."""
    per_sample: Bool[Array, '... samp']
    """Whether its gap has one amplitude per sample, as no larger than its splines."""
    functions: Int[Array, '... samp']
    """The number of amplitudes of its gap."""

    @classmethod
    def create(cls, flagged: Bool[Array, '... samp'], resolution: int) -> Self:
        n_samples = flagged.shape[-1]
        axis = flagged.ndim - 1
        edges = [(0, 0)] * axis
        t = jnp.arange(n_samples, dtype=jnp.int32)
        before = jnp.pad(flagged[..., :-1], [*edges, (1, 0)])
        after = jnp.pad(flagged[..., 1:], [*edges, (0, 1)])
        first = flagged & ~before
        start = jax.lax.cummax(jnp.where(first, t, 0), axis=axis)
        end = jax.lax.cummin(jnp.where(flagged & ~after, t, n_samples - 1), axis=axis, reverse=True)
        length = end - start + 1
        intervals = jnp.maximum(1, -(-length // resolution))
        degree = jnp.minimum(3, intervals - 1)
        per_sample = length <= intervals + degree
        functions = jnp.minimum(length, intervals + degree)
        return cls(first, length, t - start, intervals, degree, per_sample, functions)
