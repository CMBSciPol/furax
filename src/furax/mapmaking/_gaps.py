r"""The gaps of a TOD as splines whose amplitudes are solved for with the map.

Under `GapTreatment.EXPLICIT`, the flagged samples of each channel are modelled by a spline per
gap: a gap of $L$ samples is cut into $n = \lceil L / r \rceil$ intervals of about $r$ samples,
and gets the $n + p$ uniform B-splines of degree $p = \min(3, n - 1)$ over them, or one amplitude
per sample when that is fewer. A short gap is then a constant, a long one a cubic spline.

The amplitudes of a channel are packed one gap after the other in a static `budget` of slots:
under jit their count, set by the flags, is a runtime value. A channel needing more amplitudes
than the budget is dropped (see [`GapSplines.counts`][]).

The samples padding a TOD to the length of its bucket are not data: they are neither good nor
flagged, and take no gap amplitudes. Under a circulant weight, which joins the end of the data to
its start across the padding, a cubic spline over the padding, the bridge, takes slots of its own
after the budget (see [`bridge_amplitudes`][]).
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PyTree

from furax import AbstractLinearOperator, MaskOperator
from furax.math.bspline import cardinal_bspline

from ._observation import SampleRanges


@dataclass(frozen=True)
class GapSplines:
    """The spline model of the gaps of a tile."""

    spacing: int
    """The knot spacing $r$, in samples; 1 for one amplitude per flagged sample."""
    budget: int
    """The number of amplitude slots of each channel for its gaps."""
    bridge: int = 0
    """The number of amplitude slots of the cubic spline over the padding after the data, after
    the budget; 0 for none. Under a circulant weight, the padding joins the end of the data to its
    start: the spline bridges them, so that the weight sees no step there."""
    bridge_spacing: float = 1.0
    """The knot spacing of the bridge next to the data, in samples."""
    bridge_growth: float = 1.0
    """The factor by which its knot spacing grows from each end of the padding to its middle
    (see [`bridge_amplitudes`][])."""

    def counts(
        self, good: Bool[Array, '... samp'], n_samples: Int[Array, ''] | int
    ) -> Int[Array, '...']:
        """The number of amplitudes each channel needs, from its mask of good samples.

        Only the first `n_samples` samples are data, those after them padding. A channel without
        good samples has no gaps: it carries no data, so its flagged samples need no model.
        """
        return jnp.sum(
            _GapLayout.create(_flagged(good, n_samples), self.spacing).amplitudes, axis=-1
        )

    def basis(self, mask: MaskOperator, n_samples: Int[Array, ''] | int) -> 'GapSplineOperator':
        """The basis $C$ of the gaps of the mask $M$ of the good samples of a TOD.

        Only the first `n_samples` samples are data, those after them padding, which the bridge
        spans, if any. Each channel of the TOD has amplitudes of its own, even the streams of a
        demodulated TOD, which share the mask of their detector.
        """
        # the mask as stored: the streams sharing it share their layout, built once
        good = jax.tree.leaves(mask.to_boolean_mask())[0]
        basis = GapSplineOperator.create(
            _flagged(good, n_samples), self.spacing, self.budget, mask, self.bridge
        )
        if not self.bridge:
            return basis
        # only the channels holding data have something to bridge
        return basis.with_bridge(
            n_samples, jnp.any(good, axis=-1), self.budget, self.bridge_spacing, self.bridge_growth
        )


def bridge_amplitudes(length: int, spacing: float, growth: float) -> int:
    r"""The amplitudes of the bridge over a padding of `length` samples.

    Its knots are `spacing` samples apart next to each end of the padding, their spacing growing by
    `growth` towards its middle: the $j$-th knot from an end lies $s (g^j - 1) / (g - 1)$ samples
    from it. A long padding takes few more amplitudes than a short one: their number grows with the
    logarithm of its length.
    """
    intervals = 2 * float(_knot_index(np.asarray((length - 1) / 2), spacing, growth))
    return max(1, math.ceil(intervals)) + 3


def _knot_index(distance: Array | np.ndarray | float, spacing: float, growth: float) -> Array:
    """The index, fractional, of the knot of the bridge at `distance` samples from an end."""
    if growth == 1:
        return jnp.asarray(distance) / spacing
    return jnp.log1p(distance * (growth - 1) / spacing) / np.log(growth)


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


def max_gap_amplitudes(good: SampleRanges, spacing: int, max_flagged_fraction: float) -> int:
    """The most amplitudes the explicit gaps of a detector kept need, from the good ranges of all.

    It counts on the host, from the good ranges an observation's probe reports, what
    [`GapSplines.counts`][] counts with knots every `spacing` samples. The detectors with more
    than `max_flagged_fraction` of their samples flagged are left out, as they are dropped, and so
    are those without good samples, which have no gaps.
    """
    n = good.n_samples
    lengths, detectors = [], []
    for detector, ranges in enumerate(good.ranges):
        n_good = int(np.sum(ranges[:, 1] - ranges[:, 0]))
        if n_good == 0 or n - n_good > max_flagged_fraction * n:
            continue
        # the gaps lie before, between and after the good ranges
        starts = np.concatenate([[0], ranges[:, 1]])
        stops = np.concatenate([ranges[:, 0], [n]])
        lengths.append((stops - starts)[stops > starts])
        detectors.append(np.full(len(lengths[-1]), detector))
    none = np.zeros(0, int)  # for an observation without gaps
    *_, amplitudes = _splines(np.concatenate([none, *lengths]), spacing, np)
    owners = np.concatenate([none, *detectors])
    return int(np.bincount(owners, weights=amplitudes, minlength=1).max())


class GapSplineOperator(AbstractLinearOperator):
    """The gap basis $C$: from the amplitudes of each channel, `(..., budget)`, to the TOD."""

    index: Int[Array, '... samp 4']
    """The slot of each of the four B-splines under each sample."""
    values: Float[Array, '... samp 4']
    """The values of those B-splines, 0 on the good samples and past the degree."""
    tod_treedef: Any = field(metadata={'static': True})
    """The pytree structure of the TOD, whose single leaf the basis fills."""

    @classmethod
    def create(
        cls,
        flagged: Bool[Array, '... samp'],
        spacing: int,
        budget: int,
        mask: MaskOperator,
        extra: int = 0,
    ) -> Self:
        """The basis of the gaps of `flagged`, on the TOD of `mask`, in `budget` slots.

        `extra` more slots follow them, left unused. Their knots are every `spacing` samples. `flagged` broadcasts to the TOD: channels that
        share their flags, as the streams of a detector do, share the layout of their amplitudes.
        """
        treedef = jax.tree.structure(mask.in_structure)
        structure = _amplitude_structure(mask, budget + extra)
        layout = _GapLayout.create(flagged, spacing)
        dtype = structure.dtype
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
        first_slot = jnp.cumsum(layout.amplitudes, axis=-1)
        first_slot = first_slot - layout.functions
        index = jnp.clip(first_slot[..., None] + local[..., None] + o, 0, budget - 1)
        return cls(index, values, treedef, in_structure=structure)

    def with_bridge(
        self,
        n_samples: Int[Array, ''] | int,
        channels: Bool[Array, '...'],
        first: int,
        spacing: float,
        growth: float,
    ) -> Self:
        """The basis with a cubic spline over the samples from `n_samples` on.

        The spline takes the slots from `first` on, on the `channels` given. Its knots are
        `spacing` samples apart next to each end of the span, growing by `growth` towards its middle
        (see [`bridge_amplitudes`][]); with more of them than slots, they are spread wider.
        """
        slots = self.in_structure.shape[-1] - first
        n = self.index.shape[-2]
        dtype = self.values.dtype
        t = jnp.arange(n)
        span = (t >= n_samples) & channels[..., None]
        # uniform cubic B-splines in the knot index, mirrored at the middle of the span
        length = n - n_samples
        middle = (length - 1) / 2
        distance = (t - n_samples).astype(dtype)
        total = 2 * _knot_index(middle, spacing, growth)
        intervals = jnp.clip(jnp.ceil(total), 1, slots - 3)
        index = jnp.where(
            distance <= middle,
            _knot_index(distance, spacing, growth),
            total - _knot_index(length - 1 - distance, spacing, growth),
        )
        u = index * intervals / jnp.maximum(total, 1e-12)
        offset = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, intervals.astype(jnp.int32) - 1)
        offset = offset[:, None]
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
        def apply(x):
            shape = self.index.shape
            gathered = jnp.take_along_axis(x, self.index.reshape(*shape[:-2], -1), axis=-1)
            return jnp.sum(self.values * gathered.reshape(shape), axis=-1)

        # the channels sharing the layout, such as the streams of a detector, are mapped over:
        # broadcasting the layout to them would hold its index once per channel
        for _ in range(x.ndim - (self.index.ndim - 1)):
            apply = jax.vmap(apply)
        return jax.tree.unflatten(self.tod_treedef, [apply(x)])


class GapBasisOperator(AbstractLinearOperator):
    """The gap basis $C$ of a mask, built when applied.

    [`GapSplineOperator`][] holds several values per sample, which this holds none of: a tile
    keeps its mask, and builds the basis of its gaps only to apply it.
    """

    mask: MaskOperator
    """The mask $M$ of the good samples."""
    n_samples: Int[Array, '']
    """The number of samples of the observation, before those padding it."""
    gaps: GapSplines = field(metadata={'static': True})
    """The spline model of the gaps."""

    @classmethod
    def create(cls, gaps: GapSplines, mask: MaskOperator, n_samples: Int[Array, '']) -> Self:
        """The basis of the gaps of `mask`, as [`GapSplines.basis`][] builds it."""
        structure = _amplitude_structure(mask, gaps.budget + gaps.bridge)
        return cls(mask, n_samples, gaps, in_structure=structure)

    def mv(self, x: Array) -> PyTree[Array]:
        return self.gaps.basis(self.mask, self.n_samples).mv(x)


def _amplitude_structure(mask: MaskOperator, budget: int) -> jax.ShapeDtypeStruct:
    """The amplitudes of the gaps of the TOD of `mask`: `budget` slots in place of its samples."""
    (leaf,) = jax.tree.leaves(mask.in_structure)
    return jax.ShapeDtypeStruct((*leaf.shape[:-1], budget), leaf.dtype)


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

    @property
    def amplitudes(self) -> Int[Array, '... samp']:
        """The number of amplitudes of its gap on its first sample, 0 on the others."""
        return jnp.where(self.first, self.functions, 0)

    @classmethod
    def create(cls, flagged: Bool[Array, '... samp'], spacing: int) -> Self:
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
        intervals, degree, functions = _splines(length, spacing)
        per_sample = length <= intervals + degree
        return cls(first, length, t - start, intervals, degree, per_sample, functions)


def _splines[T: (Array, np.ndarray)](length: T, spacing: int, xp: Any = jnp) -> tuple[T, T, T]:
    """The knot intervals, the degree and the number of amplitudes of gaps of `length` samples.

    A gap of L samples takes n = ⌈L / r⌉ intervals and the n + p B-splines of degree
    p = min(3, n − 1) over them, or one amplitude per sample when that is fewer. `xp` is the array
    module: `np` on the host.
    """
    intervals = xp.maximum(1, -(-length // spacing))
    degree = xp.minimum(3, intervals - 1)
    return intervals, degree, xp.minimum(length, intervals + degree)
