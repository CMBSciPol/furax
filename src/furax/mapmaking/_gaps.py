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
    """The number of amplitude slots of each channel."""

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

        Only the first `n_samples` samples are data, those after them padding. Each channel of
        the TOD has amplitudes of its own, even the streams of a demodulated TOD, which share the
        mask of their detector.
        """
        # the mask as stored: the streams sharing it share their layout, built once
        good = jax.tree.leaves(mask.to_boolean_mask())[0]
        return GapSplineOperator.create(_flagged(good, n_samples), self.spacing, self.budget, mask)


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
    ) -> Self:
        """The basis of the gaps of `flagged`, on the TOD of `mask`, in `budget` slots.

        Their knots are every `spacing` samples. `flagged` broadcasts to the TOD: channels that
        share their flags, as the streams of a detector do, share the layout of their amplitudes.
        """
        (leaf,), treedef = jax.tree.flatten(mask.in_structure)
        layout = _GapLayout.create(flagged, spacing)
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
        first_slot = jnp.cumsum(layout.amplitudes, axis=-1)
        first_slot = first_slot - layout.functions
        index = jnp.clip(first_slot[..., None] + local[..., None] + o, 0, budget - 1)
        structure = jax.ShapeDtypeStruct((*leaf.shape[:-1], budget), dtype)
        return cls(index, values, treedef, in_structure=structure)

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
