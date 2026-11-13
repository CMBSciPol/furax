"""What a run reports per observation and per detector, from the per-tile results of its buckets.

The buckets hold what they know of each detector per detector tile, laid out over the devices:
the template amplitudes solved for, the detectors dropped and the noise parameters. The functions
of this module gather them to every process, in observation order (see
[`TileLayout.to_observation_order`][furax.mapmaking.layout.TileLayout.to_observation_order]).
"""

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jaxtyping import Array, PyTree

from furax.core import StackedEnsemble, lift

from ._model import DROP_REASONS, SHARED_TEMPLATES, TEMPLATES
from .layout import TilePlacement


def template_amplitudes(
    buckets: Sequence[StackedEnsemble], x: dict[str, PyTree[Array]], placement: TilePlacement
) -> dict[str, np.ndarray]:
    """The template amplitudes solved for, by template, per observation, from the unknowns `x`.

    Every device holds the same replicated copy after the gather, so the host array is
    complete on every process; then back to observation order.
    """
    amplitudes = {}
    if (per_detector := x.get(TEMPLATES)) is not None:
        bundle = buckets[0].template.explicit[TEMPLATES]
        assert bundle is not None
        operator = bundle.operator
        per_bucket = [placement.gather(a) for a in per_detector]
        for name in per_bucket[0]:
            # the detector axis follows the stream axis, if any (see `Template`), after the
            # tile axis
            axis = 1 + operator.templates[name].per.index('det')
            amplitudes[name] = placement.layout.to_observation_order(
                [a[name] for a in per_bucket], detector_axis=axis
            )
    if (shared := x.get(SHARED_TEMPLATES)) is not None:  # one set per segment, no detector axis
        amplitudes |= jax.tree.map(
            lambda *leaves: placement.layout.to_observation_order(leaves),
            *[placement.gather(a) for a in shared],
        )
    return amplitudes


def detector_drops(
    buckets: Sequence[StackedEnsemble], placement: TilePlacement
) -> dict[str, np.ndarray]:
    """Per detector, whether it was dropped, by reason (see [`DROP_REASONS`][]).

    Each array is `(n_observations, detector)`, in observation order, its detector axis
    padded with `False`.
    """
    per_bucket = []
    for tiles in buckets:
        flags = lift(lambda tile: {r: getattr(tile.model, r) for r in DROP_REASONS}, tiles)
        assert isinstance(flags, StackedEnsemble)
        leaves, treedef = jax.tree.flatten(placement.gather(flags.stacked))
        # a flag the same in every tile, as when none is dropped, is held once
        leaves = [
            v if stride == 1 else np.broadcast_to(v, (tiles.size, *v.shape))
            for v, stride in zip(leaves, flags.strides, strict=True)
        ]
        per_bucket.append(jax.tree.unflatten(treedef, leaves))
    return {
        reason: placement.layout.to_observation_order(
            [flags[reason] for flags in per_bucket], detector_axis=-1
        )
        for reason in DROP_REASONS
    }


def noise_parameters(
    buckets: Sequence[StackedEnsemble], placement: TilePlacement
) -> dict[str, np.ndarray]:
    """The noise model parameters of every detector, in observation order.

    Each parameter is `(n_observations, [stream,] detector)`, its detector axis padded.
    """
    per_bucket = []
    for tiles in buckets:
        # the parameters of the detectors are held per tile: `(tile, ..., detector)`
        noise = lift(lambda tile: tile.model.noise_model, tiles)
        assert isinstance(noise, StackedEnsemble)
        leaves, _ = jax.tree_util.tree_flatten_with_path(placement.gather(noise.stacked))
        per_bucket.append(
            {
                _parameter_name(path): leaf
                for (path, leaf), stride in zip(leaves, noise.strides, strict=True)
                if stride == 1
            }
        )
    return {
        name: placement.layout.to_observation_order([p[name] for p in per_bucket], detector_axis=-1)
        for name in per_bucket[0]
    }


def _parameter_name(path: tuple[Any, ...]) -> str:
    """The name of a leaf of a noise model, from its path: `sigma`, or `<key>/sigma` if nested."""
    return '/'.join(str(getattr(k, 'name', getattr(k, 'key', k))) for k in path)
