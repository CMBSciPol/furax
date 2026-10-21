"""A simulated sky scanned into the TOD of an observation, for [`InjectionConfig`][]."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

from furax import AbstractLinearOperator, tree
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, StokesType

from ._observation import ReaderField
from .config import InjectionConfig
from .results import MapMakingResults

__all__ = ['inject', 'load_sky', 'unpadded']


def load_sky(config: InjectionConfig, landscape: StokesLandscape) -> StokesType:
    """The sky of `config` on `landscape`, with its Stokes components and dtype, scaled.

    Components that the file does not hold are zero.

    Raises:
        ValueError: If the file is not on the pixelization of the landscape.
    """
    path = Path(config.path)
    stored = MapMakingResults._load_array(path.stem, path.parent, landscape, len(config.stokes))
    if stored.shape != (len(config.stokes), *landscape.shape):
        raise ValueError(
            f'the sky {path} has the shape {stored.shape}, not '
            f'{(len(config.stokes), *landscape.shape)}: it must be on the pixelization of the '
            'landscape, with the Stokes components of `injection.stokes`'
        )
    components = dict(zip(config.stokes, stored, strict=True))
    return Stokes.class_for(landscape.stokes)(
        *(
            config.scale * jnp.asarray(components[s], landscape.dtype)
            if s in components
            else jnp.zeros(landscape.shape, landscape.dtype)
            for s in landscape.stokes
        )
    )


def inject(
    H: AbstractLinearOperator,
    sky: StokesType,
    data: dict[str, Any],
    injectable: Bool[Array, 'dets samples'],
    replace: bool,
) -> dict[str, Any]:
    """The fields read for an observation, with the sky scanned into their sample data.

    Args:
        H: The acquisition operator of the observation, from the sky to its TOD.
        sky: The sky, with the Stokes components of the TOD read.
        data: The fields read for the observation.
        injectable: Where the signal goes, the samples that are not padding: the padding stays
            zero, as read.
        replace: Whether the signal replaces the sample data rather than adds to it.
    """
    signal = jax.tree.map(lambda x: jnp.where(injectable, x, 0), H(sky))
    if not replace:
        signal = tree.add(data[ReaderField.SAMPLE_DATA], signal)
    return {**data, ReaderField.SAMPLE_DATA: signal}


def unpadded(padding: dict[str, Any], shape: tuple[int, ...]) -> Bool[Array, 'dets samples']:
    """Where the sample data of an observation, of shape `(..., dets, samples)`, is not padding.

    Args:
        padding: The padding widths read for the observation, one per axis of each field.
        shape: The shape of the sample data.
    """
    widths = jax.tree.leaves(padding[ReaderField.SAMPLE_DATA])[0]
    n_dets, n_samples = shape[-2:]
    dets = jnp.arange(n_dets) < n_dets - widths[-2]
    samples = jnp.arange(n_samples) < n_samples - widths[-1]
    return dets[:, None] & samples[None, :]
