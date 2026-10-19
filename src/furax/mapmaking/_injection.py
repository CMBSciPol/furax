"""A simulated sky scanned into the TOD of an observation, for [`InjectionConfig`][]."""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
from fastquat import Quaternion
from jaxtyping import DTypeLike

from furax import tree
from furax.obs.landscapes import StokesLandscape
from furax.obs.stokes import Stokes, StokesType, ValidStokesLiteral
from furax.obs.tod import TOD

from ._observation import ReaderField
from .acquisition import build_acquisition_operator
from .config import InjectionConfig, InjectionMode, MapMakingConfig
from .results import MapMakingResults

__all__ = ['inject', 'load_sky']


def load_sky(
    config: InjectionConfig,
    landscape: StokesLandscape,
    stokes: ValidStokesLiteral,
    dtype: DTypeLike,
) -> StokesType:
    """The sky of `config` as the Stokes components `stokes` of `landscape`, scaled.

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
    return Stokes.class_for(stokes)(
        *(
            config.scale * jnp.asarray(components[s], dtype)
            if s in components
            else jnp.zeros(landscape.shape, dtype)
            for s in stokes
        )
    )


def inject[T: TOD](
    sky: StokesType,
    landscape: StokesLandscape,
    data: dict[str, Any],
    tod: T,
    config: MapMakingConfig,
) -> tuple[dict[str, Any], T]:
    """The data and the TOD of an observation, with the sky scanned in.

    Args:
        sky: The sky, on `landscape`, with the Stokes components of the TOD read.
        landscape: The pixelization of the sky, with the same Stokes components.
        data: The fields read for the observation.
        tod: The TOD to add the signal to, which may differ from the sample data by a gap fill.
        config: The mapmaking configuration, which says whether to replace the data.
    """
    H = build_acquisition_operator(
        landscape,
        Quaternion.from_array(data[ReaderField.BORESIGHT_QUATERNIONS]),
        Quaternion.from_array(data[ReaderField.DETECTOR_QUATERNIONS]),
        data.get(ReaderField.HWP_ANGLES),
        demodulated=config.demodulated,
        pointing_on_the_fly=config.pointing.on_the_fly,
        pointing_batch_samples=config.pointing.batch_samples,
        pointing_interpolate=config.pointing.interpolation == 'bilinear',
        dtype=config.dtype,
    )
    signal = H(sky)
    assert config.injection is not None
    combine = (lambda _, s: s) if config.injection.mode == InjectionMode.REPLACE else tree.add
    return {
        **data,
        ReaderField.SAMPLE_DATA: combine(data[ReaderField.SAMPLE_DATA], signal),
    }, combine(tod, signal)
