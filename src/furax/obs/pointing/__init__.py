"""Pointing of detectors on the sky.

[`PointingOperator`][] samples sky maps along the pointing of detectors, which a sampler gives, see
[`sampling`][]. The pointing of a telescope can be modelled from its encoder readings, and the
model fitted to observations of point sources, see [`modeling`][].
"""

from ._fitting import FitResult, fit, residuals
from ._simulate import simulate_observations
from .modeling import (
    AbstractPointingModel,
    BasicPointingModel,
    Observations,
    SATV1PointingModel,
    SATV2PointingModel,
)
from .operator import PointingOperator
from .sampling import (
    AbstractSampler,
    AngleSampler,
    DiscretizedBeam,
    PointingRows,
    PolarizationFrame,
    PrecomputedSampler,
    QuaternionSampler,
    RotatedSampler,
    SampleIndex,
    SamplingKernel,
)

__all__ = [
    'AbstractPointingModel',
    'AbstractSampler',
    'AngleSampler',
    'DiscretizedBeam',
    'FitResult',
    'Observations',
    'BasicPointingModel',
    'PointingOperator',
    'PointingRows',
    'PolarizationFrame',
    'PrecomputedSampler',
    'QuaternionSampler',
    'RotatedSampler',
    'SampleIndex',
    'SATV1PointingModel',
    'SATV2PointingModel',
    'SamplingKernel',
    'fit',
    'residuals',
    'simulate_observations',
]
