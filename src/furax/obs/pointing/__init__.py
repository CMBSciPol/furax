"""Pointing of detectors on the sky.

[`PointingOperator`][] samples sky maps along the pointing of detectors, which a sampler such as
[`QuaternionSampler`][] gives. The pointing of a telescope can be modelled from its encoder
readings with an [`AbstractPointingModel`][], and the model fitted to observations of point
sources with [`fit`][].
"""

from ._fitting import FitResult, fit, residuals
from ._modeling import AbstractPointingModel, BasicPointingModel, Observations
from ._operator import PointingOperator, PointingTransposeOperator
from ._sampling import (
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
from ._simulate import simulate_observations

__all__ = [
    'AbstractPointingModel',
    'AbstractSampler',
    'AngleSampler',
    'BasicPointingModel',
    'DiscretizedBeam',
    'FitResult',
    'Observations',
    'PointingOperator',
    'PointingRows',
    'PointingTransposeOperator',
    'PolarizationFrame',
    'PrecomputedSampler',
    'QuaternionSampler',
    'RotatedSampler',
    'SampleIndex',
    'SamplingKernel',
    'fit',
    'residuals',
    'simulate_observations',
]
