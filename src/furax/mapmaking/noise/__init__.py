"""Noise models of a TOD: white and 1/f per detector, and low-rank across detectors.

Each model gives its covariance and weight operators, and draws realizations; the package also holds
their fits and the spectral estimates they are fitted on.
"""

from ._base import NoiseModel
from .binned import BinnedNoiseModel
from .lowrank import LowRankModesOperator, LowRankNoiseModel, ModesCore, bin_edges
from .one_over_f import OneOverFNoiseModel
from .spectra import apodization_window, padding_aware_welch
from .white import WhiteNoiseModel

__all__ = [
    'OneOverFNoiseModel',
    'BinnedNoiseModel',
    'LowRankModesOperator',
    'LowRankNoiseModel',
    'ModesCore',
    'NoiseModel',
    'WhiteNoiseModel',
    'apodization_window',
    'bin_edges',
    'padding_aware_welch',
]
