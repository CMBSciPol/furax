"""Templates of structured signals in the TOD: their bases, operator and Gram.

A [`Template`][furax.mapmaking.templates.Template] is a
[`Basis`][furax.mapmaking.templates.bases.Basis] of functions of time, seen by the detectors and
streams of a TOD through a coupling. [`build_templates`][furax.mapmaking.templates.build_templates]
makes the templates a configuration enables from an observation's data, through the builders of
each kind of template, and [`TemplateOperator`][furax.mapmaking.templates.TemplateOperator] sums
the templates of one TOD.
"""

from ._builders import (
    azimuth_hwp_synchronous_templates,
    azimuth_synchronous_templates,
    binned_azimuth_hwp_synchronous_templates,
    build_templates,
    common_mode_templates,
    hwp_synchronous_templates,
    polarization_field_couplings,
    required_reader_fields,
    spline_drift_templates,
    spline_hwp_synchronous_templates,
    t2p_templates,
    winding_couplings,
    zernike_couplings,
)
from ._operator import Template, TemplateOperator

__all__ = [
    'Template',
    'TemplateOperator',
    'build_templates',
    'required_reader_fields',
    'spline_drift_templates',
    'azimuth_synchronous_templates',
    'hwp_synchronous_templates',
    'azimuth_hwp_synchronous_templates',
    'binned_azimuth_hwp_synchronous_templates',
    'spline_hwp_synchronous_templates',
    't2p_templates',
    'common_mode_templates',
    'zernike_couplings',
    'winding_couplings',
    'polarization_field_couplings',
]
