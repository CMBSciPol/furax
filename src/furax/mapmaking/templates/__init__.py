"""Templates of structured signals in the TOD: their bases, operators and Gram.

A template turns a small set of amplitudes into a time stream, so the mapmaker can fit and remove
unwanted but predictable signals (slow drifts, scan-synchronous pickup, HWP-synchronous lines,
T-to-P leakage). Each template is described by a
[`Basis`][furax.mapmaking.templates.bases.Basis] of functions of time, built from the
observation's data (`polynomial_basis`, `t2p_basis`, ...).
[`TemplateOperator`][furax.mapmaking.templates.TemplateOperator] combines the enabled templates
into one operator, keyed by template name, and
[`StokesTemplateOperator`][furax.mapmaking.templates.StokesTemplateOperator] does the same for a
[`Stokes`][furax.obs.stokes.Stokes] TOD, one amplitude set per leg. The Gram of the templates and
the Pomme offsets live in [`furax.mapmaking.templates.gram`][] and
[`furax.mapmaking.templates.pomme`][].
"""

from ._templates import (
    AbstractTemplateOperator,
    StokesLeg,
    StokesTemplateOperator,
    TemplateOperator,
    azimuth_hwp_synchronous_basis,
    binned_azimuth_hwp_synchronous_basis,
    binned_azimuth_synchronous_basis,
    hwp_synchronous_basis,
    polynomial_basis,
    scan_synchronous_basis,
    spline_hwp_synchronous_basis,
    t2p_basis,
)

__all__ = [
    'StokesLeg',
    'polynomial_basis',
    't2p_basis',
    'scan_synchronous_basis',
    'binned_azimuth_synchronous_basis',
    'hwp_synchronous_basis',
    'azimuth_hwp_synchronous_basis',
    'binned_azimuth_hwp_synchronous_basis',
    'spline_hwp_synchronous_basis',
    'AbstractTemplateOperator',
    'TemplateOperator',
    'StokesTemplateOperator',
]
