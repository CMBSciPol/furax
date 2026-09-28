r"""Coordinate systems, built on `fastquat.Quaternion`.

Quaternions use scalar-vector storage, i.e. $(1, i, j, k)$ with the scalar part first. All angles
are in radians.

The horizon frame is right-handed, with axes (North, West, Up). The azimuth increases from North
through East, and the elevation from the horizon. A telescope or detector frame looks along its
$z$ axis, so that the boresight of a telescope pointing at (az, el) with a roll angle `roll` is
`AzElAngles(az, el, roll).to_quaternion()`.
"""

from ._angles import (
    XAXIS,
    YAXIS,
    ZAXIS,
    AzElAngles,
    IsoAngles,
    LonLatAngles,
    XiEtaAngles,
    ZSPhi,
    euler,
    gamma_angle,
    gamma_angle_cos_sin,
    polarization_angle,
    polarization_angle_cos_sin,
)
from ._geometry import azel_to_vec, tangent_basis, vec_to_azel

__all__ = [
    'XAXIS',
    'YAXIS',
    'ZAXIS',
    'AzElAngles',
    'IsoAngles',
    'LonLatAngles',
    'XiEtaAngles',
    'ZSPhi',
    'azel_to_vec',
    'euler',
    'gamma_angle',
    'gamma_angle_cos_sin',
    'polarization_angle',
    'polarization_angle_cos_sin',
    'tangent_basis',
    'vec_to_azel',
]
