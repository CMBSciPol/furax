r"""Coordinate systems, built on `fastquat.Quaternion`.

Quaternions use scalar-vector storage, i.e. $(1, i, j, k)$ with the scalar part first. All angles
are in radians.

The horizon frame is right-handed, with axes (North, West, Up). The azimuth increases from North
through East, and the elevation from the horizon. A telescope or detector frame looks along its
$z$ axis, so that the boresight of a telescope pointing at (az, el) with a roll angle `roll` is
`azel_to_quat(az, el, roll)`.

A quaternion `q_a_b` rotates the axes of frame `b` into frame `a`, so that names chain:
`q_sky_bore * q_bore_det = q_sky_det`. The frames are `sky`, the frame of the map (e.g. equatorial
or galactic), `hor`, the horizon frame, `bore`, the boresight frame, and `det`, the detector
frame.
"""

from ._angles import (
    XAXIS,
    YAXIS,
    ZAXIS,
    ZSPhi,
    azel_to_quat,
    euler,
    gamma_angle,
    gamma_angle_cos_sin,
    iso_to_quat,
    lonlat_to_quat,
    polarization_angle,
    polarization_angle_cos_sin,
    quat_to_azel,
    quat_to_iso,
    quat_to_lonlat,
    quat_to_xieta,
    xieta_to_quat,
)

__all__ = [
    'XAXIS',
    'YAXIS',
    'ZAXIS',
    'ZSPhi',
    'azel_to_quat',
    'euler',
    'gamma_angle',
    'gamma_angle_cos_sin',
    'iso_to_quat',
    'lonlat_to_quat',
    'polarization_angle',
    'polarization_angle_cos_sin',
    'quat_to_azel',
    'quat_to_iso',
    'quat_to_lonlat',
    'quat_to_xieta',
    'xieta_to_quat',
]
