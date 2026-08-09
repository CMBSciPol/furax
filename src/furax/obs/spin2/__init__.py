"""Spin-2 parallel transport for interpolated Stokes maps.

HEALPix stores $Q$ and $U$ in the local meridian basis of each pixel. Neighbouring pixels have
different meridians, so their stored values are not components of one object and combining them with
scalar interpolation weights is ill-defined. This package supplies the frame rotation that carries a
neighbour's $(Q, U)$ into the basis of the point being interpolated.
"""

from ._transport import spin2_cos_sin, spin2_cos_sin_zs

__all__ = [
    'spin2_cos_sin',
    'spin2_cos_sin_zs',
]
