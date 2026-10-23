import jax
import jax.numpy as jnp
import jax_healpy as jhp
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose

from furax.core import CompositionOperator
from furax.mapmaking.acquisition import build_acquisition_operator
from furax.math.coords import gamma_angle, polarization_angle
from furax.obs.landscapes import HealpixLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.spin2 import spin2_cos_sin

NSIDE = 4
NDET, NSAMP = 3, 10


def _transport(landscape: HealpixLandscape, qdet_full: Quaternion) -> tuple[jax.Array, jax.Array]:
    """(cos 2d, -sin 2d) carrying a pixel's Q, U into the frame of the direction sampled at."""
    indices = landscape.quat2index(qdet_full)
    theta, phi = landscape.quat2world(qdet_full)
    return spin2_cos_sin(*jhp.pix2ang(landscape.nside, indices), theta, phi)


@pytest.mark.parametrize('hwp', [False, True], ids=['no_hwp', 'hwp'])
def test_acquisition_formula(hwp: bool) -> None:
    """The acquisition is d = 0.5 (I + cos(phi) Q + sin(phi) U), its transpose the binning of d.

    Q and U are the pixel's, carried into the frame of the sampled direction first. Without a
    HWP, phi = 2 pa, with pa the polarization angle; with one, phi = 2 (2 chi + pa - 2 gamma),
    with chi the HWP angle and gamma the detector orientation: the HWP angle is measured in the
    boresight frame, and the HWP flips the detector angle.
    """
    landscape = HealpixLandscape(NSIDE, 'IQU')
    k = jax.random.split(jax.random.key(0), 5)
    qbore = Quaternion.random(k[0], (NSAMP,))
    qdet = Quaternion.random(k[1], (NDET,))
    hwp_angles = jax.random.uniform(k[2], (NSAMP,), dtype=jnp.float64, maxval=jnp.pi)
    acq = build_acquisition_operator(landscape, qbore, qdet, hwp_angles=hwp_angles if hwp else None)
    # its pointing is last, where the composition rules of the pointing operator look for it
    assert isinstance(acq, CompositionOperator)
    assert isinstance(acq.operands[-1], PointingOperator)

    qdet_full = qbore[None, :] * qdet[:, None]  # (ndet, nsamp)
    pa = polarization_angle(qdet_full)
    phi = 2 * (2 * hwp_angles[None, :] + pa - 2 * gamma_angle(qdet)[:, None]) if hwp else 2 * pa
    indices = landscape.quat2index(qdet_full)
    cos_2d, sin_2d = _transport(landscape, qdet_full)

    sky = landscape.normal(k[3])
    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices] * cos_2d + sky.u.ravel()[indices] * sin_2d
    U_p = -sky.q.ravel()[indices] * sin_2d + sky.u.ravel()[indices] * cos_2d
    expected = 0.5 * (I_p + jnp.cos(phi) * Q_p + jnp.sin(phi) * U_p)
    assert_allclose(acq(sky), expected, rtol=1e-10)

    # the sample's (Q, U) contribution, carried back into the pixel's own frame before binning
    tod = jax.random.normal(k[4], (NDET, NSAMP), dtype=jnp.float64)
    q_s = 0.5 * jnp.cos(phi) * tod
    u_s = 0.5 * jnp.sin(phi) * tod
    flat = indices.ravel()
    zeros = jnp.zeros(len(landscape))
    binned = acq.T(tod)
    assert_allclose(binned.i, zeros.at[flat].add(0.5 * tod.ravel()), rtol=1e-10)
    assert_allclose(binned.q, zeros.at[flat].add((q_s * cos_2d - u_s * sin_2d).ravel()), rtol=1e-10)
    assert_allclose(binned.u, zeros.at[flat].add((q_s * sin_2d + u_s * cos_2d).ravel()), rtol=1e-10)
