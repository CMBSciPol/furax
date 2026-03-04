import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from furax.mapmaking.acquisition import build_acquisition_operator
from furax.math.quaternion import qmul, to_gamma_angles, to_polarization_angle
from furax.obs.landscapes import HealpixLandscape

NSIDE = 4
NDET, NSAMP = 3, 10


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def test_no_hwp_acquisition_formula() -> None:
    """No-HWP acquisition equals 0.5*(I + cos(2pa)*Q + sin(2pa)*U)."""
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    qbore = _random_unit_quats(k1, (NSAMP,))
    qdet = _random_unit_quats(k2, (NDET,))
    sky = landscape.normal(k3)

    acq = build_acquisition_operator(landscape, qbore, qdet)
    tod = acq(sky)

    # Reference: sample pixels and apply polarization angle formula directly
    qdet_full = qmul(qbore, qdet[:, None, :])  # (ndet, nsamp, 4)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)

    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices]
    U_p = sky.u.ravel()[indices]
    expected = 0.5 * (I_p + jnp.cos(2 * pa) * Q_p + jnp.sin(2 * pa) * U_p)

    assert_allclose(np.asarray(tod), np.asarray(expected), rtol=1e-10)


def test_no_hwp_acquisition_transpose_formula() -> None:
    """No-HWP acquisition transpose is A^T d: I += 0.5*d, Q += 0.5*cos(2pa)*d, U += 0.5*sin(2pa)*d."""
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    qbore = _random_unit_quats(k1, (NSAMP,))
    qdet = _random_unit_quats(k2, (NDET,))
    tod = jax.random.normal(k3, (NDET, NSAMP), dtype=jnp.float64)

    acq = build_acquisition_operator(landscape, qbore, qdet)
    sky = acq.T(tod)

    # Reference: scatter TOD into sky weighted by polarization angle
    qdet_full = qmul(qbore, qdet[:, None, :])  # (ndet, nsamp, 4)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    flat_indices = np.asarray(landscape.quat2index(qdet_full)).ravel()
    d = np.asarray(tod).ravel()
    pa_flat = np.asarray(pa).ravel()

    npix = len(landscape)
    expected_I = np.zeros(npix)
    expected_Q = np.zeros(npix)
    expected_U = np.zeros(npix)
    np.add.at(expected_I, flat_indices, 0.5 * d)
    np.add.at(expected_Q, flat_indices, 0.5 * np.cos(2 * pa_flat) * d)
    np.add.at(expected_U, flat_indices, 0.5 * np.sin(2 * pa_flat) * d)

    assert_allclose(np.asarray(sky.i), expected_I, rtol=1e-10)
    assert_allclose(np.asarray(sky.q), expected_Q, rtol=1e-10)
    assert_allclose(np.asarray(sky.u), expected_U, rtol=1e-10)


def test_hwp_acquisition_formula() -> None:
    """HWP acquisition: d = 0.5*(I + cos(phi)*Q + sin(phi)*U).

    phi = 2*(pa + 2*(alpha - gamma)) where pa is the polarization angle, alpha is the HWP angle,
    and gamma is the detector orientation angle.
    """
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.PRNGKey(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qbore = _random_unit_quats(k1, (NSAMP,))
    qdet = _random_unit_quats(k2, (NDET,))
    hwp_angles = jax.random.uniform(k4, (NSAMP,), dtype=jnp.float64, maxval=jnp.pi)
    sky = landscape.normal(k3)

    acq = build_acquisition_operator(landscape, qbore, qdet, hwp_angles=hwp_angles)
    tod = acq(sky)

    qdet_full = qmul(qbore, qdet[:, None, :])  # (ndet, nsamp, 4)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)
    gamma = to_gamma_angles(qdet)  # (ndet,)
    phi = 2 * (pa + 2 * (hwp_angles[None, :] - gamma[:, None]))  # (ndet, nsamp)

    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices]
    U_p = sky.u.ravel()[indices]
    expected = 0.5 * (I_p + jnp.cos(phi) * Q_p + jnp.sin(phi) * U_p)

    assert_allclose(np.asarray(tod), np.asarray(expected), rtol=1e-10)


def test_hwp_acquisition_transpose_formula() -> None:
    """HWP acquisition transpose: I += 0.5*d, Q += 0.5*cos(phi)*d, U += 0.5*sin(phi)*d.

    phi = 2*(pa + 2*(alpha - gamma)) where pa is the polarization angle, alpha is the HWP angle,
    and gamma is the detector orientation angle.
    """
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qbore = _random_unit_quats(k1, (NSAMP,))
    qdet = _random_unit_quats(k2, (NDET,))
    hwp_angles = jax.random.uniform(k3, (NSAMP,), dtype=jnp.float64, maxval=jnp.pi)
    tod = jax.random.normal(k4, (NDET, NSAMP), dtype=jnp.float64)

    acq = build_acquisition_operator(landscape, qbore, qdet, hwp_angles=hwp_angles)
    sky = acq.T(tod)

    qdet_full = qmul(qbore, qdet[:, None, :])  # (ndet, nsamp, 4)
    pa = np.asarray(to_polarization_angle(qdet_full))  # (ndet, nsamp)
    flat_indices = np.asarray(landscape.quat2index(qdet_full)).ravel()
    gamma = np.asarray(to_gamma_angles(qdet))  # (ndet,)
    phi = 2 * (pa + 2 * (np.asarray(hwp_angles)[None, :] - gamma[:, None]))  # (ndet, nsamp)

    d = np.asarray(tod).ravel()
    phi_flat = phi.ravel()

    npix = len(landscape)
    expected_I = np.zeros(npix)
    expected_Q = np.zeros(npix)
    expected_U = np.zeros(npix)
    np.add.at(expected_I, flat_indices, 0.5 * d)
    np.add.at(expected_Q, flat_indices, 0.5 * np.cos(phi_flat) * d)
    np.add.at(expected_U, flat_indices, 0.5 * np.sin(phi_flat) * d)

    assert_allclose(np.asarray(sky.i), expected_I, rtol=1e-10)
    assert_allclose(np.asarray(sky.q), expected_Q, rtol=1e-10)
    assert_allclose(np.asarray(sky.u), expected_U, rtol=1e-10)
