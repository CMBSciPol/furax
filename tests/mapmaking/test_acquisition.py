import jax
import jax.numpy as jnp
from fastquat import Quaternion
from numpy.testing import assert_allclose

from furax.mapmaking.acquisition import build_acquisition_operator
from furax.math.coords import to_gamma_angles, to_polarization_angle
from furax.obs.landscapes import HealpixLandscape

NSIDE = 4
NDET, NSAMP = 3, 10


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> Quaternion:
    q = jax.random.normal(key, (*shape, 4))
    return Quaternion.from_array(q / jnp.linalg.norm(q, axis=-1, keepdims=True))


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
    qdet_full = qbore.reshape((1, -1)) * qdet.reshape((-1, 1))  # (ndet, nsamp)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)

    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices]
    U_p = sky.u.ravel()[indices]
    expected = 0.5 * (I_p + jnp.cos(2 * pa) * Q_p + jnp.sin(2 * pa) * U_p)

    assert_allclose(tod, expected, rtol=1e-10)


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
    qdet_full = qbore.reshape((1, -1)) * qdet.reshape((-1, 1))  # (ndet, nsamp)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    flat_indices = landscape.quat2index(qdet_full).ravel()
    d = tod.ravel()
    npix = len(landscape)
    zeros = jnp.zeros(npix)
    expected_I = zeros.at[flat_indices].add(0.5 * d)
    expected_Q = zeros.at[flat_indices].add(0.5 * jnp.cos(2 * pa).ravel() * d)
    expected_U = zeros.at[flat_indices].add(0.5 * jnp.sin(2 * pa).ravel() * d)

    assert_allclose(sky.i, expected_I, rtol=1e-10)
    assert_allclose(sky.q, expected_Q, rtol=1e-10)
    assert_allclose(sky.u, expected_U, rtol=1e-10)


def test_hwp_acquisition_formula() -> None:
    """HWP acquisition: d = 0.5*(I + cos(phi)*Q + sin(phi)*U).

    phi = 2*(2*chi + pa - 2*gamma) where
    - pa is the polarization angle
    - chi is the HWP angle
    - gamma is the detector orientation angle

    This reflects the fact that we first rotate into the boresight frame,
    where the HWP angle is measured, before applying the HWP rotation. Also,
    the detector gamma angle is flipped as a result of the HWP.
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

    qdet_full = qbore.reshape((1, -1)) * qdet.reshape((-1, 1))  # (ndet, nsamp)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)
    gamma = to_gamma_angles(qdet)[:, None]  # (ndet, 1)
    phi = 2 * (2 * hwp_angles[None, :] + pa - 2 * gamma)  # (ndet, nsamp)

    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices]
    U_p = sky.u.ravel()[indices]
    expected = 0.5 * (I_p + jnp.cos(phi) * Q_p + jnp.sin(phi) * U_p)

    assert_allclose(tod, expected, rtol=1e-10)


def test_hwp_acquisition_transpose_formula() -> None:
    """HWP acquisition transpose: I += 0.5*d, Q += 0.5*cos(phi)*d, U += 0.5*sin(phi)*d.

    See test_hw_acquisition_formula for a description of phi.
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

    qdet_full = qbore.reshape((1, -1)) * qdet.reshape((-1, 1))  # (ndet, nsamp)
    pa = to_polarization_angle(qdet_full)  # (ndet, nsamp)
    flat_indices = landscape.quat2index(qdet_full).ravel()
    gamma = to_gamma_angles(qdet)[:, None]  # (ndet, 1)
    phi = 2 * (2 * hwp_angles[None, :] + pa - 2 * gamma)  # (ndet, nsamp)

    d = tod.ravel()
    npix = len(landscape)
    zeros = jnp.zeros(npix)
    expected_I = zeros.at[flat_indices].add(0.5 * d)
    expected_Q = zeros.at[flat_indices].add(0.5 * jnp.cos(phi).ravel() * d)
    expected_U = zeros.at[flat_indices].add(0.5 * jnp.sin(phi).ravel() * d)

    assert_allclose(sky.i, expected_I, rtol=1e-10)
    assert_allclose(sky.q, expected_Q, rtol=1e-10)
    assert_allclose(sky.u, expected_U, rtol=1e-10)
