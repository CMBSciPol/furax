import jax
import jax.numpy as jnp
import jax_healpy as jhp
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose

from furax.core import CompositionOperator
from furax.mapmaking.acquisition import build_acquisition_operator
from furax.math.coords import XiEtaAngles, gamma_angle, polarization_angle
from furax.obs.landscapes import HealpixLandscape
from furax.obs.pointing import PointingOperator
from furax.obs.sampling import DiscretizedBeam
from furax.obs.spin2 import spin2_cos_sin
from furax.obs.stokes import Stokes, StokesIQU

NSIDE = 4
NDET, NSAMP = 3, 10


def _transport(landscape: HealpixLandscape, qdet_full: Quaternion) -> tuple[jax.Array, jax.Array]:
    """(cos 2d, -sin 2d) carrying a pixel's Q, U into the frame of the direction sampled at."""
    indices = landscape.quat2index(qdet_full)
    theta, phi = landscape.quat2world(qdet_full)
    return spin2_cos_sin(*jhp.pix2ang(landscape.nside, indices), theta, phi)


def test_no_hwp_acquisition_formula() -> None:
    """No-HWP acquisition equals 0.5*(I + cos(2pa)*Q + sin(2pa)*U).

    Q and U are the pixel's, carried into the frame of the sampled direction first.
    """
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    qbore = Quaternion.random(k1, (NSAMP,))
    qdet = Quaternion.random(k2, (NDET,))
    sky = landscape.normal(k3)

    acq = build_acquisition_operator(landscape, qbore, qdet)
    tod = acq(sky)

    # Reference: sample pixels and apply polarization angle formula directly
    qdet_full = qbore[None, :] * qdet[:, None]  # (ndet, nsamp)
    pa = polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)

    cos_2d, sin_2d = _transport(landscape, qdet_full)
    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices] * cos_2d + sky.u.ravel()[indices] * sin_2d
    U_p = -sky.q.ravel()[indices] * sin_2d + sky.u.ravel()[indices] * cos_2d
    expected = 0.5 * (I_p + jnp.cos(2 * pa) * Q_p + jnp.sin(2 * pa) * U_p)

    assert_allclose(tod, expected, rtol=1e-10)


def test_no_hwp_acquisition_transpose_formula() -> None:
    """No-HWP acquisition transpose is A^T d: I += 0.5*d, Q += 0.5*cos(2pa)*d, U += 0.5*sin(2pa)*d."""
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.key(1)
    k1, k2, k3 = jax.random.split(key, 3)
    qbore = Quaternion.random(k1, (NSAMP,))
    qdet = Quaternion.random(k2, (NDET,))
    tod = jax.random.normal(k3, (NDET, NSAMP), dtype=jnp.float64)

    acq = build_acquisition_operator(landscape, qbore, qdet)
    sky = acq.T(tod)

    # Reference: scatter TOD into sky weighted by polarization angle
    qdet_full = qbore[None, :] * qdet[:, None]  # (ndet, nsamp)
    pa = polarization_angle(qdet_full)  # (ndet, nsamp)
    flat_indices = landscape.quat2index(qdet_full).ravel()
    # the sample's (Q, U) contribution, carried back into the pixel's own frame before binning
    cos_2d, sin_2d = _transport(landscape, qdet_full)
    q_s = 0.5 * jnp.cos(2 * pa) * tod
    u_s = 0.5 * jnp.sin(2 * pa) * tod

    d = tod.ravel()
    npix = len(landscape)
    zeros = jnp.zeros(npix)
    expected_I = zeros.at[flat_indices].add(0.5 * d)
    expected_Q = zeros.at[flat_indices].add((q_s * cos_2d - u_s * sin_2d).ravel())
    expected_U = zeros.at[flat_indices].add((q_s * sin_2d + u_s * cos_2d).ravel())

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

    key = jax.random.key(2)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qbore = Quaternion.random(k1, (NSAMP,))
    qdet = Quaternion.random(k2, (NDET,))
    hwp_angles = jax.random.uniform(k4, (NSAMP,), dtype=jnp.float64, maxval=jnp.pi)
    sky = landscape.normal(k3)

    acq = build_acquisition_operator(landscape, qbore, qdet, hwp_angles=hwp_angles)
    tod = acq(sky)

    qdet_full = qbore[None, :] * qdet[:, None]  # (ndet, nsamp)
    pa = polarization_angle(qdet_full)  # (ndet, nsamp)
    indices = landscape.quat2index(qdet_full)  # (ndet, nsamp)
    gamma = gamma_angle(qdet)[:, None]  # (ndet, 1)
    phi = 2 * (2 * hwp_angles[None, :] + pa - 2 * gamma)  # (ndet, nsamp)

    cos_2d, sin_2d = _transport(landscape, qdet_full)
    I_p = sky.i.ravel()[indices]
    Q_p = sky.q.ravel()[indices] * cos_2d + sky.u.ravel()[indices] * sin_2d
    U_p = -sky.q.ravel()[indices] * sin_2d + sky.u.ravel()[indices] * cos_2d
    expected = 0.5 * (I_p + jnp.cos(phi) * Q_p + jnp.sin(phi) * U_p)

    assert_allclose(tod, expected, rtol=1e-10)


def test_hwp_acquisition_transpose_formula() -> None:
    """HWP acquisition transpose: I += 0.5*d, Q += 0.5*cos(phi)*d, U += 0.5*sin(phi)*d.

    See test_hw_acquisition_formula for a description of phi.
    """
    landscape = HealpixLandscape(NSIDE, 'IQU')

    key = jax.random.key(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    qbore = Quaternion.random(k1, (NSAMP,))
    qdet = Quaternion.random(k2, (NDET,))
    hwp_angles = jax.random.uniform(k3, (NSAMP,), dtype=jnp.float64, maxval=jnp.pi)
    tod = jax.random.normal(k4, (NDET, NSAMP), dtype=jnp.float64)

    acq = build_acquisition_operator(landscape, qbore, qdet, hwp_angles=hwp_angles)
    sky = acq.T(tod)

    qdet_full = qbore[None, :] * qdet[:, None]  # (ndet, nsamp)
    pa = polarization_angle(qdet_full)  # (ndet, nsamp)
    flat_indices = landscape.quat2index(qdet_full).ravel()
    gamma = gamma_angle(qdet)[:, None]  # (ndet, 1)
    phi = 2 * (2 * hwp_angles[None, :] + pa - 2 * gamma)  # (ndet, nsamp)

    # the sample's (Q, U) contribution, carried back into the pixel's own frame before binning
    cos_2d, sin_2d = _transport(landscape, qdet_full)
    q_s = 0.5 * jnp.cos(phi) * tod
    u_s = 0.5 * jnp.sin(phi) * tod

    d = tod.ravel()
    npix = len(landscape)
    zeros = jnp.zeros(npix)
    expected_I = zeros.at[flat_indices].add(0.5 * d)
    expected_Q = zeros.at[flat_indices].add((q_s * cos_2d - u_s * sin_2d).ravel())
    expected_U = zeros.at[flat_indices].add((q_s * sin_2d + u_s * cos_2d).ravel())

    assert_allclose(sky.i, expected_I, rtol=1e-10)
    assert_allclose(sky.q, expected_Q, rtol=1e-10)
    assert_allclose(sky.u, expected_U, rtol=1e-10)


def test_last_acquisition_operand_is_pointing() -> None:
    landscape = HealpixLandscape(NSIDE, 'IQU')
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    qbore = Quaternion.random(k1, (NSAMP,))
    qdet = Quaternion.random(k2, (NDET,))
    acq = build_acquisition_operator(landscape, qbore, qdet)
    assert isinstance(acq, CompositionOperator)
    assert isinstance(acq.operands[-1], PointingOperator)


@pytest.mark.parametrize('mode', ['no-hwp', 'hwp', 'demodulated'])
class TestBeam:
    """The acquisition reads the sky through the pointing's beam, whatever the HWP setup."""

    @staticmethod
    def _acquisition(mode: str, beam: DiscretizedBeam | None) -> CompositionOperator:
        landscape = HealpixLandscape(NSIDE, 'IQU')
        k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
        qbore = Quaternion.random(k1, (NSAMP,))
        qdet = Quaternion.random(k2, (NDET,))
        hwp_angles = jax.random.uniform(k3, (NSAMP,), maxval=jnp.pi) if mode == 'hwp' else None
        acq = build_acquisition_operator(
            landscape,
            qbore,
            qdet,
            hwp_angles,
            demodulated=mode == 'demodulated',
            pointing_beam=beam,
        )
        assert isinstance(acq, CompositionOperator)
        return acq

    @staticmethod
    def _sky() -> StokesIQU:
        return HealpixLandscape(NSIDE, 'IQU').normal(jax.random.key(5))

    @staticmethod
    def _tod(acq: CompositionOperator, sky: StokesIQU) -> jax.Array:
        # demodulated TOD hold one stream per Stokes component
        tod = acq(sky)
        return tod.data if isinstance(tod, Stokes) else tod

    def test_a_unit_beam_on_the_line_of_sight_is_no_beam(self, mode) -> None:
        beam = DiscretizedBeam.create(Quaternion.ones((1,)), jnp.ones(1))
        acq = self._acquisition(mode, beam)

        pointing = acq.operands[-1]
        assert isinstance(pointing, PointingOperator)
        assert pointing.sampler.kernel.beam is not None
        sky = self._sky()
        assert_allclose(
            self._tod(acq, sky), self._tod(self._acquisition(mode, None), sky), rtol=1e-12
        )

    def test_the_beam_weighs_the_acquisition_of_each_node(self, mode) -> None:
        # nodes far enough apart to read different pixels at NSIDE = 4
        nodes = XiEtaAngles(
            jnp.array([0.0, 0.3]), jnp.array([0.0, -0.2]), jnp.zeros(2)
        ).to_quaternion()
        weights = jnp.array([0.3, 0.7])
        sky = self._sky()

        tod = self._tod(self._acquisition(mode, DiscretizedBeam.create(nodes, weights)), sky)

        node_tods = [
            self._tod(
                self._acquisition(mode, DiscretizedBeam.create(nodes[k : k + 1], jnp.ones(1))), sky
            )
            for k in range(2)
        ]
        assert not jnp.allclose(node_tods[0], node_tods[1])
        assert_allclose(tod, weights[0] * node_tods[0] + weights[1] * node_tods[1], rtol=1e-12)
