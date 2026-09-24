import operator
from functools import reduce
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np
import pytest
from fastquat import Quaternion
from numpy.testing import assert_allclose, assert_array_equal
from sotodlib import coords
from sotodlib.mapmaking.demod_mapmaker import project_rhs_demod

from furax import AbstractLinearOperator
from furax.core import IndexOperator, RavelOperator
from furax.interfaces.sotodlib import LazySOTODLibObservation
from furax.mapmaking.acquisition import build_acquisition_operator
from furax.mapmaking.config import SotodlibConfig
from furax.math.coords import gamma_angle, polarization_angle
from furax.obs import QURotationOperator
from furax.obs.landscapes import HealpixLandscape
from furax.obs.spin2 import spin2_cos_sin
from furax.obs.stokes import Stokes, StokesI

FOLDER = Path(__file__).parents[2] / 'data' / 'sotodlib'
NSIDE = 16


@pytest.fixture(scope='module')
def obs():
    return LazySOTODLibObservation(FOLDER / 'test_obs_2.h5').get_data()


@pytest.fixture(scope='module')
def demod_obs():
    config = SotodlibConfig(demodulated=True)
    return LazySOTODLibObservation(FOLDER / 'test_obs_2.h5', sotodlib_config=config).get_data()


def _acquisition(obs, *, demodulated: bool = False) -> AbstractLinearOperator:
    """The furax acquisition for this observation, sampling the pointing on the fly."""
    return build_acquisition_operator(
        HealpixLandscape(nside=NSIDE, stokes='IQU', dtype='float64'),
        Quaternion.from_array(obs.get_boresight_quaternions()),
        Quaternion.from_array(obs.get_detector_quaternions()),
        hwp_angles=None,
        demodulated=demodulated,
        pointing_on_the_fly=True,
    )


def _sotodlib_pointing(obs, hwp: bool):
    hp_geom = coords.healpix_utils.get_geometry(nside=NSIDE, ordering='RING')
    return coords.P.for_tod(obs.data, geom=hp_geom, comps='TQU', hwp=hwp)


def _in_pixel_frame(h: AbstractLinearOperator) -> AbstractLinearOperator:
    """The same acquisition, sampling Q and U in the pixel centre's frame (no transport)."""
    pointing = h.operands[-1]
    landscape = pointing.landscape
    qdet_full = pointing.sampler.quaternions()

    ravel = RavelOperator(1, -1, in_structure=landscape.structure)
    stokes_idx = jnp.arange(len(landscape.stokes))[:, None, None]
    pixels = landscape.quat2index(qdet_full)[None]
    gather = IndexOperator((stokes_idx, pixels), in_structure=ravel.out_structure)
    angles = polarization_angle(qdet_full)
    if pointing.sampler.frame == 'boresight':
        angles -= gamma_angle(pointing.sampler.qdet)[:, None]
    rot = QURotationOperator(angles=angles, in_structure=gather.out_structure)
    return reduce(operator.matmul, h.operands[:-1]) @ rot @ gather @ ravel


def _qu_magnitude(tods: Stokes | jax.Array) -> jax.Array:
    return jnp.hypot(tods.q, tods.u) if isinstance(tods, Stokes) else jnp.abs(jnp.asarray(tods))


def _assert_binning_matches_sotodlib(
    h: AbstractLinearOperator, tods: Stokes, sotodlib_map: np.ndarray
) -> None:
    r"""Compare the two binned maps, allowing for the one place the conventions differ.

    furax carries each pixel's Q and U into the frame of the direction a sample points at;
    sotodlib takes the pixel's own frame. I is untouched by that rotation and must match exactly.
    Q and U differ by it, and the difference is bounded: a sample's contribution is rotated by
    $2\delta_s$ before it is binned, so the binned difference is at most the largest rotation over
    the samples times the sum of the contribution magnitudes in a pixel,

    $$ |\Delta| = \Big|\sum_s (R_s - 1) u_s\Big| \le \max_s |R_s - 1| \sum_s |u_s|, $$

    with $|R(2\delta) u - u| = 2 |\sin\delta| \, |u|$. The binned map is not that sum: Q and U
    cancel as the polarisation angle turns, by a factor of several hundred here, so the sum has to
    be accumulated separately, through the intensity operator.
    """
    # Furax TODs assume power, so they are 2x smaller
    furax_map = 2 * h.T(tods)
    assert_allclose(furax_map.i, sotodlib_map[0], rtol=1e-5, atol=0)

    pointing = h.operands[-1]
    landscape = pointing.landscape
    qdet_full = pointing.sampler.quaternions()
    cos_2delta, _ = spin2_cos_sin(
        *jhp.pix2ang(landscape.nside, landscape.quat2index(qdet_full)),
        *landscape.quat2world(qdet_full),
    )
    max_rotation = float(jnp.sqrt(2 * (1 - cos_2delta)).max())  # max |R(2d) u - u| / |u|
    summed = pointing.as_stokes_i().T(StokesI(_qu_magnitude(tods))).i
    bound = max_rotation * float(summed.max())
    assert np.abs(furax_map.q - sotodlib_map[1]).max() < bound
    assert np.abs(furax_map.u - sotodlib_map[2]).max() < bound

    # the whole backing array at once, which pins the component order against sotodlib's TQU
    assert_allclose((2 * _in_pixel_frame(h).T(tods)).data, sotodlib_map, rtol=1e-5, atol=0)


def _furax_hit_map(h: AbstractLinearOperator) -> np.ndarray:
    """Project ones through the transpose of the Stokes I pointing operator."""
    pointing = h.operands[-1].as_stokes_i()
    return np.asarray(pointing.T(StokesI(jnp.ones(pointing.out_structure.shape))).i)


def test_acquisition_no_hwp_vs_sotodlib(obs):
    """Validate the acquisition transpose against sotodlib.

    The furax acquisition includes a LinearPolarizerOperator which applies a factor of 0.5:
        d = 0.5 * (I[pix] + Q[pix]*cos(2φ) + U[pix]*sin(2φ))

    When projecting back to the sky, H.T @ signal should satisfy:
        furax_map.{i,q,u} = 0.5 * sotodlib_P.to_map(signal).{T,Q,U}

    since sotodlib uses d = I + Q*cos(2φ) + U*sin(2φ) (no 0.5 factor).
    """
    h = _acquisition(obs)
    tods = obs.get_tods()

    pmap = _sotodlib_pointing(obs, hwp=False)
    sotodlib_map = pmap.to_map(tod=obs.data, signal=np.array(tods, dtype=np.float32))

    _assert_binning_matches_sotodlib(h, tods, sotodlib_map)


def test_demod_acquisition_vs_sotodlib(demod_obs):
    """Validate the demodulated acquisition transpose against sotodlib's project_rhs_demod.

    The furax acquisition transpose H.T maps demodulated (I, Q, U) TODs to a sky map.
    This should match sotodlib's project_rhs_demod with unit detector weights.
    """
    h = _acquisition(demod_obs, demodulated=True)
    tod_iqu = demod_obs.get_demodulated_tods(stokes='IQU')

    pmap = _sotodlib_pointing(demod_obs, hwp=True)
    sotodlib_map = project_rhs_demod(
        pmap,
        signalT=np.array(tod_iqu.i, dtype=np.float32),
        signalQ=np.array(tod_iqu.q, dtype=np.float32),
        signalU=np.array(tod_iqu.u, dtype=np.float32),
        det_weightsT=None,
        det_weightsQU=None,
    )

    _assert_binning_matches_sotodlib(h, tod_iqu, sotodlib_map)


@pytest.mark.parametrize('demodulated', [False, True], ids=['no_hwp', 'demod'])
def test_hit_map_vs_sotodlib(obs, demodulated: bool):
    """Validate the hit map against sotodlib.

    The furax hit map is computed by projecting a vector of ones through the transpose of the
    Stokes I pointing operator. This should be identical to sotodlib's pmap.to_map(signal=ones)[0],
    which accumulates sample counts per pixel. The transport rotates each sample's Q and U, and
    moves no sample to another pixel, so the two must agree exactly.
    """
    h = _acquisition(obs, demodulated=demodulated)
    pmap = _sotodlib_pointing(obs, hwp=demodulated)

    ndet, nsamp = h.operands[-1].out_structure.shape
    sotodlib_hits = pmap.to_map(signal=np.ones((ndet, nsamp), dtype=np.float32))[0]

    assert_array_equal(_furax_hit_map(h), sotodlib_hits)
