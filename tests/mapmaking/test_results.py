import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_array_almost_equal, assert_array_equal

from furax.mapmaking.results import MapMakingResults
from furax.obs.landscapes import (
    AstropyWCSLandscape,
    CARLandscape,
    HealpixLandscape,
    StokesLandscape,
    WCSProjection,
)
from furax.obs.stokes import StokesIQU

NSIDE = 4
NPIX = 12 * NSIDE**2
MAP_SHAPE = (8, 10)  # (ny, nx)


# --- fixtures ---


@pytest.fixture
def healpix_results() -> MapMakingResults:
    rng = np.random.default_rng(0)
    landscape = HealpixLandscape(NSIDE, stokes='IQU')
    sky = StokesIQU(
        i=jnp.array(rng.standard_normal(NPIX)),
        q=jnp.array(rng.standard_normal(NPIX)),
        u=jnp.array(rng.standard_normal(NPIX)),
    )
    hit_map = jnp.array(rng.integers(1, 100, NPIX))
    icov = jnp.array(rng.standard_normal((3, 3, NPIX)))
    return MapMakingResults(map=sky, landscape=landscape, hit_map=hit_map, icov=icov)


@pytest.fixture
def car_results() -> MapMakingResults:
    rng = np.random.default_rng(0)
    proj = WCSProjection(crpix=(5.5, 4.5), crval=(0.0, 0.0), cdelt=(-0.1, 0.1))
    landscape = CARLandscape(MAP_SHAPE, proj, stokes='IQU')
    sky = StokesIQU(
        i=jnp.array(rng.standard_normal(MAP_SHAPE)),
        q=jnp.array(rng.standard_normal(MAP_SHAPE)),
        u=jnp.array(rng.standard_normal(MAP_SHAPE)),
    )
    hit_map = jnp.array(rng.integers(1, 100, MAP_SHAPE))
    icov = jnp.array(rng.standard_normal((3, 3, *MAP_SHAPE)))
    return MapMakingResults(map=sky, landscape=landscape, hit_map=hit_map, icov=icov)


@pytest.fixture
def astropy_wcs_results() -> MapMakingResults:
    rng = np.random.default_rng(0)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    wcs.wcs.crpix = [5.5, 4.5]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [-0.1, 0.1]
    wcs.wcs.set()
    landscape = AstropyWCSLandscape(MAP_SHAPE, wcs, stokes='IQU')
    sky = StokesIQU(
        i=jnp.array(rng.standard_normal(MAP_SHAPE)),
        q=jnp.array(rng.standard_normal(MAP_SHAPE)),
        u=jnp.array(rng.standard_normal(MAP_SHAPE)),
    )
    hit_map = jnp.array(rng.integers(1, 100, MAP_SHAPE))
    icov = jnp.array(rng.standard_normal((3, 3, *MAP_SHAPE)))
    return MapMakingResults(map=sky, landscape=landscape, hit_map=hit_map, icov=icov)


class _UnknownLandscape(StokesLandscape):
    def __init__(self) -> None:
        super().__init__(shape=(NPIX,), stokes='IQU')

    def world2pixel(self, theta, phi):
        return (jnp.zeros_like(theta),)

    def normal(self, key):
        return StokesIQU.full(self.shape, 0.0, self.dtype)

    def uniform(self, key, minval=0.0, maxval=1.0):
        return StokesIQU.full(self.shape, 0.0, self.dtype)

    def full(self, fill_value):
        return StokesIQU.full(self.shape, fill_value, self.dtype)


@pytest.fixture
def unknown_landscape_results() -> MapMakingResults:
    rng = np.random.default_rng(0)
    landscape = _UnknownLandscape()
    sky = StokesIQU(
        i=jnp.array(rng.standard_normal(NPIX)),
        q=jnp.array(rng.standard_normal(NPIX)),
        u=jnp.array(rng.standard_normal(NPIX)),
    )
    hit_map = jnp.array(rng.integers(1, 100, NPIX))
    icov = jnp.array(rng.standard_normal((3, 3, NPIX)))
    return MapMakingResults(map=sky, landscape=landscape, hit_map=hit_map, icov=icov)


# --- HEALPix ---


def test_healpix_save_produces_fits(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    assert (tmp_path / 'map.fits').exists()
    assert (tmp_path / 'hit_map.fits').exists()
    assert (tmp_path / 'icov.fits').exists()


def test_healpix_map_roundtrip(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    i_read, q_read, u_read = hp.read_map(tmp_path / 'map.fits', field=[0, 1, 2])
    assert_array_almost_equal(i_read, healpix_results.map.i)
    assert_array_almost_equal(q_read, healpix_results.map.q)
    assert_array_almost_equal(u_read, healpix_results.map.u)


def test_healpix_hit_map_roundtrip(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    hit_map_read = hp.read_map(tmp_path / 'hit_map.fits')
    assert_array_equal(hit_map_read, healpix_results.hit_map)


def test_healpix_icov_column_names(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    from astropy.io import fits as _fits

    with _fits.open(tmp_path / 'icov.fits') as hdul:
        names = [c.name for c in hdul[1].columns]
    assert names == ['II', 'IQ', 'IU', 'QQ', 'QU', 'UU']


def test_healpix_icov_roundtrip(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    # icov is stored as upper triangle: (n_upper, npix) with n_upper=6 for IQU
    icov_read = np.array(hp.read_map(tmp_path / 'icov.fits', field=list(range(6))))
    icov = np.array(healpix_results.icov)
    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    expected = np.stack([icov[i, j] for i, j in upper])
    assert_array_almost_equal(icov_read, expected)


def test_healpix_hit_map_column_name(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    with fits.open(tmp_path / 'hit_map.fits') as hdul:
        assert hdul[1].columns[0].name == 'HITS'


def test_healpix_ordering(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    expected = 'NESTED' if healpix_results.landscape.nested else 'RING'
    with fits.open(tmp_path / 'map.fits') as hdul:
        assert hdul[1].header['ORDERING'] == expected


# --- WCS (CAR) ---


def test_wcs_save_produces_fits(car_results, tmp_path):
    car_results.save(tmp_path)
    assert (tmp_path / 'map.fits').exists()
    assert (tmp_path / 'hit_map.fits').exists()
    assert (tmp_path / 'icov.fits').exists()


def test_wcs_map_has_wcs_header(car_results, tmp_path):
    car_results.save(tmp_path)
    header = fits.open(tmp_path / 'map.fits')[0].header
    assert 'CTYPE1' in header
    assert 'CRVAL1' in header
    assert 'CRPIX1' in header
    assert 'CDELT1' in header


def test_wcs_map_roundtrip(car_results, tmp_path):
    car_results.save(tmp_path)
    data = fits.open(tmp_path / 'map.fits')[0].data  # shape (3, ny, nx)
    assert_array_almost_equal(data[0], car_results.map.i)
    assert_array_almost_equal(data[1], car_results.map.q)
    assert_array_almost_equal(data[2], car_results.map.u)


def test_wcs_hit_map_roundtrip(car_results, tmp_path):
    car_results.save(tmp_path)
    data = fits.open(tmp_path / 'hit_map.fits')[0].data
    assert_array_equal(data, car_results.hit_map)


def test_wcs_icov_roundtrip(car_results, tmp_path):
    car_results.save(tmp_path)
    # icov is stored as upper triangle: (n_upper, ny, nx)
    data = fits.open(tmp_path / 'icov.fits')[0].data
    icov = np.array(car_results.icov)
    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    expected = np.stack([icov[i, j] for i, j in upper])
    assert_array_almost_equal(data, expected)


# --- AstropyWCS ---


def test_astropy_wcs_save_produces_fits(astropy_wcs_results, tmp_path):
    astropy_wcs_results.save(tmp_path)
    assert (tmp_path / 'map.fits').exists()
    assert (tmp_path / 'hit_map.fits').exists()
    assert (tmp_path / 'icov.fits').exists()


def test_astropy_wcs_map_has_wcs_header(astropy_wcs_results, tmp_path):
    astropy_wcs_results.save(tmp_path)
    header = fits.open(tmp_path / 'map.fits')[0].header
    assert 'CTYPE1' in header
    assert 'CRVAL1' in header
    assert 'CRPIX1' in header
    assert 'CDELT1' in header


def test_astropy_wcs_map_roundtrip(astropy_wcs_results, tmp_path):
    astropy_wcs_results.save(tmp_path)
    data = fits.open(tmp_path / 'map.fits')[0].data
    assert_array_almost_equal(data[0], astropy_wcs_results.map.i)
    assert_array_almost_equal(data[1], astropy_wcs_results.map.q)
    assert_array_almost_equal(data[2], astropy_wcs_results.map.u)


# --- Unknown landscape ---


def test_unknown_landscape_saves_as_npy(unknown_landscape_results, tmp_path):
    unknown_landscape_results.save(tmp_path)
    assert (tmp_path / 'map.npy').exists()
    assert (tmp_path / 'hit_map.npy').exists()
    assert (tmp_path / 'icov.npy').exists()


def test_unknown_landscape_map_roundtrip(unknown_landscape_results, tmp_path):
    unknown_landscape_results.save(tmp_path)
    data = np.load(tmp_path / 'map.npy')
    assert_array_almost_equal(data[0], unknown_landscape_results.map.i)
    assert_array_almost_equal(data[1], unknown_landscape_results.map.q)
    assert_array_almost_equal(data[2], unknown_landscape_results.map.u)
