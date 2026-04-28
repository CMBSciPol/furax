import dataclasses
import importlib.util
import json

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

pixell_installed = importlib.util.find_spec('pixell') is not None

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
def healpix_results_with_extras(healpix_results) -> MapMakingResults:
    rng = np.random.default_rng(42)
    return dataclasses.replace(
        healpix_results,
        solver_stats={'num_steps': np.int32(7), 'converged': True, 'residual': np.float64(1e-8)},
        noise_fits=jnp.array(rng.standard_normal(NPIX)),
    )


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


# --- save() ---


def test_healpix_save(healpix_results, tmp_path):
    healpix_results.save(tmp_path)

    assert (tmp_path / 'map.fits').exists()
    assert (tmp_path / 'hit_map.fits').exists()
    assert (tmp_path / 'icov.fits').exists()

    i_read, q_read, u_read = hp.read_map(tmp_path / 'map.fits', field=[0, 1, 2])
    assert_array_almost_equal(i_read, healpix_results.map.i)
    assert_array_almost_equal(q_read, healpix_results.map.q)
    assert_array_almost_equal(u_read, healpix_results.map.u)

    with fits.open(tmp_path / 'map.fits') as hdul:
        assert hdul[1].header['ORDERING'] == 'RING'
    with fits.open(tmp_path / 'hit_map.fits') as hdul:
        assert hdul[1].columns[0].name == 'HITS'
    with fits.open(tmp_path / 'icov.fits') as hdul:
        assert [c.name for c in hdul[1].columns] == ['II', 'IQ', 'IU', 'QQ', 'QU', 'UU']

    icov_read = np.array(hp.read_map(tmp_path / 'icov.fits', field=list(range(6))))
    icov = np.array(healpix_results.icov)
    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    assert_array_almost_equal(icov_read, np.stack([icov[i, j] for i, j in upper]))


def test_wcs_save(car_results, tmp_path):
    car_results.save(tmp_path)

    header = fits.open(tmp_path / 'map.fits')[0].header
    for key in ('CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1'):
        assert key in header

    data = fits.open(tmp_path / 'map.fits')[0].data
    assert_array_almost_equal(data[0], car_results.map.i)
    assert_array_almost_equal(data[1], car_results.map.q)
    assert_array_almost_equal(data[2], car_results.map.u)

    icov_data = fits.open(tmp_path / 'icov.fits')[0].data
    icov = np.array(car_results.icov)
    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    assert_array_almost_equal(icov_data, np.stack([icov[i, j] for i, j in upper]))


def test_astropy_wcs_embeds_wcs_header(astropy_wcs_results, tmp_path):
    astropy_wcs_results.save(tmp_path)
    header = fits.open(tmp_path / 'map.fits')[0].header
    for key in ('CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1'):
        assert key in header


def test_unknown_landscape_save(unknown_landscape_results, tmp_path):
    unknown_landscape_results.save(tmp_path)
    assert (tmp_path / 'map.npy').exists()
    assert (tmp_path / 'hit_map.npy').exists()
    assert (tmp_path / 'icov.npy').exists()
    data = np.load(tmp_path / 'map.npy')
    assert_array_almost_equal(data[0], unknown_landscape_results.map.i)
    assert_array_almost_equal(data[1], unknown_landscape_results.map.q)
    assert_array_almost_equal(data[2], unknown_landscape_results.map.u)


def test_save_solver_stats(healpix_results, tmp_path):
    results = dataclasses.replace(
        healpix_results,
        solver_stats={'num_steps': np.int32(3), 'converged': True, 'loss': np.float32(0.5)},
    )
    results.save(tmp_path)
    assert (tmp_path / 'solver_stats.json').exists()
    with open(tmp_path / 'solver_stats.json') as f:
        stats = json.load(f)
    assert stats == {'num_steps': 3, 'converged': True, 'loss': pytest.approx(0.5, abs=1e-6)}


def test_save_no_solver_stats_file_when_none(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    assert not (tmp_path / 'solver_stats.json').exists()


# --- pixell loading ---


@pytest.mark.skipif(not pixell_installed, reason='pixell not installed')
class TestPixellLoading:
    def test_wcs_map_roundtrip(self, car_results, tmp_path):
        from pixell import enmap

        car_results.save(tmp_path)
        m = enmap.read_map(str(tmp_path / 'map.fits'))
        assert_array_almost_equal(m[0], car_results.map.i)
        assert_array_almost_equal(m[1], car_results.map.q)
        assert_array_almost_equal(m[2], car_results.map.u)

    def test_wcs_header_consistency(self, car_results, tmp_path):
        from pixell import enmap

        car_results.save(tmp_path)
        m = enmap.read_map(str(tmp_path / 'map.fits'))
        wcs = car_results.landscape.to_wcs()
        assert list(m.wcs.wcs.ctype) == list(wcs.wcs.ctype)
        np.testing.assert_allclose(m.wcs.wcs.crval, wcs.wcs.crval)
        np.testing.assert_allclose(m.wcs.wcs.crpix, wcs.wcs.crpix)
        np.testing.assert_allclose(m.wcs.wcs.cdelt, wcs.wcs.cdelt)


# --- load() ---


def test_healpix_load_roundtrip(healpix_results_with_extras, tmp_path):
    r = healpix_results_with_extras
    r.save(tmp_path)
    loaded = MapMakingResults.load(tmp_path, r.landscape)

    assert_array_almost_equal(loaded.map.i, r.map.i)
    assert_array_almost_equal(loaded.map.q, r.map.q)
    assert_array_almost_equal(loaded.map.u, r.map.u)
    assert_array_equal(loaded.hit_map, r.hit_map)

    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    for i, j in upper:
        assert_array_almost_equal(loaded.icov[i, j], r.icov[i, j])
    icov = np.array(loaded.icov)
    assert_array_equal(icov, icov.transpose(1, 0, 2))  # symmetric

    assert loaded.solver_stats == {'num_steps': 7, 'converged': True, 'residual': 1e-8}
    assert_array_almost_equal(loaded.noise_fits, r.noise_fits)


def test_healpix_load_missing_optionals_are_none(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    loaded = MapMakingResults.load(tmp_path, healpix_results.landscape)
    assert loaded.solver_stats is None
    assert loaded.noise_fits is None


def test_wcs_load_roundtrip(car_results, tmp_path):
    car_results.save(tmp_path)
    loaded = MapMakingResults.load(tmp_path, car_results.landscape)

    assert_array_almost_equal(loaded.map.i, car_results.map.i)
    assert_array_almost_equal(loaded.map.q, car_results.map.q)
    assert_array_almost_equal(loaded.map.u, car_results.map.u)
    assert_array_equal(loaded.hit_map, car_results.hit_map)

    icov = np.array(loaded.icov)
    assert_array_equal(icov, icov.transpose(1, 0, 2, 3))  # symmetric


def test_unknown_landscape_load_roundtrip(unknown_landscape_results, tmp_path):
    unknown_landscape_results.save(tmp_path)
    loaded = MapMakingResults.load(tmp_path, unknown_landscape_results.landscape)
    assert_array_almost_equal(loaded.map.i, unknown_landscape_results.map.i)
    assert_array_almost_equal(loaded.map.q, unknown_landscape_results.map.q)
    assert_array_almost_equal(loaded.map.u, unknown_landscape_results.map.u)
    assert_array_equal(loaded.hit_map, unknown_landscape_results.hit_map)


# --- load() field selection and errors ---


def test_load_subset_fields_skips_optionals(healpix_results_with_extras, tmp_path):
    healpix_results_with_extras.save(tmp_path)
    loaded = MapMakingResults.load(
        tmp_path, healpix_results_with_extras.landscape, fields={'map', 'hit_map', 'icov'}
    )
    assert loaded.solver_stats is None
    assert loaded.noise_fits is None
    assert_array_almost_equal(loaded.map.i, healpix_results_with_extras.map.i)


def test_load_invalid_field_raises(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    with pytest.raises(ValueError, match='Unknown fields'):
        MapMakingResults.load(
            tmp_path, healpix_results.landscape, fields={'map', 'hit_map', 'icov', 'bad_field'}
        )


def test_load_missing_required_field_raises(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    with pytest.raises(ValueError, match='Required fields cannot be excluded'):
        MapMakingResults.load(tmp_path, healpix_results.landscape, fields={'map', 'hit_map'})


def test_load_missing_directory_raises(tmp_path):
    landscape = HealpixLandscape(NSIDE, stokes='IQU')
    with pytest.raises(FileNotFoundError, match='Output directory not found'):
        MapMakingResults.load(tmp_path / 'nonexistent', landscape)


def test_load_missing_required_file_raises(healpix_results, tmp_path):
    healpix_results.save(tmp_path)
    (tmp_path / 'map.fits').unlink()
    with pytest.raises(FileNotFoundError, match='Expected file not found'):
        MapMakingResults.load(tmp_path, healpix_results.landscape)
