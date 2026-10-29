import dataclasses
import importlib.util
import json

import healpy as hp
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_array_almost_equal, assert_array_equal
from polars.testing import assert_frame_equal

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


UPPER = [(i, j) for i in range(3) for j in range(i, 3)]  # the icov entries stored


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


def _astropy_wcs_landscape() -> AstropyWCSLandscape:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    wcs.wcs.crpix = [5.5, 4.5]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [-0.1, 0.1]
    wcs.wcs.set()
    return AstropyWCSLandscape(MAP_SHAPE, wcs, stokes='IQU')


LANDSCAPES = {
    'healpix': lambda: HealpixLandscape(NSIDE, stokes='IQU'),
    'car': lambda: CARLandscape(
        MAP_SHAPE, WCSProjection(crpix=(5.5, 4.5), crval=(0.0, 0.0), cdelt=(-0.1, 0.1)), 'IQU'
    ),
    'astropy_wcs': _astropy_wcs_landscape,
    'unknown': _UnknownLandscape,
}


def _results(name: str) -> MapMakingResults:
    rng = np.random.default_rng(0)
    landscape = LANDSCAPES[name]()
    shape = landscape.shape
    return MapMakingResults(
        map=StokesIQU.from_array(jnp.array(rng.standard_normal((3, *shape)))),
        landscape=landscape,
        hit_map=jnp.array(rng.integers(1, 100, shape)),
        icov=jnp.array(rng.standard_normal((3, 3, *shape))),
    )


@pytest.mark.parametrize('name', list(LANDSCAPES))
def test_save_load_roundtrip(name, tmp_path):
    results = _results(name)
    results.save(tmp_path)
    loaded = MapMakingResults.load(tmp_path, results.landscape)

    assert_array_almost_equal(loaded.map.data, results.map.data)
    assert_array_equal(loaded.hit_map, results.hit_map)
    # the upper triangle is stored, the lower one mirrored from it
    icov = np.asarray(loaded.icov)
    for i, j in UPPER:
        assert_array_almost_equal(icov[i, j], results.icov[i, j])
    assert_array_equal(icov, np.swapaxes(icov, 0, 1))
    # nothing more was saved than what the results hold
    assert not (tmp_path / 'solver_stats.json').exists()
    assert loaded.solver_stats is None
    assert loaded.observations is None
    assert loaded.detectors is None


def test_healpix_files_are_healpy_maps(tmp_path):
    results = _results('healpix')
    results.save(tmp_path)
    i_q_u = hp.read_map(tmp_path / 'map.fits', field=[0, 1, 2])
    assert_array_almost_equal(i_q_u, results.map.data)
    with fits.open(tmp_path / 'map.fits') as hdul:
        assert hdul[1].header['ORDERING'] == 'RING'
    with fits.open(tmp_path / 'hit_map.fits') as hdul:
        assert hdul[1].columns[0].name == 'HITS'
    with fits.open(tmp_path / 'icov.fits') as hdul:
        assert [c.name for c in hdul[1].columns] == ['II', 'IQ', 'IU', 'QQ', 'QU', 'UU']


@pytest.mark.parametrize('name', ['car', 'astropy_wcs'])
def test_wcs_files_embed_the_wcs(name, tmp_path):
    results = _results(name)
    results.save(tmp_path)
    with fits.open(tmp_path / 'map.fits') as hdul:
        header, data = hdul[0].header, hdul[0].data
    for key in ('CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1'):
        assert key in header
    assert_array_almost_equal(data, results.map.data)


def test_unknown_landscape_is_saved_as_npy(tmp_path):
    results = _results('unknown')
    results.save(tmp_path)
    assert_array_almost_equal(np.load(tmp_path / 'map.npy'), results.map.data)
    assert (tmp_path / 'hit_map.npy').exists()
    assert (tmp_path / 'icov.npy').exists()


@pytest.mark.skipif(not pixell_installed, reason='pixell not installed')
def test_wcs_map_is_a_pixell_map(tmp_path):
    from pixell import enmap

    results = _results('car')
    results.save(tmp_path)
    m = enmap.read_map(str(tmp_path / 'map.fits'))
    assert_array_almost_equal(m, results.map.data)
    wcs = results.landscape.to_wcs()
    assert list(m.wcs.wcs.ctype) == list(wcs.wcs.ctype)
    np.testing.assert_allclose(m.wcs.wcs.crval, wcs.wcs.crval)
    np.testing.assert_allclose(m.wcs.wcs.crpix, wcs.wcs.crpix)
    np.testing.assert_allclose(m.wcs.wcs.cdelt, wcs.wcs.cdelt)


def test_extras_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    results = dataclasses.replace(
        _results('healpix'),
        solver_stats={'num_steps': 7, 'converged': True, 'residual': 1e-8},
        # all-null columns, as from a run that read no flags, must keep their type
        observations=pl.DataFrame(
            {
                'name': ['obs0', 'obs1'],
                'loaded': [True, False],
                'n_detectors': [2, 0],
                'n_samples': [100, 0],
                'probed_detectors': [2, 0],
                'probed_samples': [100, 0],
                'valid_sample_fraction': [None, None],
                'load_seconds': [1.5, None],
                'error': [None, None],
            },
            schema_overrides={'valid_sample_fraction': pl.Float64, 'error': pl.String},
        ),
        detectors=pl.DataFrame(
            {
                'observation': [0, 0],
                'detector': [0, 1],
                'name': ['a', 'b'],
                'xi': [0.1, None],
                'sigma_q': rng.standard_normal(2),
            }
        ),
    )
    results.save(tmp_path)
    with open(tmp_path / 'solver_stats.json') as f:
        assert json.load(f) == results.solver_stats
    loaded = MapMakingResults.load(tmp_path, results.landscape)
    assert loaded.solver_stats == results.solver_stats
    assert_frame_equal(loaded.observations, results.observations)
    assert_frame_equal(loaded.detectors, results.detectors)


def test_load_missing_directory_raises(tmp_path):
    landscape = HealpixLandscape(NSIDE, stokes='IQU')
    with pytest.raises(FileNotFoundError, match='Output directory not found'):
        MapMakingResults.load(tmp_path / 'nonexistent', landscape)


def test_load_missing_required_file_raises(tmp_path):
    results = _results('healpix')
    results.save(tmp_path)
    (tmp_path / 'map.fits').unlink()
    with pytest.raises(FileNotFoundError, match='Expected file not found'):
        MapMakingResults.load(tmp_path, results.landscape)
