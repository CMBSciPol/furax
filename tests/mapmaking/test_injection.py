"""A simulated sky scanned into the data, to measure what a mapmaking run does to a signal."""

from dataclasses import replace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from furax.mapmaking._injection import load_sky, unpadded
from furax.mapmaking._observation import ReaderField
from furax.mapmaking.config import (
    HealpixConfig,
    InjectionConfig,
    InjectionMode,
    LandscapeConfig,
    MapMakingConfig,
    NoiseFitConfig,
    NoiseSource,
    PointingConfig,
    PolynomialConfig,
    PolynomialOrders,
    SotodlibConfig,
    T2PConfig,
    TemplatesConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import MultiObservationMapMaker
from furax.mapmaking.results import MapMakingResults
from furax.obs.landscapes import HealpixLandscape, WCSLandscape
from tests.mapmaking.helpers import FakeLazyGroundObservation

NSIDE = 16


def _config(injection: InjectionConfig | None = None) -> MapMakingConfig:
    return MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=NSIDE)),
        weighting=WeightingConfig(
            mode=WeightingMode.DIAGONAL,
            source=NoiseSource.PRECOMPUTED,
            fitting=NoiseFitConfig(nperseg=256),
        ),
        # keep every observed pixel so that the maps can be compared
        hits_cut=0.0,
        cond_cut=0.0,
        injection=injection,
    )


def _observations():
    # ground observations carry a precomputed noise model
    return [FakeLazyGroundObservation(seed=i, n_dets=8, n_samples=1024) for i in range(2)]


@pytest.fixture
def sky(tmp_path):
    """A random IQU sky of unit scale, saved as a HEALPix map; the sky and its path."""
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(3, hp.nside2npix(NSIDE)))
    path = tmp_path / 'sky.fits'
    hp.write_map(str(path), maps, overwrite=True)
    return maps, str(path)


def _run(config):
    return MultiObservationMapMaker(_observations(), config=config).run()


def _well_observed(result):
    return np.asarray(result.hit_map) > 10


def test_the_map_of_the_sky_alone_is_the_sky(sky):
    maps, path = sky
    result = _run(_config(InjectionConfig(path, mode=InjectionMode.REPLACE)))
    observed = _well_observed(result)
    assert observed.sum() > 10
    assert_allclose(result.map.data[:, observed], maps[:, observed], atol=1e-4)


@pytest.mark.parametrize('scale', [1.0, 3.0])
def test_the_map_is_linear_in_the_data_and_the_sky(sky, scale):
    # the weights are precomputed: they do not depend on the data, so the map is linear in them
    maps, path = sky
    data = _run(_config())
    both = _run(_config(InjectionConfig(path, scale=scale, mode=InjectionMode.ADD)))
    observed = _well_observed(data)
    assert_allclose(
        both.map.data[:, observed] - data.map.data[:, observed],
        scale * maps[:, observed],
        atol=1e-4,
    )


def test_components_missing_from_the_file_are_zero(sky, tmp_path):
    maps, _ = sky
    path = tmp_path / 'qu.fits'
    hp.write_map(str(path), maps[1:], overwrite=True)
    result = _run(_config(InjectionConfig(str(path), stokes='QU', mode=InjectionMode.REPLACE)))
    observed = _well_observed(result)
    assert_allclose(result.map.data[0, observed], 0.0, atol=1e-4)
    assert_allclose(result.map.data[1:, observed], maps[1:, observed], atol=1e-4)


def test_a_sky_on_another_pixelization_is_rejected(tmp_path):
    path = tmp_path / 'sky.fits'
    hp.write_map(str(path), np.zeros((3, hp.nside2npix(2 * NSIDE))), overwrite=True)
    config = _config(InjectionConfig(str(path)))
    with pytest.raises(ValueError, match='pixelization of the landscape'):
        MultiObservationMapMaker(_observations(), config=config)


def test_replacing_the_data_needs_weights_that_are_not_fit_from_them():
    config = _config(InjectionConfig('sky.fits', mode=InjectionMode.REPLACE))
    config.weighting = WeightingConfig(mode=WeightingMode.DIAGONAL, source=NoiseSource.FIT)
    with pytest.raises(ValueError, match='precomputed'):
        config.__post_init__()


def test_no_injection_leaves_the_data_alone():
    maker = MultiObservationMapMaker(_observations(), config=_config())
    assert maker.sky is None
    assert jnp.all(jnp.isfinite(maker.run().map.data))


def _demodulated_config(injection, templates=None):
    config = _config(injection)
    config.landscape = LandscapeConfig(stokes='QU', healpix=HealpixConfig(nside=NSIDE))
    config.sotodlib = SotodlibConfig(demodulated=True)
    config.templates = templates
    return config


def test_the_sky_is_scanned_into_the_streams_of_demodulated_data(sky):
    maps, path = sky
    result = _run(_demodulated_config(InjectionConfig(path, mode=InjectionMode.REPLACE)))
    observed = _well_observed(result)
    assert_allclose(result.map.data[:, observed], maps[1:, observed], atol=1e-4)


def test_the_temperature_of_the_sky_feeds_the_t2p_template(sky):
    # The map has no I, but the T2P template is built from the I stream, which carries the sky.
    maps, path = sky
    templates = TemplatesConfig(t2p=T2PConfig())
    result = _run(_demodulated_config(InjectionConfig(path, mode=InjectionMode.REPLACE), templates))
    observed = _well_observed(result)
    q_u = np.asarray(result.map.data)[:, observed]
    assert np.all(np.isfinite(q_u))
    # the template fits a leakage coefficient to the I and Q or U of the sky, which are correlated by
    # chance on so few pixels: it absorbs some of the signal, which a map of the sky alone shows
    loss = np.abs(q_u - maps[1:, observed])
    assert 1e-3 < loss.max() < 0.5


def test_a_wcs_sky_is_read_on_the_landscape_it_was_written_on(tmp_path):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    wcs.wcs.crpix = [4.5, 3.5]
    wcs.wcs.cdelt = [-1.0, 1.0]
    landscape = WCSLandscape.from_wcs((6, 8), wcs, 'IQU', np.float64)
    stored = np.random.default_rng(2).normal(size=(3, 6, 8))
    path = tmp_path / 'sky.fits'
    fits.PrimaryHDU(stored, header=fits.Header(wcs.to_header())).writeto(path)

    sky = load_sky(InjectionConfig(str(path), scale=2.0), landscape.with_stokes('QU'))

    assert_allclose(sky.data, 2.0 * stored[1:])


@pytest.mark.parametrize('file_nested', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_a_healpix_sky_is_read_in_the_ordering_of_the_landscape(sky, tmp_path, file_nested, nested):
    maps, _ = sky
    stored = hp.reorder(maps, r2n=True) if file_nested else maps
    path = tmp_path / 'ordered.fits'
    hp.write_map(str(path), stored, nest=file_nested, overwrite=True)
    landscape = HealpixLandscape(NSIDE, 'IQU', np.float64, nested=nested)

    sky = load_sky(InjectionConfig(str(path)), landscape)

    assert_allclose(sky.data, hp.reorder(maps, r2n=True) if nested else maps)


@pytest.mark.parametrize('leading', [(), (2,)])
def test_only_the_samples_that_are_not_padding_are_injectable(leading):
    # one detector and two samples of padding, after any Stokes axis of demodulated data
    padding = {ReaderField.SAMPLE_DATA: jnp.array([*(0 for _ in leading), 1, 2])}
    expected = np.zeros((3, 5), bool)
    expected[:2, :3] = True
    assert_allclose(unpadded(padding, (*leading, 3, 5)), expected)


def _polynomial_config(injection=None, explicit=True) -> MapMakingConfig:
    config = _config(injection)
    # from order 1: the offsets of the intervals are degenerate with the intensity map
    polynomial = PolynomialConfig(legendre=PolynomialOrders(1, 2), explicit=explicit)
    config.templates = TemplatesConfig(polynomial=polynomial)
    return config


@pytest.mark.parametrize('explicit', [True, False])
def test_the_response_is_the_map_of_the_data_with_the_sky_added_minus_that_of_the_data(
    sky, explicit
):
    # Polynomial templates do not depend on the data, so the two are the same linear response.
    _, path = sky
    response = _run(
        _polynomial_config(InjectionConfig(path, mode=InjectionMode.RESPONSE), explicit)
    )
    added = _run(_polynomial_config(InjectionConfig(path, mode=InjectionMode.ADD), explicit))
    data = _run(_polynomial_config(None, explicit))
    observed = _well_observed(data)
    assert_allclose(
        response.injected_map.data[:, observed],
        added.map.data[:, observed] - data.map.data[:, observed],
        atol=1e-5,
    )
    # the map of the data is that of a run without injection
    assert_allclose(response.map.data, data.map.data, atol=1e-6)


def test_the_response_without_templates_is_the_sky(sky):
    maps, path = sky
    result = _run(_config(InjectionConfig(path, mode=InjectionMode.RESPONSE)))
    observed = _well_observed(result)
    assert_allclose(result.injected_map.data[:, observed], maps[:, observed], atol=1e-4)


def test_the_response_through_a_t2p_template_is_the_same_in_detector_batches(sky):
    maps, path = sky
    templates = TemplatesConfig(t2p=T2PConfig())
    config = _demodulated_config(InjectionConfig(path, mode=InjectionMode.RESPONSE), templates)
    whole, batched = [
        MultiObservationMapMaker(
            _observations(), config=replace(config, detector_batch_size=b)
        ).run()
        for b in (0, 4)
    ]
    observed = _well_observed(whole)
    assert np.all(np.isfinite(np.asarray(whole.injected_map.data)))
    assert_allclose(
        batched.injected_map.data[:, observed], whole.injected_map.data[:, observed], atol=1e-5
    )
    # The T2P basis comes from the I stream of the data, which does not carry the sky: no coefficient
    # is fitted to its correlation with the sky's own polarization. With the sky added to the data,
    # the basis carries it, and the template absorbs a part of the signal.
    data = _run(_demodulated_config(None, templates))
    added = _run(_demodulated_config(InjectionConfig(path, mode=InjectionMode.ADD), templates))
    response_loss = np.abs(whole.injected_map.data[:, observed] - maps[1:, observed]).max()
    added_loss = np.abs(
        np.asarray(added.map.data - data.map.data)[:, observed] - maps[1:, observed]
    ).max()
    assert added_loss > 100 * response_loss


def test_the_injected_map_is_saved_and_loaded(sky, tmp_path):
    _, path = sky
    config = _config(InjectionConfig(path, mode=InjectionMode.RESPONSE))
    result = _run(config)
    result.save(tmp_path / 'out')
    loaded = MapMakingResults.load(tmp_path / 'out', result.landscape)
    assert_allclose(loaded.injected_map.data, result.injected_map.data, atol=1e-6)
    assert MapMakingResults.load(tmp_path / 'out', result.landscape).map is not None


def test_a_run_without_injection_has_no_injected_map(tmp_path):
    result = _run(_config())
    assert result.injected_map is None
    result.save(tmp_path / 'out')
    assert MapMakingResults.load(tmp_path / 'out', result.landscape).injected_map is None
