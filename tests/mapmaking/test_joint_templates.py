"""Joint sky + template-amplitude solve in the multi-observation mapmaker.

Backed by the file-free synthetic observations (no sotodlib/toast, no fixtures):
``FakeLazyObservation`` for HWP-synchronous templates (needs only ``hwp_angles``) and
``FakeLazyGroundObservation`` for the azimuth/interval families (azimuth, scanning intervals).
"""

import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from furax.mapmaking.config import (
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    MapMakingConfig,
    NoiseFitConfig,
    PointingConfig,
    ScanSynchronousConfig,
    TemplatesConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import MultiObservationMapMaker
from tests.mapmaking.helpers import FakeLazyGroundObservation, FakeLazyObservation


def _config(templates: TemplatesConfig) -> MapMakingConfig:
    return MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=16)),
        weighting=WeightingConfig(mode=WeightingMode.DIAGONAL, fitting=NoiseFitConfig(nperseg=256)),
        templates=templates,
        # keep every observed pixel so the map comparisons below are meaningful
        hits_cut=0.0,
        cond_cut=0.0,
    )


def _hwp_obs(n_obs: int = 2, n_dets: int = 8, n_samples: int = 1024):
    return [FakeLazyObservation(seed=i, n_dets=n_dets, n_samples=n_samples) for i in range(n_obs)]


def _ground_obs(n_obs: int = 2, n_dets: int = 8, n_samples: int = 1024):
    return [
        FakeLazyGroundObservation(seed=i, n_dets=n_dets, n_samples=n_samples) for i in range(n_obs)
    ]


def test_explicit_returns_amplitudes():
    n_harmonics = 3
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(n_harmonics, explicit=True)))
    res = MultiObservationMapMaker(_hwp_obs(), config=cfg).run()
    assert res.template_amplitudes is not None
    amps = res.template_amplitudes['hwp_synchronous']
    # (n_obs, n_dets, K) with K = 2 * n_harmonics (sin + cos)
    assert amps.shape == (2, 8, 2 * n_harmonics)
    assert jnp.all(jnp.isfinite(amps))


def test_implicit_returns_no_amplitudes():
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=False)))
    res = MultiObservationMapMaker(_hwp_obs(), config=cfg).run()
    assert res.template_amplitudes is None
    assert jnp.all(jnp.isfinite(res.map.i))


@pytest.mark.parametrize(
    'observations, family',
    [
        (_hwp_obs, lambda e: TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=e))),
        (
            _ground_obs,
            lambda e: TemplatesConfig(scan_synchronous=ScanSynchronousConfig(explicit=e)),
        ),
    ],
    ids=['hwp_synchronous', 'scan_synchronous'],
)
def test_explicit_and_implicit_give_the_same_map(observations, family):
    # Marginalising the amplitudes (implicit deprojection) is exactly equivalent to solving
    # them jointly and discarding them: the recovered map must match.
    obs = observations()
    explicit = MultiObservationMapMaker(obs, config=_config(family(True))).run()
    implicit = MultiObservationMapMaker(obs, config=_config(family(False))).run()
    for stoke in 'iqu':
        assert_allclose(
            getattr(explicit.map, stoke), getattr(implicit.map, stoke), rtol=1e-4, atol=1e-6
        )


def test_mixed_explicit_and_implicit():
    # One explicit family (solved + returned) and one implicit family (deprojected into W').
    cfg = _config(
        TemplatesConfig(
            hwp_synchronous=HWPSynchronousConfig(2, explicit=True),
            scan_synchronous=ScanSynchronousConfig(explicit=False),
        )
    )
    res = MultiObservationMapMaker(_ground_obs(), config=cfg).run()
    assert set(res.template_amplitudes) == {'hwp_synchronous'}
    assert jnp.all(jnp.isfinite(res.template_amplitudes['hwp_synchronous']))


@pytest.mark.parametrize('explicit', [True, False], ids=['explicit', 'implicit'])
def test_runs_and_produces_a_map(explicit):
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=explicit)))
    maker = MultiObservationMapMaker(_hwp_obs(), config=cfg)
    res = maker.run()
    assert res.map.i.shape == maker.landscape.shape
    assert res.solver_stats['num_steps'] >= 1
