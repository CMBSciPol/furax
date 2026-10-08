"""Joint sky + template-amplitude solve in the multi-observation mapmaker.

Backed by the file-free synthetic observations (no sotodlib/toast, no fixtures):
``FakeLazyObservation`` for HWP-synchronous templates (needs only ``hwp_angles``) and
``FakeLazyGroundObservation`` for the azimuth/interval templates (azimuth, scanning intervals).
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.mapmaking import ReaderField
from furax.mapmaking.config import (
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    MapMakingConfig,
    NoiseFitConfig,
    PointingConfig,
    PolynomialConfig,
    PolynomialOrders,
    ScanSynchronousConfig,
    SolverConfig,
    SotodlibConfig,
    T2PConfig,
    TemplatesConfig,
    TwoLevelConfig,
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
    assert jnp.all(jnp.isfinite(res.map.data))


@pytest.mark.parametrize(
    'observations, template',
    [
        (_hwp_obs, lambda e: TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=e))),
        (
            _ground_obs,
            lambda e: TemplatesConfig(scan_synchronous=ScanSynchronousConfig(explicit=e)),
        ),
    ],
    ids=['hwp_synchronous', 'scan_synchronous'],
)
def test_explicit_and_implicit_give_the_same_map(observations, template):
    # Marginalising the amplitudes (implicit deprojection) is exactly equivalent to solving
    # them jointly and discarding them: the recovered map must match.
    obs = observations()
    explicit = MultiObservationMapMaker(obs, config=_config(template(True))).run()
    implicit = MultiObservationMapMaker(obs, config=_config(template(False))).run()
    assert_allclose(explicit.map.data, implicit.map.data, rtol=1e-4, atol=1e-6)


def test_mixed_explicit_and_implicit():
    # One explicit template (solved + returned) and one implicit template (deprojected into W').
    cfg = _config(
        TemplatesConfig(
            hwp_synchronous=HWPSynchronousConfig(2, explicit=True),
            scan_synchronous=ScanSynchronousConfig(explicit=False),
        )
    )
    res = MultiObservationMapMaker(_ground_obs(), config=cfg).run()
    assert set(res.template_amplitudes) == {'hwp_synchronous'}
    assert jnp.all(jnp.isfinite(res.template_amplitudes['hwp_synchronous']))


@pytest.mark.parametrize('spectrum', ['preconditioned', 'system'])
def test_two_level_preconditioner_with_explicit_templates(spectrum):
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)))
    cfg.solver = SolverConfig(rtol=1e-10, max_steps=500)
    expected = MultiObservationMapMaker(_hwp_obs(), config=cfg).run()
    cfg.solver.two_level = TwoLevelConfig(rank=4, spectrum=spectrum, tol=1e-8, max_restarts=50)
    res = MultiObservationMapMaker(_hwp_obs(), config=cfg).run()
    assert_allclose(res.map.data, expected.map.data, rtol=1e-6, atol=1e-8)
    amplitudes = res.template_amplitudes['hwp_synchronous']
    expected_amplitudes = expected.template_amplitudes['hwp_synchronous']
    assert_allclose(amplitudes, expected_amplitudes, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('explicit', [True, False], ids=['explicit', 'implicit'])
def test_runs_and_produces_a_map(explicit):
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=explicit)))
    maker = MultiObservationMapMaker(_hwp_obs(), config=cfg)
    res = maker.run()
    assert res.map.shape == maker.landscape.shape
    assert res.solver_stats['num_steps'] >= 1


def test_demodulated_polynomial_implicit_runs():
    # Regression: with demodulated data the implicit polynomial template is a per-Stokes-leg
    # TemplateOperator (a Family per leg), not a single-leg one. The structured Gram must
    # recurse into each Stokes leg; before that it raised NotImplementedError.
    cfg = _config(TemplatesConfig(polynomial=PolynomialConfig(explicit=False)))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    res = MultiObservationMapMaker(_ground_obs(), config=cfg).run()
    assert jnp.all(jnp.isfinite(res.map.data))


@pytest.mark.parametrize(
    ('name', 'legendre', 'groups', 'legs'),
    [
        ('polynomial', PolynomialOrders(0, 3), ['iqu'], ['i', 'q', 'u']),
        (
            'polynomial',
            {'i': PolynomialOrders(0, 3), 'q': PolynomialOrders(1, 3), 'u': PolynomialOrders(1, 3)},
            ['i', 'qu'],
            ['i', 'q', 'u'],
        ),
        ('scan_synchronous', {'qu': PolynomialOrders(2, 5)}, ['qu'], ['q', 'u']),
    ],
)
def test_demodulated_legs_fitted_alike_share_a_basis(name, legendre, groups, legs):
    # legs fitted with the same orders share one basis, so an observation stores it once rather
    # than once per leg; the amplitudes stay one set per leg, on the legs given orders only
    template = PolynomialConfig() if name == 'polynomial' else ScanSynchronousConfig()
    cfg = _config(TemplatesConfig(**{name: template}))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    template.legendre = legendre  # set after demodulation, which per-leg orders require
    maker = MultiObservationMapMaker(_ground_obs(), config=cfg)
    with jax.set_mesh(maker.mesh):
        acc = maker.build_model_and_accumulate()
    operator = acc.buckets[0].templates.implicit.operator
    assert sorted(operator.bases[name]) == groups
    assert sorted(operator.in_structure[name]) == legs


@pytest.mark.parametrize(
    ('observations', 'templates', 'demodulated'),
    [
        (_hwp_obs, None, False),
        (_hwp_obs, TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)), False),
        # Polynomial and scan-synchronous templates together are nearly degenerate on these short
        # scans, too ill-conditioned for any two runs to agree closely, so they come separately.
        (
            _ground_obs,
            TemplatesConfig(polynomial=PolynomialConfig(explicit=False), t2p=T2PConfig()),
            True,
        ),
        (
            _ground_obs,
            TemplatesConfig(
                scan_synchronous=ScanSynchronousConfig(explicit=False), t2p=T2PConfig()
            ),
            True,
        ),
    ],
    ids=['no-templates', 'explicit', 'demodulated-polynomial-and-t2p', 'demodulated-azss-and-t2p'],
)
def test_detector_batches_give_the_same_result(observations, templates, demodulated):
    # Processing an observation's detectors in batches is a memory layout, not a change of model:
    # every template, weight and Gram is per detector. 7 detectors in batches of 3 also pads two.
    n_dets = 7
    obs = observations(n_dets=n_dets)
    sotodlib = SotodlibConfig(demodulated=True) if demodulated else None
    cfg = replace(_config(None), templates=templates, sotodlib=sotodlib)
    makers = [
        MultiObservationMapMaker(obs, config=replace(cfg, detector_batch_size=batch))
        for batch in (0, 3)
    ]

    accumulated = []
    for maker in makers:
        with jax.set_mesh(maker.mesh):
            accumulated.append(maker.build_model_and_accumulate())
    whole, batched = accumulated
    assert_allclose(batched.hit_map, whole.hit_map)
    assert_allclose(batched.map_rhs.data, whole.map_rhs.data, rtol=1e-12, atol=1e-12)

    # The template-marginalised systems are ill-conditioned enough that the solves agree only to
    # about the solver tolerance.
    whole, batched = [maker.run() for maker in makers]
    scale = jnp.max(jnp.abs(whole.map.data))
    assert_allclose(batched.map.data, whole.map.data, atol=1e-4 * scale)
    if whole.template_amplitudes is not None:
        # batching pads 2 more detectors, whose amplitudes trail the real ones
        got = jax.tree.map(lambda a: a[:, :n_dets], batched.template_amplitudes)
        jax.tree.map(
            lambda a, b: assert_allclose(a, b, atol=1e-4 * jnp.max(jnp.abs(b))),
            got,
            whole.template_amplitudes,
        )


@pytest.mark.parametrize(('batch', 'n_buffered'), [(0, 8), (64, 8), (3, 9)])
def test_only_observations_larger_than_a_batch_are_padded(batch, n_buffered):
    # 8 detectors: a batch of 64 holds them all, unpadded; batches of 3 need a ninth
    maker = MultiObservationMapMaker(
        _hwp_obs(), config=replace(_config(None), detector_batch_size=batch)
    )
    assert maker.readers[0].out_structure[ReaderField.DETECTOR_QUATERNIONS].shape[0] == n_buffered


@pytest.mark.parametrize('prefetch', [False, True], ids=['read-after-round', 'prefetch'])
def test_several_observations_per_device_match_single_observations(monkeypatch, prefetch: bool):
    # More observations than devices: each device accumulates several, one round at a time,
    # reading the next round after the current one or, as with a GPU, while it computes.
    # Each observation's sums and outputs must be those of a run over it alone.
    monkeypatch.setattr(MultiObservationMapMaker, '_prefetches_reads', prefetch)
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)))
    observations = _hwp_obs(n_obs=10)

    def accumulate(observations):
        maker = MultiObservationMapMaker(observations, config=cfg)
        with jax.set_mesh(maker.mesh):
            return maker, maker.build_model_and_accumulate()

    maker, together = accumulate(observations)
    assert maker.layout.buckets[0].n_slots // maker.layout.n_devices >= 2
    alone = [accumulate([obs])[1] for obs in observations]

    assert_allclose(together.hit_map, sum(acc.hit_map for acc in alone))
    assert_allclose(
        together.map_rhs.data, sum(acc.map_rhs.data for acc in alone), rtol=1e-12, atol=1e-12
    )
    amplitude_rhs = together.buckets[0].amplitude_rhs['hwp_synchronous']
    per_observation = maker.layout.to_observation_order([np.asarray(amplitude_rhs)])
    for got, acc in zip(per_observation, alone, strict=True):
        assert_allclose(got, np.asarray(acc.buckets[0].amplitude_rhs['hwp_synchronous'])[0])
