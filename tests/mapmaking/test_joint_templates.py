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
    GapsConfig,
    GapTreatment,
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseFitConfig,
    NoiseSource,
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
from furax.mapmaking.templates import Template
from furax.mapmaking.templates.bases import OffsetBasis
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
    # With demodulated data, the implicit polynomial template has a set of amplitudes per stream,
    # and its Gram one block per detector and stream.
    cfg = _config(TemplatesConfig(polynomial=PolynomialConfig(explicit=False)))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    res = MultiObservationMapMaker(_ground_obs(), config=cfg).run()
    assert jnp.all(jnp.isfinite(res.map.data))


@pytest.mark.parametrize(
    ('name', 'legendre', 'templates'),
    [
        ('polynomial', PolynomialOrders(0, 3), {'polynomial': 'iqu'}),
        (
            'polynomial',
            {'i': PolynomialOrders(0, 3), 'q': PolynomialOrders(1, 3), 'u': PolynomialOrders(1, 3)},
            {'polynomial_i': 'i', 'polynomial_qu': 'qu'},
        ),
        ('scan_synchronous', {'qu': PolynomialOrders(2, 5)}, {'scan_synchronous_qu': 'qu'}),
    ],
)
def test_demodulated_streams_fitted_alike_form_one_template(name, legendre, templates):
    # the streams fitted with the same orders share one template, so an observation stores its
    # basis once; the amplitudes stay one set per stream, on the streams given orders only
    template = PolynomialConfig() if name == 'polynomial' else ScanSynchronousConfig()
    cfg = _config(TemplatesConfig(**{name: template}))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    template.legendre = legendre  # set after demodulation, which per-stream orders require
    maker = MultiObservationMapMaker(_ground_obs(), config=cfg)
    with jax.set_mesh(maker.mesh):
        acc = maker.build_model_and_accumulate()
    operator = acc.buckets[0].templates.implicit.operator
    assert {key: t.streams for key, t in operator.templates.items()} == templates


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
        # batching pads 2 more detectors, whose amplitudes trail the real ones; the amplitudes
        # are (obs, stream, det, ...) on a demodulated TOD, (obs, det, ...) otherwise
        axis = 2 if demodulated else 1
        got = jax.tree.map(
            lambda a: jnp.take(a, jnp.arange(n_dets), axis=axis), batched.template_amplitudes
        )
        jax.tree.map(
            lambda a, b: assert_allclose(a, b, atol=1e-4 * jnp.max(jnp.abs(b))),
            got,
            whole.template_amplitudes,
        )


def test_amplitudes_shared_by_the_detectors_are_the_same_in_detector_batches(monkeypatch):
    # A common mode, one offset per block of 64 samples, seen by every detector. Its amplitudes
    # belong to the observation, whatever the batches its detectors are processed in.
    def common_mode(_config, _data, tod_format, **_):
        basis = OffsetBasis(64, tod_format.n_samples, tod_format.dtype)
        return {'common': Template(basis, tod_format, per=())}, {}

    monkeypatch.setattr('furax.mapmaking._model.build_templates', common_mode)
    # the configuration only has to enable templates: the builder above makes them
    templates = TemplatesConfig(polynomial=PolynomialConfig(), regularization=1e-3)
    cfg = _config(templates)
    results = [
        MultiObservationMapMaker(_ground_obs(n_dets=7), config=replace(cfg, detector_batch_size=b))
        for b in (0, 3)
    ]
    whole, batched = [maker.run() for maker in results]
    amplitudes = whole.template_amplitudes['common']
    assert amplitudes.shape == (2, 1024 // 64)  # (obs, block)
    assert jnp.all(jnp.isfinite(amplitudes)) and jnp.max(jnp.abs(amplitudes)) > 0
    scale = jnp.max(jnp.abs(whole.map.data))
    assert_allclose(batched.map.data, whole.map.data, atol=1e-4 * scale)
    assert_allclose(
        batched.template_amplitudes['common'], amplitudes, atol=1e-4 * jnp.max(jnp.abs(amplitudes))
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


# ---------------------------------------------------------------------------
# Pomme combined with templates
# ---------------------------------------------------------------------------


def _pomme_config(templates: TemplatesConfig) -> MapMakingConfig:
    cfg = _config(templates)
    cfg.method = Methods.POMME
    cfg.pomme_tau = 37
    cfg.landscape.stokes = 'QU'
    templates.regularization = 1e-10
    return cfg


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
def test_pomme_explicit_and_implicit_give_the_same_map(observations, template):
    # Under Pomme the effective weight is W F; marginalising the amplitudes under it must still
    # match the joint solve. Both paths use the Pomme-filtered Gram (as weight fold and as
    # preconditioner respectively).
    obs = observations()
    explicit = MultiObservationMapMaker(obs, config=_pomme_config(template(True))).run()
    implicit = MultiObservationMapMaker(obs, config=_pomme_config(template(False))).run()
    assert_allclose(explicit.map.data, implicit.map.data, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('batch', [0, 1], ids=['one_batch', 'per_detector'])
def test_explicit_and_implicit_agree_under_a_ridge(batch):
    # Pomme removes the constant part of every template, so a scan-synchronous template starting
    # at order 0 has a singular Gram and only the ridge makes it invertible. Marginalising the
    # amplitudes uses the ridged Gram, so the joint system must carry the same ridge on its
    # amplitude block, otherwise the two solves estimate different maps. With one detector per
    # batch, the ridge is laid out over several slices per observation.
    def run(explicit: bool):
        scan = ScanSynchronousConfig(legendre=PolynomialOrders(0, 3), explicit=explicit)
        config = _pomme_config(TemplatesConfig(scan_synchronous=scan))
        config.templates.regularization = 1e-3  # large enough to tell the two systems apart
        config = replace(config, detector_batch_size=batch)
        observations = _ground_obs(n_obs=1, n_dets=2, n_samples=256)
        return MultiObservationMapMaker(observations, config=config).run()

    explicit, implicit = run(True), run(False)
    assert_allclose(explicit.map.data, implicit.map.data, rtol=1e-4, atol=1e-6)


def _toeplitz_config(
    templates: TemplatesConfig, precomputed: bool = False, demodulated: bool = False
) -> MapMakingConfig:
    sotodlib = SotodlibConfig(demodulated=True) if demodulated else None
    cfg = replace(_config(None), templates=templates, sotodlib=sotodlib)
    cfg.method = Methods.MAXL
    # the ground observations carry a 1/f noise model, correlated within the band; the others
    # fit one to white noise
    source = NoiseSource.PRECOMPUTED if precomputed else cfg.weighting.source
    cfg.weighting = replace(
        cfg.weighting, mode=WeightingMode.TOEPLITZ, source=source, correlation_length=64
    )
    cfg.gaps = GapsConfig(treatment=GapTreatment.INNER_MASK)
    cfg.solver = SolverConfig(rtol=1e-10, max_steps=1000)
    return cfg


@pytest.mark.parametrize(
    ('observations', 'template'),
    [
        (_hwp_obs, lambda e: TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=e))),
        # from order 1: the offsets of the intervals are degenerate with the intensity map
        (
            _ground_obs,
            lambda e: TemplatesConfig(
                polynomial=PolynomialConfig(legendre=PolynomialOrders(1, 2), explicit=e)
            ),
        ),
    ],
    ids=['hwp_synchronous', 'polynomial'],
)
def test_correlated_weight_explicit_and_implicit_give_the_same_map(observations, template):
    # Under a correlated weight, marginalizing the amplitudes needs the exact Gram, probed from the
    # weight; the joint solve only uses it as a preconditioner.
    obs = observations()
    precomputed = observations is _ground_obs
    maps = [
        MultiObservationMapMaker(obs, config=_toeplitz_config(template(e), precomputed)).run()
        for e in (True, False)
    ]
    assert_allclose(maps[0].map.data, maps[1].map.data, rtol=1e-6, atol=1e-8)


def test_detector_batches_give_the_same_result_under_a_correlated_weight():
    # The Gram of each detector is probed from the weight, whatever the batch it is in.
    templates = TemplatesConfig(polynomial=PolynomialConfig(legendre=PolynomialOrders(1, 2)))
    cfg = _toeplitz_config(templates, precomputed=True)
    whole, batched = [
        MultiObservationMapMaker(
            _ground_obs(n_dets=7), config=replace(cfg, detector_batch_size=batch)
        ).run()
        for batch in (0, 3)
    ]
    scale = jnp.max(jnp.abs(whole.map.data))
    assert_allclose(batched.map.data, whole.map.data, atol=1e-5 * scale)


@pytest.mark.parametrize('treatment', [GapTreatment.FILL, GapTreatment.NESTED])
def test_correlated_weight_with_templates_requires_inner_mask_gaps(treatment):
    config = _toeplitz_config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2)))
    config.gaps = GapsConfig(treatment=treatment)
    config.detector_batch_size = 0  # gap treatments other than the inner mask need every detector
    with pytest.raises(NotImplementedError, match='inner_mask'):
        MultiObservationMapMaker(_hwp_obs(), config=config)


def test_shared_amplitudes_under_a_correlated_weight_are_the_same_in_detector_batches(monkeypatch):
    # A common mode, one offset per block of 128 samples, seen by every detector: the Gram of its
    # amplitudes is probed from the weight in each batch, then summed over the batches.
    def common_mode(_config, _data, tod_format, **_):
        basis = OffsetBasis(128, tod_format.n_samples, tod_format.dtype)
        return {'common': Template(basis, tod_format, per=())}, {}

    monkeypatch.setattr('furax.mapmaking._model.build_templates', common_mode)
    # the configuration only has to enable templates: the builder above makes them
    templates = TemplatesConfig(polynomial=PolynomialConfig(), regularization=1e-3)
    cfg = _toeplitz_config(templates, precomputed=True)
    whole, batched = [
        MultiObservationMapMaker(
            _ground_obs(n_dets=7), config=replace(cfg, detector_batch_size=batch)
        ).run()
        for batch in (0, 3)
    ]
    amplitudes = whole.template_amplitudes['common']
    assert amplitudes.shape == (2, 1024 // 128)  # (obs, block)
    assert jnp.all(jnp.isfinite(amplitudes)) and jnp.max(jnp.abs(amplitudes)) > 0
    scale = jnp.max(jnp.abs(whole.map.data))
    assert_allclose(batched.map.data, whole.map.data, atol=1e-4 * scale)
    assert_allclose(
        batched.template_amplitudes['common'], amplitudes, atol=1e-4 * jnp.max(jnp.abs(amplitudes))
    )


def test_demodulated_polynomial_explicit_and_implicit_agree_under_a_correlated_weight():
    maps = [
        MultiObservationMapMaker(
            _ground_obs(),
            config=_toeplitz_config(
                TemplatesConfig(
                    polynomial=PolynomialConfig(legendre=PolynomialOrders(1, 2), explicit=e)
                ),
                precomputed=True,
                demodulated=True,
            ),
        ).run()
        for e in (True, False)
    ]
    assert_allclose(maps[0].map.data, maps[1].map.data, rtol=1e-6, atol=1e-8)
