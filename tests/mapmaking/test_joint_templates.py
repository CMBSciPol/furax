"""Joint sky + template-amplitude solve in the multi-observation mapmaker.

Backed by the file-free synthetic observations (no sotodlib/toast, no fixtures):
``FakeLazyObservation`` for HWP-synchronous templates (needs only ``hwp_angles``) and
``FakeLazyGroundObservation`` for the azimuth/interval templates (azimuth, scanning intervals).
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from furax.mapmaking import ReaderField
from furax.mapmaking.config import (
    CommonModeConfig,
    GapsConfig,
    GapTreatment,
    HWPSynchronousConfig,
    MapMakingConfig,
    NoiseFitConfig,
    NoiseSource,
    PolynomialOrders,
    ScanSynchronousConfig,
    SolverConfig,
    SotodlibConfig,
    SplineDriftConfig,
    T2PConfig,
    TemplatesConfig,
    TwoLevelConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import MultiObservationMapMaker
from tests.mapmaking.helpers import (
    FakeLazyGroundObservation,
    FakeLazyObservation,
    accumulate,
    assert_same_map,
    assert_same_system,
    healpix_config,
    probe_normal_system,
)


def _config(templates: TemplatesConfig | None) -> MapMakingConfig:
    return healpix_config(
        weighting=WeightingConfig(mode=WeightingMode.DIAGONAL, fitting=NoiseFitConfig(nperseg=256)),
        templates=templates,
        # keep every observed pixel so the map comparisons below are meaningful
        hits_cut=0.0,
        cond_cut=0.0,
    )


def _hwp_obs(n_obs: int = 2, n_dets: int = 8, n_samples: int = 1024):
    return [FakeLazyObservation(seed=i, n_dets=n_dets, n_samples=n_samples) for i in range(n_obs)]


def _ground_obs(n_obs: int = 2, n_dets: int = 8, n_samples: int = 1024, **kwargs):
    return [
        FakeLazyGroundObservation(seed=i, n_dets=n_dets, n_samples=n_samples, **kwargs)
        for i in range(n_obs)
    ]


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
    explicit = MultiObservationMapMaker(obs, config=_config(template(True))).make_maps()
    implicit = MultiObservationMapMaker(obs, config=_config(template(False))).make_maps()
    assert_same_map(explicit, implicit, rtol=1e-4, atol=1e-6)
    assert explicit.solver_stats['num_steps'] >= 1
    # only the explicit amplitudes are solved for, one set per observation and detector
    assert implicit.template_amplitudes is None
    (amplitudes,) = explicit.template_amplitudes.values()
    assert amplitudes.shape[:2] == (2, 8)
    assert jnp.all(jnp.isfinite(amplitudes))


def test_mixed_explicit_and_implicit():
    # One explicit template (solved + returned) and one implicit template (deprojected into W').
    cfg = _config(
        TemplatesConfig(
            hwp_synchronous=HWPSynchronousConfig(2, explicit=True),
            scan_synchronous=ScanSynchronousConfig(explicit=False),
        )
    )
    templates = accumulate(MultiObservationMapMaker(_ground_obs(), config=cfg)).buckets[0].templates
    assert list(templates.explicit.operator.templates) == ['hwp_synchronous']
    assert list(templates.implicit.operator.templates) == ['scan_synchronous']


def test_two_level_preconditioner_with_explicit_templates():
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)))
    cfg.solver = SolverConfig(rtol=1e-10, max_steps=500)
    expected = MultiObservationMapMaker(_hwp_obs(), config=cfg).make_maps()
    expected_amplitudes = expected.template_amplitudes['hwp_synchronous']
    for spectrum in ('preconditioned', 'system'):
        cfg.solver.two_level = TwoLevelConfig(rank=4, spectrum=spectrum, tol=1e-8, max_restarts=50)
        res = MultiObservationMapMaker(_hwp_obs(), config=cfg).make_maps()
        assert_allclose(res.map.data, expected.map.data, rtol=1e-6, atol=1e-8)
        amplitudes = res.template_amplitudes['hwp_synchronous']
        assert_allclose(amplitudes, expected_amplitudes, rtol=1e-6, atol=1e-8)


def test_spline_drift_has_amplitudes_per_detector_and_knot():
    spline = SplineDriftConfig(samples_per_knot=128, explicit=True)
    cfg = _config(TemplatesConfig(spline_drift=spline))
    acc = accumulate(MultiObservationMapMaker(_ground_obs(), config=cfg))
    # (tile, det, knot): one tile per observation, 1024 // 128 knots, plus 2
    assert acc.buckets[0].tiles.stacked.amplitude_rhs['spline_drift'].shape == (2, 8, 10)


@pytest.mark.parametrize(
    ('name', 'option', 'templates'),
    [
        ('spline_drift', 256, {'spline_drift': 'iqu'}),
        (
            'spline_drift',
            {'i': 256, 'q': 128, 'u': 128},
            {'spline_drift_i': 'i', 'spline_drift_qu': 'qu'},
        ),
        ('scan_synchronous', {'qu': PolynomialOrders(2, 5)}, {'scan_synchronous_qu': 'qu'}),
    ],
)
def test_demodulated_streams_fitted_alike_form_one_template(name, option, templates):
    # the streams fitted alike share one template, so an observation stores its basis once; the
    # amplitudes stay one set per stream, on the streams given an option only
    template = SplineDriftConfig() if name == 'spline_drift' else ScanSynchronousConfig()
    cfg = _config(TemplatesConfig(**{name: template}))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    # set after demodulation, which per-stream options require
    if name == 'spline_drift':
        template.samples_per_knot = option
    else:
        template.legendre = option
    acc = accumulate(MultiObservationMapMaker(_ground_obs(), config=cfg))
    operator = acc.buckets[0].templates.implicit.operator
    assert {key: t.streams for key, t in operator.templates.items()} == templates


def _detector_batch_systems(cfg: MapMakingConfig, observations=_ground_obs, **obs_kwargs):
    """The normal system of a run with the detectors processed whole, then in batches."""
    return [
        probe_normal_system(
            MultiObservationMapMaker(
                observations(n_dets=7, **obs_kwargs), config=replace(cfg, detector_batch_size=batch)
            )
        )
        for batch in (0, 3)
    ]


@pytest.mark.parametrize(
    ('observations', 'templates', 'demodulated'),
    [
        (_hwp_obs, None, False),
        (_hwp_obs, TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)), False),
        # Drift and scan-synchronous templates together are nearly degenerate on these short
        # scans, so they come separately.
        (
            _ground_obs,
            TemplatesConfig(
                spline_drift=SplineDriftConfig(samples_per_knot={'qu': 256}), t2p=T2PConfig()
            ),
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
    ids=['no-templates', 'explicit', 'demodulated-drift-and-t2p', 'demodulated-azss-and-t2p'],
)
def test_detector_batches_give_the_same_system(observations, templates, demodulated):
    # Processing an observation's detectors in batches is a memory layout, not a change of model:
    # every template, weight and Gram is per detector. 7 detectors in batches of 3 also pads two.
    # The systems are compared, not their solutions: the templates make them ill-conditioned, so
    # solving turns the rounding of either into a much larger difference of the maps.
    sotodlib = SotodlibConfig(demodulated=True) if demodulated else None
    cfg = replace(_config(None), templates=templates, sotodlib=sotodlib)
    whole, batched = _detector_batch_systems(cfg, observations)
    assert_same_system(batched, whole)


def test_padded_detectors_are_left_out_of_the_results():
    # batching 7 detectors by 3 pads two, whose rows must not reach the detector table, and whose
    # amplitudes trail the real ones
    templates = TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True))
    cfg = _config(templates)
    n_dets = 7
    whole, batched = [
        MultiObservationMapMaker(
            _hwp_obs(n_dets=n_dets), config=replace(cfg, detector_batch_size=batch)
        ).make_maps()
        for batch in (0, 3)
    ]
    assert_frame_equal(batched.detectors, whole.detectors, rel_tol=1e-10)
    (expected,) = whole.template_amplitudes.values()
    (got,) = batched.template_amplitudes.values()
    assert_allclose(got[:, :n_dets], expected, atol=1e-4 * jnp.max(jnp.abs(expected)))


@pytest.mark.parametrize(
    ('joint_qu', 'drift'),
    [(True, False), (False, False), (True, True)],
    ids=['joint', 'per-stream', 'with-drift'],
)
def test_t2p_field_gives_the_same_system_in_detector_batches(joint_qu, drift):
    # The amplitudes of a leakage field are shared by an observation's detectors, so its batches
    # solve for them together: batching changes the memory layout, not the model. With a
    # drift, the unknowns hold both kinds of amplitudes.
    templates = TemplatesConfig(
        t2p=T2PConfig(zernike_order=1, joint_qu=joint_qu),
        spline_drift=SplineDriftConfig(samples_per_knot={'qu': 256}) if drift else None,
    )
    cfg = replace(_config(None), templates=templates, sotodlib=SotodlibConfig(demodulated=True))
    whole, batched = _detector_batch_systems(cfg, focal_plane_radius=0.2)
    assert_same_system(batched, whole)


def test_observations_of_different_widths_give_the_same_system_in_tiles():
    # 7 and 3 detectors in one bucket: in tiles of 2 the observations take 4 and 2 tiles, the
    # last of each padded, where whole they are one tile each, the narrow one padded to 7
    # detectors. The amplitudes of the leakage field, shared by an observation's detectors, sum
    # over its tiles.
    templates = TemplatesConfig(t2p=T2PConfig(zernike_order=1))
    sotodlib = SotodlibConfig(demodulated=True)
    cfg = replace(_config(None), templates=templates, sotodlib=sotodlib, max_buckets=1)

    def observations():
        return [
            FakeLazyGroundObservation(seed=i, n_dets=n, n_samples=1024, focal_plane_radius=0.2)
            for i, n in enumerate((7, 3))
        ]

    whole, tiled = [
        probe_normal_system(
            MultiObservationMapMaker(observations(), config=replace(cfg, detector_batch_size=size))
        )
        for size in (0, 2)
    ]
    assert_same_system(tiled, whole)


_COMMON_MODE_WITH_DRIFT = TemplatesConfig(
    common_mode=CommonModeConfig(samples_per_knot=128),
    spline_drift=SplineDriftConfig(samples_per_knot={'qu': 512}),
    regularization=1e-3,
)


_T2P_QUADRUPOLE = TemplatesConfig(t2p=T2PConfig(quadrupole_order=1))
_QUADRUPOLAR = [
    pytest.param(_T2P_QUADRUPOLE, False, id='t2p-quadrupole'),
    pytest.param(_COMMON_MODE_WITH_DRIFT, False, id='common-mode-with-drift'),
    pytest.param(_COMMON_MODE_WITH_DRIFT, True, id='common-mode-with-drift-correlated'),
]


def _quadrupolar_config(templates: TemplatesConfig, correlated: bool) -> MapMakingConfig:
    if correlated:
        return _toeplitz_config(templates, precomputed=True, demodulated=True)
    return replace(_config(None), templates=templates, sotodlib=SotodlibConfig(demodulated=True))


@pytest.mark.parametrize(('templates', 'correlated'), _QUADRUPOLAR)
def test_quadrupolar_templates_give_the_same_system_in_detector_batches(templates, correlated):
    # Their amplitudes belong to the observation, as those of the Zernike field: batching the
    # detectors changes the memory layout, not the model. Next to each detector's implicit drift,
    # the Gram of the common mode is that under the marginalized weight; under a correlated
    # weight, it is probed in each batch, then summed.
    cfg = _quadrupolar_config(templates, correlated)
    whole, batched = _detector_batch_systems(cfg, focal_plane_radius=0.2)
    assert_same_system(batched, whole)


@pytest.mark.parametrize(
    ('name', 'templates'),
    [('t2p', _T2P_QUADRUPOLE), ('common_mode', _COMMON_MODE_WITH_DRIFT)],
    ids=['t2p-quadrupole', 'common-mode-with-drift'],
)
def test_amplitudes_shared_by_detectors_are_returned_per_observation(name, templates):
    cfg = _quadrupolar_config(templates, correlated=False)
    maps = MultiObservationMapMaker(
        _ground_obs(n_dets=7, focal_plane_radius=0.2), config=cfg
    ).make_maps()
    amplitudes = maps.template_amplitudes[name]
    assert jnp.all(jnp.isfinite(amplitudes)) and jnp.max(jnp.abs(amplitudes)) > 0
    assert amplitudes.shape[0] == 2  # one set of amplitudes per observation


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

    maker = MultiObservationMapMaker(observations, config=cfg)
    together = accumulate(maker)
    (bucket,) = maker.layout.buckets
    assert bucket.tiles_per_device >= 2
    alone = [accumulate(MultiObservationMapMaker([obs], config=cfg)) for obs in observations]

    assert_allclose(together.hit_map, sum(acc.hit_map for acc in alone))
    assert_allclose(
        together.map_rhs.data, sum(acc.map_rhs.data for acc in alone), rtol=1e-12, atol=1e-12
    )
    # (tile, ...), with one tile per observation
    amplitude_rhs = np.asarray(together.buckets[0].tiles.stacked.amplitude_rhs['hwp_synchronous'])
    for observation, acc in enumerate(alone):
        (segment,) = np.flatnonzero(bucket.segment_observations == observation)
        (tile,) = bucket.tiles_of(segment)
        alone_rhs = acc.buckets[0].tiles.stacked.amplitude_rhs['hwp_synchronous']
        assert_allclose(amplitude_rhs[tile], np.asarray(alone_rhs)[0])


# ---------------------------------------------------------------------------
# Pomme combined with templates
# ---------------------------------------------------------------------------


def _pomme_config(templates: TemplatesConfig) -> MapMakingConfig:
    cfg = _config(templates)
    cfg.pomme_tau = 37
    cfg.landscape.stokes = 'QU'
    templates.regularization = 1e-10
    return cfg


def test_pomme_explicit_and_implicit_give_the_same_map():
    # Under Pomme the effective weight is W F; marginalising the amplitudes under it must still
    # match the joint solve. Both paths use the Pomme-filtered Gram (as weight fold and as
    # preconditioner respectively).
    def template(explicit):
        return TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=explicit))

    obs = _hwp_obs()
    explicit = MultiObservationMapMaker(obs, config=_pomme_config(template(True))).make_maps()
    implicit = MultiObservationMapMaker(obs, config=_pomme_config(template(False))).make_maps()
    assert_same_map(explicit, implicit, rtol=1e-4, atol=1e-6)


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
        return MultiObservationMapMaker(observations, config=config).make_maps()

    explicit, implicit = run(True), run(False)
    assert_same_map(explicit, implicit, rtol=1e-4, atol=1e-6)


def _toeplitz_config(
    templates: TemplatesConfig, precomputed: bool = False, demodulated: bool = False
) -> MapMakingConfig:
    sotodlib = SotodlibConfig(demodulated=True) if demodulated else None
    cfg = replace(_config(None), templates=templates, sotodlib=sotodlib)
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
        (
            _ground_obs,
            lambda e: TemplatesConfig(
                scan_synchronous=ScanSynchronousConfig(PolynomialOrders(1, 3), explicit=e)
            ),
        ),
    ],
    ids=['hwp_synchronous', 'scan_synchronous'],
)
def test_correlated_weight_explicit_and_implicit_give_the_same_map(observations, template):
    # Under a correlated weight, marginalizing the amplitudes needs the exact Gram, probed from the
    # weight; the joint solve only uses it as a preconditioner.
    obs = observations()
    precomputed = observations is _ground_obs
    maps = [
        MultiObservationMapMaker(obs, config=_toeplitz_config(template(e), precomputed)).make_maps()
        for e in (True, False)
    ]
    assert_same_map(*maps, rtol=1e-6, atol=1e-8)


def test_detector_batches_give_the_same_system_under_a_correlated_weight():
    # The Gram of each detector is probed from the weight, whatever the batch it is in. The
    # systems are compared, not their solutions: this one has two directions 1e-10 times its
    # largest eigenvalue, so solving it turns the rounding of either (~1e-15) into ~1e-5 of the map.
    templates = TemplatesConfig(spline_drift=SplineDriftConfig(samples_per_knot={'qu': 256}))
    cfg = _toeplitz_config(templates, precomputed=True, demodulated=True)
    whole, batched = _detector_batch_systems(cfg)
    assert_same_system(batched, whole)


def test_demodulated_drift_explicit_and_implicit_agree_under_a_correlated_weight():
    # The Gram of the drift of a detector is singular here, so it takes a ridge to be factored
    templates = TemplatesConfig(
        spline_drift=SplineDriftConfig(samples_per_knot={'qu': 512}), regularization=1e-4
    )
    maps = [
        MultiObservationMapMaker(
            _ground_obs(),
            config=_toeplitz_config(
                replace(templates, spline_drift=replace(templates.spline_drift, explicit=e)),
                precomputed=True,
                demodulated=True,
            ),
        ).make_maps()
        for e in (True, False)
    ]
    assert_same_map(*maps, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_common_mode_absorbed_by_the_drifts_under_a_small_ridge_converges():
    # The detectors' drifts absorb the slow part of the common mode, which the system then holds
    # through a ridge below the rounding cutoff of the inverse of its Gram: preconditioned by an
    # inverse that left those directions out, CG would stall.
    templates = TemplatesConfig(
        common_mode=CommonModeConfig(samples_per_knot=128),
        spline_drift=SplineDriftConfig(samples_per_knot={'qu': 256}),
        regularization=1e-8,
    )
    cfg = _toeplitz_config(templates, precomputed=True, demodulated=True)
    # converges in about 300 iterations; without the floor, not in 1500
    cfg.solver = SolverConfig(rtol=1e-10, max_steps=500)
    results = MultiObservationMapMaker(
        _ground_obs(n_dets=7, focal_plane_radius=0.2), config=cfg
    ).make_maps()
    assert results.solver_stats['converged']
