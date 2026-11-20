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
from polars.testing import assert_frame_equal

from furax.core import StackedEnsemble
from furax.mapmaking import ReaderField
from furax.mapmaking._bucket import bucket_map_normal, bucket_normal
from furax.mapmaking._tile import TEMPLATES
from furax.mapmaking.config import (
    AzimuthSynchronousConfig,
    CommonModeConfig,
    GapsConfig,
    GapTreatment,
    HWPSynchronousConfig,
    MapMakingConfig,
    NoiseFitConfig,
    NoiseSource,
    SolverConfig,
    SotodlibConfig,
    SplineDriftConfig,
    T2PConfig,
    TemplatesConfig,
    TwoLevelConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import MapMakingRun, MultiObservationMapMaker
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


def _azimuth_spline(n_knots: int, explicit: bool, split: bool = False) -> AzimuthSynchronousConfig:
    return AzimuthSynchronousConfig(n_knots=n_knots, split_scans=split, explicit=explicit)


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
            lambda e: TemplatesConfig(azimuth_synchronous=_azimuth_spline(6, e)),
        ),
        (
            _ground_obs,
            lambda e: TemplatesConfig(azimuth_synchronous=_azimuth_spline(6, e, split=True)),
        ),
    ],
    ids=['hwp_synchronous', 'azimuth_spline', 'azimuth_spline_split'],
)
def test_explicit_and_implicit_give_the_same_map(observations, template):
    # Marginalising the amplitudes (implicit deprojection) is exactly equivalent to solving
    # them jointly and discarding them: the recovered map must match.
    obs = observations()
    explicit = MultiObservationMapMaker(_config(template(True))).make_maps(obs)
    implicit = MultiObservationMapMaker(_config(template(False))).make_maps(obs)
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
            azimuth_synchronous=_azimuth_spline(6, False),
        )
    )
    buckets, _ = accumulate(MultiObservationMapMaker(cfg).prepare(_ground_obs()))
    templates = buckets[0].template
    assert list(templates.explicit[TEMPLATES].operator.templates) == ['hwp_synchronous']
    assert list(templates.implicit.operator.templates) == ['azimuth_synchronous']


def test_two_level_preconditioner_with_explicit_templates():
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)))
    cfg.solver = SolverConfig(rtol=1e-8, max_steps=1500)
    expected = MultiObservationMapMaker(cfg).make_maps(_hwp_obs())
    expected_amplitudes = expected.template_amplitudes['hwp_synchronous']
    for spectrum in ('preconditioned', 'system'):
        cfg.solver.two_level = TwoLevelConfig(rank=4, spectrum=spectrum, tol=1e-8, max_restarts=50)
        res = MultiObservationMapMaker(cfg).make_maps(_hwp_obs())
        assert_allclose(res.map.data, expected.map.data, rtol=1e-6, atol=1e-8)
        amplitudes = res.template_amplitudes['hwp_synchronous']
        assert_allclose(amplitudes, expected_amplitudes, rtol=1e-6, atol=1e-8)


def test_spline_drift_has_amplitudes_per_detector_and_knot():
    spline = SplineDriftConfig(samples_per_knot=128, explicit=True)
    cfg = _config(TemplatesConfig(spline_drift=spline))
    _, totals = accumulate(MultiObservationMapMaker(cfg).prepare(_ground_obs()))
    # (tile, det, knot): one tile per observation, 1024 // 128 knots, plus 2
    assert totals.rhs[TEMPLATES][0]['spline_drift'].shape == (2, 8, 10)


@pytest.mark.parametrize(
    ('name', 'option', 'templates'),
    [
        ('spline_drift', 256, {'spline_drift': 'iqu'}),
        (
            'spline_drift',
            {'i': 256, 'q': 128, 'u': 128},
            {'spline_drift_i': 'i', 'spline_drift_qu': 'qu'},
        ),
        ('azimuth_synchronous', {'qu': 6}, {'azimuth_synchronous_qu': 'qu'}),
    ],
)
def test_demodulated_streams_fitted_alike_form_one_template(name, option, templates):
    # the streams fitted alike share one template, so an observation stores its basis once; the
    # amplitudes stay one set per stream, on the streams given an option only
    template = SplineDriftConfig() if name == 'spline_drift' else _azimuth_spline(6, False)
    cfg = _config(TemplatesConfig(**{name: template}))
    cfg.sotodlib = SotodlibConfig(demodulated=True)
    # set after demodulation, which per-stream options require
    if name == 'spline_drift':
        template.samples_per_knot = option
    else:
        template.n_knots = option
    buckets, _ = accumulate(MultiObservationMapMaker(cfg).prepare(_ground_obs()))
    operator = buckets[0].template.implicit.operator
    assert {key: t.streams for key, t in operator.templates.items()} == templates


def _detector_batch_systems(cfg: MapMakingConfig, observations=_ground_obs, **obs_kwargs):
    """The normal system of a run with the detectors processed whole, then in batches."""
    return [
        probe_normal_system(
            MultiObservationMapMaker(replace(cfg, detector_tile_size=batch)).prepare(
                observations(n_dets=7, **obs_kwargs)
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
            TemplatesConfig(azimuth_synchronous=_azimuth_spline(6, False), t2p=T2PConfig()),
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
        MultiObservationMapMaker(replace(cfg, detector_tile_size=batch)).make_maps(
            _hwp_obs(n_dets=n_dets)
        )
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
            MultiObservationMapMaker(replace(cfg, detector_tile_size=size)).prepare(observations())
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
_PER_WAFER = TemplatesConfig(
    azimuth_synchronous=AzimuthSynchronousConfig(
        n_knots={'qu': 6}, split_scans=True, per_wafer=True, explicit=True
    ),
    common_mode=CommonModeConfig(samples_per_knot=128, per_wafer=True),
    regularization=1e-3,
)
_QUADRUPOLAR = [
    pytest.param(_T2P_QUADRUPOLE, id='t2p-quadrupole'),
    pytest.param(_COMMON_MODE_WITH_DRIFT, id='common-mode-with-drift'),
    # the wafers alternate between detectors, so every batch holds a part of each
    pytest.param(_PER_WAFER, id='per-wafer'),
]


def _quadrupolar_config(templates: TemplatesConfig) -> MapMakingConfig:
    return replace(_config(None), templates=templates, sotodlib=SotodlibConfig(demodulated=True))


@pytest.mark.parametrize('templates', _QUADRUPOLAR)
def test_quadrupolar_templates_give_the_same_system_in_detector_batches(templates):
    # Their amplitudes belong to the observation, as those of the Zernike field: batching the
    # detectors changes the memory layout, not the model. Next to each detector's implicit drift,
    # the Gram of the common mode is that under the marginalized weight, summed over the batches.
    cfg = _quadrupolar_config(templates)
    whole, batched = _detector_batch_systems(cfg, focal_plane_radius=0.2)
    assert_same_system(batched, whole)


@pytest.mark.parametrize(
    ('name', 'templates'),
    [('t2p', _T2P_QUADRUPOLE), ('common_mode', _COMMON_MODE_WITH_DRIFT)],
    ids=['t2p-quadrupole', 'common-mode-with-drift'],
)
def test_amplitudes_shared_by_detectors_are_returned_per_observation(name, templates):
    cfg = _quadrupolar_config(templates)
    maps = MultiObservationMapMaker(cfg).make_maps(_ground_obs(n_dets=7, focal_plane_radius=0.2))
    amplitudes = maps.template_amplitudes[name]
    assert jnp.all(jnp.isfinite(amplitudes)) and jnp.max(jnp.abs(amplitudes)) > 0
    assert amplitudes.shape[0] == 2  # one set of amplitudes per observation


@pytest.mark.parametrize(
    'normal',
    [
        pytest.param(lambda tiles: bucket_normal(tiles, None), id='system'),
        pytest.param(lambda tiles: bucket_map_normal(tiles, binned=True), id='binned'),
        pytest.param(lambda tiles: bucket_map_normal(tiles, binned=False), id='white'),
    ],
)
def test_the_normal_system_of_a_bucket_holds_the_arrays_of_its_tiles(normal):
    # Memory: the system of each tile reads the arrays of the tiles in place, those the tiles of
    # an observation share -- its boresight pointing, its shared templates -- held once per
    # observation rather than copied for each of its tiles.
    cfg = replace(_quadrupolar_config(_T2P_QUADRUPOLE), detector_tile_size=3)
    run = MultiObservationMapMaker(cfg).prepare(_ground_obs(n_dets=7, focal_plane_radius=0.2))
    tiles = accumulate(run)[0][0]
    members = normal(tiles).operands
    assert isinstance(members, StackedEnsemble)
    held = {id(leaf) for leaf in jax.tree.leaves(tiles.stacked)}
    arrays = [leaf for leaf in jax.tree.leaves(members.stacked) if isinstance(leaf, jax.Array)]
    assert all(id(leaf) in held for leaf in arrays)
    assert members.strides.count('segment') == tiles.strides.count('segment') > 0


@pytest.mark.parametrize(('batch', 'n_buffered'), [(0, 8), (64, 8), (3, 9)])
def test_only_observations_larger_than_a_batch_are_padded(batch, n_buffered):
    # 8 detectors: a batch of 64 holds them all, unpadded; batches of 3 need a ninth
    maker = MultiObservationMapMaker(replace(_config(None), detector_tile_size=batch)).prepare(
        _hwp_obs()
    )
    assert maker.readers[0].out_structure[ReaderField.DETECTOR_QUATERNIONS].shape[0] == n_buffered


@pytest.mark.parametrize('prefetch', [False, True], ids=['read-after-round', 'prefetch'])
def test_several_observations_per_device_match_single_observations(monkeypatch, prefetch: bool):
    # More observations than devices: each device accumulates several, one round at a time,
    # reading the next round after the current one or, as with a GPU, while it computes.
    # Each observation's sums and outputs must be those of a run over it alone.
    monkeypatch.setattr(MapMakingRun, '_prefetches_reads', prefetch)
    cfg = _config(TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(2, explicit=True)))
    observations = _hwp_obs(n_obs=10)

    maker = MultiObservationMapMaker(cfg).prepare(observations)
    _, together = accumulate(maker)
    (bucket,) = maker.layout.buckets
    assert bucket.tiles_per_device >= 2
    alone = [accumulate(MultiObservationMapMaker(cfg).prepare([obs]))[1] for obs in observations]

    assert_allclose(together.hit_map, sum(totals.hit_map for totals in alone))
    assert_allclose(
        together.rhs['map'].data,
        sum(totals.rhs['map'].data for totals in alone),
        rtol=1e-12,
        atol=1e-12,
    )
    # (tile, ...), with one tile per observation
    template_rhs = np.asarray(together.rhs[TEMPLATES][0]['hwp_synchronous'])
    for observation, totals in enumerate(alone):
        (segment,) = np.flatnonzero(bucket.segment_observations == observation)
        (tile,) = bucket.tiles_of(segment)
        alone_rhs = totals.rhs[TEMPLATES][0]['hwp_synchronous']
        assert_allclose(template_rhs[tile], np.asarray(alone_rhs)[0])


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
    explicit = MultiObservationMapMaker(_pomme_config(template(True))).make_maps(obs)
    implicit = MultiObservationMapMaker(_pomme_config(template(False))).make_maps(obs)
    assert_same_map(explicit, implicit, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('batch', [0, 1], ids=['one_batch', 'per_detector'])
def test_explicit_and_implicit_agree_under_a_ridge(batch):
    # Pomme removes the constant part of every template, so a scan-synchronous template starting
    # at order 0 has a singular Gram and only the ridge makes it invertible. Marginalising the
    # amplitudes uses the ridged Gram, so the joint system must carry the same ridge on its
    # amplitude block, otherwise the two solves estimate different maps. With one detector per
    # batch, the ridge is laid out over several slices per observation.
    def run(explicit: bool):
        scan = _azimuth_spline(5, explicit)
        config = _pomme_config(TemplatesConfig(azimuth_synchronous=scan))
        config.templates.regularization = 1e-3  # large enough to tell the two systems apart
        config = replace(config, detector_tile_size=batch)
        observations = _ground_obs(n_obs=1, n_dets=2, n_samples=256)
        return MultiObservationMapMaker(config).make_maps(observations)

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


def test_detector_batches_give_the_same_system_under_a_correlated_weight():
    # The Gram of each detector is taken from the weight, whatever the batch it is in. The
    # systems are compared, not their solutions: this one has two directions 1e-10 times its
    # largest eigenvalue, so solving it turns the rounding of either (~1e-15) into ~1e-5 of the map.
    templates = TemplatesConfig(
        spline_drift=SplineDriftConfig(samples_per_knot={'qu': 256}, explicit=True)
    )
    cfg = _toeplitz_config(templates, precomputed=True, demodulated=True)
    whole, batched = _detector_batch_systems(cfg)
    assert_same_system(batched, whole)


# the fakes scan azimuths from 0 to 10 rad (573 deg) at an elevation of 0.8 rad (46 deg)
_ACROSS_OBSERVATIONS = AzimuthSynchronousConfig(
    bin_size=120.0,
    azimuth_range=(-60.0, 660.0),
    split_scans=True,
    per_wafer=True,
    across_observations=True,
    elevations=(46.0, 60.0),
    explicit=True,
)


def test_azimuth_template_across_observations_gives_the_same_system_in_detector_batches():
    # every tile of every observation reads the same amplitudes, whose Gram sums over them all
    templates = TemplatesConfig(azimuth_synchronous=_ACROSS_OBSERVATIONS, regularization=1e-3)
    cfg = replace(_config(None), templates=templates, sotodlib=SotodlibConfig(demodulated=True))
    whole, batched = _detector_batch_systems(cfg, _ground_obs)
    assert_same_system(batched, whole)


def test_azimuth_amplitudes_across_observations_are_reported_once():
    templates = TemplatesConfig(azimuth_synchronous=_ACROSS_OBSERVATIONS, regularization=1e-3)
    cfg = replace(_config(None), templates=templates, sotodlib=SotodlibConfig(demodulated=True))
    results = MultiObservationMapMaker(cfg).make_maps(_ground_obs(n_obs=3))
    (amplitudes,) = results.template_amplitudes.values()
    # (stream, wafer, direction x bin, elevation): no observation axis
    assert amplitudes.shape == (3, 7, 12, 2)
    assert jnp.all(jnp.isfinite(amplitudes))
    # the detectors are on the first two wafers, the observations at the first elevation
    assert jnp.all(jnp.any(amplitudes[:, :2, :, 0] != 0, axis=-1))
    assert not jnp.any(amplitudes[:, 2:]) and not jnp.any(amplitudes[..., 1])
