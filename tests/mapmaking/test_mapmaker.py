import importlib.util
import logging
from pathlib import Path
from typing import Literal

import equinox as eqx
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from furax.core._base import structure_equal
from furax.mapmaking import AbstractLazyObservation, MapMakingConfig, MultiObservationMapMaker
from furax.mapmaking._reader import detector_tile_structure
from furax.mapmaking.config import (
    DiagnosticsConfig,
    ExplicitGapsConfig,
    GapTreatment,
    HealpixConfig,
    LandscapeConfig,
    NestedConfig,
    NoiseFitConfig,
    NoiseSource,
    PointingConfig,
    SkyPatch,
    SolverConfig,
    SotodlibConfig,
    TwoLevelConfig,
    WCSConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.noise import WhiteNoiseModel
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import ValidStokesLiteral
from tests.mapmaking.helpers import (
    FailingLazyObservation,
    FakeLazyGroundObservation,
    FakeLazyObservation,
    GapProbingLazyGroundObservation,
    GappyLazyGroundObservation,
    ProbeFailingLazyObservation,
    accumulate,
    furax_records,
)

# Skip tests for interfaces that are not installed
sotodlib_installed = importlib.util.find_spec('sotodlib') is not None
toast_installed = importlib.util.find_spec('toast') is not None

INTERFACES = [
    pytest.param(
        'sotodlib',
        False,
        id='sotodlib',
        marks=pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed'),
    ),
    pytest.param(
        'sotodlib',
        True,
        id='sotodlib-demod',
        marks=pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed'),
    ),
    pytest.param(
        'toast',
        False,
        id='toast',
        marks=pytest.mark.skipif(not toast_installed, reason='toast is not installed'),
    ),
]
STOKES_TYPES = ['I', 'QU', 'IQU']


@pytest.mark.parametrize(
    ('landscape_type', 'stokes', 'interpolation'),
    [('healpix', 'IQU', 'nearest'), ('car', 'QU', 'bilinear')],
    ids=['healpix-nearest', 'car-bilinear'],
)
@pytest.mark.parametrize('name,demodulated', INTERFACES)
def test_interface_observations_are_mapped(
    name, demodulated, landscape_type, stokes, interpolation
):
    # The fake observations below cover the pipeline itself; this checks that the observations of
    # each interface reach it, on both pixelizations and with either pointing interpolation.
    observations = _observations(name, demodulated)
    config = _config(landscape_type, stokes, demodulated, interpolation=interpolation)
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.make_maps()
    n_stokes = len(stokes)
    assert results.hit_map.shape == maker.landscape.shape
    assert jnp.all(results.hit_map >= 0) and jnp.any(results.hit_map > 0)
    assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
    if interpolation == 'nearest':
        # the system of a binned map is block diagonal, which the preconditioner inverts exactly
        assert results.solver_stats['num_steps'] == 1


@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
@pytest.mark.parametrize('stokes', STOKES_TYPES)
def test_model_matches_the_reader_and_the_landscape(stokes, demodulated):
    observations = [FakeLazyObservation(seed=i) for i in range(2)]
    maker = MultiObservationMapMaker(observations, config=_config('healpix', stokes, demodulated))
    (bucket,) = maker.layout.buckets
    (reader,) = maker.readers
    assert reader.count == len(observations)
    tiles = accumulate(maker).buckets[0].tiles
    assert tiles.size == bucket.n_tiles
    model = tiles.template.model
    # structures compared ignoring sharding (the model is built sharded inside shard_map)
    assert structure_equal(model.map_structure, maker.landscape.structure)
    # the samples are read in float32, and cast to the dtype of the model
    tile, _ = detector_tile_structure(reader.out_structure, bucket.tile_size)
    dtype = model.tod_format.dtype
    expected = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, dtype), tile['sample_data'])
    assert structure_equal(model.tod_format.structure, expected)
    # a single white noise model covers both paths; on a demodulated TOD, its per-detector sigma
    # carries the stream axis (after the tile axis of the accumulation)
    assert isinstance(model.noise_model, WhiteNoiseModel)
    if demodulated:
        assert tiles.stacked.model.noise_model.sigma.shape[1] == len(stokes)


@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
def test_failed_observation_is_excluded(demodulated):
    # One good observation + one whose load fails: the run must complete, exclude the failed
    # observation (contributing nothing, no NaN), and report it.
    config = _config('healpix', 'IQU', demodulated=demodulated)
    mixed = [FakeLazyObservation(seed=0), FailingLazyObservation(seed=1)]
    maker = MultiObservationMapMaker(mixed, config=config)
    with furax_records(logging.WARNING) as records:
        results = maker.make_maps()

    assert results.failed_observations == ['failing_obs']
    (warning,) = [r for r in records if r.levelno == logging.WARNING]
    assert 'simulated preprocessing failure' in (warning.getMessage())
    assert results.hit_map.shape == maker.landscape.shape
    assert results.icov.shape == (3, 3, *maker.landscape.shape)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(results.map))
    assert jnp.all(jnp.isfinite(results.icov))

    # one row per observation, the failed one not loaded
    data = mixed[0].get_data()
    observations = results.observations
    assert observations['name'].to_list() == [data.name, 'failing_obs']
    assert observations['loaded'].to_list() == [True, False]
    assert observations['n_samples'].to_list() == [data.n_samples, 0]
    assert observations['probed_samples'][0] == data.n_samples
    assert observations['error'].to_list() == [
        None,
        'load failed: RuntimeError: simulated preprocessing failure',
    ]
    # one row per detector read, the failed observation's none
    detectors = results.detectors
    assert detectors['observation'].to_list() == [0] * data.n_detectors
    assert detectors['detector'].to_list() == list(range(data.n_detectors))
    assert detectors['name'].to_list() == data.detectors
    assert detectors['xi'].null_count() == 0
    sigma = [c for c in detectors.columns if c.startswith('sigma')]
    assert sigma and all(detectors[c].is_finite().all() for c in sigma)
    stats = results.solver_stats
    assert stats['num_steps'] == 1  # binned map
    assert stats['converged'] and len(stats['residuals']) == stats['num_steps'] + 1

    # the failed observation contributes nothing -> identical to mapping only the good one
    good_only = [FakeLazyObservation(seed=0)]
    expected = MultiObservationMapMaker(good_only, config=config).make_maps()
    assert eqx.tree_equal(results.hit_map, expected.hit_map)
    assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)


class TestBuckets:
    """Observations of different lengths, grouped into buckets of a common buffer shape.

    Bucketing is a layout decision only: the maps must not depend on how many buckets the
    observations are spread over, nor on the padding each bucket carries.
    """

    N_SAMPLES = (512, 640, 1536)

    def _observations(self):
        return [FakeLazyObservation(seed=i, n_samples=n) for i, n in enumerate(self.N_SAMPLES)]

    @staticmethod
    def _correlated_config(correlation_length: int) -> MapMakingConfig:
        config = _config('healpix', 'IQU', max_buckets=3)
        config.weighting = WeightingConfig(
            mode=WeightingMode.TOEPLITZ,
            source=NoiseSource.PRECOMPUTED,
            correlation_length=correlation_length,
        )
        return config

    @pytest.mark.parametrize('max_buckets', [1, 2, 3])
    def test_layout_follows_config(self, max_buckets):
        config = _config('healpix', 'IQU', max_buckets=max_buckets)
        maker = MultiObservationMapMaker(self._observations(), config=config)
        layout = maker.layout
        assert 1 <= len(layout.buckets) <= max_buckets
        covered = sorted(int(i) for b in layout.buckets for i in b.observations)
        assert covered == list(range(len(self.N_SAMPLES)))
        for bucket, reader in zip(layout.buckets, maker.readers, strict=True):
            assert reader.count == bucket.n_observations
            assert bucket.n_tiles % jax.device_count() == 0
            # each reader pads to its own bucket's envelope, not the dataset's largest
            n_samples = reader.out_structure['sample_data'].shape[-1]
            assert n_samples == bucket.envelope.sample_count
        if max_buckets == 3 and jax.device_count() == 1:
            assert len(layout.buckets) == 3  # singletons never pad on one device

    @pytest.mark.parametrize('stokes', ['I', 'IQU'])
    def test_maps_do_not_depend_on_bucketing(self, stokes):
        # Unit weights: a fitted noise level would see the padded samples and so depend on the
        # bucket's envelope, which is not what is under test here.
        observations = self._observations()
        one = _config('healpix', stokes, identity_noise=True)
        many = _config('healpix', stokes, identity_noise=True, max_buckets=3)
        one_result = MultiObservationMapMaker(observations, config=one).make_maps()
        many_result = MultiObservationMapMaker(observations, config=many).make_maps()
        assert eqx.tree_equal(one_result.hit_map, many_result.hit_map)
        assert eqx.tree_equal(one_result.map, many_result.map, rtol=1e-10, atol=1e-12)
        assert eqx.tree_equal(one_result.icov, many_result.icov, rtol=1e-10, atol=1e-12)

    def test_fitted_noise_does_not_depend_on_bucketing(self):
        # The Welch segment is longer than the shortest observation, so an unclipped fit would
        # average a segment straddling its padding, differently in each bucketing.
        observations = self._observations()
        one = _config('healpix', 'I', max_buckets=1, nperseg=1024)
        many = _config('healpix', 'I', max_buckets=3, nperseg=1024)
        maker = MultiObservationMapMaker(observations, config=one)
        assert maker.fit_config.weighting.fitting.nperseg == min(self.N_SAMPLES)
        one_result = maker.make_maps()
        many_result = MultiObservationMapMaker(observations, config=many).make_maps()
        assert eqx.tree_equal(one_result.hit_map, many_result.hit_map)
        assert eqx.tree_equal(one_result.icov, many_result.icov, rtol=1e-10, atol=1e-12)
        assert eqx.tree_equal(one_result.map, many_result.map, rtol=1e-10, atol=1e-12)

    def test_failed_observation_in_a_bucket(self):
        # the failing observation is the short one, alone in its bucket once split
        config = _config('healpix', 'IQU', max_buckets=3)
        observations = self._observations()
        observations[0] = FailingLazyObservation(seed=0, n_samples=self.N_SAMPLES[0])
        results = MultiObservationMapMaker(observations, config=config).make_maps()
        assert results.failed_observations == ['failing_obs']
        expected = MultiObservationMapMaker(observations[1:], config=config).make_maps()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)

    def test_probe_failed_observation_inherits_a_safe_shape(self):
        config = self._correlated_config(correlation_length=64)
        observations = [
            ProbeFailingLazyObservation(seed=0, n_samples=32),
            GappyLazyGroundObservation(seed=1, n_samples=128),
            GappyLazyGroundObservation(seed=2, n_samples=256),
        ]
        maker = MultiObservationMapMaker(observations, config=config)
        shapes = maker._probe_shapes
        assert maker._catalog.probe_failed.tolist() == [True, False, False]
        assert shapes[0] == shapes[2]
        # its row keeps why, before anything is read
        row = maker._catalog.to_table().row(0, named=True)
        assert not row['loaded']
        assert row['probed_samples'] == 0
        assert row['error'] == 'probe failed: RuntimeError: simulated shape-probe failure'

    def test_fft_friendly_sample_count_pads_the_buffers(self):
        config = _config('healpix', 'IQU', max_buckets=2)
        config.fft_friendly_sample_count = True
        observations = [FakeLazyObservation(seed=i, n_samples=n) for i, n in enumerate((97, 121))]
        maker = MultiObservationMapMaker(observations, config=config)
        assert [shape.sample_count for shape in maker._probe_shapes] == [97, 121]
        for bucket, reader in zip(maker.layout.buckets, maker.readers, strict=True):
            n_samples = bucket.envelope.sample_count
            assert n_samples in (100, 125)
            assert reader.out_structure['sample_data'].shape[-1] == n_samples
        accumulate(maker)

    def test_welch_segment_is_clipped_to_the_true_length_of_padded_buffers(self):
        config = _config('healpix', 'I', nperseg=1024)
        config.fft_friendly_sample_count = True
        observations = [FakeLazyObservation(seed=0, n_samples=97)]
        maker = MultiObservationMapMaker(observations, config=config)
        assert maker.layout.buckets[0].envelope.sample_count == 100
        assert maker.fit_config.weighting.fitting.nperseg == 97

    def test_bucket_envelope_covers_toeplitz_support(self):
        config = self._correlated_config(correlation_length=64)
        observations = [GappyLazyGroundObservation(seed=0, n_samples=32)]
        observations.extend(GappyLazyGroundObservation(seed=i, n_samples=256) for i in (1, 2))
        maker = MultiObservationMapMaker(observations, config=config)
        assert min(bucket.envelope.sample_count for bucket in maker.layout.buckets) >= 2 * 64 - 1
        accumulate(maker)

    def test_bucket_envelope_covers_pomme_support(self):
        config = _config('healpix', 'QU', pomme_tau=64, max_buckets=3)
        observations = [FakeLazyObservation(seed=0, n_samples=32)]
        observations.extend(FakeLazyObservation(seed=i, n_samples=256) for i in (1, 2))
        maker = MultiObservationMapMaker(observations, config=config)
        assert min(bucket.envelope.sample_count for bucket in maker.layout.buckets) >= 64
        accumulate(maker)


def test_two_level_preconditioner_solves_the_same_system():
    config = _config('healpix', 'IQU')
    config.weighting = WeightingConfig(
        mode=WeightingMode.TOEPLITZ, source=NoiseSource.PRECOMPUTED, correlation_length=64
    )
    config.solver = SolverConfig(rtol=1e-10, max_steps=500)
    observations = [GappyLazyGroundObservation(seed=i, n_samples=256) for i in range(2)]
    expected = MultiObservationMapMaker(observations, config=config).make_maps()
    for spectrum, which in [
        ('preconditioned', 'small'),
        ('system', 'small'),
        ('preconditioned', 'large'),
    ]:
        config.solver.two_level = TwoLevelConfig(
            rank=4, which=which, spectrum=spectrum, tol=1e-8, max_restarts=50
        )
        results = MultiObservationMapMaker(observations, config=config).make_maps()
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-8), (spectrum, which)


def test_solve_resumes_from_its_checkpoint(tmp_path: Path):
    config = _config('healpix', 'IQU')
    config.weighting = WeightingConfig(
        mode=WeightingMode.TOEPLITZ, source=NoiseSource.PRECOMPUTED, correlation_length=64
    )
    config.solver = SolverConfig(rtol=1e-10, max_steps=500)
    observations = [GappyLazyGroundObservation(seed=i, n_samples=256) for i in range(2)]
    expected = MultiObservationMapMaker(observations, config=config).make_maps()
    assert expected.solver_stats['num_steps'] > 6

    # a solve that runs out of steps saves its last iterate...
    config.solver = SolverConfig(rtol=1e-10, max_steps=6, checkpoint_every=4)
    stopped = MultiObservationMapMaker(observations, config=config).make_maps(tmp_path)
    assert not stopped.solver_stats['converged']
    assert int(np.load(tmp_path / 'checkpoint.npz')['n_steps']) == 6

    # ...which a resume with the same budget leaves as it is
    unchanged = MultiObservationMapMaker(observations, config=config).make_maps(
        tmp_path, resume=True
    )
    assert unchanged.solver_stats['resumed_after'] == 6
    assert unchanged.solver_stats['num_steps'] == 6
    assert len(unchanged.solver_stats['residuals']) == 1  # no step taken in this run
    assert int(np.load(tmp_path / 'checkpoint.npz')['n_steps']) == 6

    # ...which a solve with more steps resumes, deleting it once converged
    config.solver = SolverConfig(rtol=1e-10, max_steps=500, checkpoint_every=4)
    maker = MultiObservationMapMaker(observations, config=config)
    resumed = maker.make_maps(tmp_path, resume=True)
    assert resumed.solver_stats['converged']
    assert resumed.solver_stats['resumed_after'] == 6
    assert resumed.solver_stats['num_steps'] == 6 + len(resumed.solver_stats['residuals']) - 1
    assert not (tmp_path / 'checkpoint.npz').exists()
    assert eqx.tree_equal(resumed.map, expected.map, rtol=1e-6, atol=1e-8)


def _observations(name: str, demodulated: bool = False) -> list[AbstractLazyObservation]:
    folder = Path(__file__).parents[1] / 'data' / name
    if name == 'toast':
        from furax.interfaces.toast import LazyToastObservation

        files = [folder / 'test_obs.h5'] * 2
        return [LazyToastObservation(f) for f in files]
    elif name == 'sotodlib':
        from furax.interfaces.sotodlib import LazySOTODLibObservation

        sotodlib_config = SotodlibConfig(demodulated=True) if demodulated else None
        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
        return [LazySOTODLibObservation(f, sotodlib_config=sotodlib_config) for f in files]
    raise NotImplementedError


def _config(
    landscape_type: Literal['healpix', 'car'],
    stokes: ValidStokesLiteral,
    demodulated: bool = False,
    interpolation: Literal['nearest', 'bilinear'] = 'nearest',
    pomme_tau: int | None = None,
    ordering: Literal['nest', 'ring'] = 'ring',
    identity_noise: bool = False,
    max_buckets: int = 1,
    nperseg: int = 512,
) -> MapMakingConfig:
    if landscape_type == 'healpix':
        lc = LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16, ordering=ordering))
    else:
        lc = LandscapeConfig(
            stokes=stokes,
            wcs=WCSConfig(
                projection=ProjectionType.CAR,
                resolution=60.0,
                # centred on the footprint of the test observations (RA 303-310, dec -45 to -35),
                # so that samples land on the patch instead of falling off it
                patch=SkyPatch(center=(306.5, -40.0), width=20.0, height=20.0),
            ),
        )
    return MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True, interpolation=interpolation),
        landscape=lc,
        weighting=WeightingConfig(
            mode=WeightingMode.IDENTITY if identity_noise else WeightingMode.DIAGONAL,
            fitting=NoiseFitConfig(nperseg=nperseg),
        ),
        sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
        pomme_tau=pomme_tau,
        max_buckets=max_buckets,
    )


def test_nested_healpix_gives_the_ring_map():
    observations = _observations('sotodlib')
    ring = MultiObservationMapMaker(
        observations, config=_config('healpix', 'IQU', identity_noise=True)
    ).make_maps()
    nested = MultiObservationMapMaker(
        observations, config=_config('healpix', 'IQU', identity_noise=True, ordering='nest')
    ).make_maps()
    assert np.array_equal(
        hp.reorder(np.asarray(nested.hit_map), n2r=True), np.asarray(ring.hit_map)
    )
    for expected, actual in zip(
        jax.tree.leaves(ring.map), jax.tree.leaves(nested.map), strict=True
    ):
        reordered = hp.reorder(np.asarray(actual), n2r=True)
        np.testing.assert_allclose(reordered, np.asarray(expected), rtol=1e-10, atol=1e-12)


class TestGapTreatmentMapMaker:
    """End-to-end multi-observation ML mapmaking with the correlated-noise gap treatments."""

    def _config(self, treatment: GapTreatment) -> MapMakingConfig:
        cfg = MapMakingConfig(
            weighting=WeightingConfig(
                mode=WeightingMode.TOEPLITZ,
                source=NoiseSource.PRECOMPUTED,  # use the obs 1/f model (finite)
                correlation_length=128,  # < n_samples, keeps the Toeplitz solve small
            ),
            solver=SolverConfig(rtol=1e-6, atol=0, max_steps=30),
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
            pointing=PointingConfig(on_the_fly=True, interpolation='nearest'),
            detector_tile_size=0,  # the default gap filling works on whole observations
        )
        cfg.gaps.treatment = treatment
        return cfg

    @pytest.mark.parametrize('treatment', [GapTreatment.FILL, GapTreatment.NESTED])
    def test_ml_mapmaker_with_gaps(self, treatment):
        """The full solve completes and yields a finite map over gappy observations."""
        observations = [GappyLazyGroundObservation(seed=i) for i in range(2)]
        results = MultiObservationMapMaker(observations, config=self._config(treatment)).make_maps()
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(results.map))
        assert bool(jnp.all(jnp.isfinite(results.icov)))
        assert jnp.all(results.hit_map >= 0)
        assert results.solver_stats is not None

    def test_explicit_gaps_of_one_sample_give_the_nested_map(self):
        """With an amplitude per flagged sample, both treatments give the minimum-variance map."""
        observations = [GappyLazyGroundObservation(seed=i) for i in range(2)]
        maps = []
        for treatment, tile_size in [(GapTreatment.NESTED, 0), (GapTreatment.EXPLICIT, 2)]:
            config = self._config(treatment)
            config.solver = SolverConfig(rtol=1e-10, atol=0, max_steps=200)
            config.detector_tile_size = tile_size
            config.gaps.nested = NestedConfig(inner_steps=200)
            config.gaps.explicit = ExplicitGapsConfig(resolution=0.0)
            config.gaps.max_flagged_fraction = 0.5
            results = MultiObservationMapMaker(observations, config=config).make_maps()
            assert results.observations is not None
            assert results.observations['over_gap_budget'].to_list() == [0, 0]
            maps.append(results.map.data)
        atol = 1e-8 * float(jnp.abs(maps[0]).max())
        assert_allclose(maps[1], maps[0], rtol=1e-8, atol=atol)

    def _explicit_config(self, probe: bool) -> MapMakingConfig:
        config = self._config(GapTreatment.EXPLICIT)
        config.detector_tile_size = 2
        config.gaps.explicit = ExplicitGapsConfig(resolution=0.04, probe=probe)  # 4 samples
        config.gaps.max_flagged_fraction = 0.5
        return config

    def test_probed_gap_budget_gives_the_same_map(self):
        """The probed budget only sizes the amplitudes: the system, and the map, are the same."""
        observations = [GapProbingLazyGroundObservation(seed=i) for i in range(2)]
        makers = [
            MultiObservationMapMaker(observations, config=self._explicit_config(probe))
            for probe in (False, True)
        ]
        splines = [[g for g in maker._gap_splines if g is not None] for maker in makers]
        assert all(g.resolution == 4 for g in splines[0] + splines[1])  # 0.04 s at 100 Hz
        generic, probed = ([g.budget for g in gaps] for gaps in splines)
        assert all(p < g for p, g in zip(probed, generic, strict=True))
        maps = [maker.make_maps() for maker in makers]
        assert maps[1].observations is not None
        assert maps[1].observations['over_gap_budget'].to_list() == [0, 0]
        assert eqx.tree_equal(maps[1].map, maps[0].map, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize('treatment', [GapTreatment.INNER_MASK, GapTreatment.EXPLICIT])
    def test_diagnostics_report_the_residual_chi2(self, treatment):
        observations = [GapProbingLazyGroundObservation(seed=i) for i in range(2)]
        if treatment == GapTreatment.EXPLICIT:
            config = self._explicit_config(probe=True)
        else:
            config = self._config(treatment)
        config.diagnostics = DiagnosticsConfig(bands=(1.0,))
        results = MultiObservationMapMaker(observations, config=config).make_maps()
        assert results.observations is not None and results.detectors is not None
        chi2 = results.observations['chi2'].to_numpy()
        assert np.all(np.isfinite(chi2)) and np.all(chi2 > 0)
        kept = results.detectors.filter(~pl.col('dropped') & ~pl.col('over_gap_budget'))
        for column in ('chi2', 'chi2_0_1hz', 'chi2_1_nyquisthz'):
            assert kept[column].is_finite().all()
        assert results.residual_map is not None
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(results.residual_map))

    def test_diagnostics_turn_on_and_off_independently(self):
        observations = [GapProbingLazyGroundObservation(seed=i) for i in range(2)]
        config = self._config(GapTreatment.INNER_MASK)
        config.diagnostics = DiagnosticsConfig(chi2=False, bands=(), residual_map=True)
        results = MultiObservationMapMaker(observations, config=config).make_maps()
        assert results.residual_map is not None
        assert results.observations is not None and results.detectors is not None
        assert results.observations['chi2'].is_null().all()
        assert not any(c.startswith('chi2') for c in results.detectors.columns)

    def test_explicit_gaps_need_the_probed_sample_rate(self):
        class RatelessObservation(GapProbingLazyGroundObservation):
            def probe(self, *args, **kwargs):
                return super().probe(*args, **kwargs)._replace(sample_rate=None)

        config = self._explicit_config(probe=False)
        maker = MultiObservationMapMaker([RatelessObservation(seed=0)], config=config)
        with pytest.raises(ValueError, match='sample rate'):
            _ = maker._gap_splines

    def test_detectors_over_the_gap_budget_are_dropped(self):
        """An underestimating probe drops the detectors it does not fit, and counts them."""
        observations = [
            GapProbingLazyGroundObservation(seed=0, claimed=0),
            GapProbingLazyGroundObservation(gappy=False, seed=1),
        ]
        config = self._explicit_config(probe=True)
        results = MultiObservationMapMaker(observations, config=config).make_maps()
        assert results.observations is not None and results.detectors is not None
        n_gappy = results.observations['n_detectors'][0]
        assert results.observations['over_gap_budget'].to_list() == [n_gappy, 0]
        assert results.detectors['over_gap_budget'].sum() == n_gappy

        expected = MultiObservationMapMaker(observations[1:], config=config).make_maps()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)

    def test_detectors_flagged_above_the_threshold_are_dropped(self):
        """An observation whose detectors are all flagged above the threshold contributes nothing."""
        config = self._config(GapTreatment.NESTED)
        config.gaps.max_flagged_fraction = 0.1  # the gappy observation flags 15% of its samples
        mixed = [GappyLazyGroundObservation(seed=0), FakeLazyGroundObservation(seed=1)]
        results = MultiObservationMapMaker(mixed, config=config).make_maps()
        assert results.observations is not None and results.detectors is not None
        n_gappy = results.observations['n_detectors'][0]
        assert results.observations['dropped_detectors'].to_list() == [n_gappy, 0]
        assert results.detectors['dropped'].sum() == n_gappy

        expected = MultiObservationMapMaker([mixed[1]], config=config).make_maps()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)

    def test_gap_fill_skips_failed_observation(self):
        """The ``real & valid`` gate: a failed load is read as filler and never gap-filled."""
        config = self._config(GapTreatment.FILL)
        mixed = [GappyLazyGroundObservation(seed=0), FailingLazyObservation(seed=1)]
        results = MultiObservationMapMaker(mixed, config=config).make_maps()
        assert results.failed_observations == ['failing_obs']
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(results.map))

        good_only = [GappyLazyGroundObservation(seed=0)]
        expected = MultiObservationMapMaker(good_only, config=config).make_maps()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)
