import importlib.util
from pathlib import Path
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from furax.core._base import structure_equal
from furax.mapmaking import (
    AbstractLazyObservation,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    GapTreatment,
    HealpixConfig,
    LandscapeConfig,
    Methods,
    NoiseFitConfig,
    NoiseSource,
    PointingConfig,
    SkyPatch,
    SotodlibConfig,
    TemplatesConfig,
    WCSConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import (
    ATOPMapMaker,
    BinnedMapMaker,
    MapMaker,
    MLMapmaker,
)
from furax.mapmaking.noise import WhiteNoiseModel
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import ValidStokesLiteral
from tests.mapmaking.helpers import (
    FailingLazyObservation,
    FakeGroundObservation,
    FakeLazyObservation,
    GappyLazyGroundObservation,
)

# Skip tests for interfaces that are not installed
sotodlib_installed = importlib.util.find_spec('sotodlib') is not None
toast_installed = importlib.util.find_spec('toast') is not None

# Parameters for all the tests below.
# Tests are parametrized over:
#   - PARAMS: observation interface (sotodlib, toast) and demodulation flag
#   - STOKES_TYPES: Stokes components ('I', 'QU', 'IQU')
#   - LANDSCAPE_TYPES: output map projection (healpix, CAR)
# Add more entries to any of these lists to extend coverage.

PARAMS = [
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
LANDSCAPE_TYPES = ['healpix', 'car']


@pytest.mark.parametrize('landscape_type', LANDSCAPE_TYPES)
@pytest.mark.parametrize('stokes', STOKES_TYPES)
@pytest.mark.parametrize('name,demodulated', PARAMS)
class TestMultiObsMapMaker:
    def test_model_vs_reader_structure(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        reader = ObservationReader.from_observations(
            observations, demodulated=demodulated, stokes=stokes
        )
        with jax.set_mesh(maker.mesh):
            model = maker.build_model_and_accumulate().buckets[0].model
        n_obs = jax.tree.leaves(model)[0].shape[0]
        assert n_obs == len(observations) == reader.count
        # structures compared ignoring sharding (the model is built sharded inside shard_map)
        assert structure_equal(model.map_structure, maker.landscape.structure)
        assert structure_equal(model.tod_structure, reader.out_structure['sample_data'])

    def test_full_mapmaker(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.hit_map.shape == maker.landscape.shape
        assert jnp.all(results.hit_map >= 0)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
        assert results.solver_stats is not None
        num_steps = results.solver_stats['num_steps']
        assert num_steps == 1, (
            f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
        )

    def test_bilinear_mapmaker_runs(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated, interpolation='bilinear')
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)


@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
@pytest.mark.parametrize('stokes', STOKES_TYPES)
class TestFakeObsMapMaker:
    """Interface-agnostic pipeline coverage backed by the synthetic observation.

    Unlike the classes above, these are *not* gated on sotodlib/toast being
    installed and need no committed ``.h5`` fixtures: they exercise the binned
    mapmaker end-to-end (both the modulated and demodulated paths) in a minimal
    install (``[dev,mapmaking]`` only), where every interface-parametrized test
    is skipped. The fake observation only supports the on-the-fly + healpix
    path, so coverage is limited to that.
    """

    def test_model_vs_reader_structure(self, stokes, demodulated):
        observations = [FakeLazyObservation()]
        config = _config('healpix', stokes, demodulated=demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        reader = ObservationReader.from_observations(
            observations, demodulated=demodulated, stokes=stokes
        )
        with jax.set_mesh(maker.mesh):
            model = maker.build_model_and_accumulate().buckets[0].model
        n_obs = jax.tree.leaves(model)[0].shape[0]
        assert n_obs == len(observations) == reader.count
        # structures compared ignoring sharding (the model is built sharded inside shard_map)
        assert structure_equal(model.map_structure, maker.landscape.structure)
        assert structure_equal(model.tod_structure, reader.out_structure['sample_data'])

    def test_full_binned_mapmaker_multi_obs(self, stokes, demodulated):
        # Two observations (distinct noise seeds) so the multi-observation
        # accumulation path is exercised without any interface or data file.
        observations = [FakeLazyObservation(seed=i) for i in range(2)]
        config = _config('healpix', stokes, demodulated=demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.hit_map.shape == maker.landscape.shape
        assert jnp.all(results.hit_map >= 0)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
        assert results.solver_stats is not None
        num_steps = results.solver_stats['num_steps']
        assert num_steps == 1, (
            f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
        )

    def test_failed_observation_excluded(self, stokes, demodulated):
        # One good observation + one whose load fails: the run must complete, exclude the failed
        # observation (contributing nothing, no NaN), and report it.
        config = _config('healpix', stokes, demodulated=demodulated)
        mixed = [FakeLazyObservation(seed=0), FailingLazyObservation(seed=1)]
        results = MultiObservationMapMaker(mixed, config=config).run()

        assert results.failed_observations == ['failing_obs']
        assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(results.map))
        assert jnp.all(jnp.isfinite(results.icov))

        # the failed observation contributes nothing -> identical to mapping only the good one
        good_only = [FakeLazyObservation(seed=0)]
        expected = MultiObservationMapMaker(good_only, config=config).run()
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

    @pytest.mark.parametrize('max_buckets', [1, 2, 3])
    def test_layout_follows_config(self, max_buckets):
        config = _config('healpix', 'IQU', max_buckets=max_buckets)
        maker = MultiObservationMapMaker(self._observations(), config=config)
        layout = maker.layout
        assert 1 <= len(layout.buckets) <= max_buckets
        covered = sorted(int(i) for b in layout.buckets for i in b.observations)
        assert covered == list(range(len(self.N_SAMPLES)))
        for bucket, reader in zip(layout.buckets, maker.readers, strict=True):
            assert reader.count == bucket.n_real
            assert bucket.n_slots % jax.device_count() == 0
            # each reader pads to its own bucket's envelope, not the dataset's largest
            n_samples = reader.out_structure['sample_data'].shape[-1]
            assert n_samples == bucket.shape.sample_count
        if max_buckets == 3 and jax.device_count() == 1:
            assert len(layout.buckets) == 3  # singletons never pad on one device

    @pytest.mark.parametrize('stokes', ['I', 'IQU'])
    def test_maps_do_not_depend_on_bucketing(self, stokes):
        # Unit weights: a fitted noise level would see the padded samples and so depend on the
        # bucket's envelope, which is not what is under test here.
        observations = self._observations()
        one = _config('healpix', stokes, identity_noise=True)
        many = _config('healpix', stokes, identity_noise=True, max_buckets=3)
        one_result = MultiObservationMapMaker(observations, config=one).run()
        many_result = MultiObservationMapMaker(observations, config=many).run()
        assert eqx.tree_equal(one_result.hit_map, many_result.hit_map)
        assert eqx.tree_equal(one_result.map, many_result.map, rtol=1e-10, atol=1e-12)
        assert eqx.tree_equal(one_result.icov, many_result.icov, rtol=1e-10, atol=1e-12)

    def test_failed_observation_in_a_bucket(self):
        # the failing observation is the short one, alone in its bucket once split
        config = _config('healpix', 'IQU', max_buckets=3)
        observations = self._observations()
        observations[0] = FailingLazyObservation(seed=0, n_samples=self.N_SAMPLES[0])
        results = MultiObservationMapMaker(observations, config=config).run()
        assert results.failed_observations == ['failing_obs']
        expected = MultiObservationMapMaker(observations[1:], config=config).run()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('demodulated', [False, True], ids=['modulated', 'demodulated'])
class TestNoiseModelSelection:
    """Noise-model *selection* logic. This is generic mapmaker behaviour that
    does not depend on the specific interface or sample values, so it is backed
    by the synthetic observation -- no sotodlib/toast install or ``.h5`` fixture
    required.
    """

    def _observations(self):
        return [FakeLazyObservation(seed=i) for i in range(2)]

    def test_white_noise_models_binned_or_demodulated(self, demodulated):
        stokes = 'IQU'
        observations = self._observations()
        config = _config('healpix', stokes, demodulated=demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        with jax.set_mesh(maker.mesh):
            noise_model = maker.build_model_and_accumulate().buckets[0].model.noise_model
        # A single WhiteNoiseModel covers both paths. The demodulated TOD is a single-array Stokes,
        # so its per-detector sigma carries the leading Stokes axis (here after the observation-stack
        # axis added by the accumulation scan).
        assert isinstance(noise_model, WhiteNoiseModel)
        if demodulated:
            assert noise_model.sigma.shape[1] == len(stokes)

    def test_identity_builds_unit_white_noise(self, demodulated):
        observations = self._observations()
        config = _config('healpix', 'IQU', demodulated, identity_noise=True)
        maker = MultiObservationMapMaker(observations, config=config)
        with jax.set_mesh(maker.mesh):
            model = maker.build_model_and_accumulate().buckets[0].model
        noise_leaves = jax.tree.leaves(
            model.noise_model,
            is_leaf=lambda x: isinstance(x, WhiteNoiseModel),
        )
        assert noise_leaves
        for nm in noise_leaves:
            assert isinstance(nm, WhiteNoiseModel)
            assert jnp.allclose(nm.sigma, 1.0)

    @pytest.mark.parametrize('method', [Methods.BINNED, Methods.MAXL])
    def test_identity_full_mapmaker(self, demodulated, method):
        observations = self._observations()
        config = _config('healpix', 'IQU', demodulated, method=method, identity_noise=True)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        assert results.icov.shape == (3, 3, *maker.landscape.shape)
        assert results.solver_stats is not None


ATOP_PARAMS = [
    pytest.param(
        'sotodlib',
        id='sotodlib',
        marks=pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed'),
    ),
    pytest.param(
        'toast',
        id='toast',
        marks=pytest.mark.skipif(not toast_installed, reason='toast is not installed'),
    ),
]
ATOP_TAU = 10


@pytest.mark.parametrize('landscape_type', LANDSCAPE_TYPES)
@pytest.mark.parametrize('name', ATOP_PARAMS)
class TestATOPMapMaker:
    """Test ATOP support in MultiObservationMapMaker."""

    def test_atop_full_mapmaker(self, name, landscape_type):
        """ATOP runs end-to-end and produces a QU map with the correct shape."""
        observations = _observations(name)
        config = _config(landscape_type, stokes='QU', method=Methods.ATOP, atop_tau=ATOP_TAU)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        assert results.icov.shape == (2, 2, *maker.landscape.shape)


class TestATOPStokesValidation:
    """ATOP Stokes-config normalisation/validation. Pure construction-time logic
    that never reads sample data, so it is backed by the synthetic observation
    (or an empty list) rather than an interface or ``.h5`` fixture.
    """

    def _base_config(self, stokes: ValidStokesLiteral) -> MapMakingConfig:
        return MapMakingConfig(
            method=Methods.ATOP,
            atop_tau=ATOP_TAU,
            landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16)),
        )

    def test_iqu_stokes_falls_back_to_qu(self):
        """stokes='IQU' with ATOP is converted to 'QU'."""
        maker = MultiObservationMapMaker([FakeLazyObservation()], config=self._base_config('IQU'))
        assert maker.config.landscape.stokes == 'QU'

    def test_i_stokes_raises(self):
        with pytest.raises(ValueError, match='cannot be reduced to a supported type'):
            MultiObservationMapMaker([], config=self._base_config('I'))

    def test_iquv_stokes_raises(self):
        with pytest.raises(ValueError, match='cannot be reduced to a supported type'):
            MultiObservationMapMaker([], config=self._base_config('IQUV'))

    def test_atop_with_templates_raises(self):
        config = self._base_config('QU')
        config.templates = TemplatesConfig.full_defaults()
        with pytest.raises(NotImplementedError, match='ATOP combined with templates'):
            MultiObservationMapMaker([], config=config)


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
    method: Methods = Methods.BINNED,
    atop_tau: int = 0,
    identity_noise: bool = False,
    max_buckets: int = 1,
) -> MapMakingConfig:
    if landscape_type == 'healpix':
        lc = LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16))
    else:
        lc = LandscapeConfig(
            stokes=stokes,
            wcs=WCSConfig(
                projection=ProjectionType.CAR,
                resolution=60.0,
                patch=SkyPatch(center=(0.0, 0.0), width=20.0, height=20.0),
            ),
        )
    return MapMakingConfig(
        method=method,
        pointing=PointingConfig(on_the_fly=True, interpolation=interpolation),
        landscape=lc,
        weighting=WeightingConfig(
            mode=WeightingMode.IDENTITY if identity_noise else WeightingMode.DIAGONAL,
            fitting=NoiseFitConfig(nperseg=512),
        ),
        sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
        atop_tau=atop_tau,
        max_buckets=max_buckets,
    )


class TestSingleObsSolverGuards:
    """Solver/weighting compatibility on the single-observation ``MapMaker`` path.

    The direct binned solvers invert only the per-pixel (block-Jacobi) system, which is exact for
    nearest-neighbour pointing but drops the off-diagonal pixel coupling that bilinear interpolation
    introduces -- so bilinear pointing is restricted to the iterative ML solver. The ML solver in
    turn accepts any weighting mode (identity / diagonal / Toeplitz), not just Toeplitz.
    """

    def _config(
        self,
        mode: WeightingMode,
        interpolation: Literal['nearest', 'bilinear'] = 'nearest',
        method: Methods = Methods.MAXL,
    ) -> MapMakingConfig:
        return MapMakingConfig(
            method=method,
            pointing=PointingConfig(on_the_fly=True, interpolation=interpolation),
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
            # PRECOMPUTED sidesteps fitting a model to the synthetic white TODs (yields NaN).
            weighting=WeightingConfig(mode=mode, source=NoiseSource.PRECOMPUTED),
        )

    @pytest.mark.parametrize('mode', [WeightingMode.IDENTITY, WeightingMode.DIAGONAL])
    def test_ml_mapmaker_accepts_diagonal_weighting(self, mode: WeightingMode) -> None:
        """ML runs with identity/diagonal weighting (previously rejected via ``binned=True``)."""
        obs = FakeGroundObservation(n_dets=4, n_samples=1024, sample_rate=100.0)
        res = MLMapmaker(config=self._config(mode)).make_map(obs)
        assert bool(jnp.all(jnp.isfinite(res['map'])))

    def test_ml_mapmaker_accepts_bilinear_pointing(self) -> None:
        """Bilinear pointing runs under the ML solver."""
        obs = FakeGroundObservation(n_dets=4, n_samples=1024, sample_rate=100.0)
        cfg = self._config(WeightingMode.IDENTITY, interpolation='bilinear')
        res = MLMapmaker(config=cfg).make_map(obs)
        assert bool(jnp.all(jnp.isfinite(res['map'])))

    @pytest.mark.parametrize('maker_cls', [BinnedMapMaker, ATOPMapMaker])
    def test_direct_solvers_reject_bilinear_pointing(self, maker_cls: type[MapMaker]) -> None:
        """The direct binned solvers refuse bilinear pointing at construction time."""
        cfg = self._config(WeightingMode.DIAGONAL, interpolation='bilinear', method=Methods.BINNED)
        with pytest.raises(ValueError, match='does not support bilinear pointing'):
            maker_cls(config=cfg)


class TestGapTreatmentMapMaker:
    """End-to-end multi-observation ML mapmaking with the correlated-noise gap treatments."""

    def _config(self, treatment: GapTreatment) -> MapMakingConfig:
        cfg = MapMakingConfig.for_method('ml')
        cfg.weighting.source = NoiseSource.PRECOMPUTED  # use the obs 1/f model (finite)
        cfg.weighting.correlation_length = 128  # < n_samples, keeps the Toeplitz solve small
        cfg.landscape = LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8))
        cfg.pointing = PointingConfig(on_the_fly=True, interpolation='nearest')
        cfg.solver.max_steps = 30
        cfg.gaps.treatment = treatment
        return cfg

    @pytest.mark.parametrize('treatment', [GapTreatment.FILL, GapTreatment.NESTED])
    def test_ml_mapmaker_with_gaps(self, treatment):
        """The full solve completes and yields a finite map over gappy observations."""
        observations = [GappyLazyGroundObservation(seed=i) for i in range(2)]
        results = MultiObservationMapMaker(observations, config=self._config(treatment)).run()
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(results.map))
        assert bool(jnp.all(jnp.isfinite(results.icov)))
        assert jnp.all(results.hit_map >= 0)
        assert results.solver_stats is not None

    def test_gap_fill_skips_failed_observation(self):
        """The ``real & valid`` gate: a failed load is read as filler and never gap-filled."""
        config = self._config(GapTreatment.FILL)
        mixed = [GappyLazyGroundObservation(seed=0), FailingLazyObservation(seed=1)]
        results = MultiObservationMapMaker(mixed, config=config).run()
        assert results.failed_observations == ['failing_obs']
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(results.map))

        good_only = [GappyLazyGroundObservation(seed=0)]
        expected = MultiObservationMapMaker(good_only, config=config).run()
        assert eqx.tree_equal(results.hit_map, expected.hit_map)
        assert eqx.tree_equal(results.map, expected.map, rtol=1e-6, atol=1e-6)
