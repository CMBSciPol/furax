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
from furax.mapmaking.mapmaker import MLMapmaker, get_obs_distribution_to_process
from furax.mapmaking.noise import WhiteNoiseModel
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import Stokes, ValidStokesType
from tests.mapmaking.helpers import (
    FailingLazyObservation,
    FakeGroundObservation,
    FakeLazyObservation,
)


class TestObsDistribution:
    def test_single_process_divisible(self):
        # 8 obs, 1 proc, 4 local devs → no padding needed
        start, n_owned, n_pad = get_obs_distribution_to_process(8, rank=0, n_proc=1, n_local=4)
        assert (start, n_owned, n_pad) == (0, 8, 0)

    def test_single_process_needs_padding(self):
        # 10 obs, 1 proc, 4 local devs → pad to 12
        start, n_owned, n_pad = get_obs_distribution_to_process(10, rank=0, n_proc=1, n_local=4)
        assert (start, n_owned, n_pad) == (0, 10, 2)

    def test_multi_process_even_distribution(self):
        # 10 obs, 2 procs, 2 local devs → 5 obs each, chunk=6 (pad to multiple of 2)
        assert get_obs_distribution_to_process(10, rank=0, n_proc=2, n_local=2) == (0, 5, 1)
        assert get_obs_distribution_to_process(10, rank=1, n_proc=2, n_local=2) == (5, 5, 1)

    def test_multi_process_uneven_distribution(self):
        # 11 obs, 2 procs, 2 local devs → proc 0 gets 6, proc 1 gets 5
        # chunk = ceil(11/2)=6, padded to multiple of 2 → 6
        assert get_obs_distribution_to_process(11, rank=0, n_proc=2, n_local=2) == (0, 6, 0)
        assert get_obs_distribution_to_process(11, rank=1, n_proc=2, n_local=2) == (6, 5, 1)

    def test_no_process_left_empty(self):
        # 85 obs, 20 procs, 1 local dev → base=4, remainder=5
        # first 5 procs own 5 obs, rest own 4; no proc gets 0
        for rank in range(20):
            _, n_owned, _ = get_obs_distribution_to_process(85, rank=rank, n_proc=20, n_local=1)
            assert n_owned > 0

    def test_fewer_obs_than_procs_raises(self):
        # n_obs < n_proc is always an error regardless of rank
        with pytest.raises(ValueError, match='Not enough observations'):
            get_obs_distribution_to_process(3, rank=3, n_proc=4, n_local=1)

    def test_chunk_always_divisible_by_local_devices(self):
        # n_owned + n_pad must be divisible by n_local_dev
        n_obs, n_local_dev, n_procs = 7, 2, 3
        for rank in range(n_procs):
            _, n_owned, n_pad = get_obs_distribution_to_process(
                n_obs, rank=rank, n_proc=n_procs, n_local=n_local_dev
            )
            assert (n_owned + n_pad) % n_local_dev == 0

    def test_all_procs_cover_all_obs(self):
        # All processes together must cover every real observation exactly once
        n_obs, n_local_dev, n_procs = 10, 2, 2
        starts = []
        total_real = 0
        for rank in range(n_procs):
            start, n_owned, _ = get_obs_distribution_to_process(
                n_obs, rank=rank, n_proc=n_procs, n_local=n_local_dev
            )
            starts.append(start)
            total_real += n_owned
        assert starts == [0, 5]
        assert total_real == n_obs


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
            model, _, _ = maker.build_model_and_accumulate()
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
            model, _, _ = maker.build_model_and_accumulate()
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
            noise_model = maker.build_model_and_accumulate()[0].noise_model
        if demodulated:
            assert isinstance(noise_model, Stokes.class_for(stokes))
            assert all(
                isinstance(getattr(noise_model, stoke.lower()), WhiteNoiseModel) for stoke in stokes
            )
        else:
            assert isinstance(noise_model, WhiteNoiseModel)

    def test_identity_builds_unit_white_noise(self, demodulated):
        observations = self._observations()
        config = _config('healpix', 'IQU', demodulated, identity_noise=True)
        maker = MultiObservationMapMaker(observations, config=config)
        with jax.set_mesh(maker.mesh):
            model, _, _ = maker.build_model_and_accumulate()
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

    def _base_config(self, stokes: ValidStokesType) -> MapMakingConfig:
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
    stokes: ValidStokesType,
    demodulated: bool = False,
    interpolation: Literal['nearest', 'bilinear'] = 'nearest',
    method: Methods = Methods.BINNED,
    atop_tau: int = 0,
    identity_noise: bool = False,
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
    )


class TestSingleObsTemplates:
    """End-to-end coverage of the single-observation template path (MapMaker.make_map)."""

    def test_ml_mapmaker_runs_with_templates(self) -> None:
        # Regression: the template preconditioner build jits the template system operator;
        # a composite furax operator carries unhashable leaves, so it must be wrapped in a
        # lambda (mapmaker.py). Also locks the structured (n_dets, *shape) amplitude layout.
        n_dets = 4
        obs = FakeGroundObservation(n_dets=n_dets, n_samples=1024, sample_rate=100.0)

        cfg = MapMakingConfig.for_method('ml')
        cfg.weighting.source = NoiseSource.PRECOMPUTED  # use the obs noise model (finite)
        cfg.landscape = LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8))
        cfg.templates = TemplatesConfig.full_defaults()
        cfg.templates.ground = None  # ground template needs pointing/landscape, out of scope here
        cfg.templates.spline_hwpss.samples_per_knot = 256  # 1024 // 256 -> n_knots=4

        res = MLMapmaker(config=cfg).make_map(obs)

        assert bool(jnp.all(jnp.isfinite(res['map'])))
        # amplitudes are structured (n_dets, *basis_shape), not flat
        assert res['template_polynomial'].shape == (n_dets, 2, 4)  # intervals x (orders 0..3)
        assert res['template_azhwp_synchronous'].shape == (n_dets, 4, 9)  # orders x (DC+2*4)
        assert res['template_spline_hwpss'].shape == (n_dets, 6, 2)  # (n_knots + 2) x cos/sin
        for key, value in res.items():
            if key.startswith('template_') and not key.startswith('template_reg'):
                assert bool(jnp.all(jnp.isfinite(value))), key
