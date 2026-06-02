import importlib.util
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from furax.mapmaking import (
    AbstractLazyObservation,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    AzimuthHWPSynchronousConfig,
    BinAzimuthHWPSynchronousConfig,
    BinAzSynchronousConfig,
    BinsConfig,
    HealpixConfig,
    HWPSynchronousConfig,
    LandscapeConfig,
    LegendreOrders,
    Methods,
    NoiseFitConfig,
    PointingConfig,
    PolyConfig,
    ScanSynchronousConfig,
    SkyPatch,
    SotodlibConfig,
    TemplatesConfig,
    WCSConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.mapmaker import get_obs_distribution_to_process
from furax.mapmaking.noise import WhiteNoiseModel
from furax.obs.landscapes import ProjectionType
from furax.obs.stokes import Stokes, StokesIQU, ValidStokesType


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
        model = maker.build_model()
        n_obs = jax.tree.leaves(model)[0].shape[0]
        assert n_obs == len(observations) == reader.count
        assert model.map_structure == maker.landscape.structure
        assert model.tod_structure == reader.out_structure['sample_data']

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


@pytest.mark.parametrize('name,demodulated', PARAMS)
class TestNoiseModelSelection:
    def test_white_noise_models_binned_or_demodulated(self, name, demodulated):
        stokes = 'IQU'
        observations = _observations(name, demodulated)
        config = _config('healpix', stokes, demodulated=demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        noise_model = maker.build_model().noise_model
        if demodulated:
            assert isinstance(noise_model, Stokes.class_for(stokes))
            assert all(
                isinstance(getattr(noise_model, stoke.lower()), WhiteNoiseModel) for stoke in stokes
            )
        else:
            assert isinstance(noise_model, WhiteNoiseModel)

    def test_identity_builds_unit_white_noise(self, name, demodulated):
        observations = _observations(name, demodulated)
        config = _config('healpix', 'IQU', demodulated, identity_noise=True)
        maker = MultiObservationMapMaker(observations, config=config)
        noise_leaves = jax.tree.leaves(
            maker.build_model().noise_model, is_leaf=lambda x: isinstance(x, WhiteNoiseModel)
        )
        assert noise_leaves
        for nm in noise_leaves:
            assert isinstance(nm, WhiteNoiseModel)
            assert jnp.allclose(nm.sigma, 1.0)

    @pytest.mark.parametrize('method', [Methods.BINNED, Methods.MAXL])
    def test_identity_full_mapmaker(self, name, demodulated, method):
        observations = _observations(name, demodulated)
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

    def test_atop_iqu_stokes_falls_back_to_qu(self, name, landscape_type):
        """stokes='IQU' with ATOP is converted to 'QU'."""
        observations = _observations(name)
        config = _config(landscape_type, stokes='IQU', method=Methods.ATOP, atop_tau=ATOP_TAU)
        maker = MultiObservationMapMaker(observations, config=config)
        assert maker.config.landscape.stokes == 'QU'


TWOSTEP_PARAMS = ATOP_PARAMS

# Template families exercised by the two-step mapmaker, with the amplitude block
# key(s) each one produces. ``hwp_synchronous`` needs only ``hwp_angles``; the
# others also pull ``azimuth`` (and scan masks when splitting) from the reader.
# Templates are marked ``explicit=True`` so their amplitudes are solved in the two-step CG
# and returned (the default ``explicit=False`` would marginalise them into the weight, leaving
# no amplitudes to assert on — that path is covered by the marginalisation tests below).
TEMPLATE_PARAMS = [
    pytest.param(
        TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(n_harmonics=2, explicit=True)),
        ['hwp_synchronous'],
        id='hwp',
    ),
    pytest.param(
        TemplatesConfig(polynomial=PolyConfig(legendre=LegendreOrders(0, 3), explicit=True)),
        ['polynomial'],
        id='poly',
    ),
    pytest.param(
        TemplatesConfig(
            scan_synchronous=ScanSynchronousConfig(legendre=LegendreOrders(0, 3), explicit=True)
        ),
        ['scan_synchronous'],
        id='scan',
    ),
    pytest.param(
        TemplatesConfig(
            binaz_synchronous=BinAzSynchronousConfig(bins=BinsConfig(n_bins=4), explicit=True)
        ),
        ['binaz_synchronous'],
        id='binaz',
    ),
    pytest.param(
        TemplatesConfig(
            azhwp_synchronous=AzimuthHWPSynchronousConfig(
                legendre=LegendreOrders(0, 2), n_harmonics=2, explicit=True
            )
        ),
        ['azhwp_synchronous'],
        id='azhwp',
    ),
    # NOTE: ``split_scans=True`` (blocks azhwp_synchronous_left/right) is wired but not
    # exercised here — the bundled sotodlib fixtures carry no left/right scan flags.
    pytest.param(
        TemplatesConfig(
            binazhwp_synchronous=BinAzimuthHWPSynchronousConfig(
                bins=BinsConfig(n_bins=4), n_harmonics=2, explicit=True
            )
        ),
        ['binazhwp_synchronous'],
        id='binazhwp',
    ),
]


# Template families requiring azimuth — directly (scan/binaz/azhwp), or via
# azimuth-derived scanning intervals (polynomial). The bundled toast fixtures carry
# no azimuth, so only ``hwp_synchronous`` is exercisable on toast.
_NEEDS_AZIMUTH = {
    'scan_synchronous',
    'binaz_synchronous',
    'azhwp_synchronous',
    'binazhwp_synchronous',
    'polynomial',
}


@pytest.mark.parametrize('landscape_type', LANDSCAPE_TYPES)
@pytest.mark.parametrize('stokes', STOKES_TYPES)
@pytest.mark.parametrize('templates, amplitude_keys', TEMPLATE_PARAMS)
@pytest.mark.parametrize('name', TWOSTEP_PARAMS)
def test_twostep_full_mapmaker(name, templates, amplitude_keys, stokes, landscape_type):
    if name == 'toast' and set(amplitude_keys) & _NEEDS_AZIMUTH:
        pytest.xfail('toast fixtures provide no azimuth required by this template family')
    observations = _observations(name)
    config = _config(
        landscape_type,
        stokes=stokes,
        method=Methods.TWOSTEP,
        templates=templates,
    )
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.run()
    n_stokes = len(stokes)
    assert results.hit_map.shape == maker.landscape.shape
    assert jnp.all(results.hit_map >= 0)
    assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
    assert results.solver_stats is not None
    assert 'amplitude' in results.solver_stats
    assert 'map' in results.solver_stats
    assert results.template_amplitudes is not None
    for key in amplitude_keys:
        assert key in results.template_amplitudes


# Demodulated two-step: HWP-free templates act per Stokes leg (StokesIQU amplitudes).
DEMOD_TEMPLATE_PARAMS = [
    pytest.param(
        TemplatesConfig(
            scan_synchronous=ScanSynchronousConfig(legendre=LegendreOrders(3, 7), explicit=True)
        ),
        'scan_synchronous',
        id='scan',
    ),
    pytest.param(
        TemplatesConfig(
            binaz_synchronous=BinAzSynchronousConfig(bins=BinsConfig(n_bins=4), explicit=True)
        ),
        'binaz_synchronous',
        id='binaz',
    ),
    # Per-Stokes polynomial order: degree 3 for I (dsT), degree 1 for Q/U (demodQ/U).
    pytest.param(
        TemplatesConfig(
            polynomial=PolyConfig(
                legendre=LegendreOrders(0, 3), legendre_qu=LegendreOrders(0, 1), explicit=True
            )
        ),
        'polynomial',
        id='poly',
    ),
]


@pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed')
@pytest.mark.parametrize('landscape_type', LANDSCAPE_TYPES)
@pytest.mark.parametrize('templates, amplitude_key', DEMOD_TEMPLATE_PARAMS)
def test_twostep_demodulated_mapmaker(templates, amplitude_key, landscape_type):
    """Two-step on demodulated data: templates act per Stokes leg, yielding StokesIQU
    amplitudes."""
    observations = _observations('sotodlib', demodulated=True)
    config = _config(
        landscape_type,
        stokes='IQU',
        demodulated=True,
        method=Methods.TWOSTEP,
        templates=templates,
    )
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.run()
    assert results.icov.shape == (3, 3, *maker.landscape.shape)
    assert results.template_amplitudes is not None
    amps = results.template_amplitudes[amplitude_key]
    # Per-Stokes amplitudes: one independent block per (I, Q, U) leg.
    assert isinstance(amps, StokesIQU)


@pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed')
def test_marginalization_matches_explicit():
    """Marginalising a template (explicit=False) yields the same map as solving it explicitly
    and discarding the amplitudes — the template-marginalisation equivalence.

    Projection-agnostic, so healpix (nside=16) suffices: the sim footprint covers a handful of
    pixels there. The shared CAR fixture (60' / 20deg patch) catches a single hit pixel, which a
    4-order Legendre template makes degenerate — explicit A⁻¹ and marginalised PCG then resolve
    the null space differently, so the maps need not agree. Equivalence itself is independent of
    the map basis.
    """
    observations = _observations('sotodlib')

    def run_map(explicit: bool):
        templates = TemplatesConfig(
            polynomial=PolyConfig(legendre=LegendreOrders(0, 3), explicit=explicit)
        )
        config = _config('healpix', stokes='IQU', method=Methods.TWOSTEP, templates=templates)
        return MultiObservationMapMaker(observations, config=config).run()

    explicit_res = run_map(True)
    marginal_res = run_map(False)

    # explicit path returns amplitudes; marginalised path folds them into the weight
    assert explicit_res.template_amplitudes is not None
    assert marginal_res.template_amplitudes is None

    # identical map up to the solvers' tolerances (explicit: direct A⁻¹ + amplitude CG;
    # marginalised: PCG on the marginalised system)
    expl = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(explicit_res.map)])
    marg = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(marginal_res.map)])
    rel = jnp.linalg.norm(marg - expl) / jnp.linalg.norm(expl)
    assert float(rel) < 1e-3, f'relative map difference {float(rel):.2e}'


@pytest.mark.skipif(not sotodlib_installed, reason='sotodlib is not installed')
def test_marginalized_only_is_filter_bin():
    """With every template marginalised there are no explicit amplitudes: the amplitude CG is
    skipped (filter+bin) and a map is still produced."""
    observations = _observations('sotodlib')
    templates = TemplatesConfig(
        polynomial=PolyConfig(legendre=LegendreOrders(0, 3))
    )  # explicit=False
    config = _config('healpix', stokes='IQU', method=Methods.TWOSTEP, templates=templates)
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.run()
    assert results.template_amplitudes is None
    assert results.solver_stats['amplitude']['num_steps'] == 0
    assert results.hit_map.shape == maker.landscape.shape  # map produced
    assert jnp.all(
        jnp.isfinite(jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(results.map)]))
    )


class TestTwoStepValidation:
    """Test that invalid configurations raise errors when using TwoStep."""

    def _base(self, **overrides) -> MapMakingConfig:
        kw: dict = dict(
            method=Methods.TWOSTEP,
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=16)),
            templates=TemplatesConfig(hwp_synchronous=HWPSynchronousConfig()),
        )
        kw.update(overrides)
        return MapMakingConfig(**kw)

    def test_requires_binned(self):
        with pytest.raises(ValueError, match='TwoStep requires a white noise model'):
            MultiObservationMapMaker(
                [], config=self._base(weighting=WeightingConfig(mode=WeightingMode.TOEPLITZ))
            )

    def test_rejects_hwp_templates_when_demodulated(self):
        # demodulated two-step is supported, but HWP-coupled templates are not (the HWP
        # signal is demodulated out); this is rejected at config construction.
        with pytest.raises(ValueError, match='HWP-coupled templates'):
            self._base(sotodlib=SotodlibConfig(demodulated=True))

    def test_requires_templates(self):
        with pytest.raises(ValueError, match='TwoStep requires at least one active template'):
            MultiObservationMapMaker([], config=self._base(templates=None))


class TestATOPStokesValidation:
    """Test that invalid Stokes configurations raise errors when using ATOP."""

    def _base_config(self, stokes: ValidStokesType) -> MapMakingConfig:
        return MapMakingConfig(
            method=Methods.ATOP,
            atop_tau=ATOP_TAU,
            landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16)),
        )

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
    templates: TemplatesConfig | None = None,
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
        templates=templates,
    )
