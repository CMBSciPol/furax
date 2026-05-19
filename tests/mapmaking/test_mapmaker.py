import importlib.util
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from furax.core import CompositionOperator
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
    NoiseConfig,
    NoiseFitConfig,
    PointingConfig,
    SkyPatch,
    SotodlibConfig,
    WCSConfig,
)
from furax.mapmaking.mapmaker import get_obs_distribution_to_process
from furax.mapmaking.noise import WhiteNoiseModel
from furax.obs.landscapes import ProjectionType
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import Stokes, ValidStokesType


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
    """Test the multi-observation mapmaker.

    Use a class in order to parametrize over multiple tests at once.
    """

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

    def test_last_acquisition_operand_is_pointing(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        h = maker.build_model().H
        assert isinstance(h, CompositionOperator)
        assert isinstance(h.operands[-1], PointingOperator)

    @pytest.mark.parametrize('fit_models', [True, False])
    def test_white_noise_models_binned_or_demodulated(
        self, name, demodulated, stokes, landscape_type, fit_models
    ):
        observations = _observations(name, demodulated)
        config = _config(
            landscape_type, stokes, demodulated=demodulated, fit_noise_model=fit_models
        )
        maker = MultiObservationMapMaker(observations, config=config)
        noise_model = maker.build_model().noise_model
        if demodulated:
            # In demodulated case each block has a Stokes pytree of per-component WhiteNoiseModel's
            assert isinstance(noise_model, Stokes.class_for(stokes))
            assert all(
                isinstance(getattr(noise_model, stoke.lower()), WhiteNoiseModel) for stoke in stokes
            )
        else:
            assert isinstance(noise_model, WhiteNoiseModel)

    def test_rhs_shape(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        model = maker.distribute(maker.build_model())
        indices = maker.distribute(maker.get_padded_read_indices())
        reader = maker.get_reader(['metadata', 'sample_data'])
        rhs = maker.accumulate_rhs(model, indices, reader)
        assert rhs.shape == maker.landscape.shape

    def test_hits_are_nonnegative(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        blocks = maker.distribute(maker.build_model())
        hits = maker.accumulate_hits(blocks)
        assert hits.shape == maker.landscape.shape
        assert jnp.all(hits >= 0)

    def test_full_mapmaker(self, name, demodulated, stokes, landscape_type):
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated)
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
        assert results.solver_stats is not None
        num_steps = results.solver_stats['num_steps']
        assert num_steps == 1, (
            f'Expected CG to converge in 1 iteration (binned map), got {num_steps}'
        )

    def test_bilinear_pointing_is_interpolated(self, name, demodulated, stokes, landscape_type):
        """PointingConfig(interpolation='bilinear') sets interpolate=True on the PointingOperator."""
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated, interpolation='bilinear')
        maker = MultiObservationMapMaker(observations, config=config)
        h = maker.build_model().H
        assert isinstance(h, CompositionOperator)
        assert isinstance(h.operands[-1], PointingOperator)
        assert h.operands[-1].interpolate

    def test_bilinear_mapmaker_runs(self, name, demodulated, stokes, landscape_type):
        """Mapmaker runs end-to-end with bilinear interpolation."""
        observations = _observations(name, demodulated)
        config = _config(landscape_type, stokes, demodulated, interpolation='bilinear')
        maker = MultiObservationMapMaker(observations, config=config)
        results = maker.run()
        n_stokes = len(stokes)
        assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)


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
        from furax.interfaces.sotodlib.observation import LazySOTODLibObservation

        sotodlib_config = SotodlibConfig(demodulated=True) if demodulated else None
        files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
        return [LazySOTODLibObservation(f, sotodlib_config=sotodlib_config) for f in files]
    raise NotImplementedError


def _config(
    landscape_type: Literal['healpix', 'car'],
    stokes: ValidStokesType,
    demodulated: bool = False,
    fit_noise_model: bool = True,
    interpolation: Literal['nearest', 'bilinear'] = 'nearest',
    method: Methods = Methods.BINNED,
    atop_tau: int = 0,
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
        noise=NoiseConfig(fit_from_data=fit_noise_model, fitting=NoiseFitConfig(nperseg=512)),
        sotodlib=SotodlibConfig(demodulated=True) if demodulated else None,
        atop_tau=atop_tau,
    )
