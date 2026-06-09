"""Regression tests for ``MapMakingConfig.double_precision=False``.

Two layers of coverage:

1. **In-process dtype plumbing**: verifies that ``ObservationReader`` honours
   its ``dtype`` parameter and that ``MultiObservationMapMaker.get_reader``
   forwards ``config.dtype``. These run under the normal session-wide
   ``jax_enable_x64=True`` because they only inspect declared structures.

2. **End-to-end run with x64 off**: an ``insubprocess`` test that flips
   ``jax_enable_x64`` off (the configuration a real float32 user would use),
   runs a tiny mapmaker under ``double_precision=False``, and asserts it
   completes with float32 outputs. The ``insubprocess`` marker isolates it in
   its own process so the flag flip neither fights the session-autouse
   ``enable_x64`` fixture in ``tests/conftest.py`` nor leaks into other tests.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Bool, Float

from furax.mapmaking import (
    AbstractLazyObservation,
    AbstractSatelliteObservation,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    HealpixConfig,
    LandscapeConfig,
    MapMakingConfig,
    Methods,
    NoiseFitConfig,
    PointingConfig,
    SolverConfig,
    WeightingConfig,
)
from furax.mapmaking.noise import NoiseModel
from furax.obs.landscapes import ProjectionType, StokesLandscape

N_DETS = 1
N_SAMPS = 1024
SAMPLE_RATE = 100.0


class _FakeSatelliteObservation(AbstractSatelliteObservation[None]):
    """Self-contained synthetic observation used by the in-process tests.

    Like the real interfaces, the getters return host (numpy) arrays; the reader
    moves them to device through ``io_callback`` and the reader's ``dtype``
    parameter is what decides the final dtype that flows into the operator chain.
    """

    def __init__(self) -> None:  # type: ignore[override]
        # bypass AbstractObservation.__init__ — we don't need an underlying
        # ``data`` container for these tests.
        pass

    @classmethod
    def from_file(cls, filename, requested_fields=None) -> _FakeSatelliteObservation:
        return cls()

    @property
    def name(self) -> str:
        return 'fake_obs'

    @property
    def telescope(self) -> str:
        return 'fake_telescope'

    @property
    def n_samples(self) -> int:
        return N_SAMPS

    @property
    def detectors(self) -> list[str]:
        return [f'det{i:02d}' for i in range(N_DETS)]

    @property
    def sample_rate(self) -> float:
        return SAMPLE_RATE

    def get_tods(self) -> Float[np.ndarray, 'dets samps']:
        # Non-zero data so the white-noise PSD fit yields a finite sigma.
        return np.random.default_rng(0).normal(size=(N_DETS, N_SAMPS)).astype(np.float32)

    def get_detector_offset_angles(self) -> Float[np.ndarray, ' dets']:
        return np.zeros(N_DETS, dtype=np.float64)

    def get_hwp_angles(self) -> Float[np.ndarray, ' a']:
        # Sweeping HWP angle at 2 Hz, wrapped to [0, 2pi).
        t = np.arange(N_SAMPS) / SAMPLE_RATE
        return np.asarray((2 * np.pi * 2.0 * t) % (2 * np.pi), dtype=np.float64)

    def get_sample_mask(self) -> Bool[np.ndarray, 'dets samps']:
        return np.ones((N_DETS, N_SAMPS), dtype=bool)

    def get_timestamps(self) -> Float[np.ndarray, ' a']:
        return np.asarray(1.7e9 + np.arange(N_SAMPS) / SAMPLE_RATE, dtype=np.float64)

    def get_wcs_shape_and_kernel(self, resolution_arcmin, projection=ProjectionType.CAR):
        raise NotImplementedError

    def get_pointing_and_spin_angles(self, landscape: StokesLandscape):
        raise NotImplementedError

    def get_noise_model(self) -> None | NoiseModel:
        return None

    def get_boresight_quaternions(self) -> Float[np.ndarray, 'samp 4']:
        # Sweep the boresight so samples hit a range of sky pixels.
        phi = np.linspace(0.0, np.pi / 4, N_SAMPS)
        q = np.zeros((N_SAMPS, 4), dtype=np.float64)
        q[:, 0] = np.cos(phi / 2)
        q[:, 3] = np.sin(phi / 2)
        return q

    def get_detector_quaternions(self) -> Float[np.ndarray, 'det 4']:
        q = np.zeros((N_DETS, 4), dtype=np.float64)
        q[:, 0] = 1.0
        return q


class _FakeLazyObservation(AbstractLazyObservation[None]):
    interface_class = _FakeSatelliteObservation

    def __init__(self) -> None:  # type: ignore[override]
        self.file = Path('<synthetic>')

    def get_data(self, requested_fields=None) -> _FakeSatelliteObservation:
        return _FakeSatelliteObservation()


# ----------------------------------------------------------------------------
# In-process dtype plumbing tests
# ----------------------------------------------------------------------------


REQUIRED_FIELDS = [
    'sample_data',
    'timestamps',
    'hwp_angles',
    'detector_quaternions',
    'boresight_quaternions',
]


class TestObservationReaderDtype:
    """Verify ObservationReader honours its ``dtype`` parameter."""

    def test_default_dtype_is_float64(self) -> None:
        """Backwards compatibility: no dtype argument → float64 structures.

        External callers that construct an ObservationReader directly (i.e. not
        via MultiObservationMapMaker) must continue to see float64, otherwise
        we would silently break the historical contract.
        """
        reader = ObservationReader.from_observations(
            [_FakeLazyObservation()], requested_fields=REQUIRED_FIELDS
        )
        assert reader.dtype == jnp.float64
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == jnp.float64, (
                f'expected float64 for {field}, got {reader.out_structure[field].dtype}'
            )

    def test_explicit_float32_propagates_to_all_float_fields(self) -> None:
        """``dtype=jnp.float32`` must apply to every float ShapeDtypeStruct.

        The original fix attempt only touched ``sample_data``; under
        ``jax_enable_x64=False`` JAX rejects any float64 ShapeDtypeStruct,
        so timestamps/HWP/quaternions/noise_model_fits must also be float32.
        """
        reader = ObservationReader.from_observations(
            [_FakeLazyObservation()],
            requested_fields=REQUIRED_FIELDS,
            dtype=jnp.float32,
        )
        assert reader.dtype == jnp.float32
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == jnp.float32, (
                f'expected float32 for {field}, got {reader.out_structure[field].dtype}'
            )

    def test_bool_masks_remain_bool(self) -> None:
        """Sanity check: changing ``dtype`` must not affect boolean masks."""
        reader = ObservationReader.from_observations(
            [_FakeLazyObservation()],
            requested_fields=['valid_sample_masks', 'sample_data'],
            dtype=jnp.float32,
        )
        assert reader.out_structure['valid_sample_masks'].dtype == jnp.bool


class TestMapMakerForwardsDtype:
    """``MultiObservationMapMaker.get_reader`` must use ``config.dtype``."""

    @pytest.mark.parametrize(
        ('double_precision', 'expected_dtype'),
        [(True, jnp.float64), (False, jnp.float32)],
    )
    def test_get_reader_uses_config_dtype(self, double_precision: bool, expected_dtype) -> None:
        config = MapMakingConfig(
            method=Methods.BINNED,
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
            weighting=WeightingConfig(),
            pointing=PointingConfig(on_the_fly=True),
            double_precision=double_precision,
        )
        maker = MultiObservationMapMaker([_FakeLazyObservation()], config=config)
        reader = maker.get_reader(REQUIRED_FIELDS)
        assert reader.dtype == expected_dtype
        for field in REQUIRED_FIELDS:
            assert reader.out_structure[field].dtype == expected_dtype


class TestMapMakerRunsX64OnDoublePrecisionFalse:
    """End-to-end run with ``jax_enable_x64=True`` but ``double_precision=False``.

    This is the in-process counterpart to the subprocess test: rather than
    turning x64 *off*, it keeps the session-wide x64 *on* (as enforced by the
    ``enable_x64`` autouse fixture in ``tests/conftest.py``) and asks for a
    float32 pipeline via ``double_precision=False``. This is the combination
    JAX does *not* protect us from — float64 arrays are perfectly legal, so a
    stray un-downcast geometry/noise field would silently produce a float64
    result or raise a dtype-mismatch instead of being coerced to float32. The
    test asserts the run completes and every output is float32.
    """

    def test_run_produces_float32_outputs(self) -> None:
        assert jax.config.read('jax_enable_x64'), (
            'this test must run with x64 enabled (the session default)'
        )

        config = MapMakingConfig(
            method=Methods.BINNED,
            landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
            weighting=WeightingConfig(
                fitting=NoiseFitConfig(nperseg=256, mask_hwp_harmonics=False),
            ),
            pointing=PointingConfig(on_the_fly=True),
            double_precision=False,
            scanning_mask=False,
            sample_mask=False,
            hits_cut=0.0,
            cond_cut=0.0,
            solver=SolverConfig(rtol=1e-6, atol=0, max_steps=10),
        )
        maker = MultiObservationMapMaker([_FakeLazyObservation()], config=config)
        results = maker.run()

        map_dtype = jax.tree.leaves(results.map)[0].dtype
        assert map_dtype == jnp.float32, f'expected float32 map, got {map_dtype}'
        assert results.icov.dtype == jnp.float32, f'expected float32 icov, got {results.icov.dtype}'


# ----------------------------------------------------------------------------
# End-to-end run with x64 off (the configuration a real float32 user uses)
# ----------------------------------------------------------------------------


@pytest.mark.insubprocess
def test_mapmaker_runs_under_double_precision_false_and_x64_off() -> None:
    """End-to-end regression test for the float32 pipeline bug.

    Marked ``insubprocess`` so it can flip ``jax_enable_x64`` off -- the
    configuration a real float32 user runs with -- without fighting the
    session-autouse ``enable_x64`` fixture in ``tests/conftest.py`` or
    leaking the flag into the rest of the suite.
    """
    jax.config.update('jax_enable_x64', False)
    assert not jax.config.read('jax_enable_x64'), 'x64 must be OFF for this test'

    config = MapMakingConfig(
        method=Methods.BINNED,
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
        weighting=WeightingConfig(
            fitting=NoiseFitConfig(nperseg=256, mask_hwp_harmonics=False),
        ),
        pointing=PointingConfig(on_the_fly=True),
        double_precision=False,
        scanning_mask=False,
        sample_mask=False,
        hits_cut=0.0,
        cond_cut=0.0,
        solver=SolverConfig(rtol=1e-6, atol=0, max_steps=10),
    )
    maker = MultiObservationMapMaker([_FakeLazyObservation()], config=config)
    results = maker.run()

    map_dtype = jax.tree.leaves(results.map)[0].dtype
    assert map_dtype == jnp.float32, f'expected float32 map, got {map_dtype}'
    assert results.icov.dtype == jnp.float32, f'expected float32 icov, got {results.icov.dtype}'
