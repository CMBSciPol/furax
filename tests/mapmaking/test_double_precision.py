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

import jax
import jax.numpy as jnp
import pytest

from furax.mapmaking import (
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking._model import _sample_rate
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
from tests.mapmaking.helpers import FakeLazyObservation, FakeObservation

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
            [FakeLazyObservation()], requested_fields=REQUIRED_FIELDS
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
            [FakeLazyObservation()],
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
            [FakeLazyObservation()],
            requested_fields=['valid_sample_masks', 'sample_data'],
            dtype=jnp.float32,
        )
        assert reader.out_structure['valid_sample_masks'].dtype == jnp.bool


class TestObservationReaderRebasesTimestamps:
    """Reader rebases timestamps to a zero origin before the float32 cast.

    ``FakeObservation`` timestamps carry an absolute POSIX epoch whose float32 ULP
    is coarser than the (short) observation: without rebasing, every sample rounds
    to the same value once cast to float32.
    """

    def test_reader_rebases_and_keeps_samples_resolved(self) -> None:
        obs = FakeObservation()
        reader = ObservationReader.from_observations(
            [FakeLazyObservation()],
            requested_fields=['timestamps'],
            dtype=jnp.float32,
        )
        data, _, _ = reader.read(0)
        timestamps = data['timestamps']

        assert timestamps.dtype == jnp.float32
        # Starts at zero, and the samples stay distinct: strictly increasing, all
        # present, spanning the full (n_samples - 1) / sample_rate duration.
        assert timestamps[0] == 0.0
        assert jnp.all(jnp.diff(timestamps) > 0)
        assert jnp.unique(timestamps).size == obs.n_samples
        expected_span = (obs.n_samples - 1) / obs.sample_rate
        assert jnp.ptp(timestamps) == pytest.approx(expected_span, rel=1e-5)

    def test_derived_sample_rate_is_finite(self) -> None:
        # _sample_rate is (n_samples - 1) / ptp(timestamps): a collapsed axis (ptp=0)
        # makes it diverge, so check it stays finite and recovers the true rate.
        obs = FakeObservation()
        reader = ObservationReader.from_observations(
            [FakeLazyObservation()],
            requested_fields=['timestamps'],
            dtype=jnp.float32,
        )
        data, _, _ = reader.read(0)
        fs = _sample_rate(data['timestamps'])
        assert jnp.isfinite(fs)
        assert fs == pytest.approx(obs.sample_rate, rel=1e-4)


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
        maker = MultiObservationMapMaker([FakeLazyObservation()], config=config)
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
        maker = MultiObservationMapMaker([FakeLazyObservation()], config=config)
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
    maker = MultiObservationMapMaker([FakeLazyObservation()], config=config)
    results = maker.run()

    map_dtype = jax.tree.leaves(results.map)[0].dtype
    assert map_dtype == jnp.float32, f'expected float32 map, got {map_dtype}'
    assert results.icov.dtype == jnp.float32, f'expected float32 icov, got {results.icov.dtype}'
