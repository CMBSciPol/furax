"""Regression tests for ``MapMakingConfig.double_precision=False``.

The reader and the mapmaker must give every floating-point field ``config.dtype``, the TOD
excepted, which is always read in float32. The end-to-end runs check the float32 pipeline both
under the session-wide ``jax_enable_x64=True``, where a stray float64 field is legal and would go
unnoticed, and with x64 off, in a subprocess so the flag flip does not leak into other tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking import MultiObservationMapMaker, ObservationReader
from furax.mapmaking._model import _sample_rate
from furax.mapmaking.config import NoiseFitConfig, SolverConfig, WeightingConfig
from tests.mapmaking.helpers import FakeLazyObservation, FakeObservation, accumulate, healpix_config

REQUIRED_FIELDS = [
    'sample_data',
    'timestamps',
    'hwp_angles',
    'detector_quaternions',
    'boresight_quaternions',
]


@pytest.mark.parametrize(
    ('kwargs', 'dtype', 'sample_dtype'),
    [
        # callers that construct a reader directly keep the historical float64
        ({}, jnp.float64, jnp.float64),
        # under `jax_enable_x64=False` JAX rejects any float64 structure, so every float field
        # takes the dtype, not only the samples
        ({'dtype': jnp.float32}, jnp.float32, jnp.float32),
        ({'sample_dtype': jnp.float32}, jnp.float64, jnp.float32),
    ],
    ids=['default', 'float32', 'float32-samples'],
)
def test_reader_dtypes(kwargs, dtype, sample_dtype) -> None:
    fields = [*REQUIRED_FIELDS, 'valid_sample_masks']
    reader = ObservationReader.from_observations(
        [FakeLazyObservation()], requested_fields=fields, **kwargs
    )
    assert reader.dtype == dtype
    for field in REQUIRED_FIELDS:
        expected = sample_dtype if field == 'sample_data' else dtype
        assert reader.out_structure[field].dtype == expected, field
    assert reader.out_structure['valid_sample_masks'].dtype == jnp.bool
    data, _, _ = reader.read(0)
    assert data['sample_data'].dtype == sample_dtype


def test_reader_rebases_timestamps_before_the_float32_cast() -> None:
    # The fake timestamps carry an absolute POSIX epoch whose float32 ULP is coarser than the
    # (short) observation: without rebasing, every sample rounds to the same value.
    obs = FakeObservation()
    reader = ObservationReader.from_observations(
        [FakeLazyObservation()], requested_fields=['timestamps'], dtype=jnp.float32
    )
    data, _, _ = reader.read(0)
    timestamps = data['timestamps']

    assert timestamps.dtype == jnp.float32
    # Starts at zero, and the samples stay distinct: strictly increasing, spanning the full
    # (n_samples - 1) / sample_rate duration.
    assert timestamps[0] == 0.0
    assert jnp.all(jnp.diff(timestamps) > 0)
    expected_span = (obs.n_samples - 1) / obs.sample_rate
    assert jnp.ptp(timestamps) == pytest.approx(expected_span, rel=1e-5)
    # the sample rate derived from them, (n_samples - 1) / ptp, is the true one
    assert _sample_rate(timestamps) == pytest.approx(obs.sample_rate, rel=1e-4)


@pytest.mark.parametrize(
    ('double_precision', 'expected_dtype'), [(True, jnp.float64), (False, jnp.float32)]
)
def test_mapmaker_readers_use_config_dtype(double_precision: bool, expected_dtype) -> None:
    config = healpix_config(nside=8, double_precision=double_precision)
    maker = MultiObservationMapMaker([FakeLazyObservation()], config=config)
    (reader,) = maker.get_readers(REQUIRED_FIELDS)
    assert reader.dtype == expected_dtype
    for field in REQUIRED_FIELDS:
        # the TOD is always read in float32
        expected = jnp.float32 if field == 'sample_data' else expected_dtype
        assert reader.out_structure[field].dtype == expected


@pytest.mark.parametrize('detector_batch_size', [0, 3])
def test_float32_tod_accumulates_like_float64(detector_batch_size: int) -> None:
    # The TOD is read in float32 and converted to float64 before use, whole or one detector
    # batch at a time. The fake TOD is float32 at the source, so a float64 read gives the same sums.
    observations = [FakeLazyObservation(n_dets=7)]
    config = healpix_config(
        nside=8,
        weighting=WeightingConfig(fitting=NoiseFitConfig(nperseg=256)),
        detector_batch_size=detector_batch_size,
    )
    maker = MultiObservationMapMaker(observations, config=config)
    reference = MultiObservationMapMaker(observations, config=config)
    shapes = reference._probe_shapes
    reference.readers = (
        ObservationReader.from_observations(
            observations,
            requested_fields=reference.reader_fields,
            sample_dtype=jnp.float64,
            shapes=shapes,
        ),
    )
    float64, float32 = accumulate(reference), accumulate(maker)
    assert float32.map_rhs.data.dtype == jnp.float64
    assert_array_equal(float32.hit_map, float64.hit_map)
    assert_allclose(float32.map_rhs.data, float64.map_rhs.data, rtol=1e-12)


def _check_float32_run() -> None:
    """A tiny run under `double_precision=False` completes with float32 outputs."""
    config = healpix_config(
        nside=8,
        weighting=WeightingConfig(fitting=NoiseFitConfig(nperseg=256, mask_hwp_harmonics=False)),
        double_precision=False,
        scanning_mask=False,
        sample_mask=False,
        hits_cut=0.0,
        cond_cut=0.0,
        solver=SolverConfig(rtol=1e-6, atol=0, max_steps=10),
    )
    results = MultiObservationMapMaker([FakeLazyObservation()], config=config).run()
    assert jax.tree.leaves(results.map)[0].dtype == jnp.float32
    assert results.icov.dtype == jnp.float32


def test_double_precision_false_runs_in_float32_under_x64() -> None:
    # float64 arrays are legal under x64, so a stray un-downcast field would not raise here: it
    # would silently promote the result, or raise a dtype mismatch, instead of staying float32
    assert jax.config.read('jax_enable_x64')
    _check_float32_run()


@pytest.mark.insubprocess
def test_double_precision_false_runs_with_x64_off() -> None:
    # the configuration a float32 user runs with, where JAX rejects every float64 array
    jax.config.update('jax_enable_x64', False)
    _check_float32_run()
