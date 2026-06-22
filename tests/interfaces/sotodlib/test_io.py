from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from furax.interfaces.sotodlib import (
    LazyPreprocSOTODLibObservation,
    LazySOTODLibObservation,
)
from furax.mapmaking import (
    AbstractGroundObservation,
    HashedObservationMetadata,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
)
from furax.mapmaking.config import (
    HealpixConfig,
    LandscapeConfig,
    PointingConfig,
    SotodlibConfig,
)
from furax.obs.stokes import Stokes
from furax.tree import as_structure

from ._preproc_db import build_preproc_db

FOLDER = Path(__file__).parents[2] / 'data/sotodlib'
FILES = ['test_obs.h5', 'test_obs_2.h5']
OBS_IDS = ['obs_12345_sometel', 'obs_54321_someothertel']
OBS_NDET = [2, 4]
OBS_NSAMPLE = [1_000, 3_000]


@pytest.fixture
def observations():
    return [LazySOTODLibObservation(FOLDER / f) for f in FILES]


@pytest.fixture
def demod_observations():
    sotodlib_config = SotodlibConfig(demodulated=True)
    return [LazySOTODLibObservation(FOLDER / f, sotodlib_config=sotodlib_config) for f in FILES]


def test_reader_all_fields(observations) -> None:
    """Test consistency for all available fields."""
    reader = ObservationReader.from_observations(
        observations, requested_fields=AbstractGroundObservation.AVAILABLE_READER_FIELDS
    )
    ndet_max, nsample_max = max(OBS_NDET), max(OBS_NSAMPLE)

    # check output structure is as expected
    assert reader.out_structure == {
        'metadata': HashedObservationMetadata(
            uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            detector_uids=jax.ShapeDtypeStruct((ndet_max,), dtype=jnp.uint32),
        ),
        'sample_data': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample_max, 4), dtype=jnp.float64),
        'noise_model_fits': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
    }

    for i in range(len(FILES)):
        datum, padding = reader.read(i)

        # check structure
        assert as_structure(datum) == reader.out_structure

        # check padding consistency
        ndet, nsample = OBS_NDET[i], OBS_NSAMPLE[i]
        assert padding['metadata'].uid == ()
        assert padding['metadata'].telescope_uid == ()
        assert padding['metadata'].detector_uids == (ndet_max - ndet,)
        assert padding['sample_data'] == (ndet_max - ndet, nsample_max - nsample)
        assert padding['valid_sample_masks'] == (ndet_max - ndet, nsample_max - nsample)
        assert padding['valid_scanning_masks'] == (nsample_max - nsample,)
        assert padding['timestamps'] == (nsample_max - nsample,)
        assert padding['hwp_angles'] == (nsample_max - nsample,)
        assert padding['detector_quaternions'] == (ndet_max - ndet, 0)
        assert padding['boresight_quaternions'] == (nsample_max - nsample, 0)
        assert padding['noise_model_fits'] == (ndet_max - ndet, 0)


def test_probe_shape(observations) -> None:
    """The lazy observation sizes its buffers without (necessarily) a full load."""
    for i, obs in enumerate(observations):
        assert obs.probe_shape() == (OBS_NDET[i], OBS_NSAMPLE[i])


def test_lazy_preproc_observation(tmp_path) -> None:
    """Preproc-backed lazy obs loads straight from a (minimal, real) preprocessing db."""
    config = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[0], config)

    # get_data runs the full preproc load (with signal) through load_and_preprocess
    data = lazy.get_data()
    assert data.n_detectors == OBS_NDET[0]
    assert data.n_samples == OBS_NSAMPLE[0]
    assert data.get_tods().shape == (OBS_NDET[0], OBS_NSAMPLE[0])

    # probe_shape uses the signal-free load and matches the data shape exactly
    assert lazy.probe_shape() == (OBS_NDET[0], OBS_NSAMPLE[0])


def test_reader_with_preproc_observations(tmp_path) -> None:
    """The reader drives preproc-db-backed lazy observations through io_callback end to end."""
    config = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    observations = [LazyPreprocSOTODLibObservation(obs_id, config) for obs_id in OBS_IDS]

    reader = ObservationReader.from_observations(
        observations, requested_fields=['sample_data', 'timestamps', 'valid_scanning_masks']
    )
    ndet_max, nsample_max = max(OBS_NDET), max(OBS_NSAMPLE)
    assert reader.out_structure['sample_data'].shape == (ndet_max, nsample_max)

    for i in range(len(FILES)):
        datum, _ = reader.read(i)
        assert as_structure(datum) == reader.out_structure


def test_binned_mapmaker_over_preproc_db(tmp_path) -> None:
    """Bin a map straight from the preproc db, exercising the full streaming pipeline."""
    config_path = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    observations = [LazyPreprocSOTODLibObservation(obs_id, config_path) for obs_id in OBS_IDS]

    stokes = 'IQU'
    config = MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16)),
    )
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.run()

    n_stokes = len(stokes)
    assert results.hit_map.shape == maker.landscape.shape
    assert jnp.all(results.hit_map >= 0)
    assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
    # binned map: CG converges in a single iteration
    assert results.solver_stats is not None
    assert results.solver_stats['num_steps'] == 1


def test_reader_invalid_data_field_name(observations) -> None:
    """Test that passing an invalid data field name raises a ValueError."""
    # Test with single invalid field name
    with pytest.raises(
        ValueError, match="Requested data fields {'invalid_field'} are not supported"
    ):
        ObservationReader.from_observations(observations, requested_fields=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match="Requested data fields {'bad_field'} are not supported"):
        ObservationReader.from_observations(
            observations, requested_fields=['sample_data', 'bad_field']
        )


@pytest.mark.parametrize(
    'requested_fields',
    [
        ['sample_data', 'valid_sample_masks', 'detector_quaternions'],  # detector-related
        ['timestamps', 'boresight_quaternions', 'valid_scanning_masks'],  # time/boresight-related
        ['sample_data', 'timestamps', 'valid_scanning_masks', 'boresight_quaternions'],  # mixed
        ['sample_data'],  # detector data only
        ['valid_sample_masks'],  # detector flags only
        ['valid_scanning_masks'],  # scanning mask only
        ['boresight_quaternions'],  # boresight quats only
        ['detector_quaternions'],  # detector quats only
        ['timestamps'],  # timestamps only
        ['hwp_angles'],  # HWP angles only
        ['noise_model_fits'],  # optional field only
    ],
)
def test_reader_subset_of_data_fields(observations, requested_fields: list[str]) -> None:
    """Test that passing a subset of data fields loads only those fields."""
    reader = ObservationReader.from_observations(observations, requested_fields=requested_fields)
    assert set(reader.out_structure.keys()) == set(requested_fields)


def test_reader_all_fields_demod(demod_observations) -> None:
    """Test consistency for all available fields in demodulated mode."""
    stokes = 'IQU'
    reader = ObservationReader.from_observations(
        demod_observations,
        requested_fields=AbstractGroundObservation.AVAILABLE_READER_FIELDS,
        demodulated=True,
        stokes=stokes,
    )
    ndet_max, nsample_max = max(OBS_NDET), max(OBS_NSAMPLE)
    kls = Stokes.class_for(stokes)

    assert reader.out_structure == {
        'metadata': HashedObservationMetadata(
            uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            detector_uids=jax.ShapeDtypeStruct((ndet_max,), dtype=jnp.uint32),
        ),
        'sample_data': kls.structure_for((ndet_max, nsample_max), jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample_max, 4), dtype=jnp.float64),
        'noise_model_fits': kls.structure_for((ndet_max, 4), jnp.float64),
    }

    for i in range(len(FILES)):
        datum, padding = reader.read(i)

        assert as_structure(datum) == reader.out_structure

        ndet, nsample = OBS_NDET[i], OBS_NSAMPLE[i]
        assert padding['metadata'].uid == ()
        assert padding['metadata'].telescope_uid == ()
        assert padding['metadata'].detector_uids == (ndet_max - ndet,)
        assert all(
            getattr(padding['sample_data'], s) == (ndet_max - ndet, nsample_max - nsample)
            for s in stokes.lower()
        )
        assert padding['valid_sample_masks'] == (ndet_max - ndet, nsample_max - nsample)
        assert padding['valid_scanning_masks'] == (nsample_max - nsample,)
        assert padding['timestamps'] == (nsample_max - nsample,)
        assert padding['hwp_angles'] == (nsample_max - nsample,)
        assert padding['detector_quaternions'] == (ndet_max - ndet, 0)
        assert padding['boresight_quaternions'] == (nsample_max - nsample, 0)
        assert all(
            getattr(padding['noise_model_fits'], s) == (ndet_max - ndet, 0) for s in stokes.lower()
        )


@pytest.mark.parametrize(
    'requested_fields',
    [
        ['sample_data'],
        ['sample_data', 'noise_model_fits'],
        ['sample_data', 'timestamps', 'boresight_quaternions'],
    ],
)
def test_reader_subset_of_data_fields_demod(
    demod_observations, requested_fields: list[str]
) -> None:
    """Test that passing a subset of data fields loads only those fields in demodulated mode."""
    reader = ObservationReader.from_observations(
        demod_observations, requested_fields=requested_fields, demodulated=True
    )
    assert set(reader.out_structure.keys()) == set(requested_fields)
