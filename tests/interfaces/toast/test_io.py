from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from furax.interfaces.toast import LazyToastObservation
from furax.mapmaking import AbstractGroundObservation, HashedObservationMetadata, ObservationReader
from furax.tree import as_structure

# These tests do not depend only on TOAST but also on sotodlib because the test files
# are SO-specific (site, telescope, etc.). Therefore we skip if sotodlib is not installed.
pytest.importorskip('sotodlib', reason='sotodlib is not installed')

FOLDER = Path(__file__).parents[2] / 'data/toast'
FILES = ['test_obs.h5', 'test_obs.h5']
OBS_NDET = [2, 2]
OBS_NSAMPLE = [1000, 1000]


@pytest.fixture
def observations():
    return [LazyToastObservation(FOLDER / f) for f in FILES]


def test_reader_all_fields(observations) -> None:
    """Test consistency for all available fields."""
    reader = ObservationReader(
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


def test_reader_invalid_data_field_name(observations) -> None:
    """Test that passing an invalid data field name raises a ValueError."""
    # Test with single invalid field name
    with pytest.raises(
        ValueError, match="Requested data fields {'invalid_field'} are not supported"
    ):
        ObservationReader(observations, requested_fields=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match="Requested data fields {'bad_field'} are not supported"):
        ObservationReader(observations, requested_fields=['sample_data', 'bad_field'])


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
    reader = ObservationReader(observations, requested_fields=requested_fields)
    assert set(reader.out_structure.keys()) == set(requested_fields)
