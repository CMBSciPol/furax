from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.interfaces.sotodlib import SOTODLibObservationResource
from furax.mapmaking import GroundObservationReader, HashedObservationMetadata


@pytest.fixture
def reader():
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    resources = [SOTODLibObservationResource(f) for f in files]
    return GroundObservationReader(resources)


def test_reader(reader) -> None:
    ndet1 = 2
    nsample1 = 1000
    ndet2 = 14
    nsample2 = 10000

    # Verify out_structure includes all non-optional fields (noise_model_fits is optional)
    assert reader.out_structure == {
        'metadata': HashedObservationMetadata(
            uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            detector_uids=jax.ShapeDtypeStruct((ndet2,), dtype=jnp.uint32),
        ),
        'sample_data': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet2, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample2, 4), dtype=jnp.float64),
    }
    # Verify noise_model_fits is NOT included by default (it's optional)
    assert 'noise_model_fits' not in reader.out_structure

    data = [reader.read(0), reader.read(1)]
    for datum, _ in data:
        assert datum['metadata'].uid.shape == ()
        assert datum['metadata'].telescope_uid.shape == ()
        assert datum['metadata'].detector_uids.shape == (ndet2,)
        assert datum['sample_data'].shape == (ndet2, nsample2)
        assert datum['valid_sample_masks'].shape == (ndet2, nsample2)
        assert datum['valid_scanning_masks'].shape == (nsample2,)
        assert datum['timestamps'].shape == (nsample2,)
        assert datum['hwp_angles'].shape == (nsample2,)
        assert datum['detector_quaternions'].shape == (ndet2, 4)
        assert datum['boresight_quaternions'].shape == (nsample2, 4)

    padding1 = data[0][1]
    assert padding1['metadata'].uid == ()
    assert padding1['metadata'].telescope_uid == ()
    assert padding1['metadata'].detector_uids == (ndet2 - ndet1,)
    assert padding1['sample_data'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['valid_sample_masks'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['valid_scanning_masks'] == (nsample2 - nsample1,)
    assert padding1['timestamps'] == (nsample2 - nsample1,)
    assert padding1['hwp_angles'] == (nsample2 - nsample1,)
    assert padding1['detector_quaternions'] == (ndet2 - ndet1, 0)
    assert padding1['boresight_quaternions'] == (nsample2 - nsample1, 0)

    padding2 = data[1][1]
    assert padding2['metadata'].uid == ()
    assert padding2['metadata'].telescope_uid == ()
    assert padding2['metadata'].detector_uids == (0,)
    assert padding2['sample_data'] == (0, 0)
    assert padding2['valid_sample_masks'] == (0, 0)
    assert padding2['valid_scanning_masks'] == (0,)
    assert padding2['timestamps'] == (0,)
    assert padding2['hwp_angles'] == (0,)
    assert padding2['detector_quaternions'] == (0, 0)
    assert padding2['boresight_quaternions'] == (0, 0)


def test_reader_invalid_data_field_name() -> None:
    """Test that passing an invalid data field name raises a ValueError."""
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    resources = [SOTODLibObservationResource(f) for f in files]

    # Test with single invalid field name
    with pytest.raises(ValueError, match='Data field "invalid_field" NOT supported'):
        GroundObservationReader(resources, data_field_names=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match='Data field "bad_field" NOT supported'):
        GroundObservationReader(resources, data_field_names=['sample_data', 'bad_field'])


def test_reader_subset_of_data_fields() -> None:
    """Test that passing a subset of data fields loads only those fields.

    This test is thorough because SOTODLibReader._read_data_impure() conditionally loads
    different sotodlib fields based on requested data fields:
    - sample_data → loads 'signal'
    - valid_sample_masks → loads 'flags.glitch_flags'
    - valid_scanning_masks → loads 'preprocess.turnaround_flags'
    - timestamps → loads 'timestamps'
    - hwp_angles → loads 'hwp_angle'
    - boresight_quaternions → loads 'boresight' (and 'timestamps' if not already loaded)
    - detector_quaternions → loads 'focal_plane'
    - noise_model_fits → computed from noise model (optional, not loaded by default)
    """
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']
    resources = [SOTODLibObservationResource(f) for f in files]

    ndet2 = 14
    nsample2 = 10000

    # Test 1: All detector-related fields (sample_data, valid_sample_masks, detector_quaternions)
    detector_fields = ['sample_data', 'valid_sample_masks', 'detector_quaternions']
    reader = GroundObservationReader(resources, data_field_names=detector_fields)

    assert set(reader.out_structure.keys()) == set(detector_fields)
    assert reader.out_structure['sample_data'] == jax.ShapeDtypeStruct(
        (ndet2, nsample2), dtype=jnp.float64
    )
    assert reader.out_structure['valid_sample_masks'] == jax.ShapeDtypeStruct(
        (ndet2, nsample2), dtype=jnp.bool
    )
    assert reader.out_structure['detector_quaternions'] == jax.ShapeDtypeStruct(
        (ndet2, 4), dtype=jnp.float64
    )

    data, _ = reader.read(0)
    assert set(data.keys()) == set(detector_fields)
    assert data['sample_data'].shape == (ndet2, nsample2)
    assert data['valid_sample_masks'].shape == (ndet2, nsample2)
    assert data['detector_quaternions'].shape == (ndet2, 4)

    # Test 2: All time/boresight-related fields
    # (timestamps, boresight_quaternions, valid_scanning_masks)
    time_fields = ['timestamps', 'boresight_quaternions', 'valid_scanning_masks']
    reader = GroundObservationReader(resources, data_field_names=time_fields)

    assert set(reader.out_structure.keys()) == set(time_fields)
    assert reader.out_structure['timestamps'] == jax.ShapeDtypeStruct(
        (nsample2,), dtype=jnp.float64
    )
    assert reader.out_structure['boresight_quaternions'] == jax.ShapeDtypeStruct(
        (nsample2, 4), dtype=jnp.float64
    )
    assert reader.out_structure['valid_scanning_masks'] == jax.ShapeDtypeStruct(
        (nsample2,), dtype=jnp.bool
    )

    data, _ = reader.read(0)
    assert set(data.keys()) == set(time_fields)
    assert data['timestamps'].shape == (nsample2,)
    assert data['boresight_quaternions'].shape == (nsample2, 4)
    assert data['valid_scanning_masks'].shape == (nsample2,)

    # Test 3: Mixed combination
    # (sample_data + timestamps + valid_scanning_masks + boresight_quaternions)
    # This tests loading signal, timestamps, turnaround_flags, and boresight
    mixed_fields = ['sample_data', 'timestamps', 'valid_scanning_masks', 'boresight_quaternions']
    reader = GroundObservationReader(resources, data_field_names=mixed_fields)

    assert set(reader.out_structure.keys()) == set(mixed_fields)
    assert 'valid_sample_masks' not in reader.out_structure
    assert 'detector_quaternions' not in reader.out_structure

    data, _ = reader.read(0)
    assert set(data.keys()) == set(mixed_fields)
    assert data['sample_data'].shape == (ndet2, nsample2)
    assert data['timestamps'].shape == (nsample2,)
    assert data['valid_scanning_masks'].shape == (nsample2,)
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test 4: Only sample_data (tests loading 'signal' alone)
    reader = GroundObservationReader(resources, data_field_names=['sample_data'])
    assert set(reader.out_structure.keys()) == {'sample_data'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'sample_data'}
    assert data['sample_data'].shape == (ndet2, nsample2)

    # Test 5: Only valid_sample_masks (tests loading 'flags.glitch_flags' alone)
    reader = GroundObservationReader(resources, data_field_names=['valid_sample_masks'])
    assert set(reader.out_structure.keys()) == {'valid_sample_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_sample_masks'}
    assert data['valid_sample_masks'].shape == (ndet2, nsample2)

    # Test 6: Only valid_scanning_masks (tests loading 'preprocess.turnaround_flags' alone)
    reader = GroundObservationReader(resources, data_field_names=['valid_scanning_masks'])
    assert set(reader.out_structure.keys()) == {'valid_scanning_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_scanning_masks'}
    assert data['valid_scanning_masks'].shape == (nsample2,)

    # Test 7: Only timestamps (tests loading 'timestamps' alone)
    reader = GroundObservationReader(resources, data_field_names=['timestamps'])
    assert set(reader.out_structure.keys()) == {'timestamps'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'timestamps'}
    assert data['timestamps'].shape == (nsample2,)

    # Test 8: Only boresight_quaternions (tests loading 'boresight' + 'timestamps')
    reader = GroundObservationReader(resources, data_field_names=['boresight_quaternions'])
    assert set(reader.out_structure.keys()) == {'boresight_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'boresight_quaternions'}
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test 9: Only hwp_angles (tests loading 'hwp_angle' alone)
    reader = GroundObservationReader(resources, data_field_names=['hwp_angles'])
    assert set(reader.out_structure.keys()) == {'hwp_angles'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'hwp_angles'}
    assert data['hwp_angles'].shape == (nsample2,)

    # Test 10: Only detector_quaternions (tests loading 'focal_plane' alone)
    reader = GroundObservationReader(resources, data_field_names=['detector_quaternions'])
    assert set(reader.out_structure.keys()) == {'detector_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'detector_quaternions'}
    assert data['detector_quaternions'].shape == (ndet2, 4)

    # Test 11: Only noise_model_fits (optional field, computed from noise model)
    # Note: This test is skipped if the test data doesn't have noise model
    reader = GroundObservationReader(resources, data_field_names=['noise_model_fits'])
    assert set(reader.out_structure.keys()) == {'noise_model_fits'}
    try:
        data, _ = reader.read(0)
        assert set(data.keys()) == {'noise_model_fits'}
        assert data['noise_model_fits'].shape == (ndet2, 4)  # (sigma, alpha, fknee, f0)
    except ValueError as e:
        if 'Data field not available' in str(e):
            pytest.skip('Test data does not contain noise model')
        raise

    # Test 12: Combination that triggers conditional timestamps loading
    # (boresight_quaternions without explicit timestamps)
    # This tests the special case where timestamps is loaded automatically for boresight
    combo_fields = ['sample_data', 'boresight_quaternions']
    reader = GroundObservationReader(resources, data_field_names=combo_fields)
    assert set(reader.out_structure.keys()) == set(combo_fields)
    data, _ = reader.read(0)
    assert set(data.keys()) == set(combo_fields)
    assert data['sample_data'].shape == (ndet2, nsample2)
    assert data['boresight_quaternions'].shape == (nsample2, 4)
