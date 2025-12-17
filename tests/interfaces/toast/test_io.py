from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip('toast', reason='toast is not installed. Skipping tests.')
pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')

from furax.interfaces.toast import LazyToastObservation
from furax.mapmaking import HashedObservationMetadata, ObservationReader


@pytest.fixture
def reader():
    folder = Path(__file__).parents[2] / 'data/toast'
    files = [folder / 'test_obs.h5', folder / 'test_obs.h5']
    observations = [LazyToastObservation(f) for f in files]
    return ObservationReader(observations)


def test_reader(reader) -> None:
    ndet1 = 2
    nsample1 = 1000
    ndet2 = 2
    nsample2 = 1000

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
    folder = Path(__file__).parents[2] / 'data/toast'
    files = [folder / 'test_obs.h5', folder / 'test_obs.h5']
    observations = [LazyToastObservation(f) for f in files]

    # Test with single invalid field name
    with pytest.raises(
        ValueError, match="Requested data fields {'invalid_field'} are not supported"
    ):
        ObservationReader(observations, requested_fields=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match="Requested data fields {'bad_field'} are not supported"):
        ObservationReader(observations, requested_fields=['sample_data', 'bad_field'])


def test_reader_subset_of_data_fields() -> None:
    """Test that passing a subset of data fields loads only those fields.

    This test is thorough because ToastReader._read_data_impure() conditionally loads
    different TOAST data based on requested fields:
    - sample_data → loads det_data
    - valid_sample_masks → loads det_flags
    - hwp_angles → loads hwp_angle
    - boresight_quaternions → loads boresight_radec
    - valid_scanning_masks → loads scanning_interval
    - timestamps → always loaded
    - detector_quaternions → computed from detector info (always available)
    - noise_model_fits → computed from noise model (optional, not loaded by default)
    """
    folder = Path(__file__).parents[2] / 'data/toast'
    files = [folder / 'test_obs.h5', folder / 'test_obs.h5']
    observations = [LazyToastObservation(f) for f in files]

    ndet2 = 2
    nsample2 = 1000

    # Test 1: All detector-related fields (sample_data, valid_sample_masks, detector_quaternions)
    detector_fields = ['sample_data', 'valid_sample_masks', 'detector_quaternions']
    reader = ObservationReader(observations, requested_fields=detector_fields)

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
    reader = ObservationReader(observations, requested_fields=time_fields)

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
    # This tests loading both det_data and boresight_radec
    mixed_fields = ['sample_data', 'timestamps', 'valid_scanning_masks', 'boresight_quaternions']
    reader = ObservationReader(observations, requested_fields=mixed_fields)

    assert set(reader.out_structure.keys()) == set(mixed_fields)
    assert 'valid_sample_masks' not in reader.out_structure
    assert 'detector_quaternions' not in reader.out_structure

    data, _ = reader.read(0)
    assert set(data.keys()) == set(mixed_fields)
    assert data['sample_data'].shape == (ndet2, nsample2)
    assert data['timestamps'].shape == (nsample2,)
    assert data['valid_scanning_masks'].shape == (nsample2,)
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test 4: Only sample_data (tests loading det_data alone)
    reader = ObservationReader(observations, requested_fields=['sample_data'])
    assert set(reader.out_structure.keys()) == {'sample_data'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'sample_data'}
    assert data['sample_data'].shape == (ndet2, nsample2)

    # Test 5: Only valid_sample_masks (tests loading det_flags alone)
    reader = ObservationReader(observations, requested_fields=['valid_sample_masks'])
    assert set(reader.out_structure.keys()) == {'valid_sample_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_sample_masks'}
    assert data['valid_sample_masks'].shape == (ndet2, nsample2)

    # Test 6: Only boresight_quaternions (tests loading boresight_radec alone)
    reader = ObservationReader(observations, requested_fields=['boresight_quaternions'])
    assert set(reader.out_structure.keys()) == {'boresight_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'boresight_quaternions'}
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test 7: Only valid_scanning_masks (tests loading scanning_interval alone)
    reader = ObservationReader(observations, requested_fields=['valid_scanning_masks'])
    assert set(reader.out_structure.keys()) == {'valid_scanning_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_scanning_masks'}
    assert data['valid_scanning_masks'].shape == (nsample2,)

    # Test 8: Only timestamps (minimal loading)
    reader = ObservationReader(observations, requested_fields=['timestamps'])
    assert set(reader.out_structure.keys()) == {'timestamps'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'timestamps'}
    assert data['timestamps'].shape == (nsample2,)

    # Test 9: Only hwp_angles (tests loading hwp_angle alone)
    reader = ObservationReader(observations, requested_fields=['hwp_angles'])
    assert set(reader.out_structure.keys()) == {'hwp_angles'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'hwp_angles'}
    assert data['hwp_angles'].shape == (nsample2,)

    # Test 10: Only detector_quaternions (computed field, no special loading)
    reader = ObservationReader(observations, requested_fields=['detector_quaternions'])
    assert set(reader.out_structure.keys()) == {'detector_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'detector_quaternions'}
    assert data['detector_quaternions'].shape == (ndet2, 4)

    # Test 11: Only noise_model_fits (optional field, computed from noise model)
    reader = ObservationReader(observations, requested_fields=['noise_model_fits'])
    assert set(reader.out_structure.keys()) == {'noise_model_fits'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'noise_model_fits'}
    assert data['noise_model_fits'].shape == (ndet2, 4)  # (sigma, alpha, fknee, f0)
