from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip('toast', reason='toast is not installed. Skipping tests.')
pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')

from furax.interfaces.toast import ToastReader


def test_reader() -> None:
    folder = Path(__file__).parents[2] / 'data/toast'
    files = [folder / 'test_obs.h5', folder / 'test_obs.h5']

    ndet1 = 2
    nsample1 = 1000
    ndet2 = 2
    nsample2 = 1000

    reader = ToastReader(files)

    # Verify out_structure includes all fields including hwp_angles
    assert reader.out_structure == {
        'sample_data': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet2, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample2, 4), dtype=jnp.float64),
    }

    data = [reader.read(0), reader.read(1)]
    for datum, _ in data:
        assert datum['sample_data'].shape == (ndet2, nsample2)
        assert datum['valid_sample_masks'].shape == (ndet2, nsample2)
        assert datum['valid_scanning_masks'].shape == (nsample2,)
        assert datum['timestamps'].shape == (nsample2,)
        assert datum['hwp_angles'].shape == (nsample2,)
        assert datum['detector_quaternions'].shape == (ndet2, 4)
        assert datum['boresight_quaternions'].shape == (nsample2, 4)

    padding1 = data[0][1]
    assert padding1['sample_data'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['valid_sample_masks'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['valid_scanning_masks'] == (nsample2 - nsample1,)
    assert padding1['timestamps'] == (nsample2 - nsample1,)
    assert padding1['hwp_angles'] == (nsample2 - nsample1,)
    assert padding1['detector_quaternions'] == (ndet2 - ndet1, 0)
    assert padding1['boresight_quaternions'] == (nsample2 - nsample1, 0)

    padding2 = data[1][1]
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
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']

    # Test with single invalid field name
    with pytest.raises(ValueError, match='Data field "invalid_field" NOT supported'):
        ToastReader(files, data_field_names=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match='Data field "bad_field" NOT supported'):
        ToastReader(files, data_field_names=['sample_data', 'bad_field'])


@pytest.mark.parametrize(
    'new_field_names',
    [
        ['sample_data'],
        ['boresight_quaternions', 'detector_quaternions', 'hwp_angles'],
    ],
)
def test_reader_update_data_fields(new_field_names):
    folder = Path(__file__).parents[2] / 'data/toast'
    reader = ToastReader([folder / 'test_obs.h5'])
    new_reader = reader.update_data_field_names(new_field_names)
    data, _ = new_reader.read(0)
    assert set(data.keys()) == set(new_field_names)


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
    """
    folder = Path(__file__).parents[2] / 'data/toast'
    files = [folder / 'test_obs.h5', folder / 'test_obs.h5']

    ndet2 = 2
    nsample2 = 1000

    # Test 1: All detector-related fields (sample_data, valid_sample_masks, detector_quaternions)
    detector_fields = ['sample_data', 'valid_sample_masks', 'detector_quaternions']
    reader = ToastReader(files, data_field_names=detector_fields)

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
    reader = ToastReader(files, data_field_names=time_fields)

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
    reader = ToastReader(files, data_field_names=mixed_fields)

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
    reader = ToastReader(files, data_field_names=['sample_data'])
    assert set(reader.out_structure.keys()) == {'sample_data'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'sample_data'}
    assert data['sample_data'].shape == (ndet2, nsample2)

    # Test 5: Only valid_sample_masks (tests loading det_flags alone)
    reader = ToastReader(files, data_field_names=['valid_sample_masks'])
    assert set(reader.out_structure.keys()) == {'valid_sample_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_sample_masks'}
    assert data['valid_sample_masks'].shape == (ndet2, nsample2)

    # Test 6: Only boresight_quaternions (tests loading boresight_radec alone)
    reader = ToastReader(files, data_field_names=['boresight_quaternions'])
    assert set(reader.out_structure.keys()) == {'boresight_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'boresight_quaternions'}
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test 7: Only valid_scanning_masks (tests loading scanning_interval alone)
    reader = ToastReader(files, data_field_names=['valid_scanning_masks'])
    assert set(reader.out_structure.keys()) == {'valid_scanning_masks'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'valid_scanning_masks'}
    assert data['valid_scanning_masks'].shape == (nsample2,)

    # Test 8: Only timestamps (minimal loading)
    reader = ToastReader(files, data_field_names=['timestamps'])
    assert set(reader.out_structure.keys()) == {'timestamps'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'timestamps'}
    assert data['timestamps'].shape == (nsample2,)

    # Test 9: Only hwp_angles (tests loading hwp_angle alone)
    reader = ToastReader(files, data_field_names=['hwp_angles'])
    assert set(reader.out_structure.keys()) == {'hwp_angles'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'hwp_angles'}
    assert data['hwp_angles'].shape == (nsample2,)

    # Test 10: Only detector_quaternions (computed field, no special loading)
    reader = ToastReader(files, data_field_names=['detector_quaternions'])
    assert set(reader.out_structure.keys()) == {'detector_quaternions'}
    data, _ = reader.read(0)
    assert set(data.keys()) == {'detector_quaternions'}
    assert data['detector_quaternions'].shape == (ndet2, 4)
