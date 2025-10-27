from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip('sotodlib', reason='sotodlib is not installed. Skipping tests.')
pytest.importorskip('so3g', reason='so3g is not installed. Skipping tests.')

from furax.interfaces.sotodlib import SOTODLibReader


def test_reader() -> None:
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']

    ndet1 = 2
    nsample1 = 1000
    ndet2 = 14
    nsample2 = 10000

    reader = SOTODLibReader(files)

    assert reader.out_structure == {
        'sample_data': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.float32),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet2, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample2, 4), dtype=jnp.float64),
    }

    data = [reader.read(0), reader.read(1)]
    for datum, _ in data:
        assert datum['sample_data'].shape == (ndet2, nsample2)
        assert datum['valid_sample_masks'].shape == (ndet2, nsample2)
        assert datum['timestamps'].shape == (nsample2,)
        assert datum['detector_quaternions'].shape == (ndet2, 4)
        assert datum['boresight_quaternions'].shape == (nsample2, 4)

    padding1 = data[0][1]
    assert padding1['sample_data'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['valid_sample_masks'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['timestamps'] == (nsample2 - nsample1,)
    assert padding1['detector_quaternions'] == (ndet2 - ndet1, 0)
    assert padding1['boresight_quaternions'] == (nsample2 - nsample1, 0)

    padding2 = data[1][1]
    assert padding2['sample_data'] == (0, 0)
    assert padding2['valid_sample_masks'] == (0, 0)
    assert padding2['timestamps'] == (0,)
    assert padding2['detector_quaternions'] == (0, 0)
    assert padding2['boresight_quaternions'] == (0, 0)


def test_reader_invalid_data_field_name() -> None:
    """Test that passing an invalid data field name raises a ValueError."""
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']

    # Test with single invalid field name
    with pytest.raises(ValueError, match='Data field "invalid_field" NOT supported'):
        SOTODLibReader(files, data_field_names=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match='Data field "bad_field" NOT supported'):
        SOTODLibReader(files, data_field_names=['sample_data', 'bad_field'])


def test_reader_subset_of_data_fields() -> None:
    """Test that passing a subset of data fields loads only those fields."""
    folder = Path(__file__).parents[2] / 'data/sotodlib'
    files = [folder / 'test_obs.h5', folder / 'test_obs_2.h5']

    ndet2 = 14
    nsample2 = 10000

    # Test with subset of fields
    subset_fields = ['sample_data', 'timestamps', 'boresight_quaternions']
    reader = SOTODLibReader(files, data_field_names=subset_fields)

    # Check that out_structure only contains the requested fields
    assert set(reader.out_structure.keys()) == set(subset_fields)
    assert 'valid_sample_masks' not in reader.out_structure
    assert 'detector_quaternions' not in reader.out_structure

    # Verify structure of requested fields
    assert reader.out_structure['sample_data'] == jax.ShapeDtypeStruct(
        (ndet2, nsample2), dtype=jnp.float32
    )
    assert reader.out_structure['timestamps'] == jax.ShapeDtypeStruct(
        (nsample2,), dtype=jnp.float64
    )
    assert reader.out_structure['boresight_quaternions'] == jax.ShapeDtypeStruct(
        (nsample2, 4), dtype=jnp.float64
    )

    # Read data and verify only requested fields are present
    data, padding = reader.read(0)
    assert set(data.keys()) == set(subset_fields)
    assert 'valid_sample_masks' not in data
    assert 'detector_quaternions' not in data

    # Verify shapes of loaded data
    assert data['sample_data'].shape == (ndet2, nsample2)
    assert data['timestamps'].shape == (nsample2,)
    assert data['boresight_quaternions'].shape == (nsample2, 4)

    # Test with single field
    single_field = ['timestamps']
    reader_single = SOTODLibReader(files, data_field_names=single_field)
    assert set(reader_single.out_structure.keys()) == {'timestamps'}
    data_single, _ = reader_single.read(0)
    assert set(data_single.keys()) == {'timestamps'}
    assert data_single['timestamps'].shape == (nsample2,)
