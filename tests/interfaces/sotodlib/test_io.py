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
        'signal': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.float32),
        'mask': jax.ShapeDtypeStruct((ndet2, nsample2), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample2,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet2, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample2, 4), dtype=jnp.float64),
    }

    data = [reader.read(0), reader.read(1)]
    for datum, _ in data:
        assert datum['signal'].shape == (ndet2, nsample2)
        assert datum['mask'].shape == (ndet2, nsample2)
        assert datum['timestamps'].shape == (nsample2,)
        assert datum['detector_quaternions'].shape == (ndet2, 4)
        assert datum['boresight_quaternions'].shape == (nsample2, 4)

    padding1 = data[0][1]
    assert padding1['signal'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['mask'] == (ndet2 - ndet1, nsample2 - nsample1)
    assert padding1['timestamps'] == (nsample2 - nsample1,)
    assert padding1['detector_quaternions'] == (ndet2 - ndet1, 0)
    assert padding1['boresight_quaternions'] == (nsample2 - nsample1, 0)

    padding2 = data[1][1]
    assert padding2['signal'] == (0, 0)
    assert padding2['mask'] == (0, 0)
    assert padding2['timestamps'] == (0,)
    assert padding2['detector_quaternions'] == (0, 0)
    assert padding2['boresight_quaternions'] == (0, 0)
