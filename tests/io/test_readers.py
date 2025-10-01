from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from furax.io.readers import AbstractReader


@jax.tree_util.register_static
class FakeFileReader(AbstractReader):
    def _read_structure_impure(
        self, filename: str, option1: str = '', option2: float = 0.0
    ) -> jax.ShapeDtypeStruct:
        length = int(filename[0]) + len(option1)
        return jax.ShapeDtypeStruct((length,), jnp.float32)

    def _read_data_impure(self, filename: str, option1: str = '', option2: float = 0.0) -> Any:
        length = int(filename[0]) + len(option1)
        return np.full(length, float(filename[0]), jnp.float32)


@pytest.fixture
def reader() -> FakeFileReader:
    return FakeFileReader(
        ['1.fits', '2.fits', '5.fits'],
        option1=['a', 'bb', 'ccc'],
        common_keywords={'option2': 1.5},
    )


def test_init(reader: FakeFileReader) -> None:
    assert reader.args == [('1.fits',), ('2.fits',), ('5.fits',)]
    assert reader.keywords == [{'option1': 'a'}, {'option1': 'bb'}, {'option1': 'ccc'}]
    assert reader.common_keywords == {'option2': 1.5}
    assert reader.out_structure == jax.ShapeDtypeStruct((8,), jnp.float32)
    assert reader.paddings == [(6,), (4,), (0,)]


def test_read(reader: FakeFileReader) -> None:
    actual_data, padding = reader.read(0)
    assert_array_equal(actual_data, jnp.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32))
    assert_array_equal(padding, jnp.array(6, jnp.int64))

    actual_data, padding = reader.read(1)
    assert_array_equal(actual_data, jnp.array([2, 2, 2, 2, 0, 0, 0, 0], np.float32))
    assert_array_equal(padding, jnp.array(4, jnp.int64))

    actual_data, padding = reader.read(2)
    assert_array_equal(actual_data, jnp.array([5, 5, 5, 5, 5, 5, 5, 5], np.float32))
    assert_array_equal(padding, jnp.array(0, jnp.int64))


def test_buffers() -> None:
    reader1 = FakeFileReader(['1.npz', '2a.npz', '2b.npz', '3.npz', '5.npz'])
    reader2 = FakeFileReader(['2.npz', '5.npz'])
    assert (
        FakeFileReader.read.lower(reader1, 0).compile().memory_analysis().temp_size_in_bytes
        == FakeFileReader.read.lower(reader2, 0).compile().memory_analysis().temp_size_in_bytes
    )
