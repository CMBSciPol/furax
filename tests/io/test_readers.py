from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import register_static
from numpy.testing import assert_array_equal

from furax.io.readers import AbstractReader


@register_static
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
    actual_data, padding, _ = reader.read(0)
    assert_array_equal(actual_data, jnp.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32))
    assert_array_equal(padding, jnp.array(6, jnp.int64))

    actual_data, padding, _ = reader.read(1)
    assert_array_equal(actual_data, jnp.array([2, 2, 2, 2, 0, 0, 0, 0], np.float32))
    assert_array_equal(padding, jnp.array(4, jnp.int64))

    actual_data, padding, _ = reader.read(2)
    assert_array_equal(actual_data, jnp.array([5, 5, 5, 5, 5, 5, 5, 5], np.float32))
    assert_array_equal(padding, jnp.array(0, jnp.int64))


@register_static
class OverProbeReader(AbstractReader):
    """Reader whose probe (structure) shape may exceed the actual loaded shape.

    Mirrors an upper-bound ``probe_shape``: ``out_structure`` is sized from the (larger) probe,
    while each load returns fewer samples and must be padded from its actual shape at read time.
    """

    def _read_structure_impure(self, probe: int, actual: int) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((probe,), jnp.float32)

    def _read_data_impure(self, probe: int, actual: int) -> Any:
        return np.full(actual, 1.0, jnp.float32)


def test_read_pads_from_actual_shape() -> None:
    # probes 4 and 6 -> common buffer of 6; loads are smaller and padded from their actual length
    reader = OverProbeReader([4, 6], [2, 5])
    assert reader.out_structure == jax.ShapeDtypeStruct((6,), jnp.float32)

    data, _, _ = reader.read(0)
    assert_array_equal(data, jnp.array([1, 1, 0, 0, 0, 0], np.float32))

    data, _, _ = reader.read(1)
    assert_array_equal(data, jnp.array([1, 1, 1, 1, 1, 0], np.float32))


def test_read_raises_when_probe_under_estimates() -> None:
    # common buffer is 3 (max probe), but obs 0 actually loads 5 samples -> negative pad
    reader = OverProbeReader([2, 3], [5, 1])
    with pytest.raises(Exception, match='under-estimated'):
        reader.read(0)


@register_static
class FillerReader(AbstractReader):
    """Reader that raises on flagged items and substitutes finite filler."""

    def _read_structure_impure(self, size: int, fail: bool) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((size,), jnp.float32)

    def _read_data_impure(self, size: int, fail: bool) -> Any:
        if fail:
            raise RuntimeError('boom')
        return np.full(size, 1.0, jnp.float32)

    def _failure_filler(self) -> Any:
        return np.full(self.out_structure.shape, -1.0, jnp.float32)


def test_read_returns_valid_flag_and_filler_on_failure() -> None:
    reader = FillerReader([4, 4], [False, True])

    data, _, valid = reader.read(0)
    assert bool(valid) is True
    assert_array_equal(data, jnp.ones(4, np.float32))
    assert reader.failed_indices == []

    data, _, valid = reader.read(1)
    assert bool(valid) is False
    assert_array_equal(data, jnp.full(4, -1.0, np.float32))  # filler
    assert reader.failed_indices == [1]


def test_default_reader_has_no_filler() -> None:
    # Readers that do not override _failure_filler keep the original crash-on-failure behavior.
    reader = OverProbeReader([4], [4])
    assert reader._failure_filler() is None


def test_buffers() -> None:
    reader1 = FakeFileReader(['1.npz', '2a.npz', '2b.npz', '3.npz', '5.npz'])
    reader2 = FakeFileReader(['2.npz', '5.npz'])
    assert (
        FakeFileReader.read.lower(reader1, 0).compile().memory_analysis().temp_size_in_bytes
        == FakeFileReader.read.lower(reader2, 0).compile().memory_analysis().temp_size_in_bytes
    )
