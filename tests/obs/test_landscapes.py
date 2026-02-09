import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.testing import assert_array_equal

from furax.obs._samplings import Sampling
from furax.obs.landscapes import (
    FrequencyLandscape,
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
)
from furax.obs.stokes import Stokes, ValidStokesType


def test_healpix_landscape(stokes: ValidStokesType) -> None:
    nside = 64
    npixel = 12 * nside**2

    landscape = HealpixLandscape(nside, stokes)

    sky = landscape.ones()
    assert isinstance(sky, Stokes.class_for(stokes))
    assert sky.shape == (npixel,)
    for stoke in stokes:
        leaf = getattr(sky, stoke.lower())
        assert isinstance(leaf, Array)
        assert leaf.size == npixel
        assert_array_equal(leaf, 1.0)


def test_frequency_landscape(stokes: ValidStokesType) -> None:
    nside = 64
    npixel = 12 * nside**2
    frequencies = jnp.array([10, 20, 30])
    landscape = FrequencyLandscape(nside, frequencies, stokes)

    sky = landscape.ones()
    assert isinstance(sky, Stokes.class_for(stokes))
    assert sky.shape == (3, npixel)
    for stoke in stokes:
        leaf = getattr(sky, stoke.lower())
        assert isinstance(leaf, Array)
        assert leaf.size == 3 * npixel
        assert_array_equal(leaf, 1.0)


@pytest.mark.parametrize(
    'pixel, expected_index',
    [
        ((-0.5 - 1e-15, -0.5), -1),
        ((-0.5, -0.5 - 1e-15), -1),
        ((1.5 + 1e-15, -0.5), -1),
        ((1.5 - 1e-15, -0.5 - 1e-15), -1),
        ((-0.5 - 1e-15, 4.5 - 1e-15), -1),
        ((-0.5, 4.5 + 1e-15), -1),
        ((1.5 + 1e-15, 4.5 - 1e-15), -1),
        ((1.5 - 1e-15, 4.5 + 1e-15), -1),
        ((-0.5, -0.5), 0),
        ((-0.5, 0.5 - 1e-15), 0),
        ((0.5, -0.5), 0),
        ((0.5, 0.5 - 1e-15), 0),
        ((0, 0), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((1, 1), 3),
        ((0, 4), 8),
        ((1, 4), 9),
    ],
)
def test_pixel2index(pixel: tuple[float, float], expected_index: int) -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    landscape = CARStokesLandscape((5, 2), 'I')
    actual_index = landscape.pixel2index(*pixel)
    assert_array_equal(actual_index, expected_index)


def test_get_coverage() -> None:
    class CARStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return theta, phi

    samplings = Sampling(
        jnp.array([0.0, 1, 0, 1, 1, 1, 0]), jnp.array([0.0, 0, 0, 3, 0, 1, 0]), jnp.array(0.0)
    )
    landscape = CARStokesLandscape((5, 2), 'I')
    coverage = landscape.get_coverage(samplings)
    assert_array_equal(coverage, [[3, 2], [0, 1], [0, 0], [0, 1], [0, 0]])


# --- LocalLandscape tests ---


class _SimpleLandscape(StokesLandscape):
    """A trivial landscape where world2pixel is the identity (for testing)."""

    def world2pixel(self, theta, phi):
        return (theta,)


def _make_simple_parent(npixel: int = 10, stokes: ValidStokesType = 'I') -> _SimpleLandscape:
    return _SimpleLandscape((npixel,), stokes)


class TestLocalLandscapeConstruction:
    def test_basic_construction(self) -> None:
        parent = _make_simple_parent()
        indices = np.array([1, 3, 5, 7])
        local = LocalStokesLandscape(parent, indices)
        assert local.shape == (4,)
        assert local.nlocal == 4
        assert_array_equal(local.global_indices, indices)

    def test_empty_indices(self) -> None:
        parent = _make_simple_parent()
        local = LocalStokesLandscape(parent, np.array([], dtype=np.int64))
        assert local.shape == (0,)
        assert local.nlocal == 0

    def test_rejects_unsorted(self) -> None:
        parent = _make_simple_parent()
        with pytest.raises(ValueError, match='sorted'):
            LocalStokesLandscape(parent, np.array([3, 1, 5]))

    def test_rejects_duplicates(self) -> None:
        parent = _make_simple_parent()
        with pytest.raises(ValueError, match='sorted'):
            LocalStokesLandscape(parent, np.array([1, 3, 3, 5]))

    def test_rejects_2d(self) -> None:
        parent = _make_simple_parent()
        with pytest.raises(ValueError, match='1-dimensional'):
            LocalStokesLandscape(parent, np.array([[1, 2], [3, 4]]))

    def test_delegates_stokes_dtype(self) -> None:
        parent = _make_simple_parent(stokes='IQU')
        local = LocalStokesLandscape(parent, np.array([0, 2, 4]))
        assert local.stokes == 'IQU'
        assert local.dtype == parent.dtype

    def test_size(self) -> None:
        parent = _make_simple_parent(stokes='IQU')
        local = LocalStokesLandscape(parent, np.array([0, 2, 4]))
        assert local.size == 3 * 3  # 3 stokes * 3 pixels


class TestLocalLandscapeIndexConversion:
    def test_global2local(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))
        result = local.global2local(jnp.array([2, 5, 10, 15]))
        assert_array_equal(result, [0, 1, 2, 3])

    def test_global2local_missing(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))
        result = local.global2local(jnp.array([0, 3, 7, 19]))
        assert_array_equal(result, [-1, -1, -1, -1])

    def test_global2local_mixed(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))
        result = local.global2local(jnp.array([2, 3, 10, 19]))
        assert_array_equal(result, [0, -1, 2, -1])

    def test_local2global(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))
        result = local.local2global(jnp.array([0, 1, 2, 3]))
        assert_array_equal(result, [2, 5, 10, 15])

    def test_local2global_out_of_bounds(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))
        result = local.local2global(jnp.array([-1, 4, 100]))
        assert_array_equal(result, [-1, -1, -1])

    def test_roundtrip(self) -> None:
        parent = _make_simple_parent(100)
        indices = np.array([3, 17, 42, 88, 99])
        local = LocalStokesLandscape(parent, indices)
        local_idx = jnp.arange(5)
        roundtripped = local.global2local(local.local2global(local_idx))
        assert_array_equal(roundtripped, local_idx)


class TestLocalLandscapeWorldQuat:
    def test_world2index_returns_local(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10]))
        theta = jnp.array([2.0, 5.0, 7.0])
        phi = jnp.zeros(3)  # unused by _SimpleLandscape
        local_idx = local.world2index(theta, phi)
        # pixel 2 → local 0, pixel 5 → local 1, pixel 7 → not in local → -1
        assert_array_equal(local_idx, [0, 1, -1])


class TestLocalLandscapeStokesConversion:
    def test_restrict(self) -> None:
        parent = _make_simple_parent(10, stokes='IQU')
        local = LocalStokesLandscape(parent, np.array([2, 5, 8]))
        global_sky = parent.full(0)
        global_sky = jax.tree.map(lambda a: a.at[2].set(10).at[5].set(20).at[8].set(30), global_sky)
        local_sky = local.restrict(global_sky)
        assert local_sky.shape == (3,)
        assert_array_equal(local_sky.i, [10, 20, 30])
        assert_array_equal(local_sky.q, [10, 20, 30])
        assert_array_equal(local_sky.u, [10, 20, 30])

    def test_promote(self) -> None:
        parent = _make_simple_parent(10, stokes='IQU')
        local = LocalStokesLandscape(parent, np.array([2, 5, 8]))
        local_sky = local.full(7)
        global_sky = local.promote(local_sky)
        assert global_sky.shape == (10,)
        for attr in ('i', 'q', 'u'):
            arr = getattr(global_sky, attr)
            assert_array_equal(arr[jnp.array([2, 5, 8])], 7)
            mask = jnp.ones(10, dtype=bool).at[jnp.array([2, 5, 8])].set(False)
            assert_array_equal(arr[mask], 0)

    def test_promote_fill_value(self) -> None:
        parent = _make_simple_parent(6, stokes='I')
        local = LocalStokesLandscape(parent, np.array([1, 4]))
        local_sky = local.full(99)
        global_sky = local.promote(local_sky, fill_value=-1)
        assert_array_equal(global_sky.i, [-1, 99, -1, -1, 99, -1])

    def test_roundtrip_restrict_promote(self) -> None:
        parent = _make_simple_parent(10, stokes='QU')
        local = LocalStokesLandscape(parent, np.array([0, 3, 7]))
        local_sky = local.ones()
        roundtripped = local.restrict(local.promote(local_sky))
        assert_array_equal(roundtripped.q, local_sky.q)
        assert_array_equal(roundtripped.u, local_sky.u)


class TestLocalLandscapeArrayCreation:
    def test_structure(self) -> None:
        parent = _make_simple_parent(20, stokes='IQU')
        local = LocalStokesLandscape(parent, np.array([1, 5, 10]))
        struct = local.structure
        assert struct.i.shape == (3,)
        assert struct.q.shape == (3,)
        assert struct.u.shape == (3,)

    def test_ones(self) -> None:
        parent = _make_simple_parent(20, stokes='I')
        local = LocalStokesLandscape(parent, np.array([1, 5, 10]))
        sky = local.ones()
        assert sky.shape == (3,)
        assert_array_equal(sky.i, 1.0)

    def test_zeros(self) -> None:
        parent = _make_simple_parent(20, stokes='QU')
        local = LocalStokesLandscape(parent, np.array([1, 5]))
        sky = local.zeros()
        assert sky.shape == (2,)
        assert_array_equal(sky.q, 0.0)
        assert_array_equal(sky.u, 0.0)

    def test_normal(self) -> None:
        parent = _make_simple_parent(20, stokes='I')
        local = LocalStokesLandscape(parent, np.array([1, 5, 10]))
        key = jax.random.PRNGKey(0)
        sky = local.normal(key)
        assert sky.shape == (3,)

    def test_uniform(self) -> None:
        parent = _make_simple_parent(20, stokes='I')
        local = LocalStokesLandscape(parent, np.array([1, 5, 10]))
        key = jax.random.PRNGKey(0)
        sky = local.uniform(key, low=-1.0, high=1.0)
        assert sky.shape == (3,)


class TestLocalLandscapeCoverage:
    def test_get_coverage(self) -> None:
        parent = _make_simple_parent(10, stokes='I')
        local = LocalStokesLandscape(parent, np.array([0, 1, 3, 7]))
        sampling = Sampling(
            jnp.array([0.0, 1, 0, 3, 3, 7, 5]),
            jnp.zeros(7),
            jnp.array(0.0),
        )
        coverage = local.get_coverage(sampling)
        # pixel 0 hit 2x, pixel 1 hit 1x, pixel 3 hit 2x, pixel 7 hit 1x
        # pixel 5 is not in local → excluded
        assert_array_equal(coverage, [2, 1, 2, 1])

    def test_get_coverage_no_hits(self) -> None:
        parent = _make_simple_parent(10, stokes='I')
        local = LocalStokesLandscape(parent, np.array([2, 4, 6]))
        sampling = Sampling(
            jnp.array([0.0, 1.0, 3.0]),
            jnp.zeros(3),
            jnp.array(0.0),
        )
        coverage = local.get_coverage(sampling)
        assert_array_equal(coverage, [0, 0, 0])


class TestLocalLandscapeConstructors:
    def test_from_boolean_mask(self) -> None:
        parent = _make_simple_parent(10, stokes='I')
        mask = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
        local = LocalStokesLandscape.from_boolean_mask(parent, mask)
        assert_array_equal(local.global_indices, [1, 3, 5])
        assert local.nlocal == 3

    def test_from_sampling(self) -> None:
        parent = _make_simple_parent(10, stokes='I')
        sampling = Sampling(
            jnp.array([0.0, 3.0, 3.0, 7.0, 0.0]),
            jnp.zeros(5),
            jnp.array(0.0),
        )
        local = LocalStokesLandscape.from_sampling(parent, sampling)
        assert_array_equal(local.global_indices, [0, 3, 7])
        assert local.nlocal == 3


class TestLocalLandscapeJIT:
    def test_global2local_jit(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))

        @jax.jit
        def f(ll, idx):
            return ll.global2local(idx)

        result = f(local, jnp.array([2, 7, 15]))
        assert_array_equal(result, [0, -1, 3])

    def test_local2global_jit(self) -> None:
        parent = _make_simple_parent(20)
        local = LocalStokesLandscape(parent, np.array([2, 5, 10, 15]))

        @jax.jit
        def f(ll, idx):
            return ll.local2global(idx)

        result = f(local, jnp.array([0, 2, 3]))
        assert_array_equal(result, [2, 10, 15])
