import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal

import furax.tree as ftree
from furax.core import AbstractLinearOperator, CompositionOperator
from furax.obs.landscapes import (
    CARLandscape,
    HealpixLandscape,
    LocalStokesLandscape,
    StokesLandscape,
    WCSProjection,
)
from furax.obs.operators import QURotationOperator
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import ValidStokesLiteral

NSIDE = 4
NDET, NSAMP = 3, 10

# Full-sphere CAR landscape (180×360 at 1°/pixel, crval at RA=180° to avoid wrap issues)
_CAR_PROJECTION = WCSProjection(crpix=(180.5, 90.5), crval=(180.0, 0.0), cdelt=(-1.0, 1.0))


def _make_landscape(landscape_type: str, stokes: ValidStokesLiteral) -> StokesLandscape:
    if landscape_type == 'healpix':
        return HealpixLandscape(NSIDE, stokes)
    return CARLandscape((180, 360), _CAR_PROJECTION, stokes)


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Generate random unit quaternions."""
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
class TestAsExpandedOperator:
    def test_mv(self, stokes, frame, landscape_type) -> None:
        """PointingOperator.mv is equivalent to as_expanded_operator().mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, batch_size=2)
        sky = landscape.normal(key3)

        tod_direct = pointing_op(sky)
        tod_expanded = pointing_op.as_expanded_operator()(sky)

        assert tree_equal(tod_direct, tod_expanded, rtol=1e-10, atol=0)

    def test_transpose_mv(self, stokes, frame, landscape_type) -> None:
        """PointingOperator.T.mv is equivalent to as_expanded_operator().T.mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, batch_size=2)
        tod = pointing_op.out_structure
        tod = jax.tree.map(lambda s: jax.random.normal(key3, s.shape, s.dtype), tod)

        sky_direct = pointing_op.T(tod)
        sky_expanded = pointing_op.as_expanded_operator().T(tod)

        assert tree_equal(sky_direct, sky_expanded, rtol=1e-10, atol=0)

    def test_mv_interpolate(self, stokes, frame, landscape_type) -> None:
        """Interpolated PointingOperator.mv equals as_expanded_operator().mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

        pointing_op = PointingOperator.create(
            landscape, qbore, qdet, frame=frame, batch_size=2, interpolate=True
        )
        sky = landscape.normal(key3)

        assert tree_equal(pointing_op(sky), pointing_op.as_expanded_operator()(sky), rtol=1e-10)

    def test_transpose_mv_interpolate(self, stokes, frame, landscape_type) -> None:
        """Interpolated PointingOperator.T.mv equals as_expanded_operator().T.mv."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        qbore = _random_unit_quats(key1, (NSAMP,))
        qdet = _random_unit_quats(key2, (NDET,))

        pointing_op = PointingOperator.create(
            landscape, qbore, qdet, frame=frame, batch_size=2, interpolate=True
        )
        tod = jax.tree.map(
            lambda s: jax.random.normal(key3, s.shape, s.dtype), pointing_op.out_structure
        )

        sky_direct = pointing_op.T(tod)
        sky_expanded = pointing_op.as_expanded_operator().T(tod)

        assert tree_equal(sky_direct, sky_expanded, rtol=1e-10)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
class TestInterpolate:
    def test_adjoint(self, stokes, landscape_type) -> None:
        """<P x, y> = <x, P^T y> for the interpolated operator."""
        landscape = _make_landscape(landscape_type, stokes)

        key = jax.random.PRNGKey(7)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        qbore = _random_unit_quats(k1, (NSAMP,))
        qdet = _random_unit_quats(k2, (NDET,))

        op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        sky = landscape.normal(k3)
        tod = jax.tree.map(lambda s: jax.random.normal(k4, s.shape, s.dtype), op.out_structure)

        lhs = ftree.dot(op(sky), tod)
        rhs = ftree.dot(sky, op.T(tod))
        assert_array_almost_equal(lhs, rhs, decimal=10)

    def test_uniform_sky(self, landscape_type) -> None:
        """Sampling a uniform sky returns the same constant (weights always sum to 1)."""
        landscape = _make_landscape(landscape_type, 'I')

        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        qbore = _random_unit_quats(k1, (NSAMP,))
        qdet = _random_unit_quats(k2, (NDET,))

        op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
        tod = op(landscape.full(3.14))
        assert_array_almost_equal(tod.i, 3.14, decimal=10)


def test_expanded_interpolate_preserves_rotation_fusion() -> None:
    """The expanded interpolated operator keeps QURotation exposed so it fuses via algebra.

    An outer QURotation composed with `QURot(pa) @ XSampling @ Ravel` must reduce to a single
    QURotation (angles added), leaving one rotation in the chain rather than two.
    """
    landscape = _make_landscape('car', 'IQU')
    key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(3), 4)
    qbore = _random_unit_quats(key1, (NSAMP,))
    qdet = _random_unit_quats(key2, (NDET,))

    op = PointingOperator.create(landscape, qbore, qdet, interpolate=True)
    expanded = op.as_expanded_operator()
    gamma = jax.random.normal(key3, (NDET, NSAMP))
    outer = QURotationOperator(angles=gamma, in_structure=expanded.out_structure)

    reduced = (outer @ expanded).reduce()

    assert isinstance(reduced, CompositionOperator)
    leaves = jax.tree.leaves(
        reduced.operands, is_leaf=lambda x: isinstance(x, AbstractLinearOperator)
    )
    n_rotations = sum(isinstance(o, QURotationOperator) for o in leaves)
    assert n_rotations == 1

    tod = jax.tree.map(lambda s: jax.random.normal(key4, s.shape, s.dtype), expanded.out_structure)
    assert tree_equal((outer @ expanded)(op.T(tod)), reduced(op.T(tod)), rtol=1e-10)


class TestLocalLandscape:
    """A PointingOperator on a LocalStokesLandscape matches the full-sky one."""

    @pytest.fixture(scope='class')
    def keys(self) -> jax.Array:
        return jax.random.split(jax.random.key(11), 4)

    @pytest.fixture(scope='class', params=[False, True], ids=['nearest', 'bilinear'])
    def p_full(self, request: pytest.FixtureRequest, keys: jax.Array) -> PointingOperator:
        parent = HealpixLandscape(NSIDE, 'IQU')
        qbore = _random_unit_quats(keys[0], (NSAMP,))
        qdet = _random_unit_quats(keys[1], (NDET,))
        return PointingOperator.create(parent, qbore, qdet, interpolate=request.param)

    @pytest.fixture(scope='class')
    def sky(self, p_full: PointingOperator, keys: jax.Array):
        return p_full.landscape.normal(keys[2])

    @pytest.fixture(scope='class')
    def tod(self, p_full: PointingOperator, keys: jax.Array):
        return ftree.normal_like(p_full.out_structure, keys[3])

    @pytest.fixture(scope='class')
    def covered(self, p_full: PointingOperator) -> jax.Array:
        # global pixels hit by the pointing: bin a ones-TOD (I accumulates 1 per hit for
        # nearest, the interpolation weights for bilinear)
        hits = p_full.T(ftree.ones_like(p_full.out_structure))
        return jnp.flatnonzero(hits.i)

    @staticmethod
    def _local_operator(
        p_full: PointingOperator, indices: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        local = LocalStokesLandscape(p_full.landscape, indices)
        p_local = PointingOperator.create(
            local, p_full.qbore, p_full.qdet, interpolate=p_full.interpolate
        )
        return local, p_local

    @pytest.fixture(scope='class')
    def full_coverage(
        self, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        return self._local_operator(p_full, covered)

    @pytest.fixture(scope='class')
    def half_coverage(
        self, p_full: PointingOperator, covered: jax.Array
    ) -> tuple[LocalStokesLandscape, PointingOperator]:
        # samples on the dropped pixels land in the sink
        return self._local_operator(p_full, covered[::2])

    def test_mv_matches_full_sky(self, p_full, sky, full_coverage) -> None:
        local, p_local = full_coverage
        assert tree_equal(p_local(local.restrict(sky)), p_full(sky), rtol=1e-13)

    def test_transpose_matches_full_sky(self, p_full, tod, full_coverage) -> None:
        local, p_local = full_coverage
        # uncovered pixels are zero in both the full binning and the promoted local one
        assert tree_equal(local.promote(p_local.T(tod)), p_full.T(tod), rtol=1e-13)

    def test_partial_coverage_sinks_missing_pixels(self, p_full, tod, half_coverage) -> None:
        local, p_local = half_coverage
        subset = local.global_indices
        binned_local = local.promote(p_local.T(tod)).data
        if not p_full.interpolate:
            # nearest: contributions to kept pixels are identical, sink ones are discarded
            # (bilinear renormalizes the weights over covered neighbors, so values differ)
            binned_full = p_full.T(tod).data
            assert_array_almost_equal(binned_local[:, subset], binned_full[:, subset], decimal=13)
        # nothing lands outside the subset
        assert_array_equal(binned_local.at[:, subset].set(0.0), 0.0)
