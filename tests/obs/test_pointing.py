import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from numpy.testing import assert_array_almost_equal

import furax.tree as ftree
from furax.obs.landscapes import CARLandscape, HealpixLandscape, StokesLandscape, WCSProjection
from furax.obs.pointing import PointingOperator
from furax.obs.stokes import ValidStokesType

NSIDE = 4
NDET, NSAMP = 3, 10

# Full-sphere CAR landscape (180×360 at 1°/pixel, crval at RA=180° to avoid wrap issues)
_CAR_PROJECTION = WCSProjection(crpix=(180.5, 90.5), crval=(180.0, 0.0), cdelt=(-1.0, 1.0))


def _make_landscape(landscape_type: str, stokes: ValidStokesType) -> StokesLandscape:
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

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, chunk_size=2)
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

        pointing_op = PointingOperator.create(landscape, qbore, qdet, frame=frame, chunk_size=2)
        tod = pointing_op.out_structure
        tod = jax.tree.map(lambda s: jax.random.normal(key3, s.shape, s.dtype), tod)

        sky_direct = pointing_op.T(tod)
        sky_expanded = pointing_op.as_expanded_operator().T(tod)

        assert tree_equal(sky_direct, sky_expanded, rtol=1e-10, atol=0)


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
