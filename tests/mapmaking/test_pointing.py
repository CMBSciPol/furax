import jax
import jax.numpy as jnp
import pytest
from equinox import tree_equal
from jax.tree_util import register_static

from furax.mapmaking.pointing import PointingOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape
from furax.obs.stokes import ValidStokesType

NSIDE = 4
NDET, NSAMP = 3, 10


@register_static
class CARStokesLandscape(StokesLandscape):
    """Simple 2D grid landscape covering the full sphere.

    Maps (theta, phi) to (col, row) pixel coordinates where:
        col = phi / (2*pi) * ncol  in [0, ncol)
        row = theta / pi * nrow    in [0, nrow)
    """

    def world2pixel(self, theta, phi):
        nrow, ncol = self.shape
        col = phi / (2 * jnp.pi) * ncol - 0.5
        row = theta / jnp.pi * nrow - 0.5
        return col, row


def _make_landscape(landscape_type: str, stokes: ValidStokesType) -> StokesLandscape:
    dtype = jnp.float64
    if landscape_type == 'healpix':
        return HealpixLandscape(NSIDE, stokes, dtype)
    return CARStokesLandscape((5, 3), stokes, dtype)


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Generate random unit quaternions."""
    q = jax.random.normal(key, (*shape, 4), dtype=jnp.float64)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
def test_as_expanded_operator_mv(stokes, frame, landscape_type) -> None:
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
    assert tree_equal(tod_direct, tod_expanded, rtol=1e-12)


@pytest.mark.parametrize('landscape_type', ['healpix', 'car'])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
def test_as_expanded_operator_transpose_mv(stokes, frame, landscape_type) -> None:
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
    assert tree_equal(sky_direct, sky_expanded, rtol=1e-12)
