import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.mapmaking.pointing import PointingOperator
from furax.obs.landscapes import HealpixLandscape

NSIDE = 4
NDET, NSAMP = 3, 10


def _random_unit_quats(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Generate random unit quaternions."""
    q = jax.random.normal(key, (*shape, 4))
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


@pytest.mark.parametrize('flip', [False, True])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
def test_as_expanded_operator_mv(stokes, flip, frame) -> None:
    """PointingOperator.mv is equivalent to as_expanded_operator().mv."""
    landscape = HealpixLandscape(NSIDE, stokes)

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    qbore = _random_unit_quats(key1, (NSAMP,))
    qdet = _random_unit_quats(key2, (NDET,))

    pointing_op = PointingOperator.create(
        landscape, qbore, qdet, flip=flip, frame=frame, chunk_size=2
    )
    sky = landscape.normal(key3)

    tod_direct = pointing_op(sky)
    tod_expanded = pointing_op.as_expanded_operator()(sky)

    for leaf_direct, leaf_expanded in zip(
        jax.tree.leaves(tod_direct), jax.tree.leaves(tod_expanded)
    ):
        assert_allclose(np.asarray(leaf_direct), np.asarray(leaf_expanded), rtol=1e-10)


@pytest.mark.parametrize('flip', [False, True])
@pytest.mark.parametrize('frame', ['boresight', 'detector'])
def test_as_expanded_operator_transpose_mv(stokes, flip, frame) -> None:
    """PointingOperator.T.mv is equivalent to as_expanded_operator().T.mv."""
    landscape = HealpixLandscape(NSIDE, stokes)

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)
    qbore = _random_unit_quats(key1, (NSAMP,))
    qdet = _random_unit_quats(key2, (NDET,))

    pointing_op = PointingOperator.create(
        landscape, qbore, qdet, flip=flip, frame=frame, chunk_size=2
    )
    tod = pointing_op.out_structure
    tod = jax.tree.map(lambda s: jax.random.normal(key3, s.shape, s.dtype), tod)

    sky_direct = pointing_op.T(tod)
    sky_expanded = pointing_op.as_expanded_operator().T(tod)

    for leaf_direct, leaf_expanded in zip(
        jax.tree.leaves(sky_direct), jax.tree.leaves(sky_expanded)
    ):
        assert_allclose(np.asarray(leaf_direct), np.asarray(leaf_expanded), rtol=1e-10)
