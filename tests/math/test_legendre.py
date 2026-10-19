import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.math.legendre import legvander


@pytest.mark.parametrize('deg', [0, 3, 50, 200])
def test_matches_numpy_up_to_high_orders(deg):
    x = jnp.linspace(-1.0, 1.0, 513)
    expected = np.polynomial.legendre.legvander(np.asarray(x), deg)
    assert_allclose(legvander(x, deg), expected, atol=1e-12)


def test_points_of_any_shape_keep_their_shape():
    x = jr.uniform(jr.key(0), (3, 4), minval=-1, maxval=1)
    assert legvander(x, 4).shape == (3, 4, 5)
    assert_allclose(legvander(x, 4)[1, 2], legvander(x[1, 2], 4), atol=1e-12)


def test_polynomials_equal_one_at_the_end_point_and_alternate_at_the_other():
    orders = jnp.arange(8)
    assert_allclose(legvander(jnp.array(1.0), 7), 1.0)
    assert_allclose(legvander(jnp.array(-1.0), 7), (-1.0) ** orders)
