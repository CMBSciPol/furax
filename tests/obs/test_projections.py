import equinox
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal

from furax import DiagonalOperator, IndexOperator
from furax.obs.landscapes import HealpixLandscape, StokesLandscape
from furax.obs.stokes import Stokes


def test_direct(stokes) -> None:
    nside = 1
    landscape = HealpixLandscape(nside, stokes)
    x = Stokes.from_stokes(
        *(jnp.arange(12, dtype=landscape.dtype) * (i + 1) for i, stoke in enumerate(stokes))
    )
    indices = jnp.array([[2, 3, 2]])
    proj = IndexOperator(indices, in_structure=landscape.structure)

    y = proj(x)

    expected_y = Stokes.from_stokes(*(getattr(x, stoke.lower())[indices] for stoke in stokes))
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)


def test_transpose(stokes) -> None:
    nside = 1
    landscape = HealpixLandscape(nside, stokes)
    x = Stokes.from_stokes(
        *(jnp.array([[1, 2, 3]], dtype=landscape.dtype) * (i + 1) for i, stoke in enumerate(stokes))
    )
    indices = jnp.array([[2, 3, 2]])
    proj = IndexOperator(indices, in_structure=landscape.structure)

    y = proj.T(x)

    array = jnp.array([0.0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_y = Stokes.from_stokes(*[array * i for i in range(1, len(stokes) + 1)])
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)


def test_ptp(stokes) -> None:
    class MyStokesLandscape(StokesLandscape):
        def world2pixel(self, theta, phi):
            return phi.astype(np.int32)

    landscape = MyStokesLandscape((4,), stokes)
    indices = jnp.array([[0, 1, 0, 2, 3], [1, 0, 1, 1, 1]])
    op = IndexOperator(indices, in_structure=landscape.structure)

    product = (op.T @ op).reduce()
    assert isinstance(product, DiagonalOperator)

    actual_dense = product.as_matrix()

    dense = op.as_matrix()
    expected_dense = dense.T @ dense
    assert_array_equal(actual_dense, expected_dense)
