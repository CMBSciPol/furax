import equinox
import jax.numpy as jnp

from astrosim.landscapes import StokesIPyTree, StokesIQUPyTree
from astrosim.operators.hwp import HWPOperator


def test_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = HWPOperator(shape=(2, 5), stokes='I', pa=pa)
    x = StokesIPyTree(I=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    y = hwp.mv(x)

    expected_y = x
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)


def test_iqu() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = HWPOperator(shape=(5,), stokes='IQU', pa=pa)
    x = StokesIQUPyTree(
        I=jnp.array([1.0, 2, 3, 4, 5]),
        Q=jnp.array([1.0, 1, 1, 1, 1]),
        U=jnp.array([2.0, 2, 2, 2, 2]),
    )

    y = hwp.mv(x)

    expected_y = StokesIQUPyTree(
        I=x.I, Q=jnp.array([1.0, 0, -1, 0, 1]), U=jnp.array([0.0, 2, 0, -2, 0])
    )
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)


def test_symmetric() -> None:
    hwp = HWPOperator(shape=(5,), stokes='IQUV', pa=jnp.zeros(5))
    assert hwp.T is hwp
