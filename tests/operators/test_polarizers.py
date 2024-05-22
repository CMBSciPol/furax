import equinox
import jax.numpy as jnp
import numpy as np

from astrosim.landscapes import StokesIPyTree, StokesIQUPyTree, stokes_pytree_cls
from astrosim.operators.polarizers import LinearPolarizerOperator


def test_direct_i() -> None:
    polarizer = LinearPolarizerOperator(shape=(2, 5), stokes='I')
    x = StokesIPyTree(I=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    y = polarizer(x)

    expected_y = x.I / 2
    assert jnp.allclose(y, expected_y, atol=1e-15, rtol=1e-15)


def test_direct_iqu() -> None:
    theta = np.deg2rad(15)
    polarizer = LinearPolarizerOperator(shape=(2, 5), stokes='IQU', theta=theta)
    x = StokesIQUPyTree(
        I=jnp.array([1.0, 2, 3, 4, 5]),
        Q=jnp.array([1.0, 1, 1, 1, 1]),
        U=jnp.array([2.0, 2, 2, 2, 2]),
    )

    y = polarizer(x)

    expected_y = 0.5 * (x.I + np.cos(2 * theta) * x.Q + np.sin(2 * theta) * x.U)
    assert jnp.allclose(y, expected_y, atol=1e-15, rtol=1e-15)


def test_transpose(stokes) -> None:
    theta = np.deg2rad(15)
    polarizer = LinearPolarizerOperator(shape=(2, 5), stokes=stokes, theta=theta)
    x = jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]])

    y = polarizer.T(x)

    expected_cls = stokes_pytree_cls(stokes)
    assert isinstance(y, expected_cls)
    expected_y = expected_cls.from_iquv(
        0.5 * x, 0.5 * np.cos(2 * theta) * x, 0.5 * np.sin(2 * theta) * x, np.zeros_like(x)
    )
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)
