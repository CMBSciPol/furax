import equinox
import jax.numpy as jnp

from astrosim.landscapes import StokesIPyTree, StokesIQUPyTree, StokesPyTree
from astrosim.operators import IdentityOperator
from astrosim.operators.qu_rotations import QURotationOperator


def test_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(2, 5), stokes='I', angles=pa)
    x = StokesIPyTree(I=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    actual_y = hwp(x)

    expected_y = x
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_iqu() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(5,), stokes='IQU', angles=pa)
    x = StokesIQUPyTree(
        I=jnp.array([1.0, 2, 3, 4, 5]),
        Q=jnp.array([1.0, 1, 1, 1, 1]),
        U=jnp.array([2.0, 2, 2, 2, 2]),
    )

    actual_y = hwp(x)

    expected_y = StokesIQUPyTree(
        I=x.I,
        Q=jnp.array([1.0, -2, -1, 2, 1]),
        U=jnp.array([2.0, 1, -2, -1, 2]),
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_orthogonal(stokes) -> None:
    hwp = QURotationOperator.create(shape=(), stokes=stokes, angles=1.1)
    cls = StokesPyTree.class_for(stokes)
    x = cls.ones(())
    y = hwp.T(hwp(x))
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)


def test_matmul(stokes) -> None:
    hwp = QURotationOperator.create(shape=(), stokes=stokes, angles=1.1)
    assert isinstance(hwp @ hwp.T, IdentityOperator)
    assert isinstance(hwp.T @ hwp, IdentityOperator)


def test_simple() -> None:
    theta = jnp.arange(10)
    theta = 1
    phi = theta + 1
    mat = jnp.array(
        [
            [1, 0, 0, 0],
            [0, jnp.cos(4 * theta), jnp.sin(4 * theta), 0],
            [0, jnp.sin(4 * theta), -jnp.cos(4 * theta), 0],
            [0, 0, 0, -1],
        ]
    )

    def R(theta):
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, jnp.cos(theta), jnp.sin(theta), 0],
                [0, -jnp.sin(theta), jnp.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

    L = jnp.array(
        [
            [1, 1, 0, 0],
        ]
    )
    HWP = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
        ]
    )
    alpha = 0.5
    beta = 1.0
    gamma = 1.5
    H = L @ R(alpha) @ R(-beta) @ HWP @ R(beta) @ R(gamma)
    H_test = L @ HWP @ R(-alpha + 2 * beta + gamma)

    HTH = H.T @ H

    c = jnp.cos(-alpha + 2 * beta + gamma)
    s = jnp.sin(-alpha + 2 * beta + gamma)
    HTH_test = jnp.array(
        [
            [1, c, s, 0],
            [c, c**2, s * c, 0],
            [s, s * c, s**2, 0],
            [0, 0, 0, 0],
        ]
    )
