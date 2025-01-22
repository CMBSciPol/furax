import equinox
import jax.numpy as jnp
import pytest

import furax as fx
from furax import IdentityOperator
from furax.obs import QURotationOperator
from furax.obs.stokes import Stokes, StokesI, StokesIQU, ValidStokesType


def test_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(2, 5), stokes='I', angles=pa)
    x = StokesI(i=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    actual_y = hwp(x)

    expected_y = x
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_iqu() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(5,), stokes='IQU', angles=pa)
    x = StokesIQU(
        i=jnp.array([1.0, 2, 3, 4, 5]),
        q=jnp.array([1.0, 1, 1, 1, 1]),
        u=jnp.array([2.0, 2, 2, 2, 2]),
    )

    actual_y = hwp(x)

    expected_y = StokesIQU(
        i=x.i,
        q=jnp.array([1.0, -2, -1, 2, 1]),
        u=jnp.array([2.0, 1, -2, -1, 2]),
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_orthogonal(stokes: ValidStokesType) -> None:
    hwp = QURotationOperator.create(shape=(), stokes=stokes, angles=1.1)
    x = fx.tree.ones_like(hwp.out_structure())
    y = hwp.T(hwp(x))
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)


def test_matmul(stokes: ValidStokesType) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    hwp = QURotationOperator(1.1, structure)
    assert isinstance(hwp @ hwp.T, IdentityOperator)
    assert isinstance(hwp.T @ hwp, IdentityOperator)


@pytest.mark.parametrize(
    'transpose_left, transpose_right, expected_value',
    [(False, False, 3), (False, True, -1), (True, False, 1), (True, True, -3)],
)
def test_rules(stokes: ValidStokesType, transpose_left, transpose_right, expected_value) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    left = QURotationOperator(1, structure)
    if transpose_left:
        left = left.T
    right = QURotationOperator(2, structure)
    if transpose_right:
        right = right.T
    reduced_op = (left @ right).reduce()

    assert isinstance(reduced_op, QURotationOperator)
    assert reduced_op.angles == expected_value
