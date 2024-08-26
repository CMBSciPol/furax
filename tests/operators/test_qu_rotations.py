import equinox
import jax.numpy as jnp
import pytest

from furax.landscapes import StokesIPyTree, StokesIQUPyTree, StokesPyTree, ValidStokesType
from furax.operators import IdentityOperator
from furax.operators.qu_rotations import QURotationOperator


def test_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(2, 5), stokes='I', angles=pa)
    x = StokesIPyTree(i=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    actual_y = hwp(x)

    expected_y = x
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_iqu() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180]))
    hwp = QURotationOperator.create(shape=(5,), stokes='IQU', angles=pa)
    x = StokesIQUPyTree(
        i=jnp.array([1.0, 2, 3, 4, 5]),
        q=jnp.array([1.0, 1, 1, 1, 1]),
        u=jnp.array([2.0, 2, 2, 2, 2]),
    )

    actual_y = hwp(x)

    expected_y = StokesIQUPyTree(
        i=x.i,
        q=jnp.array([1.0, -2, -1, 2, 1]),
        u=jnp.array([2.0, 1, -2, -1, 2]),
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_orthogonal(stokes) -> None:
    hwp = QURotationOperator.create(shape=(), stokes=stokes, angles=1.1)
    cls = StokesPyTree.class_for(stokes)
    x = cls.ones(())
    y = hwp.T(hwp(x))
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)


def test_matmul(stokes) -> None:
    structure = StokesPyTree.structure_for((), stokes=stokes)
    hwp = QURotationOperator(1.1, structure)
    assert isinstance(hwp @ hwp.T, IdentityOperator)
    assert isinstance(hwp.T @ hwp, IdentityOperator)


@pytest.mark.parametrize(
    'transpose_left, transpose_right, expected_value',
    [(False, False, 3), (False, True, -1), (True, False, 1), (True, True, -3)],
)
def test_rules(stokes: ValidStokesType, transpose_left, transpose_right, expected_value) -> None:
    structure = StokesPyTree.structure_for((), stokes=stokes)
    left = QURotationOperator(1, structure)
    if transpose_left:
        left = left.T
    right = QURotationOperator(2, structure)
    if transpose_right:
        right = right.T
    reduced_op = (left @ right).reduce()

    assert isinstance(reduced_op, QURotationOperator)
    assert reduced_op.angles == expected_value
