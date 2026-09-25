import equinox
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import furax as fx
from furax import IdentityOperator
from furax.core import CompositionOperator
from furax.obs import QURotationOperator
from furax.obs.operators import Spin2Rotation
from furax.obs.stokes import Stokes, StokesI, StokesIQU, ValidStokesLiteral


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
        q=jnp.array([1.0, 2, -1, -2, 1]),
        u=jnp.array([2.0, -1, -2, 1, 2]),
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_orthogonal(stokes: ValidStokesLiteral) -> None:
    hwp = QURotationOperator.create(shape=(), stokes=stokes, angles=1.1)
    x = fx.tree.ones_like(hwp.out_structure)
    y = hwp.T(hwp(x))
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)


def test_matmul(stokes: ValidStokesLiteral) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    hwp = QURotationOperator(1.1, in_structure=structure)
    assert isinstance(hwp @ hwp.T, IdentityOperator)
    assert isinstance(hwp.T @ hwp, IdentityOperator)


@pytest.mark.parametrize(
    'transpose_left, transpose_right, expected_value',
    [(False, False, 3), (False, True, -1), (True, False, 1), (True, True, -3)],
)
def test_rules(stokes: ValidStokesLiteral, transpose_left, transpose_right, expected_value) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    left = QURotationOperator(1, in_structure=structure)
    if transpose_left:
        left = left.T
    right = QURotationOperator(2, in_structure=structure)
    if transpose_right:
        right = right.T
    reduced_op = (left @ right).reduce()

    assert isinstance(reduced_op, QURotationOperator)
    assert reduced_op.angles == expected_value


@pytest.mark.parametrize(
    'atomic_left, atomic_right',
    [(True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize(
    'transpose_left, transpose_right',
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_atomic_prevents_reduction(
    stokes: ValidStokesLiteral,
    atomic_left: bool,
    atomic_right: bool,
    transpose_left: bool,
    transpose_right: bool,
) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    left = QURotationOperator(1, atomic=atomic_left, in_structure=structure)
    if transpose_left:
        left = left.T
    right = QURotationOperator(2, atomic=atomic_right, in_structure=structure)
    if transpose_right:
        right = right.T
    reduced_op = (left @ right).reduce()

    assert isinstance(reduced_op, CompositionOperator) and len(reduced_op.operands) == 2


def test_atomic_identity_still_reduces(stokes: ValidStokesLiteral) -> None:
    structure = Stokes.class_for(stokes).structure_for(())
    op = QURotationOperator(1.1, atomic=True, in_structure=structure)

    assert isinstance((op @ op.T).reduce(), IdentityOperator)
    assert isinstance((op.T @ op).reduce(), IdentityOperator)


# numpy, so that it is float64 once the autouse fixture enables x64
ANGLES = np.array([-2.0, -0.3, 0.0, 0.4, 1.2, 3.0])


def _assert_rotation_allclose(actual: Spin2Rotation, expected: Spin2Rotation) -> None:
    assert_allclose(actual.cos_2angles, expected.cos_2angles, atol=1e-14)
    assert_allclose(actual.sin_2angles, expected.sin_2angles, atol=1e-14)


def test_from_angles_rotates_like_qu_rotation_operator() -> None:
    x = StokesIQU.normal(jax.random.key(0), ANGLES.shape)
    op = QURotationOperator.create(ANGLES.shape, angles=ANGLES)
    rotated = x.rotate_qu(*Spin2Rotation.from_angles(ANGLES))
    assert_allclose(rotated.data, op(x).data, atol=1e-14)


def test_from_cos_sin_doubles_the_angle() -> None:
    rotation = Spin2Rotation.from_cos_sin(jnp.cos(ANGLES), jnp.sin(ANGLES))
    _assert_rotation_allclose(rotation, Spin2Rotation.from_angles(ANGLES))


def test_compose_adds_the_angles() -> None:
    first = Spin2Rotation.from_angles(ANGLES)
    then = Spin2Rotation.from_angles(ANGLES[::-1])
    _assert_rotation_allclose(first.compose(then), Spin2Rotation.from_angles(ANGLES + ANGLES[::-1]))


def test_inverse_undoes_the_rotation() -> None:
    rotation = Spin2Rotation.from_angles(ANGLES)
    _assert_rotation_allclose(rotation.inverse(), Spin2Rotation.from_angles(-ANGLES))
    identity = rotation.compose(rotation.inverse())
    _assert_rotation_allclose(identity, Spin2Rotation.from_angles(jnp.zeros_like(ANGLES)))


@pytest.mark.parametrize('index', [np.array([4, 0]), (slice(None), None)])
def test_indexing_indexes_both_arrays(index) -> None:
    rotation = Spin2Rotation.from_angles(ANGLES)
    _assert_rotation_allclose(rotation[index], Spin2Rotation.from_angles(ANGLES[index]))


def test_broadcast_to() -> None:
    rotation = Spin2Rotation.from_angles(ANGLES[:, None]).broadcast_to((ANGLES.size, 3))
    assert rotation.cos_2angles.shape == rotation.sin_2angles.shape == (ANGLES.size, 3)


def test_is_a_pytree() -> None:
    rotation = Spin2Rotation.from_angles(ANGLES)
    doubled = jax.jit(lambda r: r.compose(r))(rotation)
    assert isinstance(doubled, Spin2Rotation)
    _assert_rotation_allclose(doubled, Spin2Rotation.from_angles(2 * ANGLES))
