import equinox
import jax.numpy as jnp

from furax.core import CompositionOperator
from furax.obs import HWPOperator, QURotationOperator
from furax.obs.operators._qu_rotations import QURotationTransposeOperator
from furax.obs.stokes import Stokes, StokesI, StokesIQUV, ValidStokesType


def test_hwp(stokes: ValidStokesType) -> None:
    hwp = HWPOperator.create(shape=(), stokes=stokes)
    cls = Stokes.class_for(stokes)
    x = cls.ones(())
    actual_y = hwp(x)

    one = jnp.array(1.0)
    expected_y = cls.from_iquv(one, one, -one, -one)
    assert equinox.tree_equal(actual_y, expected_y)


def test_hwp_orthogonal(stokes: ValidStokesType) -> None:
    hwp = HWPOperator.create(shape=(), stokes=stokes)
    x = Stokes.class_for(stokes).ones(())
    y = (hwp.T @ hwp)(x)
    assert equinox.tree_equal(y, x)


def test_qu_rotation_hwp_rule(stokes: ValidStokesType) -> None:
    in_structure = Stokes.class_for(stokes).structure_for(())
    hwp = HWPOperator(in_structure=in_structure)
    rot = QURotationOperator(jnp.array(1.0), in_structure=in_structure)
    reduced_op = (rot @ hwp).reduce()
    assert isinstance(reduced_op, CompositionOperator)
    assert reduced_op.operands[0] is hwp
    assert isinstance(reduced_op.operands[1], QURotationTransposeOperator)
    assert reduced_op.operands[1].operator is rot

    reduced_op = (rot.T @ hwp).reduce()
    assert isinstance(reduced_op, CompositionOperator)
    assert reduced_op.operands[0] is hwp
    assert reduced_op.operands[1] is rot


def test_rotating_hwp_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180])) / 2
    hwp = HWPOperator.create(shape=(2, 5), stokes='I', angles=pa)
    x = StokesI(i=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    actual_y = hwp(x)

    expected_y = x
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_rotating_hwp_iquv() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180])) / 2
    hwp = HWPOperator.create(shape=(5,), stokes='IQUV', angles=pa)
    x = StokesIQUV(
        i=jnp.array([1.0, 2, 3, 4, 5]),
        q=jnp.array([1.0, 1, 1, 1, 1]),
        u=jnp.array([2.0, 2, 2, 2, 2]),
        v=jnp.array([1.0, 5, 4, 3, 2]),
    )

    actual_y = hwp(x)

    expected_y = StokesIQUV(
        i=x.i,
        q=jnp.array([1.0, -2, -1, 2, 1]),
        u=jnp.array([-2.0, -1, 2, 1, -2]),
        v=-x.v,
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_rotating_hwp_orthogonal(stokes: ValidStokesType) -> None:
    hwp = HWPOperator.create(shape=(), stokes=stokes, angles=1.1)
    x = Stokes.class_for(stokes).ones(())
    y = (hwp.T @ hwp)(x)
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)
