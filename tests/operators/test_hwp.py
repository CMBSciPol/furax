import equinox
import jax.numpy as jnp

from astrosim.landscapes import StokesIPyTree, StokesIQUVPyTree, StokesPyTree, ValidStokesType
from astrosim.operators.hwp import HWPOperator, RotatingHWPOperator


def test_hwp(stokes: ValidStokesType) -> None:
    hwp = HWPOperator.create(shape=(), stokes=stokes)
    cls = StokesPyTree.class_for(stokes)
    x = cls.ones(())
    actual_y = hwp(x)

    one = jnp.array(1.0)
    expected_y = cls.from_iquv(one, one, -one, -one)
    assert equinox.tree_equal(actual_y, expected_y)


def test_hwp_orthogonal(stokes: ValidStokesType) -> None:
    hwp = HWPOperator.create(shape=(), stokes=stokes)
    x = StokesPyTree.class_for(stokes).ones(())
    y = (hwp.T @ hwp)(x)
    assert equinox.tree_equal(y, x)


def test_rotating_hwp_i() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180])) / 2
    hwp = RotatingHWPOperator.create(shape=(2, 5), stokes='I', angles=pa)
    x = StokesIPyTree(I=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    actual_y = hwp(x)

    expected_y = x
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_rotating_hwp_iquv() -> None:
    pa = jnp.deg2rad(jnp.array([0, 45, 90, 135, 180])) / 2
    hwp = RotatingHWPOperator.create(shape=(5,), stokes='IQUV', angles=pa)
    x = StokesIQUVPyTree(
        I=jnp.array([1.0, 2, 3, 4, 5]),
        Q=jnp.array([1.0, 1, 1, 1, 1]),
        U=jnp.array([2.0, 2, 2, 2, 2]),
        V=jnp.array([1.0, 5, 4, 3, 2]),
    )

    actual_y = hwp(x)

    expected_y = StokesIQUVPyTree(
        I=x.I,
        Q=jnp.array([1.0, -2, -1, 2, 1]),
        U=jnp.array([-2.0, -1, 2, 1, -2]),
        V=-x.V,
    )
    assert equinox.tree_equal(actual_y, expected_y, atol=1e-15, rtol=1e-15)


def test_rotating_hwp_orthogonal(stokes: ValidStokesType) -> None:
    hwp = RotatingHWPOperator.create(shape=(), stokes=stokes, angles=1.1)
    x = StokesPyTree.class_for(stokes).ones(())
    y = (hwp.T @ hwp)(x)
    assert equinox.tree_equal(y, x, atol=1e-15, rtol=1e-15)
