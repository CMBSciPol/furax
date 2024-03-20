from typing import get_args

import equinox
import jax.numpy as jnp
import pytest

from astrosim.landscapes import StokesIPyTree, StokesIQUPyTree, ValidStokesType, stokes_pytree_cls
from astrosim.operators.bolometers import BolometerOperator


def test_direct_i() -> None:
    bolo = BolometerOperator(shape=(2, 5), stokes='I')
    x = StokesIPyTree(I=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    y = bolo.mv(x)

    expected_y = x.I
    assert jnp.allclose(y, expected_y, atol=1e-15, rtol=1e-15)


def test_direct_iqu() -> None:
    bolo = BolometerOperator(shape=(2, 5), stokes='IQU')
    x = StokesIQUPyTree(
        I=jnp.array([1.0, 2, 3, 4, 5]),
        Q=jnp.array([1.0, 1, 1, 1, 1]),
        U=jnp.array([2.0, 2, 2, 2, 2]),
    )

    y = bolo.mv(x)

    expected_y = x.I + x.Q + x.U
    assert jnp.allclose(y, expected_y, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize('stokes', get_args(ValidStokesType))
def test_transpose(stokes) -> None:
    bolo = BolometerOperator(shape=(2, 5), stokes=stokes)
    x = jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]])

    y = bolo.T.mv(x)

    expected_cls = stokes_pytree_cls(stokes)
    assert isinstance(y, expected_cls)
    expected_y = expected_cls(*(len(stokes) * [x]))
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)
