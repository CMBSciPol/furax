import equinox
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from furax.core import CompositionOperator
from furax.obs import HWPOperator, LinearPolarizerOperator, QURotationOperator
from furax.obs.stokes import Stokes, StokesI, StokesIQU, ValidStokesType
from furax.tree import as_structure


def test_as_matrix(stokes: ValidStokesType) -> None:
    polarizer = LinearPolarizerOperator.create(shape=(), stokes=stokes)
    stokes_slice = {'I': slice(0, 1), 'QU': slice(1, 3), 'IQU': slice(0, 3), 'IQUV': slice(0, 4)}[
        stokes
    ]
    expected_matrix = jnp.array([[0.5, 0.5, 0, 0]])[..., stokes_slice]
    assert_array_equal(polarizer.as_matrix(), expected_matrix)


def test_transpose_as_matrix(stokes: ValidStokesType) -> None:
    polarizer = LinearPolarizerOperator.create(shape=(), stokes=stokes)
    stokes_slice = {
        'I': slice(0, 1),
        'QU': slice(1, 3),
        'IQU': slice(0, 3),
        'IQUV': slice(0, 4),
    }[stokes]
    expected_matrix = jnp.array([[0.5], [0.5], [0], [0]])[stokes_slice]
    assert_array_equal(polarizer.T.as_matrix(), expected_matrix)


def test_hwp_rule(stokes: ValidStokesType) -> None:
    polarizer = LinearPolarizerOperator.create(shape=(), stokes=stokes)
    hwp = HWPOperator.create(shape=(), stokes=stokes)
    op = polarizer @ hwp
    reduced_op = op.reduce()
    assert isinstance(reduced_op, LinearPolarizerOperator)
    assert_array_equal(reduced_op.as_matrix(), op.as_matrix())


def test_hwp_rule_with_angles(stokes: ValidStokesType) -> None:
    polarizer = LinearPolarizerOperator.create(shape=(), stokes=stokes, angles=1.0)
    hwp = HWPOperator.create(shape=(), stokes=stokes, angles=2.0)
    op = (polarizer @ hwp).reduce()
    assert isinstance(op, CompositionOperator)
    assert len(op.operands) == 2
    assert isinstance(op.operands[0], LinearPolarizerOperator)
    assert isinstance(op.operands[1], QURotationOperator)
    assert op.operands[1].angles == 3.0


def test_direct_i() -> None:
    polarizer = LinearPolarizerOperator.create(shape=(2, 5), stokes='I')
    x = StokesI(i=jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]]))

    y = polarizer(x)

    assert as_structure(y) == polarizer.out_structure
    expected_y = x.i / 2
    assert_allclose(y, expected_y, atol=1e-15, rtol=1e-15)


def test_create_direct_iqu() -> None:
    angles = np.deg2rad(15)
    polarizer = LinearPolarizerOperator.create(shape=(5,), angles=angles)
    x = StokesIQU(
        i=jnp.array([1.0, 2, 3, 4, 5]),
        q=jnp.array([1.0, 1, 1, 1, 1]),
        u=jnp.array([2.0, 2, 2, 2, 2]),
    )

    y = polarizer(x)

    assert as_structure(y) == polarizer.out_structure
    expected_y = 0.5 * (x.i + np.cos(2 * angles) * x.q - np.sin(2 * angles) * x.u)
    assert_allclose(y, expected_y, atol=1e-15, rtol=1e-15)


def test_create_transpose(stokes: ValidStokesType) -> None:
    angles = np.deg2rad(15)
    polarizer = LinearPolarizerOperator.create(shape=(2, 5), stokes=stokes, angles=angles)
    x = jnp.array([[1.0, 2, 3, 4, 5], [1, 1, 1, 1, 1]])

    y = polarizer.T(x)

    assert as_structure(y) == polarizer.T.out_structure
    expected_cls = Stokes.class_for(stokes)
    assert isinstance(y, expected_cls)
    expected_y = expected_cls.from_iquv(
        0.5 * x, 0.5 * np.cos(2 * angles) * x, -0.5 * np.sin(2 * angles) * x, 0 * x
    )
    assert equinox.tree_equal(y, expected_y, atol=1e-15, rtol=1e-15)
