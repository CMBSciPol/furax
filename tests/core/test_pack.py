import equinox
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jaxtyping import Bool, Float, PyTree
from numpy.testing import assert_array_equal

from furax import IdentityOperator
from furax.core._linear import PackOperator
from furax.obs.stokes import StokesI, StokesIQU
from furax.tree import as_structure


@pytest.mark.parametrize(
    'mask, in_structure, out_structure',
    [
        (
            jnp.array([False, True, True, False]),
            jax.ShapeDtypeStruct((4,), np.float16),
            jax.ShapeDtypeStruct((2,), np.float16),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesI.structure_for((4,), np.float32),
            StokesI.structure_for((2,), np.float32),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesIQU.structure_for((4,), np.float32),
            StokesIQU.structure_for((2,), np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            jax.ShapeDtypeStruct((2, 2), np.float16),
            jax.ShapeDtypeStruct((2,), np.float16),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesI.structure_for((2, 2), np.float32),
            StokesI.structure_for((2,), np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesIQU.structure_for((2, 2), np.float32),
            StokesIQU.structure_for((2,), np.float32),
        ),
    ],
)
def test_pack_structure(
    mask: Bool[Array, '...'],
    in_structure: PyTree[jax.ShapeDtypeStruct],
    out_structure: PyTree[jax.ShapeDtypeStruct],
) -> None:
    operator = PackOperator(mask, in_structure=in_structure)
    assert operator.out_structure == out_structure


@pytest.mark.parametrize(
    'mask, in_structure, x, expected_y',
    [
        (
            jnp.array([False, True, True, False]),
            jax.ShapeDtypeStruct((4,), np.float32),
            jnp.arange(4, dtype=np.float32),
            jnp.array([1, 2], dtype=np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            jax.ShapeDtypeStruct((2, 2), np.float32),
            jnp.arange(4, dtype=np.float32).reshape(2, 2),
            jnp.array([1, 2], dtype=np.float32),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesI.structure_for((4,), np.float32),
            StokesI(jnp.array([1, 2, 3, 4], dtype=np.float32)),
            StokesI(jnp.array([2, 3], dtype=np.float32)),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesI.structure_for((2, 2), np.float32),
            StokesI(jnp.array([[1, 2], [3, 4]], dtype=np.float32)),
            StokesI(jnp.array([2, 3], dtype=np.float32)),
        ),
        (
            jnp.array([False, True, False, True]),
            StokesIQU.structure_for((4,), np.float32),
            StokesIQU(
                jnp.array([1, 2, 3, 4], dtype=np.float32),
                jnp.array([5, 6, 7, 8], dtype=np.float32),
                jnp.array([9, 10, 11, 12], dtype=np.float32),
            ),
            StokesIQU(
                jnp.array([2, 4], dtype=np.float32),
                jnp.array([6, 8], dtype=np.float32),
                jnp.array([10, 12], dtype=np.float32),
            ),
        ),
        (
            jnp.array([[False, True], [False, True]]),
            StokesIQU.structure_for((2, 2), np.float32),
            StokesIQU(
                jnp.array([[1, 2], [3, 4]], dtype=np.float32),
                jnp.array([[5, 6], [7, 8]], dtype=np.float32),
                jnp.array([[9, 10], [11, 12]], dtype=np.float32),
            ),
            StokesIQU(
                jnp.array([2, 4], dtype=np.float32),
                jnp.array([6, 8], dtype=np.float32),
                jnp.array([10, 12], dtype=np.float32),
            ),
        ),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_pack_direct(
    mask: Bool[Array, '...'],
    in_structure: PyTree[jax.ShapeDtypeStruct],
    x: PyTree[Float[Array, '...']],
    expected_y: PyTree[Float[Array, '...']],
    do_jit: bool,
) -> None:
    operator = PackOperator(mask, in_structure=in_structure)
    func = operator.__call__
    if do_jit:
        func = jax.jit(func)
    actual_y = func(x)
    assert equinox.tree_equal(actual_y, expected_y)


def test_pack_transpose() -> None:
    mask = jnp.array([False, True, True, False])
    in_structure = jax.ShapeDtypeStruct((4,), np.float32)
    operator = PackOperator(mask, in_structure=in_structure)
    transposed_operator = operator.T
    assert transposed_operator.in_structure == operator.out_structure
    assert transposed_operator.out_structure == operator.in_structure


@pytest.mark.parametrize(
    'mask, in_structure, out_structure',
    [
        (
            jnp.array([False, True, True, False]),
            jax.ShapeDtypeStruct((2,), np.float16),
            jax.ShapeDtypeStruct((4,), np.float16),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            jax.ShapeDtypeStruct((2,), np.float16),
            jax.ShapeDtypeStruct((2, 2), np.float16),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesI.structure_for((2,), np.float32),
            StokesI.structure_for((4,), np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesI.structure_for((2,), np.float32),
            StokesI.structure_for((2, 2), np.float32),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesIQU.structure_for((2,), np.float32),
            StokesIQU.structure_for((4,), np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesIQU.structure_for((2,), np.float32),
            StokesIQU.structure_for((2, 2), np.float32),
        ),
    ],
)
def test_unpack_structure(
    mask: Bool[Array, '...'],
    in_structure: PyTree[jax.ShapeDtypeStruct],
    out_structure: PyTree[jax.ShapeDtypeStruct],
) -> None:
    operator = PackOperator(mask, in_structure=out_structure).T
    assert operator.in_structure == in_structure


@pytest.mark.parametrize(
    'mask, out_structure, x, expected_y',
    [
        (
            jnp.array([False, True, True, False]),
            jax.ShapeDtypeStruct((4,), np.float32),
            jnp.array([1, 2], dtype=np.float32),
            jnp.array([0, 1, 2, 0], dtype=np.float32),
        ),
        (
            jnp.array([False, True, True, False]),
            StokesI.structure_for((4,), np.float32),
            StokesI(jnp.array([2, 3], dtype=np.float32)),
            StokesI(jnp.array([0, 2, 3, 0], dtype=np.float32)),
        ),
        (
            jnp.array([False, True, False, True]),
            StokesIQU.structure_for((4,), np.float32),
            StokesIQU(
                jnp.array([2, 4], dtype=np.float32),
                jnp.array([6, 8], dtype=np.float32),
                jnp.array([10, 12], dtype=np.float32),
            ),
            StokesIQU(
                jnp.array([0, 2, 0, 4], dtype=np.float32),
                jnp.array([0, 6, 0, 8], dtype=np.float32),
                jnp.array([0, 10, 0, 12], dtype=np.float32),
            ),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            jax.ShapeDtypeStruct((2, 2), np.float32),
            jnp.array([1, 2], dtype=np.float32),
            jnp.array([[0, 1], [2, 0]], dtype=np.float32),
        ),
        (
            jnp.array([[False, True], [True, False]]),
            StokesI.structure_for((2, 2), np.float32),
            StokesI(jnp.array([2, 3], dtype=np.float32)),
            StokesI(jnp.array([[0, 2], [3, 0]], dtype=np.float32)),
        ),
        (
            jnp.array([[False, True], [False, True]]),
            StokesIQU.structure_for((2, 2), np.float32),
            StokesIQU(
                jnp.array([2, 4], dtype=np.float32),
                jnp.array([6, 8], dtype=np.float32),
                jnp.array([10, 12], dtype=np.float32),
            ),
            StokesIQU(
                jnp.array([[0, 2], [0, 4]], dtype=np.float32),
                jnp.array([[0, 6], [0, 8]], dtype=np.float32),
                jnp.array([[0, 10], [0, 12]], dtype=np.float32),
            ),
        ),
    ],
)
@pytest.mark.parametrize('do_jit', [False, True])
def test_unpack_direct(
    mask: Bool[Array, '...'],
    out_structure: PyTree[jax.ShapeDtypeStruct],
    x: PyTree[Float[Array, '...']],
    expected_y: PyTree[Float[Array, '...']],
    do_jit: bool,
) -> None:
    operator = PackOperator(mask, in_structure=out_structure).T
    func = operator.__call__
    if do_jit:
        func = jax.jit(func)
    actual_y = func(x)
    assert as_structure(actual_y) == operator.out_structure
    assert equinox.tree_equal(actual_y, expected_y)
    assert_array_equal(operator.as_matrix(), operator.T.as_matrix().T)


def test_pack_unpack_rule() -> None:
    mask = jnp.array([True, False])
    in_structure = jax.ShapeDtypeStruct(mask.shape, np.float32)
    pack = PackOperator(mask, in_structure=in_structure)
    reduced_operator = (pack @ pack.T).reduce()
    assert isinstance(reduced_operator, IdentityOperator)
    assert reduced_operator.in_structure == jax.ShapeDtypeStruct((1,), np.float32)
