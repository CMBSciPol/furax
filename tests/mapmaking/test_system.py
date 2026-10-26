import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpy.testing import assert_allclose

from furax import tree
from furax.core import DenseBlockDiagonalOperator
from furax.mapmaking._system import MemberOperator

N = 3


def _dense(blocks):
    return DenseBlockDiagonalOperator(
        blocks, subscripts='ij,j->i', in_structure=jax.ShapeDtypeStruct((N,), blocks.dtype)
    )


@pytest.mark.parametrize('batches', [False, True], ids=['per-slot', 'per-batch'])
def test_member_operator_applies_each_members_operator_to_its_own_amplitudes(batches):
    leading = (2, 4) if batches else (2,)
    k = jr.split(jr.key(0), 2)
    blocks = jr.normal(k[0], (*leading, N, N))
    x = jr.normal(k[1], (*leading, N))
    operators = tree.stack([_dense(b) for b in blocks.reshape(-1, N, N)])
    operators = jax.tree.map(lambda leaf: leaf.reshape(*leading, *leaf.shape[1:]), operators)
    op = MemberOperator(
        operators, batches=batches, in_structure=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )
    assert_allclose(op(x), jnp.einsum('...ij,...j->...i', blocks, x), rtol=1e-12)
