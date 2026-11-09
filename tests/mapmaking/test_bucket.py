import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpy.testing import assert_allclose

from furax.core import DenseBlockDiagonalOperator, StackedEnsemble
from furax.mapmaking._bucket import MemberOperator, segment_ensemble

N = 3
OWNER = np.array([0, 0, 1, 2])  # four tiles of three observations


def _dense(blocks):
    return DenseBlockDiagonalOperator(
        blocks, subscripts='ij,j->i', in_structure=jax.ShapeDtypeStruct((N,), blocks.dtype)
    )


@pytest.mark.parametrize('stride', [1, 'segment'], ids=['per-tile', 'per-observation'])
def test_member_operator_applies_each_tiles_operator_to_its_own_amplitudes(stride):
    k = jr.split(jr.key(0), 2)
    n_entries = len(OWNER) if stride == 1 else OWNER.max() + 1
    blocks = jr.normal(k[0], (n_entries, N, N))
    x = jr.normal(k[1], (len(OWNER), N))
    operators = StackedEnsemble(
        len(OWNER), _dense(blocks), stride, owner=jnp.asarray(OWNER), segments=OWNER.max() + 1
    )
    op = MemberOperator(operators, in_structure=jax.ShapeDtypeStruct(x.shape, x.dtype))
    per_tile = blocks if stride == 1 else blocks[OWNER]
    assert_allclose(op(x), jnp.einsum('...ij,...j->...i', per_tile, x), rtol=1e-12)


def test_member_operator_over_the_segments_of_the_tiles():
    k = jr.split(jr.key(1), 2)
    blocks = jr.normal(k[0], (OWNER.max() + 1, N, N))
    x = jr.normal(k[1], (OWNER.max() + 1, N))
    tiles = StackedEnsemble(len(OWNER), _dense(blocks), 'segment', owner=jnp.asarray(OWNER))
    op = MemberOperator(
        segment_ensemble(tiles), in_structure=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )
    assert_allclose(op(x), jnp.einsum('...ij,...j->...i', blocks, x), rtol=1e-12)
    with pytest.raises(ValueError, match='varies with the tile'):
        segment_ensemble(
            StackedEnsemble(
                len(OWNER), jnp.ones((len(OWNER), N)), owner=jnp.asarray(OWNER), segments=3
            )
        )
