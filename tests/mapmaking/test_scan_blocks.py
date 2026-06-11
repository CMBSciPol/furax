import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator
from furax.core import DiagonalOperator, HomothetyOperator, IdentityOperator
from furax.mapmaking._scan_blocks import (
    ScanAdditionOperator,
    ScanBlockColumnOperator,
    ScanBlockDiagonalOperator,
    ScanBlockRowOperator,
)

# ---------------------------------------------------------------------------
# Minimal stacked operator for testing
# ---------------------------------------------------------------------------


class _TestOp(AbstractLinearOperator):
    matrix: Inexact[Array, '...']

    def __init__(
        self,
        matrix: Inexact[Array, '...'],
        in_structure: PyTree[jax.ShapeDtypeStruct],
    ) -> None:
        object.__setattr__(self, 'matrix', matrix)
        super().__init__(in_structure=in_structure)

    def mv(self, x: PyTree[Inexact[Array, '...']]) -> PyTree[Inexact[Array, '...']]:
        return jnp.einsum('...ij,...j->...i', self.matrix, x)

    @property
    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return jax.ShapeDtypeStruct((self.matrix.shape[-2],), self.matrix.dtype)

    def transpose(self) -> AbstractLinearOperator:
        return _TestOp(
            jnp.swapaxes(self.matrix, -1, -2),
            in_structure=jax.ShapeDtypeStruct((self.matrix.shape[-2],), self.matrix.dtype),
        )


# ---------------------------------------------------------------------------
# Dimensions — N_OBS must be divisible by device count (4)
# ---------------------------------------------------------------------------

N_OBS = 4
N_IN = 3
N_OUT = 5

RNG = np.random.default_rng(seed=0)


@pytest.fixture(scope='module')
def mesh():
    return jax.make_mesh((jax.device_count(),), ('obs',))


@pytest.fixture(scope='module', autouse=True)
def set_mesh(mesh):
    with jax.set_mesh(mesh):
        yield


def _make_blocks(sharding=None, *, n_in: int = N_IN) -> _TestOp:
    matrices = RNG.standard_normal((N_OBS, N_OUT, n_in), dtype=np.float64)
    arr = jax.device_put(matrices, sharding)
    return _TestOp(arr, in_structure=jax.ShapeDtypeStruct((n_in,), jnp.float64))


def _per_obs(op: _TestOp) -> list[np.ndarray]:
    m = np.array(jax.device_get(op.matrix))
    return [m[i] for i in range(m.shape[0])]


# ---------------------------------------------------------------------------
# Structure shapes and sharding
# ---------------------------------------------------------------------------


def test_structure_shapes() -> None:
    blocks = _make_blocks()

    op = ScanBlockDiagonalOperator.create(blocks)
    assert op.in_structure.shape == (N_OBS, N_IN)
    assert op.out_structure.shape == (N_OBS, N_OUT)

    op = ScanBlockColumnOperator.create(blocks)
    assert op.in_structure.shape == (N_IN,)
    assert op.out_structure.shape == (N_OBS, N_OUT)

    op = ScanBlockRowOperator.create(blocks)
    assert op.in_structure.shape == (N_OBS, N_IN)
    assert op.out_structure.shape == (N_OUT,)

    op = ScanAdditionOperator.create(blocks)
    assert op.in_structure.shape == (N_IN,)
    assert op.out_structure.shape == (N_OUT,)


def test_structure_sharding(mesh) -> None:
    blocks = _make_blocks(P('obs'))

    # Structures whose leading axis is the stacked-block axis carry a NamedSharding
    # with P('obs', None, ...) under the active mesh; non-stacked axes (single-input
    # column, single-output row/addition) get no sharding.
    expected_stacked = jax.sharding.NamedSharding(mesh.abstract_mesh, P('obs', None))

    op = ScanBlockDiagonalOperator.create(blocks)
    for s in jax.tree.leaves(op.in_structure):
        assert s.sharding == expected_stacked
    for s in jax.tree.leaves(op.out_structure):
        assert s.sharding == expected_stacked

    op = ScanBlockColumnOperator.create(blocks)
    for s in jax.tree.leaves(op.in_structure):
        assert s.sharding is None
    for s in jax.tree.leaves(op.out_structure):
        assert s.sharding == expected_stacked

    op = ScanBlockRowOperator.create(blocks)
    for s in jax.tree.leaves(op.in_structure):
        assert s.sharding == expected_stacked
    for s in jax.tree.leaves(op.out_structure):
        assert s.sharding is None

    op = ScanAdditionOperator.create(blocks)
    for s in jax.tree.leaves(op.in_structure):
        assert s.sharding is None
    for s in jax.tree.leaves(op.out_structure):
        assert s.sharding is None


# ---------------------------------------------------------------------------
# Sharded forward
# ---------------------------------------------------------------------------


def test_sharded_diagonal_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = ScanBlockDiagonalOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = np.stack([m @ x_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op.mv(x), expected, rtol=1e-10)


def test_sharded_column_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = ScanBlockColumnOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    expected = np.stack([m @ np.array(x) for m in _per_obs(blocks)])
    assert_allclose(op.mv(x), expected, rtol=1e-10)


def test_sharded_row_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = ScanBlockRowOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = sum(m @ x_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op.mv(x), expected, rtol=1e-10)


def test_sharded_addition_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = ScanAdditionOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    expected = sum(m @ np.array(x) for m in _per_obs(blocks))
    assert_allclose(op.mv(x), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Sharded transpose
# ---------------------------------------------------------------------------


def test_sharded_diagonal_transpose() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = ScanBlockDiagonalOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockDiagonalOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = np.stack([m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T.mv(y), expected, rtol=1e-10)


def test_sharded_column_transpose_is_row() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = ScanBlockColumnOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockRowOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = sum(m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op_T.mv(y), expected, rtol=1e-10)


def test_sharded_row_transpose_is_column() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = ScanBlockRowOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockColumnOperator)
    y = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    expected = np.stack([m.T @ np.array(y) for m in _per_obs(blocks)])
    assert_allclose(op_T.mv(y), expected, rtol=1e-10)


def test_sharded_addition_transpose() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = ScanAdditionOperator.create(blocks).T
    assert isinstance(op_T, ScanAdditionOperator)
    y = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    expected = sum(m.T @ np.array(y) for m in _per_obs(blocks))
    assert_allclose(op_T.mv(y), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Output sharding
# ---------------------------------------------------------------------------


def test_sharded_output_sharding() -> None:
    blocks = _make_blocks(P('obs'))

    cases = [
        (ScanBlockDiagonalOperator, (N_OBS, N_IN), P('obs', None), P('obs', None)),
        (ScanBlockColumnOperator, (N_IN,), P(None), P('obs', None)),
        (ScanBlockRowOperator, (N_OBS, N_IN), P('obs', None), P(None)),
        (ScanAdditionOperator, (N_IN,), P(None), P(None)),
    ]
    for cls, shape, x_spec, expected_spec in cases:
        op = cls.create(blocks)
        x_struct = jax.ShapeDtypeStruct(shape, jnp.float64, sharding=x_spec)
        y = jax.eval_shape(op.mv, x_struct)
        assert y.sharding.spec == expected_spec


# ---------------------------------------------------------------------------
# Fusion rules
# ---------------------------------------------------------------------------


def test_sharded_fusion_ht_w_h() -> None:
    H = ScanBlockColumnOperator.create(_make_blocks(P('obs')))
    W = ScanBlockDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (H.T @ W @ H).reduce()
    assert isinstance(reduced, ScanAdditionOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced.mv(x), (H.T @ W @ H).mv(x), rtol=1e-10)


# ---------------------------------------------------------------------------
# (A) heterogeneous leaf batching: static scalar leaves ride alongside the scan
# ---------------------------------------------------------------------------


def test_homothety_on_block_lands_in_closed_over_map() -> None:
    # ``(-2) * block`` folds the scalar into a closed-over map (here the input-side ``pre``, since
    # ``c * op`` puts the scalar on the input side), leaving the scanned body strictly obs-stacked.
    blocks = _make_blocks(P('obs'))
    op = ((-2.0) * ScanBlockDiagonalOperator.create(blocks)).reduce()
    assert isinstance(op, ScanBlockDiagonalOperator)
    assert isinstance(op.pre, HomothetyOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = np.stack([-2.0 * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op.mv(x), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# (B) addition fusion: sum of two scan blocks collapses to one scan block
# ---------------------------------------------------------------------------


def test_addition_fusion_diagonal() -> None:
    A = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))
    B = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))
    reduced = (A + B).reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced.mv(x), jax.tree.map(jnp.add, A.mv(x), B.mv(x)), rtol=1e-10)


def test_subtraction_fusion_diagonal() -> None:
    # exercises (B) addition fusion + Homothety folding + (A) static leaf in one shot:
    # ``A - B`` -> ``A + (-1) @ B`` -> ScanBlockDiagonal(A_inner - B_inner), whose fused inner
    # operator carries the ``-1`` as a static scalar leaf.
    A = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))
    B = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))
    reduced = (A - B).reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced.mv(x), jax.tree.map(jnp.subtract, A.mv(x), B.mv(x)), rtol=1e-10)


def test_addition_fusion_row() -> None:
    A = ScanBlockRowOperator.create(_make_blocks(P('obs')))
    B = ScanBlockRowOperator.create(_make_blocks(P('obs')))
    reduced = (A + B).reduce()
    assert isinstance(reduced, ScanBlockRowOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced.mv(x), jax.tree.map(jnp.add, A.mv(x), B.mv(x)), rtol=1e-10)


def test_marginal_weight_fusion() -> None:
    # the marginalisation shape ``W - W T G T.T W`` reduces to a single ScanBlockDiagonal.
    # W: per-obs (N_OUT, N_OUT); T: per-obs (N_OUT, N_IN) amplitudes->tod; G: per-obs (N_IN, N_IN).
    W = ScanBlockDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    T = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))  # in (N_IN,) -> out (N_OUT,)
    G = ScanBlockDiagonalOperator.create(
        _TestOp(
            jnp.broadcast_to(jnp.eye(N_IN), (N_OBS, N_IN, N_IN)),
            in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64),
        )
    )
    chain = W @ T @ G @ T.T @ W
    Wm = (W - chain).reduce()
    assert isinstance(Wm, ScanBlockDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    expected = jax.tree.map(jnp.subtract, W.mv(x), chain.mv(x))
    assert_allclose(Wm.mv(x), expected, rtol=1e-10)


def test_create_carries_explicit_obs_size() -> None:
    # n is declared, not re-inferred from leaf shapes downstream; create starts with trivial maps.
    W = ScanBlockDiagonalOperator.create(_make_blocks(P('obs')))
    assert W.n_lead == N_OBS
    assert W.scanned.matrix.shape[0] == N_OBS
    assert isinstance(W.pre, IdentityOperator)
    assert isinstance(W.post, IdentityOperator)


def test_non_scalar_static_post_is_applied() -> None:
    # a NON-scalar closed-over map (a shared-across-observation diagonal, leaf shape (N_OUT,) with no
    # obs axis) is a valid post map: it must be accepted (no scalar requirement) and applied.
    blocks = _make_blocks(P('obs'))  # per-obs (N_OUT, N_IN)
    d = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    post = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OUT,), jnp.float64))
    pre = IdentityOperator(in_structure=blocks.in_structure)
    op = ScanBlockDiagonalOperator._build(blocks, pre, post, n_lead=N_OBS)  # must not raise
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    d_np = np.array(jax.device_get(d))
    expected = np.stack([d_np * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op.mv(x), expected, rtol=1e-10)


def test_transpose_roundtrips_static() -> None:
    # an output-side (post) map moves to the input side (pre) under transpose, symmetrically.
    blocks = _make_blocks(P('obs'))
    op = ((-3.0) * ScanBlockDiagonalOperator.create(blocks)).reduce()
    assert isinstance(op.pre, HomothetyOperator)
    op_T = op.T
    assert isinstance(op_T, ScanBlockDiagonalOperator)
    assert isinstance(op_T.post, HomothetyOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = np.stack([-3.0 * (m.T @ y_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T.mv(y), expected, rtol=1e-10)


def test_addition_fusion_defers_on_obs_size_mismatch() -> None:
    # ScanAddition structures are per-observation, so summing blocks with different obs sizes is
    # legal algebra; fusion must defer (stay unreduced) rather than crash on mis-stacked bodies.
    A = ScanAdditionOperator.create(_make_blocks(P('obs')))
    matrices = RNG.standard_normal((2 * N_OBS, N_OUT, N_IN), dtype=np.float64)
    bigger = _TestOp(
        jax.device_put(matrices, P('obs')), in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64)
    )
    B = ScanAdditionOperator.create(bigger)
    reduced = (A + B).reduce()
    assert not isinstance(reduced, ScanAdditionOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced.mv(x), jax.tree.map(jnp.add, A.mv(x), B.mv(x)), rtol=1e-10)


def test_check_scanned_rejects_non_obs_body_leaf() -> None:
    # a scanned body leaf that does not lead with the obs axis is a mis-stacked operator: raise.
    bad = _make_blocks()  # leaves lead with N_OBS
    pre = IdentityOperator(in_structure=bad.in_structure)
    post = IdentityOperator(in_structure=bad.out_structure)
    with pytest.raises(ValueError, match='observation axis'):
        ScanBlockDiagonalOperator._build(bad, pre, post, n_lead=N_OBS + 1)
