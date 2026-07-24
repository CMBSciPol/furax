import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, tree
from furax.core import BlockColumnOperator, BlockRowOperator, DiagonalOperator, HomothetyOperator
from furax.mapmaking.streaming import (
    AbstractStreamOperator,
    StreamAdditionOperator,
    StreamColumnOperator,
    StreamDiagonalOperator,
    StreamOperator,
    StreamRowOperator,
    StreamSegment,
    stream_block_column,
    stream_block_row,
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

    op = StreamDiagonalOperator.create(blocks)
    assert op.in_structure.shape == (N_OBS, N_IN)
    assert op.out_structure.shape == (N_OBS, N_OUT)

    op = StreamColumnOperator.create(blocks)
    assert op.in_structure.shape == (N_IN,)
    assert op.out_structure.shape == (N_OBS, N_OUT)

    op = StreamRowOperator.create(blocks)
    assert op.in_structure.shape == (N_OBS, N_IN)
    assert op.out_structure.shape == (N_OUT,)

    op = StreamAdditionOperator.create(blocks)
    assert op.in_structure.shape == (N_IN,)
    assert op.out_structure.shape == (N_OUT,)


# ---------------------------------------------------------------------------
# Sharded forward
# ---------------------------------------------------------------------------


def test_sharded_diagonal_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = StreamDiagonalOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = np.stack([m @ x_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op(x), expected, rtol=1e-10)


def test_sharded_column_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = StreamColumnOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    expected = np.stack([m @ np.array(x) for m in _per_obs(blocks)])
    assert_allclose(op(x), expected, rtol=1e-10)


def test_sharded_row_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = StreamRowOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = sum(m @ x_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op(x), expected, rtol=1e-10)


def test_sharded_addition_mv() -> None:
    blocks = _make_blocks(P('obs'))
    op = StreamAdditionOperator.create(blocks)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    expected = sum(m @ np.array(x) for m in _per_obs(blocks))
    assert_allclose(op(x), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Sharded transpose
# ---------------------------------------------------------------------------


def test_sharded_diagonal_transpose() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = StreamDiagonalOperator.create(blocks).T
    assert isinstance(op_T, StreamDiagonalOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = np.stack([m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T(y), expected, rtol=1e-10)


def test_sharded_column_transpose_is_row() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = StreamColumnOperator.create(blocks).T
    assert isinstance(op_T, StreamRowOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = sum(m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op_T(y), expected, rtol=1e-10)


def test_sharded_row_transpose_is_column() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = StreamRowOperator.create(blocks).T
    assert isinstance(op_T, StreamColumnOperator)
    y = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    expected = np.stack([m.T @ np.array(y) for m in _per_obs(blocks)])
    assert_allclose(op_T(y), expected, rtol=1e-10)


def test_sharded_addition_transpose() -> None:
    blocks = _make_blocks(P('obs'))
    op_T = StreamAdditionOperator.create(blocks).T
    assert isinstance(op_T, StreamAdditionOperator)
    y = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    expected = sum(m.T @ np.array(y) for m in _per_obs(blocks))
    assert_allclose(op_T(y), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Output sharding
# ---------------------------------------------------------------------------


def test_sharded_output_sharding() -> None:
    blocks = _make_blocks(P('obs'))

    cases = [
        (StreamDiagonalOperator, (N_OBS, N_IN), P('obs', None), P('obs', None)),
        (StreamColumnOperator, (N_IN,), P(None), P('obs', None)),
        (StreamRowOperator, (N_OBS, N_IN), P('obs', None), P(None)),
        (StreamAdditionOperator, (N_IN,), P(None), P(None)),
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
    H = StreamColumnOperator.create(_make_blocks(P('obs')))
    W = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (H.T @ W @ H).reduce()
    assert isinstance(reduced, StreamAdditionOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced(x), (H.T @ W @ H)(x), rtol=1e-10)


# ---------------------------------------------------------------------------
# Closed-over maps: scalars/shared operators fold into pre/post, not the scanned body
# ---------------------------------------------------------------------------


def test_homothety_on_block_lands_in_closed_over_map() -> None:
    # `(-2) * block` attaches the scalar as a shared segment on the input side (rightmost in
    # composition order, since `c * op` puts the scalar on the input side), leaving the stacked core.
    blocks = _make_blocks(P('obs'))
    op = ((-2.0) * StreamDiagonalOperator.create(blocks)).reduce()
    assert isinstance(op, StreamDiagonalOperator)
    assert [seg.stacked for seg in op.segments] == [True, False]
    assert isinstance(op.segments[-1].operator, HomothetyOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = np.stack([-2.0 * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op(x), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# addition fusion: sum of two streams collapses to one stream
# ---------------------------------------------------------------------------


def test_addition_fusion_diagonal() -> None:
    A = StreamDiagonalOperator.create(_make_blocks(P('obs')))
    B = StreamDiagonalOperator.create(_make_blocks(P('obs')))
    reduced = (A + B).reduce()
    assert isinstance(reduced, StreamDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced(x), tree.add(A(x), B(x)), rtol=1e-10)


def test_subtraction_fusion_diagonal() -> None:
    # `A - B` -> `A + (-1) * B`: the -1 lands in B's closed-over `pre`, then addition fusion
    # combines the two via BlockColumn/BlockDiagonal/BlockRow into a single StreamDiagonal,
    # keeping the scalar out of the strictly obs-stacked body.
    A = StreamDiagonalOperator.create(_make_blocks(P('obs')))
    B = StreamDiagonalOperator.create(_make_blocks(P('obs')))
    reduced = (A - B).reduce()
    assert isinstance(reduced, StreamDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced(x), tree.sub(A(x), B(x)), rtol=1e-10)


def test_addition_fusion_row() -> None:
    A = StreamRowOperator.create(_make_blocks(P('obs')))
    B = StreamRowOperator.create(_make_blocks(P('obs')))
    reduced = (A + B).reduce()
    assert isinstance(reduced, StreamRowOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced(x), tree.add(A(x), B(x)), rtol=1e-10)


def test_marginal_weight_fusion() -> None:
    # the marginalisation shape `W - W T G T.T W` reduces to a single StreamDiagonal.
    # W: per-obs (N_OUT, N_OUT); T: per-obs (N_OUT, N_IN) amplitudes->tod; G: per-obs (N_IN, N_IN).
    W = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    T = StreamDiagonalOperator.create(_make_blocks(P('obs')))  # in (N_IN,) -> out (N_OUT,)
    G = StreamDiagonalOperator.create(
        _TestOp(
            # obs-sharded like W and T: a per-obs operator's leaves must share the obs sharding,
            # else the fused scan sees mismatched per-shard leading axes under multiple devices.
            jax.device_put(jnp.broadcast_to(jnp.eye(N_IN), (N_OBS, N_IN, N_IN)), P('obs')),
            in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64),
        )
    )
    chain = W @ T @ G @ T.T @ W
    Wm = (W - chain).reduce()
    assert isinstance(Wm, StreamDiagonalOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    expected = tree.sub(W(x), chain(x))
    assert_allclose(Wm(x), expected, rtol=1e-10)


def test_fused_block_composes_with_sharded_block() -> None:
    # Regression: a fused block must carry the obs-axis sharding in its public structure so it
    # still composes with other sharded streams. `A - B` exercises both the homothety rule
    # (the −1 folds into a closed-over map) and the addition-fusion rule; before the fix their
    # final `_build` ran under the empty mesh, leaving an unsharded structure that raised when
    # composed. Square blocks so the product is well-defined.
    A = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    B = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    fused = (A - B).reduce()
    composed = (A @ fused).reduce()  # must not raise on the structure check
    x = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    assert_allclose(composed(x), A(tree.sub(A(x), B(x))), rtol=1e-10)


def test_create_carries_explicit_obs_size() -> None:
    # n is declared, not re-inferred from leaf shapes downstream; create starts with one stacked seg.
    W = StreamDiagonalOperator.create(_make_blocks(P('obs')))
    assert W.n_lead == N_OBS
    assert len(W.segments) == 1
    assert W.segments[0].stacked
    assert W.segments[0].operator.matrix.shape[0] == N_OBS


def test_non_scalar_static_post_is_applied() -> None:
    # a NON-scalar shared segment (a shared-across-observation diagonal, leaf shape (N_OUT,) with no
    # obs axis) is a valid output-side segment: it must be accepted and applied after the core.
    blocks = _make_blocks(P('obs'))  # per-obs (N_OUT, N_IN)
    d = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    post = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OUT,), jnp.float64))
    # composition order (post @ core): post is applied after the sliced core, so it comes first
    op = StreamDiagonalOperator._build(
        (StreamSegment(post, False), StreamSegment(blocks, True)), n_lead=N_OBS
    )
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    d_np = np.array(jax.device_get(d))
    expected = np.stack([d_np * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op(x), expected, rtol=1e-10)


def test_transpose_roundtrips_static() -> None:
    # an input-side shared map moves to the output side under transpose (segments reverse).
    blocks = _make_blocks(P('obs'))
    op = ((-3.0) * StreamDiagonalOperator.create(blocks)).reduce()
    assert isinstance(op.segments[-1].operator, HomothetyOperator)  # input side (applied first)
    op_T = op.T
    assert isinstance(op_T, StreamDiagonalOperator)
    assert isinstance(
        op_T.segments[0].operator, HomothetyOperator
    )  # now output side (applied last)
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected = np.stack([-3.0 * (m.T @ y_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T(y), expected, rtol=1e-10)


def test_addition_fusion_defers_on_obs_size_mismatch() -> None:
    # StreamAddition structures are per-observation, so summing blocks with different obs sizes is
    # legal algebra; fusion must defer (stay unreduced) rather than crash on mis-stacked bodies.
    A = StreamAdditionOperator.create(_make_blocks(P('obs')))
    matrices = RNG.standard_normal((2 * N_OBS, N_OUT, N_IN), dtype=np.float64)
    bigger = _TestOp(
        jax.device_put(matrices, P('obs')), in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64)
    )
    B = StreamAdditionOperator.create(bigger)
    reduced = (A + B).reduce()
    assert not isinstance(reduced, StreamAdditionOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced(x), tree.add(A(x), B(x)), rtol=1e-10)


def test_check_scanned_rejects_non_obs_body_leaf() -> None:
    # a stacked segment leaf that does not lead with the obs axis is a mis-tagged operator: raise.
    bad = _make_blocks()  # leaves lead with N_OBS
    with pytest.raises(ValueError, match='leading axis size'):
        StreamDiagonalOperator._build((StreamSegment(bad, True),), n_lead=N_OBS + 1)


# ---------------------------------------------------------------------------
# Mixed (per-component) streams: joint shared-sky + stacked-amplitude systems
# ---------------------------------------------------------------------------

N_AMP = 2


def _count_scans(jaxpr: jax.extend.core.Jaxpr) -> int:
    """Total `scan` primitives in a jaxpr, recursing into closed sub-jaxprs (shard_map bodies)."""
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'scan':
            n += 1
        for value in eqn.params.values():
            if isinstance(value, jax.extend.core.ClosedJaxpr):
                n += _count_scans(value.jaxpr)
            elif isinstance(value, jax.extend.core.Jaxpr):
                n += _count_scans(value)
    return n


def _joint_operands() -> tuple[
    StreamColumnOperator, StreamDiagonalOperator, StreamDiagonalOperator
]:
    """H_sky (shared sky -> tod), Te (stacked amps -> tod), W (tod weight)."""
    h_sky = StreamColumnOperator.create(_make_blocks(P('obs')))  # (N_IN,) -> (N_OBS, N_OUT)
    te = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_AMP))  # (N_OBS, N_AMP) -> ...
    w = StreamDiagonalOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    return h_sky, te, w


def test_stream_block_row_structure_and_specs() -> None:
    h_sky, te, _ = _joint_operands()
    h = stream_block_row([h_sky, te])
    assert isinstance(h, StreamOperator)
    assert h.in_stacked == [False, True]  # sky shared, amplitudes stacked
    assert h.out_stacked is True  # both map into the (stacked) tod
    in_struct = h.in_structure
    assert in_struct[0].shape == (N_IN,)
    assert in_struct[1].shape == (N_OBS, N_AMP)
    assert h.out_structure.shape == (N_OBS, N_OUT)


def test_stream_block_row_mv_and_transpose() -> None:
    h_sky, te, _ = _joint_operands()
    h = stream_block_row([h_sky, te])
    hs = _per_obs(h_sky.segments[0].operator)
    ts = _per_obs(te.segments[0].operator)

    x_sky = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    x_amp = jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs'))
    sky_np, amp_np = np.array(x_sky), np.array(jax.device_get(x_amp))
    expected = np.stack([hs[i] @ sky_np + ts[i] @ amp_np[i] for i in range(N_OBS)])
    assert_allclose(h([x_sky, x_amp]), expected, rtol=1e-10)

    # transpose: tod -> [Σ H_iᵀ y_i (shared), stack(Te_iᵀ y_i)]
    h_t = h.T
    assert isinstance(h_t, StreamOperator)
    assert h_t.out_stacked == [False, True]
    y = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    exp_sky = sum(hs[i].T @ y_np[i] for i in range(N_OBS))
    exp_amp = np.stack([ts[i].T @ y_np[i] for i in range(N_OBS)])
    out_sky, out_amp = h_t(y)
    assert_allclose(out_sky, exp_sky, rtol=1e-10)
    assert_allclose(out_amp, exp_amp, rtol=1e-10)


def test_stream_block_column_fans_out_shared_input() -> None:
    # a column shares one input across its blocks and stacks their outputs into a list; it is the
    # transpose of the block-row of the transposed operands.
    a = StreamColumnOperator.create(_make_blocks(P('obs')))  # (N_IN,) -> (N_OBS, N_OUT)
    b = StreamColumnOperator.create(_make_blocks(P('obs')))
    col = stream_block_column([a, b])
    # both output legs are obs-stacked, so the spec is uniform and collapses to a plain column
    assert isinstance(col, StreamColumnOperator)
    assert col.in_stacked is False  # single shared input
    assert col.out_stacked is True
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    out_a, out_b = col(x)
    assert_allclose(out_a, a(x), rtol=1e-10)
    assert_allclose(out_b, b(x), rtol=1e-10)


def test_mixed_normal_equations_fuse_to_single_scan() -> None:
    h_sky, te, w = _joint_operands()
    h = stream_block_row([h_sky, te])
    a = (h.T @ w @ h).reduce()
    assert isinstance(a, StreamOperator)
    assert a.in_stacked == [False, True]
    assert a.out_stacked == [False, True]
    assert sum(seg.stacked for seg in a.segments) == 1  # one stacked core: one tod pass

    # numerically identical to the old 2x2 block-of-reduced-streams assembly
    a_ss = (h_sky.T @ w @ h_sky).reduce()
    a_sa = (h_sky.T @ w @ te).reduce()
    a_as = (te.T @ w @ h_sky).reduce()
    a_aa = (te.T @ w @ te).reduce()
    a_2x2 = BlockColumnOperator([BlockRowOperator([a_ss, a_sa]), BlockRowOperator([a_as, a_aa])])

    x = [
        jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P()),
        jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs')),
    ]
    fused, ref = a(x), a_2x2(x)
    for leaf_a, leaf_b in zip(jax.tree.leaves(fused), jax.tree.leaves(ref), strict=True):
        assert_allclose(leaf_a, leaf_b, rtol=1e-9)

    # the point of the change: one scan for the fused operator vs four for the 2x2 assembly
    assert _count_scans(jax.make_jaxpr(a.mv)(x).jaxpr) == 1
    assert _count_scans(jax.make_jaxpr(a_2x2.mv)(x).jaxpr) == 4


def test_mixed_output_sharding() -> None:
    h_sky, te, w = _joint_operands()
    a = (stream_block_row([h_sky, te]).T @ w @ stream_block_row([h_sky, te])).reduce()
    x_struct = [
        jax.ShapeDtypeStruct((N_IN,), jnp.float64, sharding=P()),
        jax.ShapeDtypeStruct((N_OBS, N_AMP), jnp.float64, sharding=P('obs')),
    ]
    y = jax.eval_shape(a.mv, x_struct)
    assert 'obs' not in y[0].sharding.spec  # sky leg replicated (not obs-sharded)
    assert y[1].sharding.spec == P('obs', None)  # amplitude leg obs-sharded


def test_addition_composition_does_not_fuse() -> None:
    # a shared junction is a psum only available after the scan, so `Addition @ Addition` cannot
    # fuse into one scan: (Σᵢaᵢ)(Σⱼbⱼ) != Σᵢ aᵢbᵢ. It must stay an unreduced composition.
    a = StreamAdditionOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    b = StreamAdditionOperator.create(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (a @ b).reduce()
    assert not isinstance(reduced, AbstractStreamOperator)
    x = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    assert_allclose(reduced(x), a(b(x)), rtol=1e-10)


def test_mixed_addition_fusion() -> None:
    h_sky, te, w = _joint_operands()
    left = (stream_block_row([h_sky, te]).T @ w @ stream_block_row([h_sky, te])).reduce()
    h_sky2, te2, w2 = _joint_operands()
    right = (stream_block_row([h_sky2, te2]).T @ w2 @ stream_block_row([h_sky2, te2])).reduce()
    fused = (left + right).reduce()
    assert isinstance(fused, StreamOperator)
    assert sum(seg.stacked for seg in fused.segments) == 1
    x = [
        jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P()),
        jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs')),
    ]
    expected = tree.add(left(x), right(x))
    for leaf_a, leaf_b in zip(jax.tree.leaves(fused(x)), jax.tree.leaves(expected), strict=True):
        assert_allclose(leaf_a, leaf_b, rtol=1e-9)


def test_homothety_on_mixed_stream() -> None:
    h_sky, te, w = _joint_operands()
    a = (stream_block_row([h_sky, te]).T @ w @ stream_block_row([h_sky, te])).reduce()
    scaled = ((-2.0) * a).reduce()
    assert isinstance(scaled, StreamOperator)
    assert sum(seg.stacked for seg in scaled.segments) == 1  # scalar stays out of the stacked body
    x = [
        jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P()),
        jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs')),
    ]
    expected = tree.mul(-2.0, a(x))
    for leaf_a, leaf_b in zip(jax.tree.leaves(scaled(x)), jax.tree.leaves(expected), strict=True):
        assert_allclose(leaf_a, leaf_b, rtol=1e-9)


def test_stream_block_row_rejects_non_conforming_operand() -> None:
    h_sky, _, _ = _joint_operands()
    # a plain (non-stream) operator is not a valid operand
    dense = DiagonalOperator(
        jnp.ones(N_IN), in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64)
    )
    with pytest.raises(ValueError, match='stream operators'):
        stream_block_row([h_sky, dense])
