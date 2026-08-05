from collections.abc import Callable, Iterator

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
from furax.mapmaking.streaming import StreamOperator, StreamSegment

# ---------------------------------------------------------------------------
# Minimal stacked operator for testing
# ---------------------------------------------------------------------------


class _TestOp(AbstractLinearOperator):
    matrix: Inexact[Array, '...']

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


def _make_blocks(
    sharding: P | None = None, *, n_lead: int = N_OBS, n_in: int = N_IN, n_out: int = N_OUT
) -> _TestOp:
    matrices = RNG.standard_normal((n_lead, n_out, n_in), dtype=np.float64)
    arr = jax.device_put(matrices, sharding)
    return _TestOp(arr, in_structure=jax.ShapeDtypeStruct((n_in,), jnp.float64))


def _per_obs(op: _TestOp) -> list[np.ndarray]:
    m = np.array(jax.device_get(op.matrix))
    return [m[i] for i in range(m.shape[0])]


# ---------------------------------------------------------------------------
# The four uniform layouts
# ---------------------------------------------------------------------------

# A layout *is* its boundary spec pair; these four constructors are the uniform combinations.
_LAYOUTS = [
    pytest.param(StreamOperator.diagonal, True, True, id='diagonal'),
    pytest.param(StreamOperator.column, False, True, id='column'),
    pytest.param(StreamOperator.row, True, False, id='row'),
    pytest.param(StreamOperator.addition, False, False, id='addition'),
]


def _spec(stacked: bool) -> P:
    """Sharding of a boundary component: a stacked one carries the obs axis, a shared one does not."""
    return P('obs', None) if stacked else P(None)


def _boundary(stacked: bool, size: int) -> jax.Array:
    """Random input/output for a boundary component, shaped and sharded to match its spec."""
    shape = (N_OBS, size) if stacked else (size,)
    return jax.device_put(
        RNG.standard_normal(shape, dtype=np.float64), P('obs') if stacked else P()
    )


def _reference(
    matrices: list[np.ndarray], x: np.ndarray, *, in_stacked: bool, out_stacked: bool
) -> np.ndarray:
    """Apply the blocks slice by slice, the way the boundary spec pair says to.

    A stacked input is sliced per block and a shared one goes to every block; a stacked output is
    stacked back up and a shared one summed. That is the whole meaning of the spec pair, so one
    reference covers all four layouts.
    """
    per_slice = [m @ (x[i] if in_stacked else x) for i, m in enumerate(matrices)]
    return np.stack(per_slice) if out_stacked else sum(per_slice)


@pytest.mark.parametrize('make_stream, in_stacked, out_stacked', _LAYOUTS)
def test_layout_structure_and_mv(
    make_stream: Callable[[AbstractLinearOperator], StreamOperator],
    in_stacked: bool,
    out_stacked: bool,
) -> None:
    blocks = _make_blocks(P('obs'))
    op = make_stream(blocks)
    assert (op.in_stacked, op.out_stacked) == (in_stacked, out_stacked)
    assert op.in_structure.shape == ((N_OBS, N_IN) if in_stacked else (N_IN,))
    assert op.out_structure.shape == ((N_OBS, N_OUT) if out_stacked else (N_OUT,))

    x = _boundary(in_stacked, N_IN)
    expected = _reference(
        _per_obs(blocks),
        np.asarray(jax.device_get(x)),
        in_stacked=in_stacked,
        out_stacked=out_stacked,
    )
    assert_allclose(op(x), expected, rtol=1e-10)

    # the output carries the obs axis exactly when out_stacked says it should
    struct = jax.ShapeDtypeStruct(x.shape, jnp.float64, sharding=_spec(in_stacked))
    assert jax.eval_shape(op.mv, struct).sharding.spec == _spec(out_stacked)


@pytest.mark.parametrize('make_stream, in_stacked, out_stacked', _LAYOUTS)
def test_layout_transpose(
    make_stream: Callable[[AbstractLinearOperator], StreamOperator],
    in_stacked: bool,
    out_stacked: bool,
) -> None:
    blocks = _make_blocks(P('obs'))
    op_T = make_stream(blocks).T
    # transposing swaps the boundary: column <-> row, diagonal and addition map to themselves
    assert (op_T.in_stacked, op_T.out_stacked) == (out_stacked, in_stacked)
    y = _boundary(out_stacked, N_OUT)
    expected = _reference(
        [m.T for m in _per_obs(blocks)],
        np.asarray(jax.device_get(y)),
        in_stacked=out_stacked,
        out_stacked=in_stacked,
    )
    assert_allclose(op_T(y), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Fusion rules
# ---------------------------------------------------------------------------


def test_sharded_fusion_ht_w_h() -> None:
    H = StreamOperator.column(_make_blocks(P('obs')))
    W = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (H.T @ W @ H).reduce()
    assert (reduced.in_stacked, reduced.out_stacked) == (False, False)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced(x), (H.T @ W @ H)(x), rtol=1e-10)


# ---------------------------------------------------------------------------
# Constant segments: data that does not carry the batch axis stays out of the sliced body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'n_in, n_out, scalar_leads',
    [
        pytest.param(N_IN, N_OUT, False, id='input-side'),
        pytest.param(N_OUT, N_IN, True, id='output-side'),
    ],
)
def test_homothety_stays_out_of_the_sliced_body(n_in: int, n_out: int, scalar_leads: bool) -> None:
    # `c * block` attaches the scalar as its own constant segment: it owns a leaf, so `_try_merge`
    # will not fold it into a sliced segment where `scan` would try to slice it.
    #
    # Which end it lands on is not decided by writing `c * block` rather than `block @ c`. The
    # algebra commutes a scalar to whichever structure is smaller before `HomothetyStreamRule`
    # sees the composition, so the same expression reaches both of that rule's branches depending
    # only on the block shape -- and transposing then carries the segment across.
    blocks = _make_blocks(P('obs'), n_in=n_in, n_out=n_out)
    op = ((-2.0) * StreamOperator.diagonal(blocks)).reduce()
    assert (op.in_stacked, op.out_stacked) == (True, True)
    assert [seg.sliced for seg in op.segments] == ([False, True] if scalar_leads else [True, False])
    assert isinstance(op.segments[0 if scalar_leads else -1].operator, HomothetyOperator)
    x = jax.device_put(RNG.standard_normal((N_OBS, n_in), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    expected = np.stack([-2.0 * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op(x), expected, rtol=1e-10)

    op_T = op.T
    assert (op_T.in_stacked, op_T.out_stacked) == (True, True)
    assert isinstance(op_T.segments[-1 if scalar_leads else 0].operator, HomothetyOperator)
    y = jax.device_put(RNG.standard_normal((N_OBS, n_out), dtype=np.float64), P('obs'))
    y_np = np.array(jax.device_get(y))
    expected_T = np.stack([-2.0 * (m.T @ y_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T(y), expected_T, rtol=1e-10)


# ---------------------------------------------------------------------------
# addition fusion: sum of two streams collapses to one stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('make_stream, in_stacked, out_stacked', _LAYOUTS)
def test_addition_fusion(
    make_stream: Callable[[AbstractLinearOperator], StreamOperator],
    in_stacked: bool,
    out_stacked: bool,
) -> None:
    A = make_stream(_make_blocks(P('obs')))
    B = make_stream(_make_blocks(P('obs')))
    reduced = (A + B).reduce()
    assert (reduced.in_stacked, reduced.out_stacked) == (in_stacked, out_stacked)
    x = _boundary(in_stacked, N_IN)
    assert_allclose(reduced(x), tree.add(A(x), B(x)), rtol=1e-10)


def test_subtraction_fusion_diagonal() -> None:
    # `A - B` -> `A + (-1) * B`: the -1 becomes a constant segment on B, then addition fusion
    # blocks the two bodies up slot by slot, keeping the scalar out of the sliced body.
    A = StreamOperator.diagonal(_make_blocks(P('obs')))
    B = StreamOperator.diagonal(_make_blocks(P('obs')))
    reduced = (A - B).reduce()
    assert (reduced.in_stacked, reduced.out_stacked) == (True, True)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    assert_allclose(reduced(x), tree.sub(A(x), B(x)), rtol=1e-10)


def test_marginal_weight_fusion() -> None:
    # the marginalisation shape `W - W T G T.T W` reduces to a single StreamDiagonal.
    # W: per-obs (N_OUT, N_OUT); T: per-obs (N_OUT, N_IN) amplitudes->tod; G: per-obs (N_IN, N_IN).
    W = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_OUT))
    T = StreamOperator.diagonal(_make_blocks(P('obs')))  # in (N_IN,) -> out (N_OUT,)
    G = StreamOperator.diagonal(
        _TestOp(
            # obs-sharded like W and T: a per-obs operator's leaves must share the obs sharding,
            # else the fused scan sees mismatched per-shard leading axes under multiple devices.
            jax.device_put(jnp.broadcast_to(jnp.eye(N_IN), (N_OBS, N_IN, N_IN)), P('obs')),
            in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64),
        )
    )
    chain = W @ T @ G @ T.T @ W
    Wm = (W - chain).reduce()
    assert (Wm.in_stacked, Wm.out_stacked) == (True, True)
    x = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    expected = tree.sub(W(x), chain(x))
    assert_allclose(Wm(x), expected, rtol=1e-10)


def test_fused_block_composes_with_sharded_block() -> None:
    # Regression: a fused block must carry the obs-axis sharding in its public structure so it
    # still composes with other sharded streams. `A - B` exercises both the homothety rule
    # (the −1 becomes a constant segment) and the addition-fusion rule; before the fix their
    # final `create` ran under the empty mesh, leaving an unsharded structure that raised when
    # composed. Square blocks so the product is well-defined.
    A = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_OUT))
    B = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_OUT))
    fused = (A - B).reduce()
    composed = (A @ fused).reduce()  # must not raise on the structure check
    x = jax.device_put(RNG.standard_normal((N_OBS, N_OUT), dtype=np.float64), P('obs'))
    assert_allclose(composed(x), A(tree.sub(A(x), B(x))), rtol=1e-10)


def test_create_carries_explicit_obs_size() -> None:
    # n is declared, not re-inferred from leaf shapes downstream; create starts with one stacked seg.
    W = StreamOperator.diagonal(_make_blocks(P('obs')))
    assert W.n_lead == N_OBS
    assert len(W.segments) == 1
    assert W.segments[0].sliced
    assert W.segments[0].operator.matrix.shape[0] == N_OBS


def test_non_scalar_static_post_is_applied() -> None:
    # a NON-scalar constant segment (a shared-across-observation diagonal, leaf shape (N_OUT,) with no
    # obs axis) is a valid output-side segment: it must be accepted and applied after the core.
    blocks = _make_blocks(P('obs'))  # per-obs (N_OUT, N_IN)
    d = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    post = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OUT,), jnp.float64))
    # composition order (post @ core): post is applied after the sliced core, so it comes first
    op = StreamOperator.create(
        (StreamSegment(post, False), StreamSegment(blocks, True)),
        n_lead=N_OBS,
        in_stacked=True,
        out_stacked=True,
    )
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np = np.array(jax.device_get(x))
    d_np = np.array(jax.device_get(d))
    expected = np.stack([d_np * (m @ x_np[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op(x), expected, rtol=1e-10)


def test_addition_fusion_defers_on_obs_size_mismatch() -> None:
    # an addition stream's structures are per-observation, so summing blocks of different obs
    # sizes is legal algebra; fusion must defer rather than crash on mis-stacked bodies.
    A = StreamOperator.addition(_make_blocks(P('obs')))
    matrices = RNG.standard_normal((2 * N_OBS, N_OUT, N_IN), dtype=np.float64)
    bigger = _TestOp(
        jax.device_put(matrices, P('obs')), in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64)
    )
    B = StreamOperator.addition(bigger)
    reduced = (A + B).reduce()
    assert not isinstance(reduced, StreamOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(reduced(x), tree.add(A(x), B(x)), rtol=1e-10)


def test_stacked_segment_must_lead_with_obs_axis() -> None:
    # a sliced segment leaf that does not lead with the obs axis is a mis-tagged operator: raise.
    bad = _make_blocks()  # leaves lead with N_OBS
    with pytest.raises(ValueError, match='leading axis size'):
        StreamOperator.diagonal(bad, n_lead=N_OBS + 1)


# ---------------------------------------------------------------------------
# Shared data must be replicated along the stream axis
# ---------------------------------------------------------------------------


def test_shared_input_sharded_over_obs_is_rejected() -> None:
    # shard_map infers its in_specs from the arguments, so an obs-sharded shared input raises
    # nothing there: each shard silently sees a different slice. Catch it up front instead.
    h = StreamOperator.column(_make_blocks(P('obs'), n_in=N_OBS))
    x = jax.device_put(RNG.standard_normal((N_OBS,), dtype=np.float64), P('obs'))
    with pytest.raises(ValueError, match='sharded over'):
        h(x)


def test_shared_segment_sharded_over_obs_is_rejected() -> None:
    # same trap on the other entry point for shared data: a constant segment's own arrays
    d = jax.device_put(RNG.standard_normal((N_OBS,), dtype=np.float64), P('obs'))
    post = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OBS,), jnp.float64))
    op = StreamOperator.create(
        (StreamSegment(post, False), StreamSegment(_make_blocks(P('obs'), n_out=N_OBS), True)),
        n_lead=N_OBS,
        in_stacked=True,
        out_stacked=True,
    )
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    with pytest.raises(ValueError, match='sharded over'):
        op(x)


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


def _joint_operands() -> tuple[StreamOperator, StreamOperator, StreamOperator]:
    """H_sky (shared sky -> tod), Te (stacked amps -> tod), W (tod weight)."""
    h_sky = StreamOperator.column(_make_blocks(P('obs')))  # (N_IN,) -> (N_OBS, N_OUT)
    te = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_AMP))  # (N_OBS, N_AMP) -> ...
    w = StreamOperator.diagonal(_make_blocks(P('obs'), n_in=N_OUT))
    return h_sky, te, w


def test_stream_block_row_structure_and_specs() -> None:
    h_sky, te, _ = _joint_operands()
    h = StreamOperator.block_row([h_sky, te])
    assert isinstance(h, StreamOperator)
    assert h.in_stacked == [False, True]  # sky shared, amplitudes stacked
    assert h.out_stacked is True  # both map into the (stacked) tod
    in_struct = h.in_structure
    assert in_struct[0].shape == (N_IN,)
    assert in_struct[1].shape == (N_OBS, N_AMP)
    assert h.out_structure.shape == (N_OBS, N_OUT)


def test_stream_block_row_mv_and_transpose() -> None:
    h_sky, te, _ = _joint_operands()
    h = StreamOperator.block_row([h_sky, te])
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


def _constant_wrapped(op: StreamOperator, *, post: bool) -> StreamOperator:
    """Wrap a stream in a non-identity constant map, on the output (post) or input (pre) side.

    Built from segments rather than from `scalar * op`, whose scalar the algebra is free to commute
    to whichever side it likes; which side the constant segment sits on is what matters here.

    The constant map has to own arrays: a leafless one (a scalar homothety, say) is legally folded
    into the sliced core by `_normalize`, which is the case this helper is not about.
    """
    structure = op.segments[0].out_structure if post else op.segments[-1].in_structure
    d = jax.device_put(RNG.standard_normal(structure.shape, dtype=np.float64), P())
    shared = StreamSegment(DiagonalOperator(d, in_structure=structure), False)
    segments = (shared,) + op.segments if post else op.segments + (shared,)
    return StreamOperator.create(
        segments, n_lead=op.n_lead, in_stacked=op.in_stacked, out_stacked=op.out_stacked
    )


@pytest.mark.parametrize('post', [False, True])
def test_stream_block_row_keeps_shared_maps_out_of_the_body(post: bool) -> None:
    # operands built by a layout constructor alone have no constant segments; wrapping them
    # exercises the slots that keep a constant map beside the sliced core.
    h_sky, te, _ = _joint_operands()
    a = _constant_wrapped(h_sky, post=post)
    b = _constant_wrapped(te, post=post)
    h = StreamOperator.block_row([a, b])
    # composition order: a post map lands left of the sliced core, a pre map right of it
    assert [seg.sliced for seg in h.segments] == ([False, True] if post else [True, False])
    assert h.in_stacked == [False, True]
    x_sky = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    x_amp = jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs'))
    assert_allclose(h([x_sky, x_amp]), tree.add(a(x_sky), b(x_amp)), rtol=1e-10)


def test_structural_maps_fold_into_the_stacked_core() -> None:
    # a block row's fan-in and fan-out carry no arrays when the operands have no shared
    # segments, so `_normalize` folds those slots into the core rather than leaving them
    h_sky, te, _ = _joint_operands()
    h = StreamOperator.block_row([h_sky, te])
    assert [seg.sliced for seg in h.segments] == [True]


def _multi_stacked(n_in: int = N_IN) -> StreamOperator:
    """A body with two sliced segments, kept apart by an array-owning constant map.

    A constant map that owns arrays cannot be folded into the sliced core (it would then be sliced),
    so bodies like this survive normalization and are what the slot alignment exists for.
    """
    d = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    shared = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OUT,), jnp.float64))
    segments = (
        StreamSegment(_make_blocks(P('obs'), n_in=N_OUT), True),
        StreamSegment(shared, False),
        StreamSegment(_make_blocks(P('obs'), n_in=n_in), True),
    )
    return StreamOperator.create(segments, n_lead=N_OBS, in_stacked=False, out_stacked=True)


def test_block_row_aligns_bodies_of_unequal_shape() -> None:
    # one operand has a single sliced segment, the other two: padding with identities lets them
    # be blocked up slot for slot, and the fused body keeps the deeper operand's two sliced slots
    a = _multi_stacked()
    b = StreamOperator.column(_make_blocks(P('obs'), n_in=N_AMP))
    assert [seg.sliced for seg in a.segments] == [True, False, True]
    h = StreamOperator.block_row([a, b])
    assert [seg.sliced for seg in h.segments] == [True, False, True]
    x_a = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    x_b = jax.device_put(RNG.standard_normal((N_AMP,), dtype=np.float64), P())
    assert_allclose(h([x_a, x_b]), tree.add(a(x_a), b(x_b)), rtol=1e-10)


def test_multi_stacked_bodies_still_fuse_under_addition() -> None:
    # before slot alignment this deferred to a plain AdditionOperator, i.e. one scan per operand
    a, b = _multi_stacked(), _multi_stacked()
    fused = (a + b).reduce()
    assert isinstance(fused, StreamOperator)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    assert_allclose(fused(x), tree.add(a(x), b(x)), rtol=1e-10)
    assert _count_scans(jax.make_jaxpr(fused.mv)(x).jaxpr) == 1


def test_segment_structure_does_not_depend_on_trace_context() -> None:
    # a Python float is not an array until something traces it, so deciding the fold on "holds an
    # array" would give this operator one structure eagerly and another under jit -- and structure
    # is pytree metadata, so the two would neither compose nor share a jit cache entry
    blocks = _make_blocks(P('obs'))

    def build(scale: float | jax.Array) -> StreamOperator:
        homo = HomothetyOperator(scale, in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64))
        return StreamOperator.create(
            (StreamSegment(blocks, True), StreamSegment(homo, False)),
            n_lead=N_OBS,
            in_stacked=True,
            out_stacked=True,
        )

    expected = jax.tree.structure(build(2.0))
    assert jax.tree.structure(build(jnp.asarray(2.0))) == expected

    traced: list[jax.tree_util.PyTreeDef] = []

    @jax.jit
    def under_trace(scale: jax.Array) -> jax.Array:
        traced.append(jax.tree.structure(build(scale)))
        return scale

    under_trace(jnp.asarray(2.0))
    assert traced == [expected]


def test_array_owning_shared_map_is_never_folded() -> None:
    # a shared array is applied whole to every slice, so folding it into the sliced core would
    # compute something else -- even here, where its leading axis matches n_lead and the
    # sliceability check alone would wave it through
    d = jax.device_put(RNG.standard_normal((N_OBS,), dtype=np.float64), P())
    shared = DiagonalOperator(d, in_structure=jax.ShapeDtypeStruct((N_OBS,), jnp.float64))
    op = StreamOperator.create(
        (StreamSegment(shared, False), StreamSegment(_make_blocks(P('obs'), n_out=N_OBS), True)),
        n_lead=N_OBS,
        in_stacked=True,
        out_stacked=True,
    )
    assert [seg.sliced for seg in op.segments] == [False, True]
    x = jax.device_put(RNG.standard_normal((N_OBS, N_IN), dtype=np.float64), P('obs'))
    x_np, d_np = np.array(jax.device_get(x)), np.array(jax.device_get(d))
    blocks = _per_obs(op.segments[1].operator)
    expected = np.stack([d_np * (m @ x_np[i]) for i, m in enumerate(blocks)])
    assert_allclose(op(x), expected, rtol=1e-10)


def test_stream_block_column_fans_out_shared_input() -> None:
    # a column shares one input across its blocks and stacks their outputs into a list; it is the
    # transpose of the block-row of the transposed operands.
    a = StreamOperator.column(_make_blocks(P('obs')))  # (N_IN,) -> (N_OBS, N_OUT)
    b = StreamOperator.column(_make_blocks(P('obs')))
    col = StreamOperator.block_column([a, b])
    # both output legs are obs-stacked, so the spec is uniform and collapses to a plain column
    # a uniform per-block spec collapses back to a bare bool, so this is a plain column again
    assert (col.in_stacked, col.out_stacked) == (False, True)
    x = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
    out_a, out_b = col(x)
    assert_allclose(out_a, a(x), rtol=1e-10)
    assert_allclose(out_b, b(x), rtol=1e-10)


def test_mixed_normal_equations_fuse_to_single_scan() -> None:
    h_sky, te, w = _joint_operands()
    h = StreamOperator.block_row([h_sky, te])
    a = (h.T @ w @ h).reduce()
    assert isinstance(a, StreamOperator)
    assert a.in_stacked == [False, True]
    assert a.out_stacked == [False, True]
    assert a.sliced_count == 1  # one sliced core: one tod pass

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
    a = (
        StreamOperator.block_row([h_sky, te]).T @ w @ StreamOperator.block_row([h_sky, te])
    ).reduce()
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
    a = StreamOperator.addition(_make_blocks(P('obs'), n_in=N_OUT))
    b = StreamOperator.addition(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (a @ b).reduce()
    assert not isinstance(reduced, StreamOperator)
    x = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    assert_allclose(reduced(x), a(b(x)), rtol=1e-10)


def test_mixed_addition_fusion() -> None:
    h_sky, te, w = _joint_operands()
    left = (
        StreamOperator.block_row([h_sky, te]).T @ w @ StreamOperator.block_row([h_sky, te])
    ).reduce()
    h_sky2, te2, w2 = _joint_operands()
    right = (
        StreamOperator.block_row([h_sky2, te2]).T @ w2 @ StreamOperator.block_row([h_sky2, te2])
    ).reduce()
    fused = (left + right).reduce()
    assert isinstance(fused, StreamOperator)
    assert fused.sliced_count == 1
    x = [
        jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P()),
        jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs')),
    ]
    expected = tree.add(left(x), right(x))
    for leaf_a, leaf_b in zip(jax.tree.leaves(fused(x)), jax.tree.leaves(expected), strict=True):
        assert_allclose(leaf_a, leaf_b, rtol=1e-9)


def test_homothety_on_mixed_stream() -> None:
    h_sky, te, w = _joint_operands()
    a = (
        StreamOperator.block_row([h_sky, te]).T @ w @ StreamOperator.block_row([h_sky, te])
    ).reduce()
    scaled = ((-2.0) * a).reduce()
    assert isinstance(scaled, StreamOperator)
    assert scaled.sliced_count == 1  # scalar stays out of the sliced body
    x = [
        jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P()),
        jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs')),
    ]
    expected = tree.mul(-2.0, a(x))
    for leaf_a, leaf_b in zip(jax.tree.leaves(scaled(x)), jax.tree.leaves(expected), strict=True):
        assert_allclose(leaf_a, leaf_b, rtol=1e-9)


def _not_a_stream() -> AbstractLinearOperator:
    return DiagonalOperator(jnp.ones(N_IN), in_structure=jax.ShapeDtypeStruct((N_IN,), jnp.float64))


def _other_n_lead() -> AbstractLinearOperator:
    return StreamOperator.column(_make_blocks(n_lead=N_OBS + 1))


def _other_out_structure() -> AbstractLinearOperator:
    return StreamOperator.column(_make_blocks(n_out=N_OUT + 1))


def _other_out_stacked() -> AbstractLinearOperator:
    # same per-slice out structure as a column, but summed rather than stacked
    return StreamOperator.addition(_make_blocks())


@pytest.mark.parametrize(
    'make_operand, error, match',
    [
        (_not_a_stream, TypeError, 'must be stream operators'),
        (_other_n_lead, ValueError, 'share n_lead'),
        (_other_out_structure, ValueError, 'per-slice junction structure'),
        (_other_out_stacked, ValueError, 'junction stack spec'),
    ],
)
def test_stream_block_row_rejects_non_conforming_operand(
    make_operand: Callable[[], AbstractLinearOperator], error: type[Exception], match: str
) -> None:
    h_sky, _, _ = _joint_operands()
    with pytest.raises(error, match=match):
        StreamOperator.block_row([h_sky, make_operand()])


def test_stream_block_column_rejects_non_conforming_operand() -> None:
    # the column delegates to the row constructor on transposed operands; `.T` is a bijection on
    # stream operators, so a non-stream operand is still caught (and named side-neutrally)
    h_sky, _, _ = _joint_operands()
    with pytest.raises(TypeError, match='must be stream operators'):
        StreamOperator.block_column([h_sky, _not_a_stream()])
    with pytest.raises(ValueError, match='at least one operand'):
        StreamOperator.block_column([])


def test_create_rejects_non_prefix_spec() -> None:
    # a two-component spec cannot resolve against this operator's single-leaf input structure;
    # jax names the offending key path and subtree types, so no wrapping is needed
    with pytest.raises(ValueError, match='pytree structure error'):
        StreamOperator.create(
            (StreamSegment(_make_blocks(), True),),
            n_lead=N_OBS,
            in_stacked=[True, False],
            out_stacked=True,
        )


# ---------------------------------------------------------------------------
# Multi-device: the paths a single-shard mesh cannot reach
# ---------------------------------------------------------------------------


@pytest.mark.distributed
class TestSharded:
    """Exercises `mv` on a mesh with more than one shard.

    On the single-device mesh of a default run the sharded machinery is invisible: `psum` is the
    identity, the per-shard scan length equals `n_lead`, and every output spec describes the whole
    array. The parts that only mean something across shards -- per-component `out_specs`, the
    shared-output carry and its reduction, the scan length -- are covered here.

    The mesh is built from the first `N_SHARDS` devices rather than all of them, so the class does
    not care how many the session provides and `N_OBS` need only divide `N_SHARDS`.
    """

    N_SHARDS = 2

    @pytest.fixture(autouse=True)
    def sharded_mesh(self) -> Iterator[None]:
        """Shadow the module mesh; arrays built in a test resolve `P('obs')` against this one."""
        devices = jax.devices()[: self.N_SHARDS]
        with jax.set_mesh(jax.make_mesh((len(devices),), ('obs',), devices=devices)):
            yield

    @pytest.mark.parametrize('make_stream, in_stacked, out_stacked', _LAYOUTS)
    def test_layout_mv(
        self,
        make_stream: Callable[[AbstractLinearOperator], StreamOperator],
        in_stacked: bool,
        out_stacked: bool,
    ) -> None:
        blocks = _make_blocks(P('obs'))
        op = make_stream(blocks)
        x = _boundary(in_stacked, N_IN)
        expected = _reference(
            _per_obs(blocks),
            np.asarray(jax.device_get(x)),
            in_stacked=in_stacked,
            out_stacked=out_stacked,
        )
        # a shared output is a per-shard partial sum reduced by `psum`; a stacked one is the shards'
        # slices concatenated. Both are no-ops on one shard, so this is where they mean something.
        assert_allclose(op(x), expected, rtol=1e-10)
        struct = jax.ShapeDtypeStruct(x.shape, jnp.float64, sharding=_spec(in_stacked))
        assert jax.eval_shape(op.mv, struct).sharding.spec == _spec(out_stacked)

    def test_mixed_normal_equations(self) -> None:
        # a mixed boundary is what needs per-component out_specs: within one scan the sky leg is
        # summed across shards and the amplitude leg stays sharded.
        h_sky, te, w = _joint_operands()
        h = StreamOperator.block_row([h_sky, te])
        a = (h.T @ w @ h).reduce()
        hs = _per_obs(h_sky.segments[0].operator)
        ts = _per_obs(te.segments[0].operator)
        ws = _per_obs(w.segments[0].operator)

        x_sky = jax.device_put(RNG.standard_normal((N_IN,), dtype=np.float64), P())
        x_amp = jax.device_put(RNG.standard_normal((N_OBS, N_AMP), dtype=np.float64), P('obs'))
        sky_np, amp_np = np.array(x_sky), np.array(jax.device_get(x_amp))
        wd = [ws[i] @ (hs[i] @ sky_np + ts[i] @ amp_np[i]) for i in range(N_OBS)]
        exp_sky = sum(hs[i].T @ wd[i] for i in range(N_OBS))
        exp_amp = np.stack([ts[i].T @ wd[i] for i in range(N_OBS)])

        out_sky, out_amp = a([x_sky, x_amp])
        assert_allclose(out_sky, exp_sky, rtol=1e-9)
        assert_allclose(out_amp, exp_amp, rtol=1e-9)
        assert _count_scans(jax.make_jaxpr(a.mv)([x_sky, x_amp]).jaxpr) == 1

        y = jax.eval_shape(
            a.mv,
            [
                jax.ShapeDtypeStruct((N_IN,), jnp.float64, sharding=P()),
                jax.ShapeDtypeStruct((N_OBS, N_AMP), jnp.float64, sharding=P('obs')),
            ],
        )
        assert 'obs' not in y[0].sharding.spec  # sky leg reduced, hence replicated
        assert y[1].sharding.spec == P('obs', None)  # amplitude leg still sharded

    def test_indivisible_batch_axis_is_rejected(self) -> None:
        # the scan length is per shard, so a batch axis the shards do not divide has no valid one.
        # Blocks stay unsharded here: `P('obs')` could not place them over the mesh to begin with.
        n_lead = self.N_SHARDS + 1
        op = StreamOperator.diagonal(_make_blocks(n_lead=n_lead))
        with pytest.raises(ValueError, match='not divisible'):
            op(jnp.zeros((n_lead, N_IN)))
