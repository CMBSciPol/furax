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
from furax.core import DiagonalOperator, HomothetyOperator
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
# Multi-segment bodies: constant maps that keep sliced segments apart
# ---------------------------------------------------------------------------


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


def test_addition_composition_does_not_fuse() -> None:
    # a shared junction is a psum only available after the scan, so `Addition @ Addition` cannot
    # fuse into one scan: (Σᵢaᵢ)(Σⱼbⱼ) != Σᵢ aᵢbᵢ. It must stay an unreduced composition.
    a = StreamOperator.addition(_make_blocks(P('obs'), n_in=N_OUT))
    b = StreamOperator.addition(_make_blocks(P('obs'), n_in=N_OUT))
    reduced = (a @ b).reduce()
    assert not isinstance(reduced, StreamOperator)
    x = jax.device_put(RNG.standard_normal((N_OUT,), dtype=np.float64), P())
    assert_allclose(reduced(x), a(b(x)), rtol=1e-10)


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
