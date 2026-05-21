import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Inexact, PyTree
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator, OperatorTag
from furax.interfaces.lineax import as_lineax_operator
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


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=0)


@pytest.fixture(scope='module')
def mesh():
    return jax.make_mesh((jax.device_count(),), ('obs',))


def _make_blocks(
    rng: np.random.Generator,
    n_obs: int,
    n_out: int,
    n_in: int,
    sharding=None,
) -> _TestOp:
    matrices = rng.standard_normal((n_obs, n_out, n_in)).astype(np.float64)
    arr = (
        jax.device_put(jnp.asarray(matrices), sharding)
        if sharding is not None
        else jnp.asarray(matrices)
    )
    return _TestOp(arr, in_structure=jax.ShapeDtypeStruct((n_in,), jnp.float64))


def _per_obs(op: _TestOp) -> list[np.ndarray]:
    m = np.array(jax.device_get(op.matrix))
    return [m[i] for i in range(m.shape[0])]


# ---------------------------------------------------------------------------
# Structure shapes and sharding
# ---------------------------------------------------------------------------


def test_structure_shapes(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)

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


def test_structure_sharding_sharded(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)

    # Diagonal and Row: in_structure is obs-sharded along first dim
    for cls in (ScanBlockDiagonalOperator, ScanBlockRowOperator):
        op = cls.create(blocks)
        s = op.in_structure.sharding
        assert isinstance(s, NamedSharding)
        assert s.spec[0] is not None

    # Column and Addition: in_structure is replicated
    for cls in (ScanBlockColumnOperator, ScanAdditionOperator):
        op = cls.create(blocks)
        s = op.in_structure.sharding
        assert isinstance(s, NamedSharding)
        assert all(a is None for a in s.spec)  # replicated: P() padded to P(None,...)


# ---------------------------------------------------------------------------
# Unsharded forward
# ---------------------------------------------------------------------------


def test_unsharded_diagonal_mv(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockDiagonalOperator.create(blocks)
    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64))
    expected = np.stack([m @ np.array(x[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op.mv(x), expected, rtol=1e-5)


def test_unsharded_column_mv(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockColumnOperator.create(blocks)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    expected = np.stack([m @ np.array(x) for m in _per_obs(blocks)])
    assert_allclose(op.mv(x), expected, rtol=1e-5)


def test_unsharded_row_mv(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op = ScanBlockRowOperator.create(blocks)
    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64))
    expected = sum(m @ np.array(x[i]) for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op.mv(x), expected, rtol=1e-5)


def test_unsharded_addition_mv(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op = ScanAdditionOperator.create(blocks)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    expected = sum(m @ np.array(x) for m in _per_obs(blocks))
    assert_allclose(op.mv(x), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Unsharded transpose
# ---------------------------------------------------------------------------


def test_unsharded_diagonal_transpose(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op_T = ScanBlockDiagonalOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockDiagonalOperator)
    y = jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float64))
    expected = np.stack([m.T @ np.array(y[i]) for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


def test_unsharded_column_transpose_is_row(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op_T = ScanBlockColumnOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockRowOperator)
    y = jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float64))
    expected = sum(m.T @ np.array(y[i]) for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


def test_unsharded_row_transpose_is_column(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op_T = ScanBlockRowOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockColumnOperator)
    y = jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float64))
    expected = np.stack([m.T @ np.array(y) for m in _per_obs(blocks)])
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


def test_unsharded_addition_transpose(rng: np.random.Generator) -> None:
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    op_T = ScanAdditionOperator.create(blocks).T
    assert isinstance(op_T, ScanAdditionOperator)
    y = jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float64))
    expected = sum(m.T @ np.array(y) for m in _per_obs(blocks))
    assert_allclose(op_T.mv(y), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sharded forward
# ---------------------------------------------------------------------------


def test_sharded_diagonal_mv(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op = ScanBlockDiagonalOperator.create(blocks)
    x = jax.device_put(
        jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64)), obs_sharding
    )
    x_np = np.array(jax.device_get(x))
    expected = np.stack([m @ x_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(np.array(op.mv(x)), expected, rtol=1e-5)


def test_sharded_column_mv(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op = ScanBlockColumnOperator.create(blocks)
    x = jax.device_put(jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64)), no_sharding)
    expected = np.stack([m @ np.array(x) for m in _per_obs(blocks)])
    assert_allclose(np.array(op.mv(x)), expected, rtol=1e-5)


def test_sharded_row_mv(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op = ScanBlockRowOperator.create(blocks)
    x = jax.device_put(
        jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64)), obs_sharding
    )
    x_np = np.array(jax.device_get(x))
    expected = sum(m @ x_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(np.array(op.mv(x)), expected, rtol=1e-5)


def test_sharded_addition_mv(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op = ScanAdditionOperator.create(blocks)
    x = jax.device_put(jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64)), no_sharding)
    expected = sum(m @ np.array(x) for m in _per_obs(blocks))
    assert_allclose(np.array(op.mv(x)), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Sharded transpose
# ---------------------------------------------------------------------------


def test_sharded_diagonal_transpose(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op_T = ScanBlockDiagonalOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockDiagonalOperator)
    y = jax.device_put(
        jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float64)), obs_sharding
    )
    y_np = np.array(jax.device_get(y))
    expected = np.stack([m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks))])
    assert_allclose(np.array(op_T.mv(y)), expected, rtol=1e-5)


def test_sharded_column_transpose_is_row(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op_T = ScanBlockColumnOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockRowOperator)
    y = jax.device_put(
        jnp.asarray(rng.standard_normal((N_OBS, N_OUT)).astype(np.float64)), obs_sharding
    )
    y_np = np.array(jax.device_get(y))
    expected = sum(m.T @ y_np[i] for i, m in enumerate(_per_obs(blocks)))
    assert_allclose(np.array(op_T.mv(y)), expected, rtol=1e-5)


def test_sharded_row_transpose_is_column(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op_T = ScanBlockRowOperator.create(blocks).T
    assert isinstance(op_T, ScanBlockColumnOperator)
    y = jax.device_put(jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float64)), no_sharding)
    expected = np.stack([m.T @ np.array(y) for m in _per_obs(blocks)])
    assert_allclose(np.array(op_T.mv(y)), expected, rtol=1e-5)


def test_sharded_addition_transpose(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    op_T = ScanAdditionOperator.create(blocks).T
    assert isinstance(op_T, ScanAdditionOperator)
    y = jax.device_put(jnp.asarray(rng.standard_normal((N_OUT,)).astype(np.float64)), no_sharding)
    expected = sum(m.T @ np.array(y) for m in _per_obs(blocks))
    assert_allclose(np.array(op_T.mv(y)), expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Output sharding
# ---------------------------------------------------------------------------


def test_specs(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    obs_spec = P(mesh.axis_names)

    for cls in (
        ScanBlockDiagonalOperator,
        ScanBlockColumnOperator,
        ScanBlockRowOperator,
        ScanAdditionOperator,
    ):
        assert cls.create(blocks)._in_specs == obs_spec

    assert ScanBlockDiagonalOperator.create(blocks)._out_specs == obs_spec
    assert ScanBlockColumnOperator.create(blocks)._out_specs == obs_spec
    assert ScanBlockRowOperator.create(blocks)._out_specs == P()
    assert ScanAdditionOperator.create(blocks)._out_specs == P()


def test_sharded_output_sharding(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)

    cases = [
        (ScanBlockDiagonalOperator, jnp.ones((N_OBS, N_IN), jnp.float64), obs_sharding),
        (ScanBlockColumnOperator, jnp.ones((N_IN,), jnp.float64), no_sharding),
        (ScanBlockRowOperator, jnp.ones((N_OBS, N_IN), jnp.float64), obs_sharding),
        (ScanAdditionOperator, jnp.ones((N_IN,), jnp.float64), no_sharding),
    ]
    for cls, x, x_sharding in cases:
        op = cls.create(blocks)
        y = op.mv(jax.device_put(x, x_sharding))
        assert y.sharding == NamedSharding(mesh, op._out_specs)


# ---------------------------------------------------------------------------
# Fusion rules (unsharded)
# ---------------------------------------------------------------------------


def test_fusion_diag_diag(rng: np.random.Generator) -> None:
    left = ScanBlockDiagonalOperator.create(_make_blocks(rng, N_OBS, N_IN, N_IN))
    right = ScanBlockDiagonalOperator.create(_make_blocks(rng, N_OBS, N_IN, N_IN))
    reduced = (left @ right).reduce()
    assert isinstance(reduced, ScanBlockDiagonalOperator)
    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64))
    assert_allclose(reduced.mv(x), (left @ right).mv(x), rtol=1e-5)


def test_fusion_diag_column(rng: np.random.Generator) -> None:
    left = ScanBlockDiagonalOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_OUT))
    right = ScanBlockColumnOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_IN))
    reduced = (left @ right).reduce()
    assert isinstance(reduced, ScanBlockColumnOperator)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    assert_allclose(reduced.mv(x), (left @ right).mv(x), rtol=1e-5)


def test_fusion_row_diag(rng: np.random.Generator) -> None:
    left = ScanBlockRowOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_IN))
    right = ScanBlockDiagonalOperator.create(_make_blocks(rng, N_OBS, N_IN, N_IN))
    reduced = (left @ right).reduce()
    assert isinstance(reduced, ScanBlockRowOperator)
    x = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64))
    assert_allclose(reduced.mv(x), (left @ right).mv(x), rtol=1e-5)


def test_fusion_row_column(rng: np.random.Generator) -> None:
    left = ScanBlockRowOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_IN))
    right = ScanBlockColumnOperator.create(_make_blocks(rng, N_OBS, N_IN, N_IN))
    reduced = (left @ right).reduce()
    assert isinstance(reduced, ScanAdditionOperator)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    assert_allclose(reduced.mv(x), (left @ right).mv(x), rtol=1e-5)


def test_fusion_ht_w_h(rng: np.random.Generator) -> None:
    H = ScanBlockColumnOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_IN))
    W = ScanBlockDiagonalOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_OUT))
    reduced = (H.T @ W @ H).reduce()
    assert isinstance(reduced, ScanAdditionOperator)
    x = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    assert_allclose(reduced.mv(x), (H.T @ W @ H).mv(x), rtol=1e-5)


def test_sharded_fusion_ht_w_h(rng: np.random.Generator, mesh) -> None:
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    H = ScanBlockColumnOperator.create(_make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding))
    W = ScanBlockDiagonalOperator.create(
        _make_blocks(rng, N_OBS, N_OUT, N_OUT, sharding=obs_sharding)
    )
    reduced = (H.T @ W @ H).reduce()
    assert isinstance(reduced, ScanAdditionOperator)
    x = jax.device_put(jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64)), no_sharding)
    assert_allclose(np.array(reduced.mv(x)), np.array((H.T @ W @ H).mv(x)), rtol=1e-5)


# ---------------------------------------------------------------------------
# Lineax CG integration
# ---------------------------------------------------------------------------


def _make_spd_blocks(rng: np.random.Generator, n_obs: int, n: int, sharding=None) -> _TestOp:
    """Each block is M @ M.T + I — guaranteed SPD."""
    m = rng.standard_normal((n_obs, n, n)).astype(np.float64)
    spd = m @ m.swapaxes(-1, -2) + np.eye(n)
    arr = jax.device_put(jnp.asarray(spd), sharding) if sharding is not None else jnp.asarray(spd)
    return _TestOp(arr, in_structure=jax.ShapeDtypeStruct((n,), jnp.float64))


def test_cg_scan_addition_unsharded(rng: np.random.Generator) -> None:
    # A = H^T W H fused to ScanAdditionOperator — SPD when W is PD
    H_blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN)
    W_blocks = _make_spd_blocks(rng, N_OBS, N_OUT)
    H = ScanBlockColumnOperator.create(H_blocks)
    W = ScanBlockDiagonalOperator.create(W_blocks)
    A = (H.T @ W @ H).reduce()
    assert isinstance(A, ScanAdditionOperator)

    lx_A = as_lineax_operator(A, tags=OperatorTag.SYMMETRIC | OperatorTag.POSITIVE_SEMIDEFINITE)
    b = jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64))
    sol = lx.linear_solve(lx_A, b, solver=lx.CG(rtol=1e-4, atol=1e-4))
    assert_allclose(A.mv(sol.value), b, rtol=1e-3, atol=1e-3)


def test_cg_scan_diagonal_unsharded(rng: np.random.Generator) -> None:
    # Block-diagonal SPD operator: each obs solved independently via CG
    blocks = _make_spd_blocks(rng, N_OBS, N_IN)
    A = ScanBlockDiagonalOperator.create(blocks)

    lx_A = as_lineax_operator(A, tags=OperatorTag.SYMMETRIC | OperatorTag.POSITIVE_SEMIDEFINITE)
    b = jnp.asarray(rng.standard_normal((N_OBS, N_IN)).astype(np.float64))
    sol = lx.linear_solve(lx_A, b, solver=lx.CG(rtol=1e-4, atol=1e-4))
    assert_allclose(A.mv(sol.value), b, rtol=1e-3, atol=1e-3)


def test_cg_scan_addition_sharded(rng: np.random.Generator, mesh) -> None:
    # Sharded H^T W H — in/out are replicated (P()), lineax CG sees regular arrays
    obs_sharding = NamedSharding(mesh, P('obs'))
    no_sharding = NamedSharding(mesh, P())
    H_blocks = _make_blocks(rng, N_OBS, N_OUT, N_IN, sharding=obs_sharding)
    W_blocks = _make_spd_blocks(rng, N_OBS, N_OUT, sharding=obs_sharding)
    H = ScanBlockColumnOperator.create(H_blocks)
    W = ScanBlockDiagonalOperator.create(W_blocks)
    A = (H.T @ W @ H).reduce()
    assert isinstance(A, ScanAdditionOperator)

    lx_A = as_lineax_operator(A, tags=OperatorTag.SYMMETRIC | OperatorTag.POSITIVE_SEMIDEFINITE)
    b = jax.device_put(jnp.asarray(rng.standard_normal((N_IN,)).astype(np.float64)), no_sharding)
    sol = lx.linear_solve(lx_A, b, solver=lx.CG(rtol=1e-4, atol=1e-4))
    assert_allclose(np.array(A.mv(sol.value)), np.array(b), rtol=1e-3, atol=1e-3)
