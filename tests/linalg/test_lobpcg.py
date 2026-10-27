"""Tests for the LOBPCG eigenvalue solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.linalg import LOBPCGResult, lobpcg_standard
from furax.tree import as_structure, stacked_gram, stacked_normal_like


def _diagonal_op(d):
    return DiagonalOperator(d, in_structure=as_structure(d))


def _dense_operator(mat):
    structure = jax.ShapeDtypeStruct(mat.shape[:1], mat.dtype)
    return DenseBlockDiagonalOperator(mat, in_structure=structure)


def _laplacian(n):
    """1D Laplacian, with eigenvalues 2 - 2 cos(π j / (n + 1)), j = 1..n."""
    return 2 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1)


@pytest.mark.parametrize('largest', [False, True])
def test_lobpcg_diagonal(largest):
    d = jnp.arange(1.0, 9.0)
    A = _diagonal_op(d)
    X = stacked_normal_like(A.in_structure, 2, jax.random.key(0))

    result = lobpcg_standard(A, X, tol=1e-10, largest=largest)

    assert isinstance(result, LOBPCGResult)
    assert result.eigenvectors.shape == (2, 8)
    assert_allclose(result.eigenvalues, [7.0, 8.0] if largest else [1.0, 2.0], rtol=1e-10)
    assert result.converged.all()
    assert result.iterations < 100
    U = result.eigenvectors
    assert_allclose(U @ U.T, jnp.eye(2), atol=1e-12)
    assert_allclose(U * d, U * result.eigenvalues[:, None], atol=1e-9)


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_lobpcg_matches_eigh(dtype):
    """Smallest eigenpairs of a random Hermitian matrix, with residual norms that are true."""
    B = jax.random.normal(jax.random.key(0), (30, 30), dtype)
    mat = (B + B.conj().T) / 2
    A = _dense_operator(mat)
    X = stacked_normal_like(A.in_structure, 3, jax.random.key(1))

    result = lobpcg_standard(A, X, tol=1e-10, maxiter=300)

    assert result.converged.all()
    assert_allclose(result.eigenvalues, jnp.linalg.eigvalsh(mat)[:3], atol=1e-9)
    U = result.eigenvectors
    assert_allclose(stacked_gram(U, U), jnp.eye(3), atol=1e-12)
    residuals = jnp.linalg.norm(U @ mat.T - result.eigenvalues[:, None] * U, axis=1)
    assert_allclose(result.residual_norms, residuals, atol=1e-12)


def test_lobpcg_pytree():
    d1, d2 = jnp.array([1.0, 4.0, 6.0]), jnp.array([[2.0, 5.0], [3.0, 7.0]])
    A = BlockDiagonalOperator(
        {
            'a': _diagonal_op(d1),
            'b': _diagonal_op(d2),
        }
    )
    X = stacked_normal_like(A.in_structure, 2, jax.random.key(0))

    result = lobpcg_standard(A, X, tol=1e-10)

    assert result.eigenvectors['a'].shape == (2, 3)
    assert result.eigenvectors['b'].shape == (2, 2, 2)
    assert_allclose(result.eigenvalues, [1.0, 2.0], rtol=1e-10)


def test_lobpcg_preconditioner_accelerates_convergence():
    """An approximate inverse makes an ill-conditioned problem converge in a few iterations."""
    n, k = 100, 2
    mat = _laplacian(n)
    A = _dense_operator(mat)
    M = _dense_operator(jnp.linalg.inv(mat + 0.01 * jnp.eye(n)))
    X = stacked_normal_like(A.in_structure, k, jax.random.key(0))
    expected = 2 - 2 * jnp.cos(jnp.pi * jnp.arange(1, k + 1) / (n + 1))

    plain = lobpcg_standard(A, X, tol=1e-8, maxiter=30)
    preconditioned = lobpcg_standard(A, X, tol=1e-8, maxiter=30, preconditioner=M)

    assert not plain.converged.all()
    assert preconditioned.converged.all()
    assert preconditioned.iterations < 20
    assert_allclose(preconditioned.eigenvalues, expected, rtol=1e-8)


def test_lobpcg_repeated_eigenvalues():
    """Search directions that become dependent are dropped: no spurious zero eigenvalue."""
    d = jnp.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0])
    A = _diagonal_op(d)
    X = stacked_normal_like(A.in_structure, 3, jax.random.key(0))

    result = lobpcg_standard(A, X, tol=1e-10)

    assert result.converged.all()
    assert_allclose(result.eigenvalues, [1.0, 1.0, 1.0], rtol=1e-10)
    assert_allclose(result.eigenvectors[:, 3:], 0, atol=1e-9)


def test_lobpcg_null_space_converges():
    """Zero eigenvalues converge: the tolerance floor scales with ||A||, not the wanted values."""
    d = jnp.concatenate([jnp.zeros(2), jnp.arange(1.0, 101.0)])
    A = _diagonal_op(d)
    X = stacked_normal_like(A.in_structure, 2, jax.random.key(2))

    result = lobpcg_standard(A, X, tol=1e-10, maxiter=300)

    assert result.converged.all()
    assert result.iterations < 300
    assert_allclose(result.eigenvalues, 0, atol=1e-12)


@pytest.mark.parametrize(
    'X0',
    [
        jnp.eye(8)[:2],
        jnp.stack([jnp.eye(8)[0], jax.random.normal(jax.random.key(0), (8,))]),
        jax.random.normal(jax.random.key(1), (4, 8)).at[:, 5:].set(0),
    ],
    ids=['exact', 'one-exact', 'low-dimensional'],
)
def test_lobpcg_degenerate_start(X0):
    """Starts that leave few search directions alive give orthonormal, finite eigenvectors."""
    d = jnp.arange(1.0, 9.0)
    A = _diagonal_op(d)
    k = X0.shape[0]

    result = lobpcg_standard(A, X0, tol=1e-10)

    U = result.eigenvectors
    assert jnp.isfinite(U).all()
    assert_allclose(U @ U.T, jnp.eye(k), atol=1e-12)
    assert_allclose(U * d, U * result.eigenvalues[:, None], atol=1e-9)


def test_lobpcg_jit():
    d = jnp.arange(1.0, 9.0)
    A = _diagonal_op(d)
    X = stacked_normal_like(A.in_structure, 2, jax.random.key(0))

    result = jax.jit(lambda X: lobpcg_standard(A, X, tol=1e-10))(X)

    assert_allclose(result.eigenvalues, [1.0, 2.0], rtol=1e-10)


def test_lobpcg_too_many_vectors():
    d = jnp.arange(1.0, 4.0)
    A = _diagonal_op(d)
    with pytest.raises(ValueError, match='number of vectors'):
        lobpcg_standard(A, jnp.ones((4, 3)))


@pytest.mark.distributed
def test_lobpcg_sharded():
    """Eigenvectors keep the explicit sharding of the operator input."""
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ('i',), axis_types=(AxisType.Explicit,))
    d = jnp.arange(1.0, 4 * n + 1)
    with jax.set_mesh(mesh):
        d_s = jax.device_put(d, NamedSharding(mesh, P('i')))
        A = _diagonal_op(d_s)
        compute = jax.jit(
            lambda key: lobpcg_standard(A, stacked_normal_like(A.in_structure, 2, key), tol=1e-10)
        )
        result = compute(jax.random.key(0))

    assert result.eigenvectors.sharding.spec == P(None, 'i'), result.eigenvectors.sharding.spec
    assert_allclose(result.eigenvalues, [1.0, 2.0], rtol=1e-10)
    U = np.asarray(result.eigenvectors)
    assert_allclose(U @ U.T, np.eye(2), atol=1e-12)
