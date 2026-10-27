"""Tests for the randomized Nyström approximation and preconditioner."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

from furax import BlockDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.linalg import LowRankTerms, make_nystrom_preconditioner, randomized_nystrom
from furax.tree import as_structure, stacked_gram


def _diagonal_op(d):
    return DiagonalOperator(d, in_structure=as_structure(d))


def _psd_operator(eigenvalues, key):
    """Dense operator with the given eigenvalues and random eigenvectors."""
    n = eigenvalues.size
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n), eigenvalues.dtype))
    mat = (Q * eigenvalues) @ Q.T
    structure = jax.ShapeDtypeStruct((n,), mat.dtype)
    return DenseBlockDiagonalOperator(mat, in_structure=structure), mat


def test_nystrom_full_rank_is_exact():
    d = jnp.arange(1.0, 6.0)
    A = _diagonal_op(d)

    terms = randomized_nystrom(A, rank=5, key=jax.random.key(0), oversampling=0)

    assert isinstance(terms, LowRankTerms)
    assert_allclose(terms.eigenvalues, d, rtol=1e-10)
    U = terms.eigenvectors
    assert_allclose(U @ U.T, jnp.eye(5), atol=1e-12)
    assert_allclose(U * d, U * terms.eigenvalues[:, None], atol=1e-10)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_nystrom_low_rank_operator(dtype):
    """An operator of rank below the number of samples is recovered, its null space excluded."""
    eigenvalues = jnp.zeros(40, dtype).at[:4].set(jnp.array([10.0, 5.0, 2.0, 1.0], dtype))
    # On GPU, float32 matrix products default to TF32, whose rounding would dominate.
    with jax.default_matmul_precision('highest'):
        A, mat = _psd_operator(eigenvalues, jax.random.key(0))
        terms = randomized_nystrom(A, rank=4, key=jax.random.key(1))

    rtol = 1e-4 if dtype == jnp.float32 else 1e-10
    assert terms.eigenvalues.dtype == dtype
    assert_allclose(terms.eigenvalues, [1.0, 2.0, 5.0, 10.0], rtol=rtol)
    U = terms.eigenvectors
    assert_allclose(U @ U.T, jnp.eye(4), atol=10 * rtol)
    assert_allclose((U.T * terms.eigenvalues) @ U, mat, atol=10 * rtol)


def test_nystrom_pytree():
    d1, d2 = jnp.array([1.0, 5.0]), jnp.array([[3.0], [0.5]])
    A = BlockDiagonalOperator(
        {
            'a': _diagonal_op(d1),
            'b': _diagonal_op(d2),
        }
    )

    terms = randomized_nystrom(A, rank=2, key=jax.random.key(0))

    assert terms.eigenvectors['a'].shape == (2, 2)
    assert terms.eigenvectors['b'].shape == (2, 2, 1)
    assert_allclose(terms.eigenvalues, [3.0, 5.0], rtol=1e-10)


def test_nystrom_jit():
    d = jnp.arange(1.0, 6.0)
    A = _diagonal_op(d)
    key = jax.random.key(0)

    expected = randomized_nystrom(A, rank=2, key=key)
    result = jax.jit(lambda key: randomized_nystrom(A, rank=2, key=key))(key)

    assert_allclose(result.eigenvalues, expected.eigenvalues, rtol=1e-12)


@pytest.mark.parametrize('rank', [0, 4])
def test_nystrom_invalid_rank(rank):
    d = jnp.arange(1.0, 4.0)
    A = _diagonal_op(d)
    with pytest.raises(ValueError, match='rank'):
        randomized_nystrom(A, rank=rank, key=jax.random.key(0))


@pytest.mark.parametrize('mu', [0.0, 0.1])
def test_nystrom_preconditioner_formula(mu):
    U = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    terms = LowRankTerms(jnp.array([2.0, 4.0]), U)

    preconditioner = make_nystrom_preconditioner(terms, mu=mu)

    expected = jnp.diag(jnp.array([1.0, (2 + mu) / (4 + mu), 1.0]))
    assert preconditioner.is_symmetric
    assert_allclose(preconditioner.as_matrix(), expected, rtol=1e-12)


@pytest.mark.parametrize('zero', [0.0, 1e-17], ids=['exact', 'rounding'])
def test_nystrom_preconditioner_ignores_zero_eigenvalues(zero):
    """Eigenvalues that are zero up to rounding leave their eigenvectors in the complement."""
    terms = LowRankTerms(jnp.array([zero, 1.0, 2.0, 4.0]), jnp.eye(5)[:4])

    preconditioner = make_nystrom_preconditioner(terms)

    expected = jnp.diag(jnp.array([1.0, 1.0, 0.5, 0.25, 1.0]))
    assert_allclose(preconditioner.as_matrix(), expected, rtol=1e-12)


def test_nystrom_preconditioner_reduces_condition_number():
    """Capturing the top of the spectrum maps it close to the smallest captured eigenvalue."""
    eigenvalues = jnp.concatenate([jnp.logspace(4, 0, 10), jnp.full(30, 1e-2)])
    A, mat = _psd_operator(eigenvalues, jax.random.key(0))
    mu = 1e-2
    terms = randomized_nystrom(A, rank=10, key=jax.random.key(1))

    preconditioner = make_nystrom_preconditioner(terms, mu=mu).as_matrix()

    system = mat + mu * jnp.eye(40)
    assert jnp.linalg.cond(system) > 1e5
    assert jnp.linalg.cond(preconditioner @ system) < 100


@pytest.mark.distributed
def test_nystrom_sharded():
    """Eigenvectors keep the explicit sharding of the operator input."""
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ('i',), axis_types=(AxisType.Explicit,))
    d = jnp.arange(1.0, 4 * n + 1)
    with jax.set_mesh(mesh):
        d_s = jax.device_put(d, NamedSharding(mesh, P('i')))
        A = _diagonal_op(d_s)
        compute = jax.jit(lambda key: randomized_nystrom(A, rank=2, key=key, oversampling=4 * n))
        terms = compute(jax.random.key(0))

    assert terms.eigenvectors.sharding.spec == P(None, 'i'), terms.eigenvectors.sharding.spec
    assert_allclose(terms.eigenvalues, [4.0 * n - 1, 4.0 * n], rtol=1e-10)
    U = np.asarray(terms.eigenvectors)
    assert_allclose(stacked_gram(U, U), np.eye(2), atol=1e-12)
