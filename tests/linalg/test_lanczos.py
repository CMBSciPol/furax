"""Tests for Lanczos eigenvalue solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Key
from numpy.testing import assert_allclose, assert_array_equal

from furax import BlockDiagonalOperator, DenseBlockDiagonalOperator, DiagonalOperator
from furax.linalg._lanczos import lanczos_eigh, lanczos_tr, lanczos_tridiag
from furax.tree import as_structure, normal_like


def _random_hermitian_operator(n: int, key: Key[Array, ''], dtype=None, *, pd=False):
    """Random Hermitian operator of size n.

    Returns (A, v0, eigenvalues) with eigenvalues sorted ascending.
    """
    key_mat, key_v0 = jax.random.split(key)
    B = jax.random.normal(key_mat, (n, n), dtype=dtype)
    if pd:
        mat = B @ B.conj().T + n * jnp.eye(n, dtype=B.dtype)
    else:
        mat = (B + B.conj().T) / 2
    eigenvalues = jnp.linalg.eigvalsh(mat)
    v0 = jax.random.normal(key_v0, (n,), dtype=dtype)
    A = DenseBlockDiagonalOperator(mat, in_structure=as_structure(v0))
    return A, v0, eigenvalues


class TestLanczosTridiag:
    """Tests for Lanczos tridiagonalization."""

    def test_lanczos_tridiag_orthonormal_vectors(self):
        """Lanczos basis is orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_hermitian_operator(30, jax.random.key(0), pd=True)
        _, _, V, _, _ = lanczos_tridiag(A, v0, m=12)
        assert_allclose(V @ V.T, jnp.eye(12), atol=1e-10)

    def test_lanczos_tridiag_eigenvalue_relation(self):
        """Tridiagonal T has same eigenvalues as A when Krylov space is full (m == n)."""
        A, v0, true_eigenvalues = _random_hermitian_operator(15, jax.random.key(1), pd=True)
        alpha, beta, _, _, _ = lanczos_tridiag(A, v0, m=15)
        eigenvalues = jax.scipy.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=True)
        assert_allclose(eigenvalues, true_eigenvalues, atol=1e-10)

    def test_lanczos_tridiag_breakdown(self):
        """An invariant Krylov subspace gives β ≈ 0 and the basis stays orthonormal."""
        d = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.key(0))
        _, beta, V, _, _ = lanczos_tridiag(A, v0, m=6)
        assert_allclose(beta[2], 0, atol=1e-14)  # the Krylov subspace of v0 has dimension 3
        assert_allclose(V @ V.T, jnp.eye(6), atol=1e-10)

    def test_lanczos_tridiag_exact_breakdown(self):
        """An exactly zero residual gives β = 0 and a random continuation vector."""
        d = jnp.ones(3)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = jnp.array([1.0, 0.0, 0.0])
        alpha, beta, V, _, _ = lanczos_tridiag(A, v0, m=2)
        assert_allclose(alpha, jnp.ones(2))
        assert_array_equal(beta, jnp.zeros(1))
        assert_allclose(V @ V.T, jnp.eye(2), atol=1e-14)

    def test_lanczos_tridiag_breakdown_last_step(self):
        """With m == n, a breakdown at the last step leaves a zero residual, not NaN."""
        d = jnp.ones(2)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = jnp.array([1.0, 0.0])
        with jax.debug_nans(True):
            alpha, beta, V, beta_last, v_last = lanczos_tridiag(A, v0, m=2)
        assert_allclose(alpha, jnp.ones(2))
        assert_array_equal(beta, jnp.zeros(1))
        assert_allclose(V @ V.T, jnp.eye(2), atol=1e-14)
        assert beta_last == 0
        assert_array_equal(v_last, jnp.zeros(2))

    def test_lanczos_tridiag_key(self):
        """The restart vector after a breakdown depends on the key."""
        d = jnp.ones(3)
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = jnp.array([1.0, 0.0, 0.0])
        _, _, V1, _, _ = lanczos_tridiag(A, v0, m=2, key=jax.random.key(0))
        _, _, V2, _, _ = lanczos_tridiag(A, v0, m=2, key=jax.random.key(1))
        assert not jnp.allclose(V1[1], V2[1])


class TestLanczosEigh:
    """Integration tests for Lanczos eigenvalue solver."""

    def test_lanczos_eigh_eigenvalues(self):
        """lanczos_eigh returns accurate eigenvalues when Krylov space is full (m == n)."""
        A, v0, true_eigenvalues = _random_hermitian_operator(15, jax.random.key(0), pd=True)
        result = lanczos_eigh(A, v0, k=5, m=15)

        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigenvalues[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(5), atol=1e-10)

    def test_lanczos_pytree_structure(self):
        """Test Lanczos with PyTree-structured operators."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )

        v0 = normal_like(A.in_structure, jax.random.key(3))
        result = lanczos_eigh(A, v0, k=3)

        assert_allclose(result.eigenvalues, jnp.array([1.0, 2.0, 3.0]), atol=1e-10)

    def test_lanczos_eigenvectors_orthonormal(self):
        """Ritz vectors are orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_hermitian_operator(30, jax.random.key(7), pd=True)
        result = lanczos_eigh(A, v0, k=4, m=12)

        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(4), atol=1e-10)

    def test_lanczos_eigh_repeated_eigenvalues(self):
        """With m == n, repeated eigenvalues are recovered with their multiplicity."""
        d = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.key(0))
        result = lanczos_eigh(A, v0, k=6, m=6)

        assert_allclose(result.eigenvalues, d, atol=1e-10)
        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(6), atol=1e-10)

    @pytest.mark.parametrize(
        'solver',
        [
            lambda A, v0: lanczos_tridiag(A, v0, m=6),
            lambda A, v0: lanczos_eigh(A, v0, k=2, m=6),
            lambda A, v0: lanczos_tr(A, v0, k=2, m=6),
        ],
        ids=['tridiag', 'eigh', 'tr'],
    )
    def test_lanczos_m_larger_than_n(self, solver):
        """m larger than the operator size is rejected."""
        A, v0, _ = _random_hermitian_operator(5, jax.random.key(0), pd=True)
        with pytest.raises(ValueError, match='must be <= the operator size'):
            solver(A, v0)


_UNCONVERGED = {
    lanczos_eigh: {'k': 5},
    lanczos_tr: {'k': 2, 'which': 'SA', 'max_restarts': 0},
}


@pytest.mark.parametrize('solver', [lanczos_eigh, lanczos_tr])
class TestLanczosStartingVector:
    """The starting vector is either given or drawn from a random key."""

    def test_key_only(self, solver):
        """With only a key, the solver finds eigenpairs of the operator."""
        A, _, true_eigenvalues = _random_hermitian_operator(10, jax.random.key(0), pd=True)
        result = solver(A, key=jax.random.key(1), k=2, m=10)

        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigenvalues), axis=1)
        assert_allclose(min_dist, jnp.zeros(2), atol=1e-10)
        residuals = (
            A.as_matrix() @ result.eigenvectors.T - result.eigenvectors.T * result.eigenvalues
        )
        assert_allclose(residuals, 0, atol=1e-10)

    def test_v0_takes_precedence(self, solver):
        """With both, the given v0 is used and the key does not draw another one."""
        A, v0, _ = _random_hermitian_operator(10, jax.random.key(0), pd=True)
        expected = solver(A, v0, k=2, m=4)
        result = solver(A, v0, key=jax.random.key(1), k=2, m=4)
        assert_allclose(result.eigenvalues, expected.eigenvalues)
        assert_allclose(result.eigenvectors, expected.eigenvectors)

    def test_key_draws_restart_vectors(self, solver):
        """After an invariant subspace is found, the iteration continues from the key."""
        d = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])  # Krylov subspace of dimension 3
        # Unconverged Ritz values from the 2 vectors after the restart depend on its direction.
        kwargs = _UNCONVERGED[solver]
        result1 = solver(A, v0, key=jax.random.key(0), m=5, **kwargs)
        result2 = solver(A, v0, key=jax.random.key(1), m=5, **kwargs)
        assert not jnp.allclose(result1.eigenvalues, result2.eigenvalues)

    def test_requires_v0_or_key(self, solver):
        """Without v0 or key there is no starting vector."""
        A, _, _ = _random_hermitian_operator(10, jax.random.key(0), pd=True)
        with pytest.raises(ValueError, match='v0 or a random key'):
            solver(A, k=2, m=4)


class TestLanczosThickRestart:
    """Tests for the thick-restart Lanczos method."""

    def test_tr_smallest_algebraic(self):
        """TR with which='SA' finds k smallest (algebraic) eigenvalues."""
        A, v0, true_eigenvalues = _random_hermitian_operator(20, jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=8, which='SA')
        assert_allclose(result.eigenvalues, true_eigenvalues[:2], atol=1e-10)

    def test_tr_largest_algebraic(self):
        """TR with which='LA' finds k largest (algebraic) eigenvalues."""
        A, v0, true_eigenvalues = _random_hermitian_operator(20, jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=8, which='LA')
        assert_allclose(result.eigenvalues, true_eigenvalues[-2:], atol=1e-10)

    def test_tr_largest_magnitude(self):
        """TR with which='LM' finds k largest |λ| (differs from LA for indefinite A)."""
        A, v0, true_eigenvalues = _random_hermitian_operator(20, jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=8, which='LM')
        expected = jnp.sort(true_eigenvalues[jnp.argsort(jnp.abs(true_eigenvalues))[-2:]])
        assert_allclose(jnp.sort(result.eigenvalues), expected, atol=1e-10)

    def test_tr_both_ends(self):
        """TR with which='BE' returns half from each end of the spectrum."""
        A, v0, true_eigenvalues = _random_hermitian_operator(20, jax.random.key(42))
        result = lanczos_tr(A, v0, k=4, m=10, which='BE')
        expected = jnp.concatenate([true_eigenvalues[:2], true_eigenvalues[-2:]])
        assert_allclose(result.eigenvalues, expected, atol=1e-10)

    def test_tr_smallest_magnitude(self):
        """TR with which='SM' targets smallest |λ|.

        Smallest-magnitude eigenvalues are interior, which plain Lanczos resolves
        slowly, so this uses a diagonal operator (Lanczos is near-exact on it).
        """
        d = jnp.array([-5.0, -1.0, 0.5, 3.0, 8.0, -7.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=5, which='SM', tol=1e-10)
        assert_allclose(jnp.sort(result.eigenvalues), jnp.array([-1.0, 0.5]), atol=1e-10)

    def test_tr_invalid_which(self):
        """TR raises on an unknown which value."""
        A, v0, _ = _random_hermitian_operator(5, jax.random.key(0), pd=True)
        with pytest.raises(ValueError, match='which must be one of'):
            lanczos_tr(A, v0, k=2, m=4, which='smallest')

    def test_tr_eigenvectors_orthonormal(self):
        """TR Ritz vectors are orthonormal for a random SPD operator with m < n."""
        A, v0, _ = _random_hermitian_operator(30, jax.random.key(0), pd=True)
        result = lanczos_tr(A, v0, k=3, m=8)

        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(3), atol=1e-10)

    def test_tr_pytree_operator(self):
        """TR works with PyTree-structured operators."""
        d1 = jnp.array([1.0, 2.0])
        d2 = jnp.array([3.0, 4.0, 5.0])
        structure = {'a': as_structure(d1), 'b': as_structure(d2)}
        A = BlockDiagonalOperator(
            {
                'a': DiagonalOperator(d1, in_structure=structure['a']),
                'b': DiagonalOperator(d2, in_structure=structure['b']),
            }
        )
        v0 = {
            'a': jax.random.normal(jax.random.key(0), (2,)),
            'b': jax.random.normal(jax.random.key(1), (3,)),
        }
        tol = 1e-6

        result = lanczos_tr(A, v0, k=2, m=4, tol=tol)

        assert jnp.all(result.residual_norms < tol)
        true_eigs = jnp.concatenate([d1, d2])
        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigs[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(2), atol=1e-10)

    @pytest.mark.parametrize('which', ['SA', 'SM', 'LA', 'LM', 'BE'])
    def test_tr_repeated_eigenvalues(self, which):
        """With repeated eigenvalues, TR returns genuine eigenpairs, not spurious zeros."""
        d = jnp.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        A = DiagonalOperator(d, in_structure=as_structure(d))
        v0 = normal_like(as_structure(d), jax.random.key(0))
        result = lanczos_tr(A, v0, k=2, m=4, which=which)

        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - d[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(2), atol=1e-10)
        G = result.eigenvectors @ result.eigenvectors.T
        assert_allclose(G, jnp.eye(2), atol=1e-10)
        residuals = (
            A.as_matrix() @ result.eigenvectors.T - result.eigenvectors.T * result.eigenvalues
        )
        assert_allclose(residuals, 0, atol=1e-10)

    def test_tr_requires_m_greater_than_k(self):
        """TR raises when m <= k."""
        A, v0, _ = _random_hermitian_operator(5, jax.random.key(0), pd=True)

        with pytest.raises(ValueError, match='m .* must be > k'):
            lanczos_tr(A, v0, k=5, m=5)


class TestLanczosComplexHermitian:
    """Tests for Lanczos with complex Hermitian operators."""

    def test_tridiag_orthonormal_basis(self):
        """Lanczos basis is orthonormal for a complex Hermitian operator."""
        A, v0, _ = _random_hermitian_operator(20, jax.random.key(0), dtype=jnp.complex128, pd=True)
        _, _, V, _, _ = lanczos_tridiag(A, v0, m=10)
        assert_allclose(V @ V.conj().T, jnp.eye(10), atol=1e-10)

    def test_tridiag_real_alpha_beta(self):
        """Tridiagonal coefficients are real for a complex Hermitian operator."""
        A, v0, _ = _random_hermitian_operator(20, jax.random.key(1), dtype=jnp.complex128, pd=True)
        alpha, beta, _, _, _ = lanczos_tridiag(A, v0, m=10)
        assert not jnp.iscomplexobj(alpha)
        assert not jnp.iscomplexobj(beta)

    def test_eigh_eigenvalues(self):
        """lanczos_eigh returns accurate eigenvalues for a complex Hermitian operator."""
        A, v0, true_eigenvalues = _random_hermitian_operator(
            15, jax.random.key(2), dtype=jnp.complex128, pd=True
        )
        result = lanczos_eigh(A, v0, k=5, m=15)

        min_dist = jnp.min(jnp.abs(result.eigenvalues[:, None] - true_eigenvalues[None, :]), axis=1)
        assert_allclose(min_dist, jnp.zeros(5), atol=1e-10)

    def test_eigh_eigenvectors_orthonormal(self):
        """Ritz vectors are orthonormal for a complex Hermitian operator."""
        A, v0, _ = _random_hermitian_operator(20, jax.random.key(3), dtype=jnp.complex128, pd=True)
        result = lanczos_eigh(A, v0, k=4, m=12)

        G = result.eigenvectors @ result.eigenvectors.conj().T
        assert_allclose(G, jnp.eye(4), atol=1e-10)

    def test_tr_smallest_eigenvalues(self):
        """TR finds k smallest eigenvalues of a complex Hermitian operator."""
        A, v0, true_eigenvalues = _random_hermitian_operator(
            20, jax.random.key(4), dtype=jnp.complex128
        )
        result = lanczos_tr(A, v0, k=2, m=8, which='SA')
        assert_allclose(result.eigenvalues, true_eigenvalues[:2], atol=1e-10)

    def test_tr_largest_eigenvalues(self):
        """TR finds k largest eigenvalues of a complex Hermitian operator."""
        A, v0, true_eigenvalues = _random_hermitian_operator(
            20, jax.random.key(5), dtype=jnp.complex128
        )
        result = lanczos_tr(A, v0, k=2, m=8, which='LA')
        assert_allclose(result.eigenvalues, true_eigenvalues[-2:], atol=1e-10)

    def test_tr_eigenvectors_orthonormal(self):
        """TR Ritz vectors are orthonormal for a complex Hermitian operator."""
        A, v0, _ = _random_hermitian_operator(20, jax.random.key(6), dtype=jnp.complex128, pd=True)
        result = lanczos_tr(A, v0, k=3, m=8)

        G = result.eigenvectors @ result.eigenvectors.conj().T
        assert_allclose(G, jnp.eye(3), atol=1e-10)


@pytest.mark.distributed
@pytest.mark.parametrize('solver', [lanczos_eigh, lanczos_tr])
def test_lanczos_sharded(solver):
    """Eigenvectors keep the explicit sharding of the operator input, across a breakdown."""
    n = jax.device_count()
    mesh = jax.make_mesh((n,), ('i',), axis_types=(AxisType.Explicit,))
    d = jnp.repeat(jnp.arange(1.0, 4.0), n)  # 3 distinct values: breakdown after 3 steps
    with jax.set_mesh(mesh):
        d_s = jax.device_put(d, NamedSharding(mesh, P('i')))
        A = DiagonalOperator(d_s, in_structure=as_structure(d_s))
        result = jax.jit(lambda key: solver(A, key=key, k=2, m=5))(jax.random.key(0))

    assert result.eigenvectors.sharding.spec == P(None, 'i'), result.eigenvectors.sharding.spec
    U = np.asarray(result.eigenvectors)
    assert_allclose(U @ U.T, np.eye(2), atol=1e-10)
    assert_allclose(U * np.asarray(d), U * np.asarray(result.eigenvalues)[:, None], atol=1e-10)
