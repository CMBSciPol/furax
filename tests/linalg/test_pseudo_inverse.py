import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numpy.testing import assert_allclose

from furax import AbstractLinearOperator
from furax.linalg import DensePseudoInverseOperator


def test_pseudo_inverse_matches_numpy():
    a = jr.normal(jr.key(0), (2, 5, 3))
    matrix = a @ jnp.swapaxes(a, -1, -2)  # rank 3 of 5
    op = DensePseudoInverseOperator.from_dense(matrix)
    x = jr.normal(jr.key(1), (2, 5))
    expected = np.einsum('...ij,...j->...i', np.linalg.pinv(np.asarray(matrix), hermitian=True), x)
    assert_allclose(op(x), expected, rtol=1e-8, atol=1e-10)


def test_scaled_pseudo_inverse_leaves_out_directions_cancelled_up_to_rounding():
    # A difference cancelling on 3 directions up to an error far above the rounding of its largest
    # eigenvalue, over amplitudes in disparate units: normalized by the diagonal of the terms, the
    # cancelled directions are left out.
    k = jr.split(jr.key(2), 3)
    m, n_null = 8, 3
    units = jnp.logspace(-3, 3, m)
    basis, _ = jnp.linalg.qr(jr.normal(k[0], (m, m)))
    kept = basis[:, n_null:] * jnp.sqrt(jnp.linspace(0.5, 1, m - n_null))
    error = 1e-11 * jr.normal(k[1], (m, m))
    matrix = units[:, None] * (kept @ kept.T + (error + error.T)) * units
    scale = units**2 * (1 + jr.uniform(k[2], (m,)))
    op = DensePseudoInverseOperator.from_dense(matrix, scale=scale, rtol=1e-8)
    assert int(jnp.sum(op.inverse_eigenvalues != 0)) == m - n_null

    x = jr.normal(jr.key(3), (m,))
    b = matrix @ x
    assert_allclose(matrix @ op(b), b, atol=1e-8 * jnp.abs(b).max())


def test_pseudo_inverse_of_a_pytree():
    matrix = jnp.diag(jnp.array([2.0, 4.0, 0.0]))
    structure = (None, jax.ShapeDtypeStruct((3,), jnp.float64))
    op = DensePseudoInverseOperator.from_dense(matrix, structure)
    y_a, y_c = op((None, jnp.array([2.0, 4.0, 1.0])))
    assert y_a is None
    assert_allclose(y_c, [1.0, 1.0, 0.0])


def test_clamped_pseudo_inverse_inverts_the_directions_left_out_at_the_cutoff():
    # One direction cancelled below the cutoff: left out by the pseudo-inverse, inverted at the
    # cutoff when clamped, the others alike.
    basis, _ = jnp.linalg.qr(jr.normal(jr.key(4), (4, 4)))
    eigenvalues = jnp.array([1e-12, 0.5, 1.0, 2.0])
    matrix = basis @ jnp.diag(eigenvalues) @ basis.T
    left_out = DensePseudoInverseOperator.from_dense(matrix, rtol=1e-8)
    clamped = DensePseudoInverseOperator.from_dense(matrix, rtol=1e-8, clamp=True)
    cutoff = 1e-8 * 2.0
    assert_allclose(left_out.inverse_eigenvalues, [0, 2, 1, 0.5])
    assert_allclose(clamped.inverse_eigenvalues, [1 / cutoff, 2, 1, 0.5])


def test_as_matrix_is_that_of_the_operator():
    # batched over two leaves, whose flattened entries interleave the batch elements
    a = jr.normal(jr.key(5), (2, 5, 3))
    matrix = a @ jnp.swapaxes(a, -1, -2)
    structure = (
        jax.ShapeDtypeStruct((2, 2), jnp.float64),
        jax.ShapeDtypeStruct((2, 3), jnp.float64),
    )
    op = DensePseudoInverseOperator.from_dense(matrix, structure)
    assert_allclose(op.as_matrix(), AbstractLinearOperator.as_matrix(op), atol=1e-12)
