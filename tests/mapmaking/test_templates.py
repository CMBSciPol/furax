import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking.templates import ATOPProjectionOperator


class TestATOPProjectionOperator:
    def make_op(self, n_det: int, n_samp: int, tau: int) -> ATOPProjectionOperator:
        return ATOPProjectionOperator(
            tau, in_structure=jax.ShapeDtypeStruct((n_det, n_samp), jnp.float64)
        )

    @pytest.mark.parametrize(
        'tau,x_vals,expected_vals',
        [
            (2, [1, 3, 5, 7], [-1, 1, -1, 1]),
            (3, [0, 3, 6, 10, 10, 10], [-3, 0, 3, 0, 0, 0]),
            (4, [0, 0, 0, 0, 1, 1, 1, 1, 2, 4, 6, 8], [0, 0, 0, 0, 0, 0, 0, 0, -3, -1, 1, 3]),
        ],
    )
    def test_demeaning_per_interval(self, tau: int, x_vals: list, expected_vals: list):
        """Each interval has its own mean removed, not a global one."""
        x = jnp.array([x_vals], dtype=jnp.float64)
        y = self.make_op(1, len(x_vals), tau)(x)
        assert_allclose(y, [expected_vals])

    def test_tail_passed_through_unchanged(self):
        """Samples in the partial tail interval are returned as-is."""
        n_det, n_samp, tau = 1, 10, 4  # tail at indices 8, 9
        x = jnp.arange(n_det * n_samp, dtype=jnp.float64).reshape(n_det, n_samp)
        y = self.make_op(n_det, n_samp, tau)(x)
        assert_array_equal(y[0, 2 * tau :], x[0, 2 * tau :])

    def test_multiple_detectors_demeaned_independently(self):
        """Each detector row is demeaned using its own interval means."""
        tau = 4
        n_det, n_samp = 3, 8
        x = jnp.stack([jnp.full((n_samp,), float(d)) for d in range(n_det)])
        y = self.make_op(n_det, n_samp, tau)(x)
        assert_array_equal(y, jnp.zeros((n_det, n_samp)))

    @pytest.mark.parametrize('n_samp,tau', [(8, 4), (10, 4)])
    def test_idempotent(self, n_samp: int, tau: int):
        """Test that op(op(x)) == op(x): operator is a projector."""
        n_det = 2
        x = jax.random.normal(jax.random.PRNGKey(0), (n_det, n_samp))
        op = self.make_op(n_det, n_samp, tau)
        assert_allclose(op(op(x)), op(x), atol=1e-6)

    def test_tau_one(self):
        """With tau=1, every sample is its own interval, so output is all zeros."""
        n_det, n_samp = 2, 6
        x = jax.random.normal(jax.random.PRNGKey(1), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=1)(x)
        assert_allclose(y, jnp.zeros((n_det, n_samp)), atol=1e-6)

    def test_tau_equals_n_samp(self):
        """With tau=n_samp, a single interval covers all samples; output sums to zero."""
        n_det, n_samp = 2, 8
        x = jax.random.normal(jax.random.PRNGKey(2), (n_det, n_samp))
        y = self.make_op(n_det, n_samp, tau=n_samp)(x)
        assert_allclose(y.sum(axis=-1), jnp.zeros(n_det), atol=1e-6)
