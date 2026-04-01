import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from furax.mapmaking.templates import ATOPProjectionOperator


class TestATOPProjectionOperatorTail:
    """Test behaviour of ATOPProjectionOperator when n_samp is not divisible by tau."""

    tau = 4

    def _make_op(self, n_det: int, n_samp: int, tau: int) -> ATOPProjectionOperator:
        return ATOPProjectionOperator(
            tau, in_structure=jax.ShapeDtypeStruct((n_det, n_samp), jnp.float64)
        )

    def test_full_intervals_are_demeaned(self):
        """Samples in complete intervals have their interval mean subtracted."""
        n_det, n_samp = 1, 10  # 2 full intervals + tail of 2
        x = jnp.ones((n_det, n_samp))
        op = self._make_op(n_det, n_samp, self.tau)
        y = op(x)
        # mean of all-ones interval is 1, so result is 0
        assert_array_equal(y[0, : 2 * self.tau], jnp.zeros(2 * self.tau))

    def test_tail_is_passed_through_unchanged(self):
        """Samples in the partial tail interval are returned as-is."""
        n_det, n_samp = 1, 10  # tail = indices 8, 9
        x = jnp.arange(n_det * n_samp, dtype=jnp.float64).reshape(n_det, n_samp)
        op = self._make_op(n_det, n_samp, self.tau)
        y = op(x)
        assert_array_equal(y[0, 2 * self.tau :], x[0, 2 * self.tau :])

    @pytest.mark.parametrize('tail_value', [True, False])
    def test_tail_zeroed_by_interval_mask(self, tail_value):
        """The interval_mask logic in _sample_mask zeros the tail regardless of its value.

        True tail: abs(F(mask)) = abs(1) = 1 > 0.5/tau → interval_mask False → masked out.
        False tail: abs(F(mask)) = abs(0) = 0 < 0.5/tau → interval_mask True, but mask & True = False.
        """
        tau = self.tau
        n_det, n_samp = 1, 10
        # Build an all-valid mask and optionally set tail to False
        mask = jnp.ones((n_det, n_samp), dtype=jnp.bool_)
        if not tail_value:
            mask = mask.at[0, 2 * tau :].set(False)

        op = ATOPProjectionOperator(tau, in_structure=jax.ShapeDtypeStruct(mask.shape, jnp.bool_))
        interval_mask = jnp.abs(op(mask)) < 0.5 / tau
        result = jnp.logical_and(mask, interval_mask)

        assert_array_equal(result[0, 2 * tau :], jnp.zeros(n_samp - 2 * tau, dtype=jnp.bool_))
