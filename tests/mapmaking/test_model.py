import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from furax.mapmaking._model import _sample_mask
from furax.mapmaking.config import MapMakingConfig, Methods


class TestSampleMask:
    @pytest.mark.parametrize('tail_value', [True, False])
    def test_atop_tail_always_masked(self, tail_value: bool):
        """_sample_mask zeros the tail regardless of whether tail samples are valid.

        True tail:  |F(mask)| = 1 > 0.5/tau → interval_mask False → masked out.
        False tail: |F(mask)| = 0 < 0.5/tau → interval_mask True, but mask & True = False.
        """
        tau, n_det, n_samp = 4, 1, 10  # 2 full intervals + tail of 2
        mask = jnp.ones((n_det, n_samp), dtype=jnp.bool_)
        if not tail_value:
            mask = mask.at[0, 2 * tau :].set(False)

        config = MapMakingConfig(method=Methods.ATOP, atop_tau=tau)
        result = _sample_mask({'valid_sample_masks': mask}, config)

        assert_array_equal(result[0, 2 * tau :], jnp.zeros(n_samp - 2 * tau, dtype=jnp.bool_))
