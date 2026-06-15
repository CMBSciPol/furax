import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from furax.mapmaking._model import _noise_model, _noise_operator, _sample_mask
from furax.mapmaking.config import MapMakingConfig, Methods, WeightingConfig, WeightingMode
from furax.mapmaking.noise import WhiteNoiseModel


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


class TestIdentityNoise:
    @pytest.fixture
    def tod_structure(self):
        return jax.ShapeDtypeStruct((4, 200), jnp.float64)

    @pytest.fixture
    def identity_config(self):
        return MapMakingConfig(weighting=WeightingConfig(mode=WeightingMode.IDENTITY))

    def test_noise_model_creates_unit_white_noise(self, identity_config, tod_structure):
        data = {'timestamps': jnp.linspace(0, 1, 200)}
        noise_model, _ = _noise_model(data, identity_config, tod_structure=tod_structure)
        assert isinstance(noise_model, WhiteNoiseModel)
        assert_allclose(noise_model.sigma, jnp.ones(4))
        assert noise_model.sigma.dtype == tod_structure.dtype

    def test_noise_operator_acts_as_identity(self, identity_config, tod_structure):
        data = {'timestamps': jnp.linspace(0, 1, 200)}
        noise_model, fs = _noise_model(data, identity_config, tod_structure=tod_structure)
        inv_op = _noise_operator(noise_model, tod_structure, fs, 100, inverse=True)
        fwd_op = _noise_operator(noise_model, tod_structure, fs, 100, inverse=False)
        x = jax.random.normal(jax.random.key(1), (4, 200))
        assert_allclose(inv_op(x), x, rtol=1e-12)
        assert_allclose(fwd_op(x), x, rtol=1e-12)

    def test_identity_requires_tod_structure(self, identity_config):
        data = {'timestamps': jnp.linspace(0, 1, 200)}
        with pytest.raises(ValueError, match='tod_structure is required'):
            _noise_model(data, identity_config, tod_structure=None)

    def test_identity_config_yaml_round_trip(self):
        config = MapMakingConfig(weighting=WeightingConfig(mode=WeightingMode.IDENTITY))
        yaml_str = config._to_yaml()
        assert 'mode: identity' in yaml_str
        loaded = MapMakingConfig.load_dict({'weighting': {'mode': 'identity'}})
        assert loaded.weighting.mode is WeightingMode.IDENTITY

    def test_diagonal_by_default(self):
        config = WeightingConfig()
        assert config.mode is WeightingMode.DIAGONAL
