import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

from furax import IdentityOperator, MaskOperator
from furax.mapmaking import ReaderField
from furax.mapmaking._model import (
    BucketModel,
    ObservationModel,
    _noise_model,
    _noise_operator,
    _sample_mask,
    restrict_streams,
)
from furax.mapmaking.config import (
    MapMakingConfig,
    Methods,
    NestedConfig,
    WeightingConfig,
    WeightingMode,
)
from furax.mapmaking.noise import WhiteNoiseModel, padding_aware_welch
from furax.mapmaking.streaming import StreamLayout
from furax.mapmaking.weight import NestedWeightOperator, WeightOperator
from furax.obs import TODFormat
from furax.obs.stokes import StokesIQU, StokesQU


def test_padding_aware_welch_matches_unpadded_signal():
    signal = jax.random.normal(jax.random.key(0), (2, 1024), dtype=jnp.float64)
    expected_f, expected_psd = jax.scipy.signal.welch(signal, fs=100.0, nperseg=256)
    padded = jnp.pad(signal, ((0, 0), (0, 512)))
    actual_f, actual_psd = padding_aware_welch(padded, 512, fs=jnp.array(100.0), nperseg=256)
    assert_allclose(actual_f, expected_f)
    assert_allclose(actual_psd, expected_psd, rtol=1e-12, atol=1e-12)


def test_padding_aware_welch_clips_segment_to_short_signal():
    signal = jax.random.normal(jax.random.key(0), (2, 127), dtype=jnp.float64)
    expected_f, expected_psd = jax.scipy.signal.welch(signal, fs=100.0, nperseg=256)
    actual_f, actual_psd = padding_aware_welch(signal, 0, fs=jnp.array(100.0), nperseg=256)
    assert_allclose(actual_f, expected_f)
    assert_allclose(actual_psd, expected_psd, rtol=1e-12, atol=1e-12)


def test_padding_aware_welch_preserves_float32():
    with jax.enable_x64():
        signal = jax.random.normal(jax.random.key(0), (2, 256), dtype=jnp.float32)
        frequencies, psd = padding_aware_welch(
            signal, 0, fs=jnp.array(100.0, dtype=jnp.float32), nperseg=128
        )
    assert frequencies.dtype == jnp.float32
    assert psd.dtype == jnp.float32


@jax.jit
def _welch(signal, padding, valid):
    return padding_aware_welch(signal, padding, fs=jnp.array(100.0), nperseg=256, valid=valid)


def test_padding_aware_welch_skips_flagged_segments():
    nperseg, step = 256, 128
    signal = jax.random.normal(jax.random.key(0), (3, 1024), dtype=jnp.float64)
    glitched = signal.at[0, 300:340].set(1e6)
    valid = jnp.ones_like(signal, bool)
    valid = valid.at[0, 300:340].set(False)  # the segments starting at 128 and 256 hold them
    valid = valid.at[2, ::100].set(False)  # no segment is clean: the detector keeps them all

    _, psd = _welch(glitched.at[2].set(signal[2]), jnp.array(0), valid)  # padding traced

    def segment(s):
        return jax.scipy.signal.welch(signal[0, s : s + nperseg], fs=100.0, nperseg=nperseg)[1]

    starts = [s for s in range(0, 1024 - nperseg + 1, step) if s not in (128, 256)]
    assert_allclose(psd[0], sum(map(segment, starts)) / len(starts), rtol=1e-10, atol=1e-12)
    _, unmasked = padding_aware_welch(signal, 0, fs=jnp.array(100.0), nperseg=nperseg)
    assert_allclose(psd[1:], unmasked[1:], rtol=1e-12, atol=1e-12)


class TestSampleMask:
    @pytest.mark.parametrize('tail_value', [True, False])
    def test_pomme_tail_always_masked(self, tail_value: bool):
        """_sample_mask zeros the tail regardless of whether tail samples are valid.

        True tail:  |F(mask)| = 1 > 0.5/tau → interval_mask False → masked out.
        False tail: |F(mask)| = 0 < 0.5/tau → interval_mask True, but mask & True = False.
        """
        tau, n_det, n_samp = 4, 1, 10  # 2 full intervals + tail of 2
        mask = jnp.ones((n_det, n_samp), dtype=bool)
        if not tail_value:
            mask = mask.at[0, 2 * tau :].set(False)

        config = MapMakingConfig(method=Methods.POMME, pomme_tau=tau)
        result = _sample_mask({'valid_sample_masks': mask}, config)

        assert_array_equal(result[0, 2 * tau :], jnp.zeros(n_samp - 2 * tau, dtype=bool))

    def test_pomme_widens_scanning_mask_to_whole_intervals(self):
        """An interval partially flagged by the scanning mask is dropped whole.

        The weight must be constant on every interval for it to commute with the Pomme projector,
        whichever mask flags the samples.
        """
        tau, n_det, n_samp = 4, 2, 12  # 3 whole intervals
        sample_mask = jnp.ones((n_det, n_samp), dtype=bool)
        scanning = jnp.ones(n_samp, dtype=bool).at[5].set(False)  # one sample of interval 1

        config = MapMakingConfig(method=Methods.POMME, pomme_tau=tau)
        data = {'valid_sample_masks': sample_mask, 'valid_scanning_masks': scanning}
        result = _sample_mask(data, config)

        expected = jnp.ones((n_det, n_samp), dtype=bool).at[:, tau : 2 * tau].set(False)
        assert_array_equal(result, expected)

    def test_scanning_mask_applies_without_pomme(self):
        n_det, n_samp = 2, 6
        sample_mask = jnp.ones((n_det, n_samp), dtype=bool)
        scanning = jnp.ones(n_samp, dtype=bool).at[1].set(False)
        data = {'valid_sample_masks': sample_mask, 'valid_scanning_masks': scanning}
        result = _sample_mask(data, MapMakingConfig())
        assert_array_equal(result, sample_mask.at[:, 1].set(False))


class TestIdentityNoise:
    @pytest.fixture
    def tod_format(self):
        return TODFormat((4, 200), jnp.float64)

    @pytest.fixture
    def identity_config(self):
        return MapMakingConfig(weighting=WeightingConfig(mode=WeightingMode.IDENTITY))

    @pytest.mark.parametrize(('stokes', 'shape'), [(None, (4,)), ('QU', (2, 4))])
    def test_noise_model_creates_unit_white_noise(self, identity_config, stokes, shape):
        # one noise model covers every stream of a demodulated TOD, with a leading stream axis
        tod_format = TODFormat((4, 200), jnp.float64, stokes=stokes)
        data = {'timestamps': jnp.linspace(0, 1, 200)}
        noise_model, _ = _noise_model(data, identity_config, tod_format)
        assert isinstance(noise_model, WhiteNoiseModel)
        assert_allclose(noise_model.sigma, jnp.ones(shape))
        assert noise_model.sigma.dtype == tod_format.dtype

    def test_noise_operator_acts_as_identity(self, identity_config, tod_format):
        data = {'timestamps': jnp.linspace(0, 1, 200)}
        noise_model, fs = _noise_model(data, identity_config, tod_format)
        inv_op = _noise_operator(noise_model, tod_format, fs, 100, inverse=True)
        fwd_op = _noise_operator(noise_model, tod_format, fs, 100, inverse=False)
        x = jax.random.normal(jax.random.key(1), (4, 200))
        assert_allclose(inv_op(x), x, rtol=1e-12)
        assert_allclose(fwd_op(x), x, rtol=1e-12)

    def test_identity_config_yaml_round_trip(self):
        config = MapMakingConfig(weighting=WeightingConfig(mode=WeightingMode.IDENTITY))
        yaml_str = config._to_yaml()
        assert 'mode: identity' in yaml_str
        loaded = MapMakingConfig.load_dict({'weighting': {'mode': 'identity'}})
        assert loaded.weighting.mode is WeightingMode.IDENTITY

    def test_diagonal_by_default(self):
        config = WeightingConfig()
        assert config.mode is WeightingMode.DIAGONAL


def test_restrict_streams_keeps_the_streams_of_the_map():
    # streams told apart by their values: stream k of the read data holds k
    tod = StokesIQU.from_array(jnp.arange(3.0)[:, None, None] * jnp.ones((3, 2, 5)))
    fits = jnp.arange(3.0)[:, None, None] * jnp.ones((3, 2, 4))
    data = {ReaderField.SAMPLE_DATA: tod, ReaderField.NOISE_MODEL_FITS: fits}

    data, restricted = restrict_streams(data, tod, 'QU')

    assert isinstance(restricted, StokesQU)
    assert_array_equal(restricted.data, tod.data[1:])
    assert_array_equal(data[ReaderField.SAMPLE_DATA].data, tod.data[1:])
    assert_array_equal(data[ReaderField.NOISE_MODEL_FITS], fits[1:])


@pytest.fixture
def single_device_mesh():
    # the mapmaker's mesh (see `SlotPlacement`)
    mesh = jax.make_mesh((1,), ('obs',), (AxisType.Auto,), devices=jax.devices()[:1])
    with jax.set_mesh(mesh):
        yield


@pytest.mark.usefixtures('single_device_mesh')
@pytest.mark.parametrize('nested', [False, True])
def test_bucket_white_weight_is_the_plain_masked_weight(nested):
    # identity weighting has white noise, but the nested gap treatment still wraps it
    tod_format = TODFormat((2, 50), jnp.float64)
    structure = tod_format.structure
    noise_model = WhiteNoiseModel(sigma=jnp.ones(2))
    ninv = _noise_operator(noise_model, tod_format, jnp.array(1.0))
    good = jnp.ones((2, 50), bool).at[:, 20:25].set(False)
    mask = MaskOperator.from_boolean_mask(good, in_structure=structure)
    W = (
        NestedWeightOperator.create(ninv, mask, NestedConfig())
        if nested
        else WeightOperator.create(ninv, mask)
    )
    identity = IdentityOperator(in_structure=structure)
    entry = ObservationModel(identity, W, identity, noise_model, jnp.array(1.0))
    entries = jax.tree.map(lambda leaf: jax.device_put(jnp.stack([leaf, leaf]), P('obs')), entry)
    weight = BucketModel(StreamLayout(2), entries, None, ()).white_weight()

    is_nested = lambda node: isinstance(node, NestedWeightOperator)
    assert not any(is_nested(node) for node in jax.tree.leaves(weight, is_leaf=is_nested))
    x = jax.device_put(jax.random.normal(jax.random.key(0), (2, 2, 50)), P('obs'))
    assert_allclose(weight(x), jnp.where(good, x, 0))
