from pathlib import Path
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sotodlib.core import Context
from sotodlib.core.metadata.loader import LoaderError
from sotodlib.preprocess import preprocess_util as pu

from furax.interfaces.sotodlib import (
    LazyPreprocSOTODLibObservation,
    LazySOTODLibObservation,
)
from furax.interfaces.sotodlib.observation import _digest, _is_missing_record
from furax.mapmaking import (
    AbstractGroundObservation,
    HashedObservationMetadata,
    MapMakingConfig,
    MultiObservationMapMaker,
    ObservationReader,
    ReaderField,
)
from furax.mapmaking.config import (
    HealpixConfig,
    LandscapeConfig,
    PointingConfig,
    SotodlibConfig,
)
from furax.obs.stokes import Stokes
from furax.tree import as_structure

from ._preproc_db import build_preproc_db, cut_group, drop_group_record

FOLDER = Path(__file__).parents[2] / 'data/sotodlib'
FILES = ['test_obs.h5', 'test_obs_2.h5']
OBS_IDS = ['obs_12345_sometel', 'obs_54321_someothertel']
OBS_NDET = [2, 4]
OBS_NSAMPLE = [1_000, 3_000]
OBS_NINTERVAL = [8, 3]


@pytest.fixture
def observations():
    return [LazySOTODLibObservation(FOLDER / f) for f in FILES]


@pytest.fixture
def demod_observations():
    sotodlib_config = SotodlibConfig(demodulated=True)
    return [LazySOTODLibObservation(FOLDER / f, sotodlib_config=sotodlib_config) for f in FILES]


def test_demodulated_tods_keep_their_stored_precision(demod_observations) -> None:
    data = demod_observations[0].get_data([ReaderField.SAMPLE_DATA])
    tods = data.get_demodulated_tods(stokes='IQU')
    stored = [np.asarray(getattr(data.data, attr)) for attr in ('dsT', 'demodQ', 'demodU')]
    assert tods.data.dtype == stored[0].dtype
    for stream, raw in zip((tods.i, tods.q, tods.u), stored, strict=True):
        assert_array_equal(stream, 0.5 * raw)


def test_demodulated_tods_are_written_into_the_corner_of_out(demod_observations) -> None:
    data = demod_observations[0].get_data([ReaderField.SAMPLE_DATA])
    expected = data.get_demodulated_tods(stokes='IQU').data
    n_dets, n_samples = expected.shape[1:]
    out = np.full((3, n_dets + 1, n_samples + 2), np.nan, np.float32)
    tods = data.get_demodulated_tods(stokes='IQU', out=out)
    assert np.shares_memory(tods.data, out)
    assert_array_equal(out[:, :n_dets, :n_samples], expected.astype(np.float32))
    assert np.isnan(out[:, n_dets:]).all()
    assert np.isnan(out[..., n_samples:]).all()


def test_demodulated_tods_reject_a_small_out(demod_observations) -> None:
    data = demod_observations[0].get_data([ReaderField.SAMPLE_DATA])
    with pytest.raises(ValueError, match='too small'):
        data.get_demodulated_tods(stokes='IQU', out=np.empty((3, 1, 1), np.float32))


def test_reader_all_fields(observations) -> None:
    """Test consistency for all available fields."""
    reader = ObservationReader.from_observations(
        observations, requested_fields=AbstractGroundObservation.AVAILABLE_READER_FIELDS
    )
    ndet_max, nsample_max, nint_max = max(OBS_NDET), max(OBS_NSAMPLE), max(OBS_NINTERVAL)

    # check output structure is as expected
    assert reader.out_structure == {
        'metadata': HashedObservationMetadata(
            uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            detector_uids=jax.ShapeDtypeStruct((ndet_max,), dtype=jnp.uint32),
        ),
        'sample_data': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample_max, 4), dtype=jnp.float64),
        'noise_model_fits': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
        'azimuth': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'elevation': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'left_scan_mask': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'right_scan_mask': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'scanning_intervals': jax.ShapeDtypeStruct((nint_max, 2), dtype=jnp.int32),
    }

    for i in range(len(FILES)):
        datum, padding, _ = reader.read(i)

        # check structure
        assert as_structure(datum) == reader.out_structure

        # check padding consistency
        ndet, nsample, nint = OBS_NDET[i], OBS_NSAMPLE[i], OBS_NINTERVAL[i]
        assert tuple(padding['metadata'].uid) == ()
        assert tuple(padding['metadata'].telescope_uid) == ()
        assert tuple(padding['metadata'].detector_uids) == (ndet_max - ndet,)
        assert tuple(padding['sample_data']) == (ndet_max - ndet, nsample_max - nsample)
        assert tuple(padding['valid_sample_masks']) == (ndet_max - ndet, nsample_max - nsample)
        assert tuple(padding['valid_scanning_masks']) == (nsample_max - nsample,)
        assert tuple(padding['timestamps']) == (nsample_max - nsample,)
        assert tuple(padding['hwp_angles']) == (nsample_max - nsample,)
        assert tuple(padding['detector_quaternions']) == (ndet_max - ndet, 0)
        assert tuple(padding['boresight_quaternions']) == (nsample_max - nsample, 0)
        assert tuple(padding['noise_model_fits']) == (ndet_max - ndet, 0)
        assert tuple(padding['azimuth']) == (nsample_max - nsample,)
        assert tuple(padding['elevation']) == (nsample_max - nsample,)
        assert tuple(padding['left_scan_mask']) == (nsample_max - nsample,)
        assert tuple(padding['right_scan_mask']) == (nsample_max - nsample,)
        assert tuple(padding['scanning_intervals']) == (nint_max - nint, 0)

        # starts are sorted, and padding conserves the true number of intervals
        intervals = np.asarray(datum['scanning_intervals'])
        assert np.all(np.diff(intervals[:, 0]) >= 0)
        assert np.sum(intervals[:, 1] > intervals[:, 0]) == nint


def test_probe_shape(observations) -> None:
    """The lazy observation sizes its buffers without (necessarily) a full load."""
    for i, obs in enumerate(observations):
        # n_intervals stays 0 unless scanning_intervals is requested
        assert obs.probe_shape() == (OBS_NDET[i], OBS_NSAMPLE[i], 0)
        assert obs.probe_shape(intervals=True) == (
            OBS_NDET[i],
            OBS_NSAMPLE[i],
            OBS_NINTERVAL[i],
        )


@pytest.mark.xfail(reason='our test observation does not have subscan metadata')
def test_lazy_preproc_observation(tmp_path) -> None:
    """Preproc-backed lazy obs loads straight from a (minimal, real) preprocessing db."""
    config = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[0], config)

    # get_data runs the full preproc load (with signal) through load_and_preprocess
    data = lazy.get_data()
    assert data.n_detectors == OBS_NDET[0]
    assert data.n_samples == OBS_NSAMPLE[0]
    assert data.get_tods().shape == (OBS_NDET[0], OBS_NSAMPLE[0])

    # probe_shape reads an upper bound straight from the archive metadata (no pipeline run);
    # here the no-op pipeline neither cuts nor trims, so the bound matches the load exactly.
    # n_intervals is only populated (via a load) when scanning_intervals is requested.
    assert lazy.probe_shape() == (OBS_NDET[0], OBS_NSAMPLE[0], 0)
    assert lazy.probe_shape(intervals=True) == (
        OBS_NDET[0],
        OBS_NSAMPLE[0],
        OBS_NINTERVAL[0],
    )


@pytest.mark.parametrize('demodulated', [False, True])
def test_downsample(demodulated: bool) -> None:
    """The file-backed observation is downsampled by `downsample`, sizing included."""
    sotodlib_config = SotodlibConfig(demodulated=demodulated, downsample=3)
    lazy = LazySOTODLibObservation(FOLDER / FILES[0], sotodlib_config=sotodlib_config)

    # ceil(1000 / 3) == 334
    expected_nsamp = -(-OBS_NSAMPLE[0] // 3)
    assert lazy.probe_shape() == (OBS_NDET[0], expected_nsamp, 0)
    obs = lazy.get_data()
    assert obs.n_samples == expected_nsamp
    if demodulated:
        assert obs.get_demodulated_tods().shape == (OBS_NDET[0], expected_nsamp)
    else:
        assert obs.get_tods().shape == (OBS_NDET[0], expected_nsamp)


def test_lazy_preproc_probe_shape_downsample(tmp_path) -> None:
    """probe_shape ceils the sample bound by the downsample factor, matching downsample_obs."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[0], config, sotodlib_config=SotodlibConfig(downsample=3)
    )

    # ceil(1000 / 3) == 334, the same count downsample_obs produces in the real load
    expected_nsamp = -(-OBS_NSAMPLE[0] // 3)
    assert lazy.probe_shape() == (OBS_NDET[0], expected_nsamp, 0)
    assert lazy.get_data().n_samples == expected_nsamp


@pytest.mark.parametrize('downsample', [1, 3])
def test_lazy_preproc_join_groups(tmp_path, downsample: int) -> None:
    """Joining the preprocessing groups gives the observation loaded as a whole."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    sotodlib_config = SotodlibConfig(downsample=downsample)
    whole = LazyPreprocSOTODLibObservation(OBS_IDS[1], config, sotodlib_config=sotodlib_config)
    joined = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, sotodlib_config=sotodlib_config, join_groups=True
    )

    assert joined.probe_shape() == whole.probe_shape()
    expected, actual = whole.get_data(), joined.get_data()
    assert actual.n_samples == expected.n_samples == -(-OBS_NSAMPLE[1] // downsample)
    assert_array_equal(actual.detectors, expected.detectors)
    # downsampling Fourier-resamples a group at a time: equal up to rounding
    tods = expected.get_tods()
    assert_allclose(actual.get_tods(), tods, rtol=0, atol=1e-6 * np.abs(tods).max())
    assert_array_equal(actual.get_timestamps(), expected.get_timestamps())


def test_lazy_preproc_join_selected_groups(tmp_path) -> None:
    """Only the groups matching the detector selection are joined."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection={'detset': 'detset0_1'}, join_groups=True
    )

    assert lazy.probe_shape() == (OBS_NDET[1] // 2, OBS_NSAMPLE[1], 0)
    assert lazy.get_data().n_detectors == OBS_NDET[1] // 2


@pytest.fixture
def missing_group(monkeypatch):
    """Make `Context.get_meta` find no preprocessing record for the group `detset0_1`."""
    drop_group_record(monkeypatch, 'detset0_1')


@pytest.mark.usefixtures('missing_group')
def test_lazy_preproc_join_groups_skips_a_group_without_record(tmp_path, caplog) -> None:
    """A group with no preprocessing record is left out, not a reason to lose the observation."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[1], config, join_groups=True)

    assert lazy.probe_shape() == (OBS_NDET[1] // 2, OBS_NSAMPLE[1], 0)
    assert lazy.get_data().n_detectors == OBS_NDET[1] // 2
    # the group is reported in plain Python values, not as numpy scalars
    assert "No preprocessing record for {'detset': 'detset0_1'}" in caplog.text


@pytest.mark.usefixtures('missing_group')
def test_lazy_preproc_a_missing_group_still_fails_alone(tmp_path) -> None:
    """Without joining, the selected group is the observation: its missing record is an error."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection={'detset': 'detset0_1'}
    )

    with pytest.raises(LoaderError):
        lazy.probe_shape()


def test_lazy_preproc_join_groups_without_any_record_fails(tmp_path, monkeypatch) -> None:
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)

    no_record = Mock(side_effect=LoaderError('There were no metadata records found.'))
    monkeypatch.setattr(Context, 'get_meta', no_record)
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[1], config, join_groups=True)

    with pytest.raises(RuntimeError, match='no preprocessing records'):
        lazy.probe_shape()
    with pytest.raises(RuntimeError, match='no detectors left'):
        lazy.get_data()


GROUPS = ['detset0_0', 'detset0_1']  # the preprocessing groups of OBS_IDS[1], with two detsets


def _cached_names(cache_dir) -> list[str]:
    """The names of the files cached for OBS_IDS[1], leaving out those of the key."""
    (key,) = cache_dir.iterdir()
    assert (key / 'key.json').exists()
    return sorted(p.name for p in key.iterdir() if p.name != 'key.json')


@pytest.mark.parametrize(
    ('join_groups', 'selection', 'names'),
    [
        # joined: one file per group, named after the group
        (True, None, [f'{OBS_IDS[1]}__detset-{group}.h5' for group in GROUPS]),
        (True, {'detset': GROUPS[1]}, [f'{OBS_IDS[1]}__detset-{GROUPS[1]}.h5']),
        # not joined: the selection is loaded and cached as one unit
        (False, None, [f'{OBS_IDS[1]}__all.h5']),
        (False, {'detset': GROUPS[1]}, [f'{OBS_IDS[1]}__detset-{GROUPS[1]}.h5']),
    ],
)
def test_lazy_preproc_cache_layout(tmp_path, join_groups, selection, names) -> None:
    """The cache holds one file per group that is joined, or one per selection that is not."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1],
        config,
        detector_selection=selection,
        join_groups=join_groups,
        cache_dir=tmp_path / 'cache',
    )

    assert not lazy.is_cached()
    lazy.get_data()

    assert _cached_names(tmp_path / 'cache') == names
    assert sorted(p.name for p in lazy.cache_paths()) == names
    assert lazy.is_cached()


@pytest.mark.parametrize('join_groups', [False, True])
def test_lazy_preproc_cache_reads_back_what_was_loaded(tmp_path, join_groups: bool) -> None:
    """A cached observation, joined from its group files or not, equals the loaded one.

    The archive is deleted in between, so that the second read can only come from the cache.
    """
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1],
        config,
        sotodlib_config=SotodlibConfig(downsample=3),
        join_groups=join_groups,
        cache_dir=tmp_path / 'cache',
    )
    loaded = lazy.get_data()
    (tmp_path / 'preprocess_archive.h5').unlink()

    cached = lazy.get_data([ReaderField.SAMPLE_DATA, ReaderField.TIMESTAMPS])

    assert cached.name == OBS_IDS[1]
    # downsampled once, when loaded: the cached data is not downsampled again
    assert cached.n_samples == loaded.n_samples == -(-OBS_NSAMPLE[1] // 3)
    assert_array_equal(cached.detectors, loaded.detectors)
    assert_array_equal(cached.get_tods(), loaded.get_tods())
    assert_array_equal(cached.get_timestamps(), loaded.get_timestamps())


def test_lazy_preproc_joins_cached_groups_whatever_the_requested_fields(tmp_path) -> None:
    """Joining cached groups needs no field in the request: the samples are always compared."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=True, cache_dir=tmp_path / 'cache'
    )
    n_detectors = lazy.get_data().n_detectors
    (tmp_path / 'preprocess_archive.h5').unlink()

    assert lazy.get_data([]).n_detectors == n_detectors


def test_lazy_preproc_cache_is_shared_by_groups(tmp_path) -> None:
    """Runs that select or join groups differently load the group files they have in common."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    cache_dir = tmp_path / 'cache'
    alone = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection={'detset': GROUPS[0]}, cache_dir=cache_dir
    )
    joined = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=True, cache_dir=cache_dir
    )
    other = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection={'detset': GROUPS[1]}, cache_dir=cache_dir
    )

    # a group cached alone is part of the joined observation, which only loads the other group
    n_alone = alone.get_data().n_detectors
    (first,) = alone.cache_paths()
    written = first.stat().st_mtime_ns
    assert not joined.is_cached()
    assert joined.get_data().n_detectors > n_alone
    assert first.stat().st_mtime_ns == written
    assert joined.is_cached()

    # and a group cached by the joined run serves a run that selects it alone
    (tmp_path / 'preprocess_archive.h5').unlink()
    assert other.is_cached()
    assert 0 < other.get_data().n_detectors < joined.get_data().n_detectors


@pytest.mark.usefixtures('missing_group')
def test_lazy_preproc_cache_does_not_skip_a_missing_group_alone(tmp_path) -> None:
    """Without joining, a missing record is an error, and not recorded as a skipped group."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1],
        config,
        detector_selection={'detset': GROUPS[1]},
        cache_dir=tmp_path / 'cache',
    )

    with pytest.raises(LoaderError):
        lazy.get_data()

    assert not lazy.is_cached()
    assert not list((tmp_path / 'cache').glob('*/*.skip'))


def test_lazy_preproc_joins_cached_and_loaded_groups(tmp_path) -> None:
    """Groups from the cache and from preprocessing have the same fields, in either order."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    cache_dir = tmp_path / 'cache'
    alone = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection={'detset': GROUPS[1]}, cache_dir=cache_dir
    )
    joined = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=True, cache_dir=cache_dir
    )
    alone.get_data()

    # the first group is loaded, the second one read from the cache with the requested fields
    assert joined.get_data([ReaderField.TIMESTAMPS]).n_detectors == OBS_NDET[1]


def test_missing_record_is_recognised_in_the_error_of_sotodlib(tmp_path) -> None:
    """The check reads sotodlib's message: it must match the error sotodlib raises."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]])

    with pytest.raises(LoaderError) as error:
        pu.load_and_preprocess('obs_missing', config.as_posix())
    assert _is_missing_record(error.value)


@pytest.mark.parametrize('cause', ['no_record', 'cut'])
def test_lazy_preproc_cache_remembers_a_group_without_detectors(
    tmp_path, monkeypatch, caplog, cause: str
) -> None:
    """A group with no detector, for lack of a record or after the cuts, is not loaded again."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    if cause == 'no_record':
        drop_group_record(monkeypatch, GROUPS[1])
    else:
        cut_group(monkeypatch, GROUPS[1])
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=True, cache_dir=tmp_path / 'cache'
    )

    assert not lazy.is_cached()
    n_detectors = lazy.get_data().n_detectors
    assert 0 < n_detectors < OBS_NDET[1]
    assert _cached_names(tmp_path / 'cache') == [
        f'{OBS_IDS[1]}__detset-{GROUPS[0]}.h5',
        f'{OBS_IDS[1]}__detset-{GROUPS[1]}.skip',
    ]
    assert lazy.is_cached()

    # any further preprocessing load would fail
    monkeypatch.setattr(pu, 'load_and_preprocess', Mock(side_effect=AssertionError('loaded')))
    caplog.clear()
    assert lazy.get_data().n_detectors == n_detectors
    assert 'No preprocessing record' not in caplog.text


def test_lazy_preproc_cache_remembers_a_selection_cut_entirely(tmp_path, monkeypatch) -> None:
    """Without joining, a selection cut entirely is an error each time, and is not loaded again."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    loaded = cut_group(monkeypatch, GROUPS[1])
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1],
        config,
        detector_selection={'detset': GROUPS[1]},
        cache_dir=tmp_path / 'cache',
    )

    with pytest.raises(RuntimeError, match='no detectors left'):
        lazy.get_data()
    assert _cached_names(tmp_path / 'cache') == [f'{OBS_IDS[1]}__detset-{GROUPS[1]}.skip']
    assert lazy.is_cached()

    loaded.clear()
    with pytest.raises(RuntimeError, match='no detectors left'):
        lazy.get_data()
    assert loaded == []


def test_lazy_preproc_cache_skips_only_a_missing_record(tmp_path, monkeypatch) -> None:
    """Another LoaderError may be transient: it is raised, and not remembered as a skipped group."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    monkeypatch.setattr(pu, 'load_and_preprocess', Mock(side_effect=LoaderError('unreachable')))
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=True, cache_dir=tmp_path / 'cache'
    )

    with pytest.raises(LoaderError, match='unreachable'):
        lazy.get_data()
    assert not list((tmp_path / 'cache').glob('*/*.skip'))


@pytest.mark.parametrize(
    ('join_groups', 'selection', 'name'),
    [
        (False, {'wafer_slot': 'ws0'}, f'{OBS_IDS[1]}__ws0'),
        (False, {'wafer_slot': ['ws0', 'ws1']}, OBS_IDS[1]),
        (False, {'wafer.bandpass': 'f090'}, OBS_IDS[1]),
        (True, {'wafer_slot': 'ws0'}, OBS_IDS[1]),
        (False, None, OBS_IDS[1]),
    ],
)
def test_lazy_preproc_name_tells_the_wafers_of_an_observation_apart(
    tmp_path, join_groups, selection, name
) -> None:
    """Each wafer that is an observation of its own has its own name."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]])
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, detector_selection=selection, join_groups=join_groups
    )

    assert lazy.name == name


@pytest.mark.parametrize('join_groups', [False, True])
def test_lazy_preproc_cache_does_not_skip_a_group_that_fails_to_load(
    tmp_path, monkeypatch, join_groups: bool
) -> None:
    """A load error is raised as is, not recorded as a group without detectors."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    monkeypatch.setattr(pu, 'load_and_preprocess', Mock(side_effect=RuntimeError('unreadable')))
    lazy = LazyPreprocSOTODLibObservation(
        OBS_IDS[1], config, join_groups=join_groups, cache_dir=tmp_path / 'cache'
    )

    with pytest.raises(RuntimeError, match='unreadable'):
        lazy.get_data()
    assert not list((tmp_path / 'cache').glob('*/*.skip'))


@pytest.mark.parametrize(
    ('changes', 'shared'),
    [
        ({'sotodlib_config': SotodlibConfig(demodulated=True, noise_source='mapmaking')}, True),
        ({'sotodlib_config': SotodlibConfig(downsample=2)}, False),
        ({'join_groups': True}, True),
        ({'detector_selection': {'detset': 'detset0_1'}}, True),
        ({'init_config': 'copy.yaml'}, True),
        ({'init_config': 'edited.yaml'}, False),
        ({'proc_config': 'copy.yaml'}, False),
    ],
)
def test_lazy_preproc_cache_key(tmp_path, changes: dict, shared: bool) -> None:
    """Observations share a cache directory if and only if they load the same data."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    text = config.read_text()
    (tmp_path / 'copy.yaml').write_text(text)
    (tmp_path / 'edited.yaml').write_text(text + 'extra: 1\n')
    for name in ('init_config', 'proc_config'):
        if name in changes:
            changes[name] = tmp_path / changes[name]
    reference = LazyPreprocSOTODLibObservation(OBS_IDS[1], config, cache_dir=tmp_path)
    other = LazyPreprocSOTODLibObservation(
        **{
            'observation_id': OBS_IDS[1],
            'init_config': config,
            'cache_dir': tmp_path,
            **changes,
        }
    )

    keys = (_digest(obs._cache_inputs) for obs in (reference, other))
    assert (len(set(keys)) == 1) == shared


def test_lazy_preproc_cache_key_covers_context(tmp_path) -> None:
    """Editing the context of the configs in place changes the cache key."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])
    before = LazyPreprocSOTODLibObservation(OBS_IDS[0], config, cache_dir=tmp_path)
    key = _digest(before._cache_inputs)
    context = tmp_path / 'context.yaml'
    context.write_text(context.read_text() + 'extra: 1\n')
    after = LazyPreprocSOTODLibObservation(OBS_IDS[0], config, cache_dir=tmp_path)

    assert _digest(after._cache_inputs) != key


def test_lazy_preproc_cache_key_ignores_the_tod_location(tmp_path) -> None:
    """Where the raw TOD files are stored does not change the loaded data, nor the cache key."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])
    key = _digest(LazyPreprocSOTODLibObservation(OBS_IDS[0], config)._cache_inputs)
    context = tmp_path / 'context.yaml'
    context.write_text(context.read_text() + "obsfiledb.prefix: '/data/tod'\n")

    assert _digest(LazyPreprocSOTODLibObservation(OBS_IDS[0], config)._cache_inputs) == key


def test_lazy_preproc_cache_unreadable_file(tmp_path, caplog) -> None:
    """A cached file that cannot be read is loaded again and replaced."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[0], config, cache_dir=tmp_path / 'cache')
    expected = lazy.get_data().get_tods()
    (path,) = lazy.cache_paths()
    path.write_bytes(b'not hdf5')

    assert_array_equal(lazy.get_data().get_tods(), expected)
    assert 'Cannot read' in caplog.text
    (tmp_path / 'preprocess_archive.h5').unlink()
    assert_array_equal(lazy.get_data().get_tods(), expected)


def test_lazy_preproc_cache_write_failure(tmp_path, caplog) -> None:
    """An unwritable cache leaves the observation loaded, uncached."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])
    cache_dir = tmp_path / 'cache'
    cache_dir.write_text('not a directory')
    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[0], config, cache_dir=cache_dir)

    assert lazy.get_data().n_detectors == OBS_NDET[0]
    assert 'Could not write' in caplog.text


def test_reader_with_preproc_observations(tmp_path) -> None:
    """The reader drives preproc-db-backed lazy observations through io_callback end to end."""
    config = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    observations = [LazyPreprocSOTODLibObservation(obs_id, config) for obs_id in OBS_IDS]

    reader = ObservationReader.from_observations(
        observations, requested_fields=['sample_data', 'timestamps', 'valid_scanning_masks']
    )
    ndet_max, nsample_max = max(OBS_NDET), max(OBS_NSAMPLE)
    assert reader.out_structure['sample_data'].shape == (ndet_max, nsample_max)

    for i in range(len(FILES)):
        datum, _, _ = reader.read(i)
        assert as_structure(datum) == reader.out_structure


def test_preproc_context_cache(tmp_path, monkeypatch) -> None:
    """Repeated loads on one thread build the sotodlib Context once (not per load)."""
    from sotodlib.core import Context

    config = build_preproc_db(tmp_path, [FOLDER / FILES[0]])

    n_builds = 0
    original_init = Context.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal n_builds
        n_builds += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Context, '__init__', counting_init)

    lazy = LazyPreprocSOTODLibObservation(OBS_IDS[0], config)
    lazy.probe_shape()
    lazy.probe_shape()
    lazy.get_data()
    # three loads on this thread, but the Context (and its obsdb/obsfiledb opens) is built once
    assert n_builds == 1


def test_binned_mapmaker_over_preproc_db(tmp_path) -> None:
    """Bin a map straight from the preproc db, exercising the full streaming pipeline."""
    config_path = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    observations = [LazyPreprocSOTODLibObservation(obs_id, config_path) for obs_id in OBS_IDS]

    stokes = 'IQU'
    config = MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes=stokes, healpix=HealpixConfig(nside=16)),
    )
    maker = MultiObservationMapMaker(observations, config=config)
    results = maker.make_maps()

    n_stokes = len(stokes)
    assert results.hit_map.shape == maker.landscape.shape
    assert jnp.all(results.hit_map >= 0)
    assert results.icov.shape == (n_stokes, n_stokes, *maker.landscape.shape)
    # binned map: CG converges in a single iteration
    assert results.solver_stats is not None
    assert results.solver_stats['num_steps'] == 1


def test_mapmaker_reads_each_observation_once(tmp_path, monkeypatch) -> None:
    """A full run loads (and preprocesses) each observation exactly once, not twice."""
    from furax.interfaces.sotodlib.observation import SOTODLibObservation

    config_path = build_preproc_db(tmp_path, [FOLDER / f for f in FILES])
    observations = [LazyPreprocSOTODLibObservation(obs_id, config_path) for obs_id in OBS_IDS]

    n_loads = 0
    original = SOTODLibObservation.from_preproc_group.__func__

    def counting(cls, *args, **kwargs):
        nonlocal n_loads
        n_loads += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(SOTODLibObservation, 'from_preproc_group', classmethod(counting))

    config = MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=16)),
    )
    MultiObservationMapMaker(observations, config=config).make_maps()

    # one load per observation (the fused build/accumulate pass), not two
    assert n_loads == len(observations)


def test_reader_invalid_data_field_name(observations) -> None:
    """Test that passing an invalid data field name raises a ValueError."""
    # Test with single invalid field name
    with pytest.raises(
        ValueError, match="Requested data fields {'invalid_field'} are not supported"
    ):
        ObservationReader.from_observations(observations, requested_fields=['invalid_field'])

    # Test with mix of valid and invalid field names
    with pytest.raises(ValueError, match="Requested data fields {'bad_field'} are not supported"):
        ObservationReader.from_observations(
            observations, requested_fields=['sample_data', 'bad_field']
        )


@pytest.mark.parametrize(
    'requested_fields',
    [
        ['sample_data', 'valid_sample_masks', 'detector_quaternions'],  # detector-related
        ['timestamps', 'boresight_quaternions', 'valid_scanning_masks'],  # time/boresight-related
        ['sample_data', 'timestamps', 'valid_scanning_masks', 'boresight_quaternions'],  # mixed
        ['sample_data'],  # detector data only
        ['valid_sample_masks'],  # detector flags only
        ['valid_scanning_masks'],  # scanning mask only
        ['boresight_quaternions'],  # boresight quats only
        ['detector_quaternions'],  # detector quats only
        ['timestamps'],  # timestamps only
        ['hwp_angles'],  # HWP angles only
        ['noise_model_fits'],  # optional field only
    ],
)
def test_reader_subset_of_data_fields(observations, requested_fields: list[str]) -> None:
    """Test that passing a subset of data fields loads only those fields."""
    reader = ObservationReader.from_observations(observations, requested_fields=requested_fields)
    assert set(reader.out_structure.keys()) == set(requested_fields)


def test_reader_all_fields_demod(demod_observations) -> None:
    """Test consistency for all available fields in demodulated mode."""
    stokes = 'IQU'
    reader = ObservationReader.from_observations(
        demod_observations,
        requested_fields=AbstractGroundObservation.AVAILABLE_READER_FIELDS,
        stokes=stokes,
    )
    ndet_max, nsample_max, nint_max = max(OBS_NDET), max(OBS_NSAMPLE), max(OBS_NINTERVAL)
    kls = Stokes.class_for(stokes)

    assert reader.out_structure == {
        'metadata': HashedObservationMetadata(
            uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            telescope_uid=jax.ShapeDtypeStruct((), dtype=jnp.uint32),
            detector_uids=jax.ShapeDtypeStruct((ndet_max,), dtype=jnp.uint32),
        ),
        'sample_data': kls.structure_for((ndet_max, nsample_max), jnp.float64),
        'valid_sample_masks': jax.ShapeDtypeStruct((ndet_max, nsample_max), dtype=jnp.bool),
        'valid_scanning_masks': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'timestamps': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'hwp_angles': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'detector_quaternions': jax.ShapeDtypeStruct((ndet_max, 4), dtype=jnp.float64),
        'boresight_quaternions': jax.ShapeDtypeStruct((nsample_max, 4), dtype=jnp.float64),
        'noise_model_fits': jax.ShapeDtypeStruct((len(stokes), ndet_max, 4), dtype=jnp.float64),
        'azimuth': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'elevation': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.float64),
        'left_scan_mask': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'right_scan_mask': jax.ShapeDtypeStruct((nsample_max,), dtype=jnp.bool),
        'scanning_intervals': jax.ShapeDtypeStruct((nint_max, 2), dtype=jnp.int32),
    }

    for i in range(len(FILES)):
        datum, padding, _ = reader.read(i)

        assert as_structure(datum) == reader.out_structure

        ndet, nsample, nint = OBS_NDET[i], OBS_NSAMPLE[i], OBS_NINTERVAL[i]
        assert tuple(padding['metadata'].uid) == ()
        assert tuple(padding['metadata'].telescope_uid) == ()
        assert tuple(padding['metadata'].detector_uids) == (ndet_max - ndet,)
        assert tuple(padding['sample_data'].data) == (0, ndet_max - ndet, nsample_max - nsample)
        assert tuple(padding['valid_sample_masks']) == (ndet_max - ndet, nsample_max - nsample)
        assert tuple(padding['valid_scanning_masks']) == (nsample_max - nsample,)
        assert tuple(padding['timestamps']) == (nsample_max - nsample,)
        assert tuple(padding['hwp_angles']) == (nsample_max - nsample,)
        assert tuple(padding['detector_quaternions']) == (ndet_max - ndet, 0)
        assert tuple(padding['boresight_quaternions']) == (nsample_max - nsample, 0)
        assert tuple(padding['noise_model_fits']) == (0, ndet_max - ndet, 0)
        assert tuple(padding['azimuth']) == (nsample_max - nsample,)
        assert tuple(padding['elevation']) == (nsample_max - nsample,)
        assert tuple(padding['left_scan_mask']) == (nsample_max - nsample,)
        assert tuple(padding['right_scan_mask']) == (nsample_max - nsample,)
        assert tuple(padding['scanning_intervals']) == (nint_max - nint, 0)

        # starts are sorted, and padding conserves the true number of intervals
        intervals = np.asarray(datum['scanning_intervals'])
        assert np.all(np.diff(intervals[:, 0]) >= 0)
        assert np.sum(intervals[:, 1] > intervals[:, 0]) == nint


@pytest.mark.parametrize(
    'requested_fields',
    [
        ['sample_data'],
        ['sample_data', 'noise_model_fits'],
        ['sample_data', 'timestamps', 'boresight_quaternions'],
    ],
)
def test_reader_subset_of_data_fields_demod(
    demod_observations, requested_fields: list[str]
) -> None:
    """Test that passing a subset of data fields loads only those fields in demodulated mode."""
    reader = ObservationReader.from_observations(
        demod_observations, requested_fields=requested_fields, stokes='IQU'
    )
    assert set(reader.out_structure.keys()) == set(requested_fields)
