import logging
import os
import time
from pathlib import Path

import pytest

from furax.mapmaking.config import SotodlibConfig
from so_mapmaking.cache import fill, list_keys, prune
from so_mapmaking.stage import Stager, stage_config
from so_mapmaking.util import preproc_observations
from tests.interfaces.sotodlib._preproc_db import build_preproc_db, cut_group, drop_group_record

FOLDER = Path(__file__).parents[1] / 'data/sotodlib'
FILES = ['test_obs.h5', 'test_obs_2.h5']
OBS_IDS = ['obs_12345_sometel', 'obs_54321_someothertel']
KEY = '0123456789abcdef'


@pytest.fixture(autouse=True)
def restore_furax_logger():
    yield
    logging.getLogger('furax').handlers.clear()


@pytest.fixture
def config(tmp_path) -> Path:
    return build_preproc_db(tmp_path, [FOLDER / f for f in FILES])


def test_fill_caches_what_the_map_run_reads(tmp_path, config, capsys) -> None:
    cache_dir = tmp_path / 'cache'
    # no wafer or band: the test detectors have neither
    assert fill(cache_dir, config, obsid=OBS_IDS, wafer='', band='') is None

    observations = preproc_observations(OBS_IDS, config, None, '', '', SotodlibConfig(), cache_dir)
    assert all(obs.is_cached() for obs in observations)
    fill(cache_dir, config, obsid=OBS_IDS, wafer='', band='', dry_run=True)
    assert 'Cached 2 of 2 observations' in capsys.readouterr().err


def test_staged_configs_share_the_cache(tmp_path, config) -> None:
    """A config staged anywhere is keyed as its source, and loads from its staged copies."""
    cache_dir = tmp_path / 'cache'
    staged = []
    for dest in (tmp_path / 'stage_a', tmp_path / 'stage_b'):
        dest.mkdir()
        staged.append(stage_config(Stager(dest), config))
    observations = [
        preproc_observations([OBS_IDS[0]], path, None, '', '', None, cache_dir)[0]
        for path in (config, *staged)
    ]

    assert len({obs.cache_key_path for obs in observations}) == 1
    assert observations[1].get_data([]).n_detectors > 0
    assert observations[0].is_cached()


def test_staged_config_follows_its_source(tmp_path, config) -> None:
    """Editing the source of a staged config changes the key of the staged config with it."""
    dest = tmp_path / 'stage'
    dest.mkdir()
    staged = stage_config(Stager(dest), config)
    before = preproc_observations([OBS_IDS[0]], staged, None, '', '', None, tmp_path)[
        0
    ].cache_key_path
    config.write_text(config.read_text() + 'extra: 1\n')
    after, source = (
        preproc_observations([OBS_IDS[0]], path, None, '', '', None, tmp_path)[0]
        for path in (staged, config)
    )

    assert after.cache_key_path == source.cache_key_path != before


def test_fill_reports_observations_it_cannot_cache(tmp_path, config, capsys) -> None:
    cache_dir = tmp_path / 'cache'
    status = fill(cache_dir, config, obsid=[OBS_IDS[0], 'obs_missing'], wafer='', band='')

    assert status == 1
    assert 'Failed to cache 1 observations: obs_missing' in capsys.readouterr().err
    assert not list(cache_dir.glob('*/*.skip'))
    assert len(list(cache_dir.glob('*/*.h5'))) == 1


def test_list_keys(tmp_path, config, capsys) -> None:
    cache_dir = tmp_path / 'cache'
    fill(cache_dir, config, obsid=OBS_IDS, wafer='', band='')
    (cache_dir / KEY).mkdir()
    partial = cache_dir / KEY.replace('0', 'f')
    partial.mkdir()
    (partial / 'key.json').write_text('{"preprocess_configs": [{"archive": null}]}')
    capsys.readouterr()

    list_keys(cache_dir)
    out = capsys.readouterr().out
    assert '2 files' in out
    assert str(tmp_path / 'preprocess.sqlite') in out
    assert f'{KEY}: 0 files' in out
    assert 'no key.json' in out
    assert 'archives: ?' in out
    assert 'downsample: ?' in out


def test_fill_and_list_treat_a_skipped_group_as_cached(tmp_path, monkeypatch, capsys) -> None:
    """A group with no record is recorded as skipped: it is cached, and holds no data to list."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    drop_group_record(monkeypatch, 'detset0_1')
    cache_dir = tmp_path / 'cache'

    assert fill(cache_dir, config, obsid=[OBS_IDS[1]], wafer='all', band='') is None
    assert len(list(cache_dir.glob('*/*.skip'))) == 1
    capsys.readouterr()

    fill(cache_dir, config, obsid=[OBS_IDS[1]], wafer='all', band='', dry_run=True)
    assert 'Cached 1 of 1 observations' in capsys.readouterr().err
    list_keys(cache_dir)
    assert ': 1 files' in capsys.readouterr().out


def test_fill_does_not_fail_an_observation_cut_entirely(tmp_path, monkeypatch, capsys) -> None:
    """An observation without a detector left is recorded as such: it is cached, not failed."""
    config = build_preproc_db(tmp_path, [FOLDER / FILES[1]], n_detsets=2)
    cut_group(monkeypatch, 'detset0_0')
    cut_group(monkeypatch, 'detset0_1')
    cache_dir = tmp_path / 'cache'

    assert fill(cache_dir, config, obsid=[OBS_IDS[1]], wafer='all', band='') is None
    assert len(list(cache_dir.glob('*/*.skip'))) == 2
    assert 'Nothing to cache' in capsys.readouterr().err


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('temporary', ['a.h5.tmp.1', 'a.skip.tmp.1'])
def test_prune_temporary_files(tmp_path, dry_run: bool, temporary: str) -> None:
    directory = tmp_path / KEY
    directory.mkdir()
    old, young, cached = (directory / name for name in (temporary, 'b.h5.tmp.2', 'c.h5'))
    for path in (old, young, cached):
        path.touch()
    two_days_ago = time.time() - 48 * 3600
    for path in (old, cached):
        os.utime(path, (two_days_ago, two_days_ago))

    assert prune(tmp_path, dry_run=dry_run) is None
    assert old.exists() == dry_run
    assert young.exists()
    assert cached.exists()


def test_prune_keys(tmp_path) -> None:
    (tmp_path / KEY).mkdir()
    (tmp_path / KEY / 'c.h5').touch()
    (tmp_path / 'other').mkdir()

    assert prune(tmp_path, key=[KEY, 'other', '..']) == 1
    assert not (tmp_path / KEY).exists()
    assert (tmp_path / 'other').exists()
