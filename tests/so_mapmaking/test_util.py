import logging
from pathlib import Path

import pytest

from so_mapmaking.util import (
    EarlyRecords,
    obsdir_observations,
    preproc_observations,
    resolve_obsids,
    setup_logger,
)
from tests.interfaces.sotodlib._preproc_db import build_preproc_db

SOTODLIB_LOGGERS = ('sotodlib', 'sotodlib.core.metadata.loader')


@pytest.fixture
def restore_loggers():
    levels = {name: logging.getLogger(name).level for name in SOTODLIB_LOGGERS}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    for handler in logging.getLogger('furax').handlers:
        handler.close()
    logging.getLogger('furax').handlers.clear()


@pytest.mark.usefixtures('restore_loggers')
@pytest.mark.parametrize('loglevel', ['info', 'warning', 'error'])
def test_setup_logger_silences_sotodlib(loglevel: str) -> None:
    setup_logger(loglevel, None)

    assert logging.getLogger('sotodlib').getEffectiveLevel() == logging.WARNING
    loader = logging.getLogger('sotodlib.core.metadata.loader')
    assert not loader.isEnabledFor(logging.ERROR)
    assert logging.getLogger('sotodlib.core.context').getEffectiveLevel() == logging.WARNING


@pytest.mark.usefixtures('restore_loggers')
def test_setup_logger_debug_keeps_sotodlib() -> None:
    setup_logger('debug', None)

    assert logging.getLogger('sotodlib').isEnabledFor(logging.INFO)
    assert logging.getLogger('sotodlib.core.metadata.loader').isEnabledFor(logging.ERROR)


def _console_handler() -> logging.Handler:
    (console,) = [
        h
        for h in logging.getLogger('furax').handlers
        if type(h) is logging.StreamHandler  # a FileHandler is a StreamHandler too
    ]
    return console


def _passes(handler: logging.Handler, level: int) -> bool:
    record = logging.LogRecord('furax.so_mapmaking', level, __file__, 0, 'message', None, None)
    return bool(handler.filter(record))


@pytest.mark.usefixtures('restore_loggers')
def test_setup_logger_keeps_the_records_logged_before_the_file(tmp_path) -> None:
    setup_logger('info', None)
    logging.getLogger('furax.so_mapmaking').warning('logged early')

    setup_logger('info', tmp_path / 'run.log')
    logging.getLogger('furax.so_mapmaking').info('logged late')
    for handler in logging.getLogger('furax').handlers:
        handler.flush()

    text = (tmp_path / 'run.rank0.log').read_text()
    assert text.count('logged early') == 1
    assert text.index('logged early') < text.index('logged late')
    # the file takes the records over: later calls do not write them again
    assert not any(isinstance(h, EarlyRecords) for h in logging.getLogger('furax').handlers)


@pytest.mark.usefixtures('restore_loggers')
@pytest.mark.parametrize(
    ('rank', 'level', 'shown'),
    [
        (0, logging.INFO, True),
        (0, logging.WARNING, True),
        (1, logging.INFO, False),
        (1, logging.WARNING, True),
        (1, logging.ERROR, True),
    ],
)
def test_console_shows_the_records_below_warning_of_rank_0_only(
    rank: int, level: int, shown: bool
) -> None:
    setup_logger('info', None, process_index=rank)

    assert _passes(_console_handler(), level) == shown


@pytest.mark.usefixtures('restore_loggers')
def test_log_file_of_a_rank_holds_its_info_records(tmp_path) -> None:
    setup_logger('info', tmp_path / 'run.log', process_index=1)
    logging.getLogger('furax.so_mapmaking').info('from rank 1')
    for handler in logging.getLogger('furax').handlers:
        handler.flush()

    assert 'from rank 1' in (tmp_path / 'run.rank1.log').read_text()


@pytest.mark.parametrize('from_file', [False, True])
def test_resolve_obsids_keeps_each_id_once(tmp_path, from_file: bool) -> None:
    obsids = ['obs_2', 'obs_1', 'obs_2', 'obs_3', 'obs_1']
    obsids_file = tmp_path / 'obsids.txt'
    obsids_file.write_text('\n'.join(obsids) + '\n\n')

    if from_file:
        resolved = resolve_obsids(None, obsids_file)
    else:
        resolved = resolve_obsids(obsids, None)
    assert resolved == ['obs_2', 'obs_1', 'obs_3']


@pytest.mark.parametrize(
    ('obsids', 'expected'),
    [(['obs_b', 'obs_missing', 'obs_a'], ['obs_b', 'obs_a']), ([], ['obs_a', 'obs_b'])],
)
def test_obsdir_observations(tmp_path, obsids: list[str], expected: list[str]) -> None:
    """The files of the given ids that exist, or else every file of the directory."""
    for name in ('obs_b', 'obs_a'):
        (tmp_path / f'{name}.h5').touch()

    observations = obsdir_observations(tmp_path, obsids, None)
    assert [obs.name for obs in observations] == expected


@pytest.fixture
def preproc_config(tmp_path) -> Path:
    folder = Path(__file__).parents[1] / 'data/sotodlib'
    return build_preproc_db(tmp_path, [folder / 'test_obs.h5'])


@pytest.mark.parametrize(
    ('wafers', 'join_wafers', 'selections', 'join_groups'),
    [
        # the test detectors have no wafer slot, so the requested wafers cannot be checked
        (('ws0', 'ws1'), False, [{'wafer_slot': 'ws0'}, {'wafer_slot': 'ws1'}], [False, False]),
        (('ws0', 'ws1'), True, [{'wafer_slot': ['ws0', 'ws1']}], [True]),
        (('ws0',), True, [{'wafer_slot': 'ws0'}], [True]),
        (None, True, [{}], [True]),
        ((), False, [{}], [False]),
    ],
)
def test_preproc_observations_wafers(
    preproc_config, wafers, join_wafers, selections, join_groups
) -> None:
    observations = preproc_observations(
        ['obs_12345_sometel'], preproc_config, None, wafers, '', join_wafers, None, None
    )

    assert [obs.detector_selection for obs in observations] == selections
    assert [obs.join_groups for obs in observations] == join_groups


def test_preproc_observations_all_wafers_needs_wafer_slots(preproc_config) -> None:
    assert (
        preproc_observations(
            ['obs_12345_sometel'], preproc_config, None, None, '', False, None, None
        )
        == []
    )
