import logging

import pytest

from so_mapmaking.util import resolve_obsids, setup_logger

SOTODLIB_LOGGERS = ('sotodlib', 'sotodlib.core.metadata.loader')


@pytest.fixture
def restore_loggers():
    levels = {name: logging.getLogger(name).level for name in SOTODLIB_LOGGERS}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
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
