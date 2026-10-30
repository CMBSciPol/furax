import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from so_mapmaking._outdir import claim, earlier_run

FOLDER = Path(__file__).parents[1] / 'data/sotodlib'


@pytest.mark.parametrize(
    'files, expected',
    [
        ({}, None),
        ({'furax-so-map.rank0.log': ''}, None),
        ({'mapmaking_config.yaml': ''}, 'an earlier run (no status)'),
        ({'map.fits': ''}, 'an earlier run (no status)'),
        (
            {'status.json': '{"state": "done", "started": "2026-10-29T10:00:00+00:00"}'},
            'an earlier run (done, started 2026-10-29T10:00:00+00:00)',
        ),
        ({'status.json': '{'}, 'an earlier run (unreadable status.json)'),
    ],
)
def test_earlier_run(tmp_path, files: dict[str, str], expected: str | None) -> None:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    assert earlier_run(tmp_path) == expected
    assert earlier_run(tmp_path / 'missing') is None


def test_only_the_first_claim_of_a_directory_succeeds(tmp_path) -> None:
    outdir = tmp_path / 'maps'
    assert claim(outdir, started='first')
    assert not claim(outdir, started='second')
    assert json.loads((outdir / 'status.json').read_text()) == {
        'state': 'running',
        'started': 'first',
    }
    assert [p.name for p in outdir.iterdir()] == ['status.json']


def test_run_records_a_done_run_and_keeps_it(tmp_path) -> None:
    outdir = tmp_path / 'maps'
    assert _run(outdir).returncode == 0
    status = _status(outdir)
    assert status['state'] == 'done'
    assert 'finished' in status
    assert 'furax' in json.loads((outdir / 'provenance.json').read_text())

    refused = _run(outdir)
    assert refused.returncode == 1
    assert '--overwrite' in refused.stderr
    assert _status(outdir) == status

    # a stale product goes, the file of a batch scheduler stays
    (outdir / 'amplitudes.npz').write_bytes(b'')
    (outdir / 'slurm.out').write_text('')
    assert _run(outdir, '--overwrite').returncode == 0
    assert _status(outdir)['state'] == 'done'
    assert not (outdir / 'amplitudes.npz').exists()
    assert (outdir / 'slurm.out').exists()


def test_run_rejects_an_invalid_config_without_touching_outdir(tmp_path) -> None:
    # Pomme maps Q and U only, and the default landscape is IQU
    mapmaking_config = _config_file(tmp_path, {'method': 'Pomme'})
    outdir = tmp_path / 'maps'

    rejected = _run(outdir, '--mapmaking-config', str(mapmaking_config))
    assert rejected.returncode == 1
    assert 'invalid mapmaking config' in rejected.stderr
    assert not outdir.exists()


def test_run_refuses_an_output_path_that_is_a_file(tmp_path) -> None:
    outdir = tmp_path / 'maps'
    outdir.write_text('')

    refused = _run(outdir)
    assert refused.returncode == 1
    assert 'is not a directory' in refused.stderr
    assert outdir.read_text() == ''


def test_run_refuses_an_output_directory_it_cannot_create(tmp_path) -> None:
    parent = tmp_path / 'file'
    parent.write_text('')

    refused = _run(parent / 'maps')
    assert refused.returncode == 1
    assert 'cannot prepare' in refused.stderr


def test_run_stops_when_the_observations_cannot_be_loaded(tmp_path) -> None:
    outdir = tmp_path / 'maps'
    command = [sys.executable, '-m', 'so_mapmaking.run', '--outdir', str(outdir)]
    # a preproc config that does not exist fails the load of the observations
    ids = tmp_path / 'obsids.txt'
    ids.write_text('obs\n')
    command += ['--obsids-file', str(ids), '--init-config', str(tmp_path / 'missing.yaml')]
    stopped = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    assert stopped.returncode == 1
    assert 'loading the observations failed' in stopped.stderr
    assert not outdir.exists()


def test_run_records_a_failed_run(tmp_path) -> None:
    # the mapmaker loads the sky to inject when it starts: a missing file fails the run
    sky = tmp_path / 'missing.fits'
    mapmaking_config = _config_file(tmp_path, {'injection': {'path': str(sky)}})
    outdir = tmp_path / 'maps'

    assert _run(outdir, '--mapmaking-config', str(mapmaking_config)).returncode == 1
    status = _status(outdir)
    assert status['state'] == 'failed'
    assert 'missing.fits' in status['error']
    assert (outdir / 'mapmaking_config.yaml').exists()
    assert (outdir / 'provenance.json').exists()
    assert (outdir / 'furax-so-map.rank0.log').exists()


def _run(outdir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Runs `furax-so-map` on a test observation, in a fresh interpreter.

    The command initializes `jax.distributed`, which must come before any use of the JAX backend,
    and the test session has already started it.
    """
    command = [sys.executable, '-m', 'so_mapmaking.run', '--obsdir', str(FOLDER)]
    command += ['--obsid', 'test_obs', '--outdir', str(outdir), *args]
    return subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)


def _status(outdir: Path) -> dict[str, str]:
    return json.loads((outdir / 'status.json').read_text())


def _config_file(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / 'mapmaking.yaml'
    path.write_text(yaml.safe_dump(config))
    return path
