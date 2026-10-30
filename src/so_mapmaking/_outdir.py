"""The record a `furax-so-map` run keeps in its output directory, beside the maps.

A run writes `status.json` when it starts, with the state `running`, and rewrites it last, with the
state `done` or `failed`: a directory whose state is `done` holds every product of the run. A run
killed by a signal, e.g. at the end of its time limit, stays `running`.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

STATUS_FILE = 'status.json'
LOG_FILE = 'furax-so-map.log'


class RunState(StrEnum):
    """The state of a run, as `status.json` records it."""

    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'


def earlier_run(outdir: Path) -> str | None:
    """A description of the run that wrote into `outdir` earlier, or `None` if there was none."""
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.mapmaking import MapMakingResults

    try:
        status = json.loads((outdir / STATUS_FILE).read_text())
    except (FileNotFoundError, NotADirectoryError):
        # Files of the mapmaker mark a run older than `status.json`, or one that never wrote it
        return 'an earlier run (no status)' if MapMakingResults.exists(outdir) else None
    except (OSError, ValueError):
        return f'an earlier run (unreadable {STATUS_FILE})'
    return f'an earlier run ({status.get("state", "?")}, started {status.get("started", "?")})'


def remove_run_files(outdir: Path) -> None:
    """Delete the files a run writes into `outdir`: its status, default logs and mapmaker files.

    Other files, such as the output of a batch scheduler, are kept.
    """
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.mapmaking import MapMakingResults

    MapMakingResults.remove(outdir)
    logs = outdir.glob('furax-so-map.rank*.log')  # LOG_FILE, as setup_logger suffixes it
    for path in [outdir / STATUS_FILE, *logs]:
        path.unlink(missing_ok=True)


def claim(outdir: Path, **fields: Any) -> bool:
    """Claim `outdir` for a run by creating its `status.json`, in the state `running`.

    Returns `True` if the claim succeeded, `False` otherwise: `status.json` already exists, so the
    directory belongs to another run, and is left alone.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL (exclusive): the file is the claim
        os.close(os.open(outdir / STATUS_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    write_status(outdir, RunState.RUNNING, **fields)
    return True


def write_status(outdir: Path, state: RunState, **fields: Any) -> None:
    """Write `status.json` with the given state and fields, e.g. `started`, creating `outdir`."""
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.mapmaking.results import write_json

    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / STATUS_FILE, {'state': state, **fields})
