"""The provenance of a mapmaking run: the code, environment and command that produced it."""

import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import jax

_PACKAGES = ('jax', 'jaxlib', 'numpy', 'sotodlib', 'so3g', 'toast', 'litebird_sim', 'pixell')
"""Packages whose versions are recorded, when installed."""


def utc_now() -> str:
    """The current time, in ISO 8601 format and UTC."""
    return datetime.now(UTC).isoformat(timespec='seconds')


def provenance(started: str) -> dict[str, Any]:
    """The provenance of a run started at `started`, finishing now, as JSON values.

    It records the furax version and, for a furax run from a git checkout, its commit and whether
    the tracked files differed from it; the versions of the main dependencies; the command line,
    host and devices; and the start and end times.
    """
    return {
        'furax': {'version': _version('furax'), **_git_state(Path(__file__).parent)},
        'packages': {name: v for name in _PACKAGES if (v := _version(name)) is not None},
        'python': platform.python_version(),
        'command': sys.argv,
        'host': socket.gethostname(),
        'processes': jax.process_count(),
        'devices': jax.device_count(),
        'platform': jax.devices()[0].platform,
        'started': started,
        'finished': utc_now(),
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_state(path: Path) -> dict[str, Any]:
    """The commit of the git checkout holding `path`, and whether its tracked files differ."""

    def git(*args: str) -> str:
        out = subprocess.run(
            ['git', '-C', str(path), *args], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip()

    try:
        # an installed copy may sit in another repository (a virtual environment inside a
        # project): the checkout must track this package's files to be furax's own
        git('ls-files', '--error-unmatch', '__init__.py')
        commit = git('rev-parse', 'HEAD')
        branch = git('rev-parse', '--abbrev-ref', 'HEAD')
        dirty = git('status', '--porcelain', '--untracked-files=no') != ''
    except (OSError, subprocess.SubprocessError):
        return {}  # not a git checkout (an installed package), or git unavailable
    return {'commit': commit, 'branch': branch, 'dirty': dirty}
