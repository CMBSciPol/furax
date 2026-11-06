"""On-disk cache of preprocessed observations, as [`LazyPreprocSOTODLibObservation`][] uses it.

The cache directory holds one subdirectory per cache key. A key is a hash of the inputs that
change the loaded data (see [`cache_inputs`][]), and its subdirectory holds:

- `<observation_id>__<group>.h5`: the data of a preprocessing group.
- `<observation_id>__<group>.skip`: a group that is skipped, with nothing to cache.
- `key.json`: the inputs the key was derived from.
- `*.tmp.*`: files being written, or left by an interrupted write.

The `furax-so-cache` command reads this layout without importing furax, which pulls in JAX: it
repeats the key pattern, the file suffixes and the `key.json` fields, and `test_cache.py` checks
that they agree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import numpy as np
import sotodlib
import yaml

__all__ = [
    'PreprocCache',
    'cache_inputs',
]

logger = logging.getLogger(__name__)

# Version of the cache, part of its key. Bump it when furax changes the data it loads, e.g. which
# timestreams downsample_obs resamples or how preprocessing groups are joined, so that existing
# caches are not reused.
_CACHE_VERSION = 2

# Context entries that locate the raw TOD files, excluded from the cache key (see _read_context)
_KEY_EXCLUDED_CONTEXT_KEYS = ('obsfiledb.prefix',)


class PreprocCache:
    """The cache directory of one cache key.

    The files are written to a temporary name and renamed into place, so concurrent runs can share
    a cache and never read a partial file.

    Args:
        cache_dir: Cache directory, which may hold the directories of other keys.
        inputs: The inputs that determine the cached data, as [`cache_inputs`][] returns them.
    """

    def __init__(self, cache_dir: Path, inputs: dict[str, Any]) -> None:
        self.inputs = inputs
        self.directory = cache_dir / _digest(inputs)

    def group_path(self, observation_id: str, selection: dict[str, Any]) -> Path:
        """The file of the preprocessing group of a detector selection. It need not exist yet."""
        group = '_'.join(f'{k}-{_join_values(v)}' for k, v in sorted(selection.items())) or 'all'
        name = re.sub(r'[^\w.+-]', '-', f'{observation_id}__{group}')
        return self.directory / f'{name}.h5'

    def is_skipped(self, path: Path) -> bool:
        """Whether the group of the file is recorded as skipped."""
        return _skip_path(path).exists()

    def is_complete(self, paths: Collection[Path]) -> bool:
        """Whether there is at least one group, and each is cached or recorded as skipped."""
        return bool(paths) and all(path.exists() or self.is_skipped(path) for path in paths)

    def write(self, path: Path, write: Callable[[Path], object]) -> bool:
        """Caches a group by calling `write` on a temporary file; tells whether it is cached.

        A failed write is logged and leaves no file: the group is then not cached.
        """
        self._write_key()
        _write_atomically(path, write)
        return path.exists()

    def skip(self, path: Path) -> None:
        """Records the group of the file as skipped."""
        self._write_key()
        _write_atomically(_skip_path(path), lambda tmp: tmp.touch())

    def _write_key(self) -> None:
        """Records the inputs of the key next to the files, once."""
        key = self.directory / 'key.json'
        if not key.exists():
            _write_atomically(key, lambda tmp: tmp.write_text(_dumps(self.inputs)))


def cache_inputs(init_config: Path, proc_config: Path | None, downsample: int) -> dict[str, Any]:
    """The inputs that determine the data loaded from the preprocessing configs.

    Args:
        init_config: Base-layer preprocessing config file.
        proc_config: Optional second-layer preprocessing config file.
        downsample: Downsampling factor of the loaded data.
    """
    paths = [init_config] + ([proc_config] if proc_config else [])
    configs = [_canonical_config(path) for path in paths]
    return {
        'preprocess_configs': configs,
        'contexts': [_read_context(config) for config in configs],
        'downsample': downsample,
        'sotodlib': sotodlib.__version__,
        'cache_version': _CACHE_VERSION,
    }


def _canonical_config(path: Path) -> dict[str, Any]:
    """A preprocessing config, with the paths sotodlib resolves against the cwd made absolute.

    Two configs with the same text but relative paths read different archives from different
    working directories, so they must not share a cache key. A config written by
    `furax-so-stage` reads node-local copies of the databases of the config it names as its
    source, and is that config here.
    """
    doc = yaml.safe_load(path.read_text())
    # the entry so_mapmaking.stage writes in the configs it stages
    source = doc.get('furax_staged_from')
    if isinstance(source, str):
        return _canonical_config(Path(source))
    if isinstance(doc.get('context_file'), str):
        doc['context_file'] = Path(doc['context_file']).resolve().as_posix()
    archive = doc.get('archive')
    if isinstance(archive, dict) and isinstance(archive.get('index'), str):
        archive['index'] = Path(archive['index']).resolve().as_posix()
    return doc


def _read_context(config: dict[str, Any]) -> Any:
    """The context of a canonical preprocessing config, `None` if it names none.

    The entries that only say where the raw TOD files are stored are left out: the TOD content is
    not part of the cache key, so they cannot tell data apart, and moving the files must not
    invalidate a cache.
    """
    path = config.get('context_file')
    if not isinstance(path, str):
        return None
    context = yaml.safe_load(Path(path).read_text())
    if isinstance(context, dict):
        for key in _KEY_EXCLUDED_CONTEXT_KEYS:
            context.pop(key, None)
    return context


def _digest(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(_dumps(inputs).encode()).hexdigest()[:16]


def _dumps(inputs: dict[str, Any]) -> str:
    # sorted keys: the key must not depend on the order of the entries in the configs
    return json.dumps(inputs, sort_keys=True, indent=2, default=str)


def _join_values(value: Any) -> str:
    return '+'.join(str(v) for v in np.atleast_1d(value))


def _skip_path(path: Path) -> Path:
    return path.with_suffix('.skip')


def _write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Writes `path` through a temporary file in its directory, renamed into place when complete.

    The rename is atomic, so a concurrent reader sees either no file or all of it, and of two
    concurrent writers of the same content the last one wins. A failed write is logged and
    leaves no file.
    """
    tmp = path.with_name(f'{path.name}.tmp.{uuid.uuid4().hex}')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write(tmp)
        # flush the data to storage before the name appears, or a crash could leave the final
        # name on a partial file
        with tmp.open('rb') as file:
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except Exception:
        # not only I/O errors: AxisManager.save also fails on fields it cannot encode, and the
        # observation is loaded either way
        logger.warning('Could not write %s', path, exc_info=True)
    finally:
        if tmp.exists():
            tmp.unlink()
