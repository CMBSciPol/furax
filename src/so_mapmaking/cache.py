"""Manage the cache of preprocessed observations written by `furax-so-map --cache-dir`."""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import shutil
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cyclopts import App

from .util import enable_x64, preproc_observations, resolve_obsids, setup_logger

if TYPE_CHECKING:
    from furax.interfaces.sotodlib import LazyPreprocSOTODLibObservation

app = App(help='Manage the cache of preprocessed observations of `furax-so-map --cache-dir`.')

# A cache key, as LazyPreprocSOTODLibObservation names the directories of its cache
_KEY = re.compile(r'[0-9a-f]{16}')


@app.command
def fill(
    cache_dir: Path,
    init_config: Path,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    mapmaking_config: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    nproc: int = 1,
    dry_run: bool = False,
    loglevel: str = 'info',
    log_path: Path | None = None,
):
    """Preprocess observations into the cache, without mapmaking.

    Pass the arguments of the `furax-so-map` run to prepare: that run then finds every
    observation in the cache. Observations already cached are skipped. With `--wafer all`, each
    observation is saved with its wafers joined, as that run maps it; a run with a single wafer
    uses a cache of its own.

    Args:
        cache_dir: Cache directory.
        init_config: Base-layer preprocessing config file.
        proc_config: Optional second-layer preprocessing config file.
        obsid: Observation id(s) to cache.
        obsids_file: Text file with one obsid per line.
        mapmaking_config: Mapmaking config file, of which only the sotodlib section matters.
        wafer: Wafer slot selection, or 'all' to join every wafer of an observation.
        band: Wafer bandpass selection.
        nproc: Number of worker processes (single node).
        dry_run: Only report the cache key and which observations are missing.
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path.
    """
    enable_x64()
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.mapmaking import MapMakingConfig

    logger = setup_logger(loglevel, log_path)
    obsids = resolve_obsids(obsid, obsids_file)
    if not obsids:
        logger.warning('no observations to cache')
        return
    config = MapMakingConfig.load_yaml(mapmaking_config) if mapmaking_config else MapMakingConfig()
    observations = preproc_observations(
        obsids, init_config, proc_config, wafer, band, config.sotodlib, cache_dir
    )
    missing = [obs for obs in observations if not _cache_path(obs).exists()]
    logger.info(f'cache key: {_cache_path(observations[0]).parent}')
    logger.info(f'{len(observations) - len(missing)} of {len(observations)} observations cached')
    if dry_run:
        for obs in missing:
            logger.info(f'missing: {obs.name}')
        return

    nproc = max(1, min(nproc, len(missing)))
    if nproc == 1:
        failed = [obs.name for obs in missing if not _fill_one(obs)]
    else:
        failed = _fill_in_parallel(missing, nproc, loglevel, log_path)
    if failed:
        logger.error(f'{len(failed)} observations not cached: {" ".join(failed)}')
        return 1


@app.command(name='list')
def list_keys(cache_dir: Path) -> None:
    """List the cache keys, with the inputs and the size of each.

    Args:
        cache_dir: Cache directory.
    """
    for directory in sorted(_key_directories(cache_dir)):
        stats = _stats(directory.glob('*.h5'))
        size = sum(stat.st_size for stat in stats) / 2**30
        print(f'{directory.name}: {len(stats)} observations, {size:.1f} GiB')
        if stats:
            last = max(stat.st_mtime for stat in stats)
            print(f'  last written: {time.strftime("%Y-%m-%d %H:%M", time.localtime(last))}')
        try:
            inputs = json.loads((directory / 'key.json').read_text())
        except FileNotFoundError:
            print('  inputs unknown (no key.json)')
            continue
        for line in _describe(inputs):
            print(f'  {line}')


@app.command
def prune(
    cache_dir: Path,
    key: list[str] | None = None,
    older_than: float = 24.0,
    dry_run: bool = False,
) -> int | None:
    """Delete the temporary files left by interrupted writes, and the given cache keys.

    Args:
        cache_dir: Cache directory.
        key: Cache keys to delete, as `list` prints them. A run reading a deleted key fails.
        older_than: Age in hours past which a temporary file is deleted. Younger ones may still
            be written by a running job.
        dry_run: Only print what would be deleted.
    """
    cutoff = time.time() - older_than * 3600
    for directory in _key_directories(cache_dir):
        for tmp in directory.glob('*.tmp.*'):
            try:
                if tmp.stat().st_mtime >= cutoff:
                    continue
                print(f'delete {tmp}')
                if not dry_run:
                    tmp.unlink()
            except FileNotFoundError:  # renamed into place, or deleted, since the glob
                pass
    status = None
    for name in key or []:
        directory = cache_dir / name
        if not (_KEY.fullmatch(name) and directory.is_dir()):
            print(f'no cache key {name} in {cache_dir}')
            status = 1
            continue
        print(f'delete {directory}')
        if not dry_run:
            shutil.rmtree(directory)
    return status


def _cache_path(observation: LazyPreprocSOTODLibObservation) -> Path:
    path = observation.cache_path
    assert path is not None  # preproc_observations was given a cache_dir
    return path


def _fill_in_parallel(
    observations: list[LazyPreprocSOTODLibObservation],
    nproc: int,
    loglevel: str,
    log_path: Path | None,
) -> list[str]:
    """Loads observations into the cache in `nproc` processes; returns those not cached."""
    logger = logging.getLogger('furax.so_mapmaking')
    # spawn, not fork: a process forked after the JAX backend has started can deadlock
    context = multiprocessing.get_context('spawn')
    failed = []
    broken = False
    with ProcessPoolExecutor(nproc, context, _init_worker, (loglevel, log_path)) as executor:
        # one task per observation, so that a worker free takes the next one
        futures = [executor.submit(_fill_one, obs) for obs in observations]
        for obs, future in zip(observations, futures, strict=True):
            try:
                if not future.result():
                    failed.append(obs.name)
            except BrokenProcessPool:
                broken = True
                if not _cache_path(obs).exists():  # it may have been cached before the break
                    failed.append(obs.name)
    if broken:
        # e.g. killed out of memory: the pool stops, and its pending observations with it
        logger.error('a worker process died')
    return failed


def _fill_one(observation: LazyPreprocSOTODLibObservation) -> bool:
    """Loads an observation into the cache, and tells whether it is cached."""
    logger = logging.getLogger('furax.so_mapmaking')
    start = time.perf_counter()
    try:
        observation.get_data([])
    except Exception:
        logger.exception(f'{observation.name}: could not load')
        return False
    if not _cache_path(observation).exists():  # the write failed, and was logged
        return False
    logger.info(f'{observation.name}: cached ({time.perf_counter() - start:.0f} s)')
    return True


def _init_worker(loglevel: str, log_path: Path | None) -> None:
    enable_x64()
    # the workers are numbered from 1: log each to its own rank file
    setup_logger(loglevel, log_path, process_index=multiprocessing.current_process()._identity[0])


def _key_directories(cache_dir: Path) -> list[Path]:
    if not cache_dir.is_dir():
        return []
    return [path for path in cache_dir.iterdir() if path.is_dir() and _KEY.fullmatch(path.name)]


def _stats(paths: Iterable[Path]) -> list[os.stat_result]:
    """The stats of the files that still exist, as a concurrent prune may delete them."""
    stats = []
    for path in paths:
        try:
            stats.append(path.stat())
        except FileNotFoundError:
            pass
    return stats


def _describe(inputs: dict[str, Any]) -> list[str]:
    """The inputs of a cache key, summarised one per line, with `?` for those missing."""
    archives = []
    for config in inputs.get('preprocess_configs') or []:
        archive = config.get('archive') if isinstance(config, dict) else None
        archives.append(archive.get('index') if isinstance(archive, dict) else None)
    return [
        f'archives: {", ".join(str(archive or "?") for archive in archives) or "?"}',
        f'detector selection: {inputs.get("detector_selection", "?")}',
        f'join groups: {inputs.get("join_groups", "?")}',
        f'downsample: {inputs.get("downsample", "?")}',
        f'sotodlib {inputs.get("sotodlib", "?")}, cache version {inputs.get("cache_version", "?")}',
    ]


if __name__ == '__main__':
    app()
