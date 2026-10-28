from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from . import _preproc as pp

if TYPE_CHECKING:
    from furax.interfaces.sotodlib import LazyPreprocSOTODLibObservation, LazySOTODLibObservation
    from furax.mapmaking.config import SotodlibConfig


def enable_x64() -> None:
    """Enables JAX's 64-bit mode, before furax is imported.

    jax_healpy warns when it is imported with the 64-bit mode off. A command that takes the
    precision from its config sets it again once the config is loaded.
    """
    # Deferred: importing this module stays free of JAX
    import jax

    jax.config.update('jax_enable_x64', True)


def setup_logger(loglevel: str, log_path: Path | None, process_index: int = 0) -> logging.Logger:
    level = loglevel.upper()
    # Configure the package-root logger so every ``furax.*`` submodule (mapmaker, io.readers, ...)
    # propagates its records to these handlers. Configuring ``furax.mapmaker`` alone would drop
    # sibling loggers such as ``furax.io.readers`` (per-observation read logging).
    logger = logging.getLogger('furax')
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        f'%(asctime)s - [rank {process_index}] - %(levelname)s - %(message)s'
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_path is not None:
        log_path = log_path.with_stem(f'{log_path.stem}.rank{process_index}')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    _quiet_sotodlib(level)
    return logger


def _quiet_sotodlib(level: str) -> None:
    """Keep sotodlib out of the logs, unless the run is debugged.

    sotodlib prints its INFO records on import, and its metadata loader logs the error of a missing
    record before raising it, which furax handles and reports itself (e.g. a skipped group).
    """
    debug = logging.getLevelNamesMapping()[level] <= logging.DEBUG
    logging.getLogger('sotodlib').setLevel(logging.DEBUG if debug else logging.WARNING)
    loader = logging.getLogger('sotodlib.core.metadata.loader')
    loader.setLevel(logging.DEBUG if debug else logging.CRITICAL)


def detector_selection(wafer: str | None, band: str | None) -> dict[str, str] | None:
    if wafer is None and band is None:
        return None
    det_select = {}
    if wafer:
        det_select['wafer_slot'] = wafer
    if band:
        det_select['wafer.bandpass'] = band
    return det_select


def resolve_obsids(obsid: list[str] | None, obsids_file: Path | None) -> list[str]:
    """The observation ids given on the command line, or else in the file, one per line.

    An id given more than once is kept once, at its first position: an observation listed twice
    would otherwise enter the map twice.
    """
    if obsid:
        obsids = obsid
    elif obsids_file is not None:
        obsids = [line.strip() for line in obsids_file.read_text().splitlines() if line.strip()]
    else:
        obsids = []
    return list(dict.fromkeys(obsids))


def preproc_observations(
    obsids: list[str],
    init_config: Path,
    proc_config: Path | None,
    wafer: str,
    band: str,
    sotodlib_config: SotodlibConfig | None,
    cache_dir: Path | None,
) -> list[LazyPreprocSOTODLibObservation]:
    """The observations of the preprocessing database, as `furax-so-map` reads them.

    Args:
        obsids: Observation ids.
        init_config: Base-layer preprocessing config file.
        proc_config: Optional second-layer preprocessing config file.
        wafer: Wafer slot selection, or 'all' to join every wafer of an observation.
        band: Wafer bandpass selection.
        sotodlib_config: The sotodlib section of the mapmaking config.
        cache_dir: Directory of cached observations, if any.
    """
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.interfaces.sotodlib import LazyPreprocSOTODLibObservation

    # Normalise the configs (cwd-sensitive paths -> absolute) instead of chdir'ing the whole
    # process: sotodlib then loads them from anywhere, and init/proc may live under different
    # roots. The temp copies must outlive every lazy read, so clean them up only at exit.
    stage_dir = Path(tempfile.mkdtemp(prefix='furax-so-'))
    atexit.register(shutil.rmtree, stage_dir, ignore_errors=True)
    init_config = pp.normalize_config(init_config, stage_dir)
    proc_config = pp.normalize_config(proc_config, stage_dir) if proc_config else None
    join_groups = wafer == 'all'
    det_select = detector_selection(None if join_groups else wafer, band)
    return [
        LazyPreprocSOTODLibObservation(
            obs_id,
            init_config,
            proc_config,
            det_select,
            sotodlib_config=sotodlib_config,
            join_groups=join_groups,
            cache_dir=cache_dir,
        )
        for obs_id in obsids
    ]


def obsdir_observations(
    obsdir: Path, obsids: list[str], sotodlib_config: SotodlibConfig | None
) -> list[LazySOTODLibObservation]:
    """The observations saved as `<obsid>.h5` files in `obsdir`, all of them without `obsids`.

    Args:
        obsdir: Directory of observation files.
        obsids: Observation ids, whose missing files are skipped with a warning.
        sotodlib_config: The sotodlib section of the mapmaking config.
    """
    # Deferred: importing furax pulls in JAX, which the CLI parse path does not need
    from furax.interfaces.sotodlib import LazySOTODLibObservation

    if obsids:
        obsfiles = []
        for obs_id in obsids:
            obsfile = obsdir / f'{obs_id}.h5'
            if obsfile.exists():
                obsfiles.append(obsfile)
            else:
                logging.getLogger('furax.so_mapmaking').warning(f'{obsfile} not found, skipping')
    else:
        obsfiles = sorted(obsdir.glob('*.h5'))
    return [LazySOTODLibObservation(f, sotodlib_config=sotodlib_config) for f in obsfiles]
