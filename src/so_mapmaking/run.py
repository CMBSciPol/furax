import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter
from typedload.exceptions import TypedloadException

from ._outdir import (
    LOG_FILE,
    STATUS_FILE,
    DirectoryInUse,
    RunState,
    claim,
    earlier_run,
    has_checkpoint,
    remove_run_files,
    run_lock,
    write_status,
)
from .util import (
    Band,
    Wafer,
    enable_x64,
    obsdir_observations,
    preproc_observations,
    resolve_obsids,
    setup_logger,
)

logger = logging.getLogger('furax.so_mapmaking')
app = App(help='Run the mapmaker on SO observations, loaded straight from the preproc db.')


@app.default
def run(
    init_config: Path | None = None,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path | None = None,
    mapmaking_config: Path | None = None,
    wafer: Annotated[
        tuple[Wafer, ...] | None, Parameter(consume_multiple=True, negative_iterable='')
    ] = None,
    join_wafers: bool = True,
    band: Band = 'f090',
    obsdir: Path | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    resume: bool = False,
    loglevel: Annotated[str, Parameter(env_var='LOGLEVEL')] = 'info',
    log_path: Path | None = None,
):
    """Run the mapmaker, loading observations directly from the preprocessing database.

    Observations are loaded from the preproc archive at mapmaking time, or from ``cache_dir``
    where an earlier run saved them. Pass ``obsdir`` instead to map pre-dumped ``.h5`` files
    (legacy path).

    The run records its state in `status.json` in `outdir`: `running` from the start, then `done`
    once every product is written, or `failed`. The mapmaker writes the mapmaking config and
    the provenance at the start too, so that a run that does not finish leaves them behind.

    Args:
        init_config: Base-layer preprocessing config file (preproc-db mode).
        proc_config: Optional second-layer preprocessing config file.
        obsid: Observation id(s) to map.
        obsids_file: Text file with one obsid per line.
        outdir: Output directory for maps.
        mapmaking_config: Mapmaking config file.
        wafer: Wafer slots to map, every wafer of an observation by default (preproc-db mode). A
            wafer an observation does not have is skipped.
        join_wafers: Join the selected wafers of an observation into one observation, so that the
            focal plane is fitted at once. With `--no-join-wafers`, each wafer is an observation
            of its own, with its own noise and template fit.
        band: Wafer bandpass to map.
        obsdir: Legacy mode: directory of prepared .h5 files. Mutually exclusive with
            ``init_config``.
        cache_dir: Directory of preprocessed observations (preproc-db mode). The first run saves
            each observation there once loaded, and later runs with the same preprocessing load
            it back instead of preprocessing again. Each wafer of an observation is saved in a
            file of its own, shared by the runs that select or join the wafers differently. Delete
            the cache after rerunning the preprocessing.
        overwrite: Map into an `outdir` that holds the output of an earlier run, after deleting
            the files that run wrote. Other files, such as the output of a batch scheduler, are
            kept.
        resume: Map into an `outdir` that holds the checkpoint of an earlier run, keeping its
            files, and resume its solve from there. The checkpoints are saved every
            `solver.checkpoint_every` steps of the mapmaking config; a solve that runs out of
            steps saves its last iterate, to be resumed with a larger `solver.max_steps`, which
            bounds the steps of all the runs.
        loglevel: Logging level (debug, info, warning, error). Defaults to `$LOGLEVEL`, else info.
        log_path: Log output path, one file per process. Defaults to `furax-so-map.log` in `outdir`.
    """
    # Before importing furax, so that jax_healpy does not warn; the config sets the precision below
    enable_x64()

    # Deferred, so that `--help` stays fast
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mhu

    from furax.distributed import maybe_init
    from furax.mapmaking import MapMakingConfig, MapMakingResults, MultiObservationMapMaker
    from furax.mapmaking._provenance import utc_now

    run_start = time.perf_counter()
    maybe_init()  # before the first use of the JAX backend, jax.process_index() below
    rank = jax.process_index()
    outdir = (outdir or Path.cwd()).resolve()
    # Log to the console only until the output directory is known to be free, so that a run
    # refused, or stopped by an invalid input, leaves the directory as it found it
    setup_logger(loglevel, None, process_index=rank)

    if outdir.exists() and not outdir.is_dir():
        logger.error(f'Output directory {outdir} is not a directory')
        return 1
    if (init_config is None) == (obsdir is None):
        logger.error('Specify exactly one of --init-config (preproc db) or --obsdir (legacy)')
        return 1
    if cache_dir is not None and init_config is None:
        logger.error('Option --cache-dir requires --init-config')
        return 1
    if overwrite and resume:
        logger.error('Pass either --overwrite or --resume')
        return 1
    try:
        config = MapMakingConfig.load_yaml(mapmaking_config) if mapmaking_config else None
    except (OSError, yaml.YAMLError, TypedloadException) as e:
        logger.error(f'Invalid mapmaking config {mapmaking_config}: {e}')
        return 1
    config = config or MapMakingConfig()
    jax.config.update('jax_enable_x64', config.double_precision)

    # A process that fails to load its observations must not leave the others waiting at the
    # collectives below: every process learns of the failure, and of the observation counts
    failed = False
    try:
        obsids = resolve_obsids(obsid, obsids_file)
        if init_config is not None:
            observations = preproc_observations(
                obsids,
                init_config,
                proc_config,
                wafers=wafer,
                band=band,
                join_wafers=join_wafers,
                sotodlib_config=config.sotodlib,
                cache_dir=cache_dir,
            )
        else:
            assert obsdir is not None
            observations = obsdir_observations(obsdir, obsids, config.sotodlib)
    except Exception:
        logger.exception('Loading the observations failed')
        observations, failed = [], True
    counts = mhu.process_allgather(np.array([len(observations), failed])).reshape(-1, 2)
    if counts[:, 1].any():
        if not failed:  # the process that failed has logged why
            logger.error('Loading the observations failed on another process')
        return 1
    if not counts[:, 0].any():
        logger.warning('No observations to map')
        return

    with ExitStack() as stack:
        # Process 0 clears or claims the directory; the others wait for its verdict before they
        # open their logs there
        started = utc_now()
        refused = False
        if rank == 0:
            try:
                # Held until the run returns. A live run on the directory keeps its lock, so
                # neither --overwrite nor --resume can take the directory from it.
                if not stack.enter_context(run_lock(outdir)):
                    raise DirectoryInUse
                if overwrite:
                    if outdir.is_dir():
                        remove_run_files(outdir)
                    earlier = None
                elif resume:
                    # The earlier run's status gives way to the claim of this one
                    if has_checkpoint(outdir):
                        # The products of the earlier run are stale once this one starts
                        MapMakingResults.remove(outdir)
                        (outdir / STATUS_FILE).unlink(missing_ok=True)
                        earlier = None
                    else:
                        raise FileNotFoundError('it holds no checkpoint to resume from')
                else:
                    earlier = earlier_run(outdir)
                # Claiming is exclusive: a run started at the same time on the directory loses it
                if earlier is None and not claim(outdir, started=started):
                    earlier = 'a run that started at the same time'
                if earlier is not None:
                    logger.error(
                        f'Output directory {outdir} holds {earlier}: pass --overwrite or --resume'
                    )
                    refused = True
            except DirectoryInUse:
                logger.error(f'Output directory {outdir} is in use by a run that is still going')
                refused = True
            except OSError as e:
                # The other processes are waiting for the verdict below
                logger.error(f'Cannot prepare {outdir}: {e}')
                refused = True
        if mhu.broadcast_one_to_all(np.int32(refused)):
            return 1
        setup_logger(loglevel, log_path or outdir / LOG_FILE, process_index=rank)
        if mapmaking_config is None:
            logger.warning('No mapmaking configuration specified, using defaults')
        logger.info(f'Found {len(observations)} observations')

        try:
            maker = MultiObservationMapMaker(observations, config=config)
            logger.info('Loaded config and set up mapmaker')
            maker.run(outdir, resume=resume)
        except Exception as e:
            logger.exception('Mapmaking failed')
            # Any process may be the one that fails, and the others may then never return
            write_status(
                outdir,
                RunState.FAILED,
                started=started,
                finished=utc_now(),
                error=repr(e),
                rank=rank,
            )
            return 1
        if rank == 0:
            write_status(outdir, RunState.DONE, started=started, finished=utc_now())
        logger.info(f'Finished mapmaking in {time.perf_counter() - run_start:.0f} s')


if __name__ == '__main__':
    app()
