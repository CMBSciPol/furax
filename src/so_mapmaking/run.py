from pathlib import Path

import yaml
from cyclopts import App
from typedload.exceptions import TypedloadException

from ._outdir import (
    LOG_FILE,
    RunState,
    claim,
    earlier_run,
    remove_run_files,
    write_status,
)
from .util import (
    enable_x64,
    obsdir_observations,
    preproc_observations,
    resolve_obsids,
    setup_logger,
)

app = App(help='Run the mapmaker on SO observations, loaded straight from the preproc db.')


@app.default
def run(
    init_config: Path | None = None,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path | None = None,
    mapmaking_config: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    obsdir: Path | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    loglevel: str = 'info',
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
        wafer: Wafer slot selection, or 'all' to map every wafer of an observation as one
            observation, so that the whole focal plane is fitted at once (preproc-db mode).
        band: Wafer bandpass selection.
        obsdir: Legacy mode: directory of prepared .h5 files. Mutually exclusive with
            ``init_config``.
        cache_dir: Directory of preprocessed observations (preproc-db mode). The first run saves
            each observation there once loaded, and later runs with the same preprocessing load
            it back instead of preprocessing again. With `--wafer all`, an observation is saved
            with its wafers joined, and serves only runs with `--wafer all`. Delete the cache
            after rerunning the preprocessing.
        overwrite: Map into an `outdir` that holds the output of an earlier run, after deleting
            the files that run wrote. Other files, such as the output of a batch scheduler, are
            kept.
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path, one file per process. Defaults to `furax-so-map.log` in `outdir`.
    """
    # Before importing furax, so that jax_healpy does not warn; the config sets the precision below
    enable_x64()

    # Deferred, so that `--help` stays fast
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mhu

    from furax.distributed import maybe_init
    from furax.mapmaking import MapMakingConfig, MultiObservationMapMaker
    from furax.mapmaking._provenance import utc_now

    maybe_init()  # before the first use of the JAX backend, jax.process_index() below
    rank = jax.process_index()
    outdir = (outdir or Path.cwd()).resolve()
    # Log to the console only until the output directory is known to be free, so that a run
    # refused, or stopped by an invalid input, leaves the directory as it found it
    logger = setup_logger(loglevel, None, process_index=rank)

    if outdir.exists() and not outdir.is_dir():
        logger.error(f'{outdir} is not a directory')
        return 1
    if (init_config is None) == (obsdir is None):
        logger.error('specify exactly one of --init-config (preproc db) or --obsdir (legacy)')
        return 1
    if cache_dir is not None and init_config is None:
        logger.error('--cache-dir requires --init-config')
        return 1
    try:
        config = MapMakingConfig.load_yaml(mapmaking_config) if mapmaking_config else None
    except (OSError, yaml.YAMLError, TypedloadException) as e:
        logger.error(f'invalid mapmaking config {mapmaking_config}: {e}')
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
                obsids, init_config, proc_config, wafer, band, config.sotodlib, cache_dir
            )
        else:
            assert obsdir is not None
            observations = obsdir_observations(obsdir, obsids, config.sotodlib)
    except Exception:
        logger.exception('loading the observations failed')
        observations, failed = [], True
    counts = mhu.process_allgather(np.array([len(observations), failed])).reshape(-1, 2)
    if counts[:, 1].any():
        logger.error('loading the observations failed')
        return 1
    if not counts[:, 0].any():
        logger.warning('no observations to map')
        return

    # Process 0 clears or claims the directory; the others wait for its verdict before they open
    # their logs there
    started = utc_now()
    refused = False
    if rank == 0:
        try:
            if overwrite:
                if outdir.is_dir():
                    remove_run_files(outdir)
                earlier = None
            else:
                earlier = earlier_run(outdir)
            # Claiming is exclusive: a run started at the same time on the directory loses it
            if earlier is None and not claim(outdir, started=started):
                earlier = 'a run that started at the same time'
            if earlier is not None:
                logger.error(
                    f'{outdir} holds the output of {earlier}; pass --overwrite to replace it'
                )
                refused = True
        except OSError as e:
            # The other processes are waiting for the verdict below
            logger.error(f'cannot prepare {outdir}: {e}')
            refused = True
    if mhu.broadcast_one_to_all(np.int32(refused)):
        return 1
    logger = setup_logger(loglevel, log_path or outdir / LOG_FILE, process_index=rank)
    if mapmaking_config is None:
        logger.warning('no mapmaking configuration specified, using defaults')
    logger.info(f'found {len(observations)} observations')

    try:
        maker = MultiObservationMapMaker(observations, config=config, logger=logger)
        logger.info('loaded config and set up mapmaker')
        maker.run(outdir)
    except Exception as e:
        logger.exception('mapmaking failed', exc_info=e)
        # Any process may be the one that fails, and the others may then never return
        write_status(
            outdir, RunState.FAILED, started=started, finished=utc_now(), error=repr(e), rank=rank
        )
        return 1
    if rank == 0:
        write_status(outdir, RunState.DONE, started=started, finished=utc_now())
    logger.info('finished mapmaking')


if __name__ == '__main__':
    app()
