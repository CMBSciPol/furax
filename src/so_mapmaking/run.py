from pathlib import Path

from cyclopts import App

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
    loglevel: str = 'info',
    log_path: Path | None = None,
):
    """Run the mapmaker, loading observations directly from the preprocessing database.

    Observations are loaded from the preproc archive at mapmaking time, or from ``cache_dir``
    where an earlier run saved them. Pass ``obsdir`` instead to map pre-dumped ``.h5`` files
    (legacy path).

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
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path.
    """
    # Before importing furax, so that jax_healpy does not warn; the config sets the precision below
    enable_x64()

    # Deferred, so that `--help` stays fast
    import jax

    from furax.distributed import maybe_init
    from furax.mapmaking import MapMakingConfig, MultiObservationMapMaker

    maybe_init()  # before the first use of the JAX backend, jax.process_index() below
    logger = setup_logger(loglevel, log_path, process_index=jax.process_index())

    if (init_config is None) == (obsdir is None):
        logger.error('specify exactly one of --init-config (preproc db) or --obsdir (legacy)')
        return 1
    if cache_dir is not None and init_config is None:
        logger.error('--cache-dir requires --init-config')
        return 1

    if mapmaking_config is None:
        config = MapMakingConfig()
        logger.warning('no mapmaking configuration specified, using defaults')
    else:
        config = MapMakingConfig.load_yaml(mapmaking_config)

    jax.config.update('jax_enable_x64', config.double_precision)

    obsids = resolve_obsids(obsid, obsids_file)
    if init_config is not None:
        observations = preproc_observations(
            obsids, init_config, proc_config, wafer, band, config.sotodlib, cache_dir
        )
    else:
        assert obsdir is not None
        observations = obsdir_observations(obsdir, obsids, config.sotodlib)
    if not observations:
        logger.warning('no observations to map')
        return
    logger.info(f'found {len(observations)} observations')

    outdir = (outdir or Path.cwd()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    maker = MultiObservationMapMaker(observations, config=config, logger=logger)
    logger.info('loaded config and set up mapmaker')

    try:
        maker.run(outdir)
        logger.info('finished mapmaking')
    except Exception as e:
        logger.exception('mapmaking failed', exc_info=e)
        return 1


if __name__ == '__main__':
    app()
