import os
from pathlib import Path
from typing import Any

from cyclopts import App

from .util import detector_selection, resolve_obsids, setup_logger

app = App(help='Run the mapmaker on SO observations, loaded straight from the preproc db.')


def _chdir_to_config_root(layers: list[Path]) -> list[Path]:
    """Resolve config paths against the launch dir, then chdir to the preproc config root.

    Configs typically live in ``<root>/preprocessing/satpy/...``; the archive index they
    reference is relative to ``<root>``. Returns the resolved (absolute) config paths so
    callers can keep using them after the chdir. Mirrors ``furax-so-prepare``.
    """
    resolved = [layer.resolve() for layer in layers]
    roots = {layer.parents[2] for layer in resolved}
    if len(roots) > 1:
        raise ValueError('all preproc configs must share the same root directory')
    os.chdir(roots.pop())
    return resolved


@app.default  # type: ignore[untyped-decorator]
def run(  # type: ignore[no-untyped-def]
    init_config: Path | None = None,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path | None = None,
    mapmaking_config: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    downsample: int = 1,
    obsdir: Path | None = None,
    loglevel: str = 'info',
    log_path: Path | None = None,
):
    """Run the mapmaker, loading observations directly from the preprocessing database.

    Observations are streamed from the preproc archive at mapmaking time (no intermediate
    binary files). Pass ``obsdir`` instead to map pre-dumped ``.h5`` files (legacy path).

    Args:
        init_config: Base-layer preprocessing config file (preproc-db mode).
        proc_config: Optional second-layer preprocessing config file.
        obsid: Observation id(s) to map.
        obsids_file: Text file with one obsid per line.
        outdir: Output directory for maps.
        mapmaking_config: Mapmaking config file.
        wafer: Wafer slot selection.
        band: Wafer bandpass selection.
        downsample: Downsampling factor applied after preprocessing.
        obsdir: Legacy mode: directory of prepared .h5 files. Mutually exclusive with
            ``init_config``.
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path.
    """
    # Defer JAX (and the mapmaker stack it pulls in) to call time so that
    # merely importing this module stays cheap and backend-free.
    import jax

    from furax.distributed import maybe_init

    maybe_init()  # must run before the JAX backend is touched

    from furax.interfaces.sotodlib import LazyPreprocSOTODLibObservation, LazySOTODLibObservation
    from furax.mapmaking import (
        AbstractLazyObservation,
        MapMakingConfig,
        MultiObservationMapMaker,
    )

    logger = setup_logger(loglevel, log_path, process_index=jax.process_index())

    if (init_config is None) == (obsdir is None):
        logger.error('specify exactly one of --init-config (preproc db) or --obsdir (legacy)')
        return 1

    if mapmaking_config is None:
        config = MapMakingConfig()
        logger.warning('no mapmaking configuration specified, using defaults')
    else:
        config = MapMakingConfig.load_yaml(mapmaking_config)

    if config.double_precision:
        jax.config.update('jax_enable_x64', True)

    obsids = resolve_obsids(obsid, obsids_file)
    observations: list[AbstractLazyObservation[Any]]

    # Resolve outdir against the launch directory now: the init-config branch chdir's to the
    # preproc config root below, so a relative outdir would otherwise be written there instead.
    outdir = (outdir or Path.cwd()).resolve()

    if init_config is not None:
        layers = [init_config] + ([proc_config] if proc_config else [])
        try:
            resolved = _chdir_to_config_root(layers)
        except ValueError as e:
            logger.error(str(e))
            return 1
        init_config = resolved[0]
        proc_config = resolved[1] if proc_config else None
        if not obsids:
            logger.warning('no observations to map')
            return
        det_select = detector_selection(wafer, band)
        observations = [
            LazyPreprocSOTODLibObservation(
                obs_id,
                init_config,
                proc_config,
                det_select,
                downsample,
                sotodlib_config=config.sotodlib,
            )
            for obs_id in obsids
        ]
        mapped = obsids
    else:
        assert obsdir is not None
        if obsids:
            obsfiles = []
            for obs_id in obsids:
                obsfile = obsdir / f'{obs_id}.h5'
                if not obsfile.exists():
                    logger.warning(f'{obsfile} not found, skipping')
                    continue
                obsfiles.append(obsfile)
        else:
            obsfiles = sorted(obsdir.glob('*.h5'))
        if len(obsfiles) == 0:
            logger.warning('no observations to map')
            return
        observations = [
            LazySOTODLibObservation(f, sotodlib_config=config.sotodlib) for f in obsfiles
        ]
        mapped = [f.stem for f in obsfiles]

    logger.info(f'found {len(observations)} observations')

    outdir.mkdir(parents=True, exist_ok=True)

    maker = MultiObservationMapMaker(observations, config=config, logger=logger)
    logger.info('loaded config and set up mapmaker')

    try:
        results = maker.run(outdir)
        logger.info('finished mapmaking')
    except Exception as e:
        logger.exception('mapmaking failed', exc_info=e)
        return 1

    # Record the observations that actually made it into the maps (load/preproc failures excluded).
    if jax.process_index() == 0:
        failed = set(results.failed_observations or ())
        with open(outdir / 'mapped_obsids.txt', 'w') as f:
            for name in mapped:
                if name not in failed:
                    f.write(f'{name}\n')


if __name__ == '__main__':
    app()
