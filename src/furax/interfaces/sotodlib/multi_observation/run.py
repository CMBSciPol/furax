from pathlib import Path

from furax.interfaces.sotodlib import LazySOTODLibObservation
from furax.mapmaking import MapMakingConfig, MultiObservationMapMaker

from .util import resolve_obsids, setup_logger


def run(  # type: ignore[no-untyped-def]
    obsdir: Path,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path = Path.cwd(),
    mapmaking_config: Path | None = None,
    loglevel: str = 'info',
    log_path: Path | None = None,
):
    """Run the mapmaker on prepared binary observation files.

    Args:
        obsdir: Directory containing prepared .h5 files.
        obsid: Observation id(s) to map. If not specified, use all .h5 files in obsdir.
        obsids_file: Text file with one obsid per line.
        outdir: Output directory for maps.
        mapmaking_config: Mapmaking config file.
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path.
    """
    logger = setup_logger(loglevel, log_path)

    obsids = resolve_obsids(obsid, obsids_file)
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

    observations = [LazySOTODLibObservation(f) for f in obsfiles]

    logger.info(f'found {len(observations)} observations')

    if mapmaking_config is None:
        config = MapMakingConfig()
        logger.warning('no mapmaking configuration specified, using defaults')
    else:
        config = MapMakingConfig.load_yaml(mapmaking_config)

    maker = MultiObservationMapMaker(observations, config=config, logger=logger)
    logger.info('loaded config and set up mapmaker')

    outdir.mkdir(parents=True, exist_ok=True)

    try:
        maker.run(outdir)
        logger.info('finished mapmaking')
    except Exception as e:
        logger.exception('mapmaking failed', exc_info=e)
        return 1
