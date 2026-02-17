from pathlib import Path
from typing import Annotated

import typer

from furax.interfaces.sotodlib import LazySOTODLibObservation
from furax.mapmaking import MapMakingConfig, MultiObservationMapMaker

from .util import resolve_obsids, setup_logger

app = typer.Typer()


@app.command()  # type: ignore[misc]
def run(
    obsdir: Annotated[Path, typer.Option(help='Directory containing prepared .h5 files.')],
    obsid: Annotated[
        list[str] | None,
        typer.Option(
            help='Observation id(s) to map. If not specified, use all .h5 files in obsdir.'
        ),
    ] = None,
    obsids_file: Annotated[
        Path | None, typer.Option(help='Text file with one obsid per line.')
    ] = None,
    outdir: Annotated[Path, typer.Option(help='Output directory for maps.')] = Path.cwd(),
    mapmaking_config: Annotated[Path | None, typer.Option(help='Mapmaking config file.')] = None,
    verbose: Annotated[
        int, typer.Option('--verbose', '-v', count=True, help='Increase verbosity.')
    ] = 1,
    log_path: Annotated[Path | None, typer.Option(help='Log output path.')] = None,
) -> None:
    """Run the mapmaker on prepared binary observation files."""

    logger = setup_logger(verbose, log_path)

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
        raise typer.Abort()

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
        raise typer.Exit(code=1)
