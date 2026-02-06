import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from sotodlib.core.axisman import AxisManager
from sotodlib.mapmaking.utils import downsample_obs
from sotodlib.preprocess.preprocess_util import preproc_or_load_group

from .util import detector_selection, resolve_obsids, setup_logger, standard_obsdir

app = typer.Typer()


@app.command()  # type: ignore[misc]
def prepare(
    init_config: Annotated[Path, typer.Option(help='Base layer preprocessing config file.')],
    proc_config: Annotated[
        Path | None, typer.Option(help='Second layer preprocessing config file.')
    ] = None,
    obsid: Annotated[list[str] | None, typer.Option(help='Observation id(s) to process.')] = None,
    obsids_file: Annotated[
        Path | None, typer.Option(help='Text file with one obsid per line.')
    ] = None,
    outdir: Annotated[
        Path | None,
        typer.Option(help='Output directory. Defaults to preproc archive index parent.'),
    ] = None,
    wafer: Annotated[str, typer.Option(help='Wafer slot selection.')] = 'ws0',
    band: Annotated[str, typer.Option(help='Wafer bandpass selection.')] = 'f090',
    downsample: Annotated[int, typer.Option(help='Downsampling factor.')] = 1,
    verbose: Annotated[
        int, typer.Option('--verbose', '-v', count=True, help='Increase verbosity.')
    ] = 1,
    log_path: Annotated[Path | None, typer.Option(help='Log output path.')] = None,
    overwrite: Annotated[bool, typer.Option(help='Overwrite existing files.')] = False,
    dry_run: Annotated[bool, typer.Option(help='Stop before any actual processing.')] = False,
) -> None:
    """Load observations via preprocessing config and save to binary files."""
    logger = setup_logger(verbose, log_path)

    obsids = resolve_obsids(obsid, obsids_file)
    if len(obsids) == 0:
        logger.warning('no observations to prepare')
        raise typer.Abort()

    det_select = detector_selection(wafer, band)

    # cd to where the preproc config is for relative paths to work
    # config is typically in `vx/preprocessing/satpy/...`, need to be in `vx` directory
    layers = [init_config] + ([proc_config] if proc_config else [])
    roots = {layer.parent.resolve() for layer in layers}
    if len(roots) > 1:
        logger.error('all preproc configs must share the same root directory')
        raise typer.Exit(code=1)
    os.chdir(init_config.parents[2])

    if outdir is None:
        # use last configuration layer to determine output directory
        config_last = yaml.safe_load(layers[-1].read_text())
        outdir = Path(config_last['archive']['index']).parent
        logger.info(f'using default outdir: {outdir}')

    obsdir = standard_obsdir(outdir, wafer, band, downsample)
    obsdir.mkdir(parents=True, exist_ok=True)
    logger.info(f'writing observations to /.../{obsdir.relative_to(outdir.parent)}')

    if dry_run:
        raise typer.Exit()

    for obs_id in obsids:
        filename = f'{obs_id}.h5'
        obsfile = obsdir / filename
        if obsfile.exists() and not overwrite:
            logger.info(f'{filename} already exists, skipping')
            continue

        obs, *_ = preproc_or_load_group(
            obs_id,
            init_config.resolve().as_posix(),
            det_select,
            configs_proc=proc_config.resolve().as_posix() if proc_config else None,
            save_archive=False,
            save_proc_aman=False,
        )
        if not isinstance(obs, AxisManager):
            logger.error(f'preproc_or_load_group failed for {obs_id}')
            continue
        logger.info(f'successfully processed {obs_id}')

        if downsample > 1:
            obs = downsample_obs(obs, downsample, logger=logger)
            logger.info(f'downsampled {obs_id} by factor {downsample}')

        obs.save(obsfile.resolve().as_posix(), overwrite=overwrite)
        logger.info(f'saved observation to {filename}')
