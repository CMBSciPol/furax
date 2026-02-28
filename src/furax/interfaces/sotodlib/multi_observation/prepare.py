import os
from pathlib import Path

import yaml
from sotodlib.core.axisman import AxisManager
from sotodlib.mapmaking.utils import downsample_obs
from sotodlib.preprocess.preprocess_util import preproc_or_load_group

from .util import detector_selection, resolve_obsids, setup_logger, standard_obsdir


def prepare(  # type: ignore[no-untyped-def]
    init_config: Path,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    downsample: int = 1,
    loglevel: str = 'info',
    log_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
):
    """Load observations via preprocessing config and save to binary files.

    Args:
        init_config: Base layer preprocessing config file.
        proc_config: Second layer preprocessing config file.
        obsid: Observation id(s) to process.
        obsids_file: Text file with one obsid per line.
        outdir: Output directory. Defaults to preproc archive index parent.
        wafer: Wafer slot selection.
        band: Wafer bandpass selection.
        downsample: Downsampling factor.
        loglevel: Logging level (debug, info, warning, error).
        log_path: Log output path.
        overwrite: Overwrite existing files.
        dry_run: Stop before any actual processing.
    """
    logger = setup_logger(loglevel, log_path)

    obsids = resolve_obsids(obsid, obsids_file)
    if len(obsids) == 0:
        logger.warning('no observations to prepare')
        return

    det_select = detector_selection(wafer, band)

    # cd to where the preproc config is for relative paths to work
    # config is typically in `vx/preprocessing/satpy/...`, need to be in `vx` directory
    layers = [init_config] + ([proc_config] if proc_config else [])
    roots = {layer.parent.resolve() for layer in layers}
    if len(roots) > 1:
        logger.error('all preproc configs must share the same root directory')
        return 1
    os.chdir(init_config.parents[3])

    if outdir is None:
        # use last configuration layer to determine output directory
        config_last = yaml.safe_load(layers[-1].read_text())
        outdir = Path(config_last['archive']['index']).parent
        logger.info(f'using default outdir: {outdir}')

    obsdir = standard_obsdir(outdir, wafer, band, downsample)
    obsdir.mkdir(parents=True, exist_ok=True)
    logger.info(f'writing observations to /.../{obsdir.relative_to(outdir.parent)}')

    if dry_run:
        return

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
