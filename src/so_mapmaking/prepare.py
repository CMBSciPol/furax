import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, current_process
from pathlib import Path

import yaml
from cyclopts import App

from .util import detector_selection, resolve_obsids, setup_logger, standard_obsdir

app = App(help='Preprocess SO observations and dump them to binary files.')


@dataclass(frozen=True)
class _Context:
    """Constant per-run state shared by all workers."""

    obsdir: Path
    init_config: str  # resolved posix path
    proc_config: str | None  # resolved posix path
    det_select: dict[str, str] | None
    downsample: int
    overwrite: bool
    loglevel: str
    log_path: Path | None


def _worker_index() -> int:
    # Pool names workers "ForkPoolWorker-N"; map to 0-based index for logging
    ident = current_process()._identity
    return ident[0] - 1 if ident else 0


def _process_obs(obs_id: str, ctx: _Context) -> None:
    # sotodlib is slow to import; defer it out of the CLI/parse path
    from sotodlib.core.axisman import AxisManager
    from sotodlib.mapmaking.utils import downsample_obs
    from sotodlib.preprocess.preprocess_util import preproc_or_load_group

    logger = setup_logger(ctx.loglevel, ctx.log_path, process_index=_worker_index())

    obsfile = ctx.obsdir / f'{obs_id}.h5'
    if obsfile.exists() and not ctx.overwrite:
        logger.info(f'{obsfile.name} already exists, skipping')
        return

    obs, *_ = preproc_or_load_group(
        obs_id,
        ctx.init_config,
        ctx.det_select,
        configs_proc=ctx.proc_config,
        save_archive=False,
        save_proc_aman=False,
    )
    if not isinstance(obs, AxisManager):
        logger.error(f'preproc_or_load_group failed for {obs_id}')
        return
    logger.info(f'successfully processed {obs_id}')

    if ctx.downsample > 1:
        obs = downsample_obs(obs, ctx.downsample, logger=logger)
        logger.info(f'downsampled {obs_id} by factor {ctx.downsample}')

    obs.save(obsfile.resolve().as_posix(), overwrite=ctx.overwrite)
    logger.info(f'saved observation to {obsfile.name}')


@app.default
def prepare(  # type: ignore[no-untyped-def]
    init_config: Path,
    proc_config: Path | None = None,
    obsid: list[str] | None = None,
    obsids_file: Path | None = None,
    outdir: Path | None = None,
    wafer: str = 'ws0',
    band: str = 'f090',
    downsample: int = 1,
    nproc: int = 1,
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
        nproc: Number of worker processes (single node).
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
        return

    ctx = _Context(
        obsdir=obsdir,
        init_config=init_config.resolve().as_posix(),
        proc_config=proc_config.resolve().as_posix() if proc_config else None,
        det_select=det_select,
        downsample=downsample,
        overwrite=overwrite,
        loglevel=loglevel,
        log_path=log_path,
    )
    work = partial(_process_obs, ctx=ctx)

    nproc = max(1, min(nproc, len(obsids)))
    if nproc == 1:
        for obs_id in obsids:
            work(obs_id)
    else:
        logger.info(f'processing {len(obsids)} observations with {nproc} workers')
        with Pool(nproc) as pool:
            pool.map(work, obsids)


if __name__ == '__main__':
    app()
