import logging
from pathlib import Path


def setup_logger(
    loglevel: str | None, log_path: Path | None, process_index: int = 0
) -> logging.Logger:
    """Layer CLI options onto the shared 'furax-mapmaking' logger.

    The base console handler, formatter and per-rank stamping are owned by
    ``furax.mapmaking._logger`` (configured when the mapmaker is imported). This only adds the
    CLI-specific bits: an explicit level override when given on the command line, and a per-rank
    file handler when ``log_path`` is set. The logger is fetched by name so this stays jax-free
    for the ``prepare`` CLI; that path never imports the furax mapmaker, so a matching console
    handler is added as a fallback when the base config is absent.
    """
    logger = logging.getLogger('furax-mapmaking')
    if loglevel is not None:
        logger.setLevel(logging.getLevelName(loglevel.upper()))

    formatter = logging.Formatter(
        f'%(asctime)s - [rank {process_index}] - %(levelname)s - %(message)s'
    )

    if not logger.handlers:
        # prepare path: furax._logger never ran, so stand up a console handler here.
        logger.propagate = False
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_path is not None:
        log_path = log_path.with_stem(f'{log_path.stem}.rank{process_index}')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


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
    if obsid:
        return obsid
    if obsids_file is not None:
        return [line.strip() for line in obsids_file.read_text().splitlines() if line.strip()]
    return []


def standard_obsdir(outdir: Path, wafer: str, band: str, downsample: int = 1) -> Path:
    path = outdir / 'obs' / wafer / band
    if downsample > 1:
        path = path / f'ds{downsample}'
    return path
