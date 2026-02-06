import logging
from pathlib import Path


def setup_logger(verbose: int, log_path: Path | None) -> logging.Logger:
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, len(levels) - 1)]
    logger = logging.getLogger('furax.mapmaker')
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
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
