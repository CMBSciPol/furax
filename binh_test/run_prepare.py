from pathlib import Path
import yaml
import os

from so_mapmaking.prepare import prepare


WAFERS = [
    "ws0",
    "ws1",
    "ws2",
    "ws3",
    "ws4",
    "ws5",
    "ws6",
]

BANDS = [
    "f090",
    "f150",
]


with open("binh_test/prepare_config.yaml") as f:
    cfg = yaml.safe_load(f)


# Slurm array index
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])


# map task -> wafer/band
wafer = WAFERS[task_id // len(BANDS)]
band = BANDS[task_id % len(BANDS)]


print(
    f"Running task {task_id}: "
    f"wafer={wafer}, band={band}"
)


prepare(
    init_config=Path(cfg["init_config"]),

    obsids_file=Path(cfg["obsids_file"]),

    wafer=wafer,
    band=band,

    outdir=Path(cfg["outdir"]),

    downsample=cfg.get("downsample", 1),

    nproc=cfg.get("nproc", 1),

    overwrite=cfg.get("overwrite", False),

    dry_run=cfg.get("dry_run", False),

    loglevel=cfg.get("loglevel", "info"),

    log_path=Path(
        f"logs/preprocess_{wafer}_{band}.log"
    ),
)