import os
from pathlib import Path
import yaml

from so_mapmaking.prepare import prepare


with open("binh_test/prepare_config.yaml") as f:
    cfg = yaml.safe_load(f)


task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
n_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])


if task_id >= n_tasks:
    raise RuntimeError(
        f"Invalid task_id={task_id}, n_tasks={n_tasks}"
    )


with open(cfg["task_file"]) as f:
    tasks = [
        line.strip()
        for line in f
        if line.strip()
    ]


# distribute tasks among Slurm array jobs
my_tasks = tasks[task_id::n_tasks]


print(
    f"Array task {task_id}/{n_tasks}: "
    f"processing {len(my_tasks)} observations"
)


for task in my_tasks:

    wafer, band, obs_id = task.split()

    print(
        f"Running wafer={wafer}, "
        f"band={band}, "
        f"obsid={obs_id}"
    )

    prepare(
        init_config=Path(cfg["init_config"]),

        obsid=[obs_id],

        wafer=wafer,
        band=band,

        outdir=Path(cfg["outdir"]),

        downsample=cfg.get("downsample", 1),

        overwrite=cfg.get("overwrite", False),

        dry_run=cfg.get("dry_run", False),

        loglevel=cfg.get("loglevel", "info"),

        log_path=Path(
            f"logs/preprocess_array_{task_id}.log"
        ),
    )